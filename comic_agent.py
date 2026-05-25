"""
ComicWeaver - 简化版漫画生成 Agent 工作流
基于 LangGraph + ComfyUI + 智谱 GLM-4-Flash

Agent 流水线（5 个 Agent）:
  ScriptAgent          → 将用户输入扩充为结构化剧本（自然语言）
  CharacterPromptAgent → 将角色自然语言描述转换为 Danbooru Tag
  CharacterDesignAgent → 生成角色人设图（半身像/胸像，通过 generate_character.json）
  StoryboardAgent      → 设计分镜构图（镜头、姿势、场景等 Prompt）
  ImageAgent           → 生成最终分镜图（通过 generate_picture.json + IPAdapter）
"""
import os
import sys
import json
import time
import uuid
import urllib.parse
import requests
import random
import re
from typing import Any, Optional, Tuple
from datetime import datetime

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# ============================================================================
# 配置常量
# ============================================================================

ZHIPUAI_API_KEY = "2487ed1dfc9f456fbd5630be90a18575.kPSrC0ZoSbjHBwgL"
COMFYUI_SERVER = "127.0.0.1:8188"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
CHARACTER_DIR = os.path.join(OUTPUT_DIR, "characters")
PANEL_DIR = os.path.join(OUTPUT_DIR, "panels")
LOG_FILE = os.path.join(OUTPUT_DIR, "workflow_log.json")

CHARACTER_WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "generate_character.json")
PANEL_WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "generate_picture.json")

# 固定画风前缀（来自 generate_character.json 节点 38）
STYLE_PREFIX = "masterpiece, best quality, ultra high res, hyper-detailed, anime realism, 8K, newest"

# 负向 Prompt
NEGATIVE_CHARACTER = (
    "lowres, worst quality, bad quality, bad anatomy, simple background, blurry, "
    "cropped, cropped head, head out of frame, forehead cut off, top of head cut off, "
    "bad framing, bad composition, extra limbs, deformed hands, bad face, sketch, "
    "jpeg artifacts, signature, watermark, old, oldest, photorealistic, 3d render, realistic"
)
NEGATIVE_PANEL = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
    "signature, watermark, username, blurry, bad feet, mutation, deformed, extra limbs, "
    "extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, "
    "cross-eyed, mutated hands, polar lowres, bad face, photorealistic, 3d render, "
    "realistic, western comic style"
)


# ============================================================================
# 工具函数
# ============================================================================

def _try_repair_json(text: str) -> str:
    """尝试修复常见的 LLM JSON 格式错误（缺失逗号、多余逗号等）。"""
    # 1. } 或 ] 后面直接跟 "  → 缺失逗号
    repaired = re.sub(r'([}\]])\s*\n?\s*"', r'\1, "', text)
    # 2. " 后面直接跟 { → 缺失逗号
    repaired = re.sub(r'"\s*\n?\s*\{', r'", {', repaired)
    # 3. 数字后面直接跟 " → 缺失逗号
    repaired = re.sub(r'(\d)\s+\n?\s*"', r'\1, "', repaired)
    # 4. true/false/null 后面直接跟 " → 缺失逗号
    repaired = re.sub(r'(true|false|null)\s+\n?\s*"', r'\1, "', repaired)
    # 5. 数字后面跟 true/false/null → 缺失逗号
    repaired = re.sub(r'(\d)\s+(true|false|null)', r'\1, \2', repaired)
    # 6. " 后面跟 true/false/null → 缺失逗号
    repaired = re.sub(r'"\s+(true|false|null)', r'", \1', repaired)
    # 7. } 后面跟 { → 缺失逗号
    repaired = re.sub(r'\}\s*\n?\s*\{', r'}, {', repaired)
    # 8. ] 后跟 { / [ → 缺失逗号
    repaired = re.sub(r'\]\s*\n?\s*\{', r'], {', repaired)
    repaired = re.sub(r'\]\s*\n?\s*\[', r'], [', repaired)
    # 9. 多余尾逗号（在 } 或 ] 之前）→ 移除
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    return repaired


def extract_json(text: str) -> Any:
    """从 LLM 输出中提取 JSON，尽力修复格式错误，失败则抛出异常。"""
    # 尝试匹配 markdown 代码块中的 JSON
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        block = m.group(1).strip()
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            try:
                return json.loads(_try_repair_json(block))
            except json.JSONDecodeError:
                pass  # fall through to boundary search below

    # 找到最外层 JSON 边界（数组或对象），逐个尝试解析
    first_brace = text.find("{")
    first_bracket = text.find("[")
    candidates = []
    if first_bracket >= 0:
        candidates.append(("[", "]", first_bracket))
    if first_brace >= 0:
        candidates.append(("{", "}", first_brace))
    candidates.sort(key=lambda x: x[2])

    last_error = None
    for op, cl, start in candidates:
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == op:
                depth += 1
            elif text[i] == cl:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if depth == 0:
            chunk = text[start:end]
            # 先尝试直接解析
            try:
                return json.loads(chunk)
            except json.JSONDecodeError as e:
                last_error = e
                # 再尝试修复后解析
                try:
                    return json.loads(_try_repair_json(chunk))
                except json.JSONDecodeError:
                    continue

    if last_error:
        raise ValueError(
            f"JSON 解析失败: {last_error}\n"
            f"原始输出（前 800 字符）:\n{text[:800]}"
        )
    raise ValueError(f"无法从 LLM 输出中提取 JSON:\n{text[:500]}")


# ============================================================================
# API 客户端
# ============================================================================

class ZhipuAIHTTPClient:
    """智谱 AI HTTP 客户端。"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://open.bigmodel.cn/api/paas/v4"

    def chat(self, model: str = "glm-4-flash", messages: list = None,
             temperature: float = 0.7, max_tokens: int = 2048) -> dict:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        resp = requests.post(url, headers=headers, json=data, timeout=120)
        resp.raise_for_status()
        return resp.json()


class ComfyUIClient:
    """ComfyUI HTTP 客户端。"""

    def __init__(self, server: str = COMFYUI_SERVER):
        self.server = server
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, workflow: dict) -> dict:
        resp = requests.post(
            f"http://{self.server}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
            headers={"Content-Type": "application/json"}, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_history(self, prompt_id: str) -> dict:
        resp = requests.get(f"http://{self.server}/history/{prompt_id}", timeout=30)
        return resp.json()

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url = f"http://{self.server}/view?{urllib.parse.urlencode(params)}"
        resp = requests.get(url, timeout=30)
        return resp.content

    def upload_image(self, image_data: bytes, filename: str,
                     folder_type: str = "input", subfolder: str = "") -> dict:
        files = {
            "image": (filename, io.BytesIO(image_data), "image/png"),
            "type": (None, folder_type),
        }
        if subfolder:
            files["subfolder"] = (None, subfolder)
        resp = requests.post(f"http://{self.server}/upload/image", files=files, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def wait_for_image(self, prompt_id: str, timeout: int = 300) -> Tuple[bytes, str]:
        """轮询等待 ComfyUI 生成完成，返回 (image_data, filename)。"""
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                for node_output in history[prompt_id].get("outputs", {}).values():
                    if "images" in node_output and node_output["images"]:
                        info = node_output["images"][0]
                        return (
                            self.get_image(info["filename"], info.get("subfolder", ""), info.get("type", "output")),
                            info["filename"],
                        )
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "error" or status.get("completed", False):
                    raise RuntimeError(f"ComfyUI 执行失败: {status.get('messages', [])}")
            time.sleep(2)
        raise TimeoutError(f"ComfyUI 生成超时: {prompt_id}")

    def generate(self, workflow_path: str, overrides: dict, output_dir: str,
                 seed: Optional[int] = None) -> Tuple[bytes, str, str, int]:
        """
        执行 ComfyUI 工作流，返回 (image_data, filename, saved_path, seed)。
        overrides: {node_id: {input_key: value}}
        seed: 传入 None 则随机生成，否则使用指定种子（用于固定画风）。
        """
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        for node_id, inputs in overrides.items():
            if node_id in workflow:
                workflow[node_id]["inputs"].update(inputs)

        # 设置种子（传入或随机）
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        for nid in ("3", "25", "seed_node"):
            if nid in workflow and "seed" in workflow[nid].get("inputs", {}):
                workflow[nid]["inputs"]["seed"] = seed
                break

        prompt_id = self.queue_prompt(workflow)["prompt_id"]
        image_data, filename = self.wait_for_image(prompt_id)

        os.makedirs(output_dir, exist_ok=True)
        saved_path = os.path.join(output_dir, filename)
        with open(saved_path, "wb") as f:
            f.write(image_data)

        return image_data, filename, saved_path, seed


# ============================================================================
# LangGraph State
# ============================================================================

class ComicState(TypedDict):
    user_input: str
    target_pages: int
    target_panels_per_page: int
    comfyui_available: bool

    # Agent 输出
    script_data: dict                # ScriptAgent 输出
    character_prompts: list          # CharacterPromptAgent 输出: [{name, role, gender, age_range, core_tags, appearance_desc, personality}]
    character_images: list           # CharacterDesignAgent 输出: [{name, image_path, comfyui_input_filename}]
    panels: list                     # StoryboardAgent 输出: [{panel_id, page_id, shot_size, ...}]
    panel_images: dict               # ImageAgent 输出: {panel_id: image_path}


# ============================================================================
# 全局客户端（单例）
# ============================================================================

_llm = ZhipuAIHTTPClient(ZHIPUAI_API_KEY)
_comfyui: Optional[ComfyUIClient] = None
_log_entries: list = []
_log_stages: list = []       # 结构化阶段日志
_log_current_stage: dict = {}  # 当前阶段累积数据


def _log(msg: str) -> None:
    """控制台输出 + 简单日志。"""
    print(msg)
    _log_entries.append({"timestamp": datetime.now().isoformat(), "message": msg})


def _log_stage_begin(stage: str, agent: str) -> None:
    """开始一个结构化阶段。"""
    global _log_current_stage
    _log_current_stage = {
        "stage": stage,
        "agent": agent,
        "start_time": datetime.now().isoformat(),
    }


def _log_llm(model: str, temperature: float, max_tokens: int,
             system_prompt: str, user_prompt: str, response: str,
             parsed: Any = None, retry: bool = False) -> None:
    """记录一次完整的 LLM 调用。"""
    call = {
        "type": "llm_call",
        "retry": retry,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": response,
        "parsed_output": parsed,
    }
    _log_current_stage.setdefault("llm_calls", []).append(call)


def _log_comfyui(character: str, workflow_path: str, overrides: dict,
                 positive_prompt: str, negative_prompt: str,
                 output_file: str, duration_seconds: float,
                 success: bool, error: str = "") -> None:
    """记录一次完整的 ComfyUI 图像生成。"""
    gen = {
        "type": "comfyui_generation",
        "character": character,
        "workflow_path": os.path.basename(workflow_path),
        "overrides": overrides,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "output_file": output_file,
        "duration_seconds": round(duration_seconds, 1),
        "success": success,
        "error": error,
    }
    _log_current_stage.setdefault("generations", []).append(gen)


def _log_stage_end() -> None:
    """结束当前阶段，归档到 _log_stages。"""
    global _log_current_stage
    if _log_current_stage:
        _log_current_stage["end_time"] = datetime.now().isoformat()
        _log_stages.append(_log_current_stage)
    _log_current_stage = {}


def _save_log() -> None:
    """保存完整日志（含结构化阶段数据）。"""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "start_time": _log_stages[0]["start_time"] if _log_stages else "",
            "end_time": datetime.now().isoformat(),
            "stages": _log_stages,
        }, f, ensure_ascii=False, indent=2)


def _check_comfyui() -> bool:
    """检查 ComfyUI 是否运行，是则初始化全局客户端。"""
    global _comfyui
    try:
        resp = requests.get(f"http://{COMFYUI_SERVER}/system_stats", timeout=5)
        if resp.status_code == 200:
            _comfyui = ComfyUIClient()
            _log("[系统] ComfyUI 已连接")
            return True
    except Exception:
        pass
    _log("[系统] ComfyUI 未运行，跳过图像生成")
    return False


# ============================================================================
# Agent 1: ScriptAgent — 剧本解析
# ============================================================================

def script_node(state: ComicState) -> dict:
    """
    将用户输入的故事扩充为结构化剧本：
    - 标题、类型、基调、摘要
    - 角色列表（自然语言描述外貌、性格）
    - 场景列表（地点、时间、氛围、关键动作、建议分镜数）
    """
    _log("\n" + "─" * 50)
    _log("[ScriptAgent] 开始解析剧本...")
    _log_stage_begin("script", "ScriptAgent")

    system_prompt = """You are a professional manga script analyst.

Parse the user's story into structured JSON.  Output ONLY valid JSON — no extra text, no markdown fences.

Rules:
- Characters: natural language appearance (hair, eyes, skin, clothing, accessories), personality, backstory
- Scenes: break the story into key moments, estimate panels per scene
- NEVER output Stable Diffusion / Danbooru tags — those will be generated downstream

Required JSON structure:
{
  "title": "...",
  "genre": "action|romance|fantasy|sci-fi|slice-of-life|...",
  "tone": "dark|lighthearted|dramatic|mysterious|...",
  "summary": "...",
  "characters": [
    {
      "name": "...",
      "role": "protagonist|antagonist|supporting",
      "gender": "male|female",
      "age_range": "child|teenager|young_adult|adult|elder",
      "appearance": "detailed natural language (hair color, eye color, skin tone, clothing, accessories)",
      "personality": "...",
      "backstory": "..."
    }
  ],
  "scenes": [
    {
      "scene_id": 1,
      "location": "...",
      "time_of_day": "dawn|morning|afternoon|dusk|night",
      "weather": "...",
      "atmosphere": "...",
      "characters_present": ["name1", "name2"],
      "key_action": "...",
      "emotion_intensity": 0.5,
      "suggested_panels": 2
    }
  ]
}"""

    user_prompt = (
        f"Parse this story into manga script:\n\n{state['user_input']}\n\n"
        f"Target: {state['target_pages']} pages, ~{state['target_panels_per_page']} panels/page"
    )

    resp = _llm.chat(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.6, max_tokens=2500,
    )
    content = resp["choices"][0]["message"]["content"]
    script_data = extract_json(content)

    _log_llm(model="glm-4-flash", temperature=0.6, max_tokens=2500,
             system_prompt=system_prompt, user_prompt=user_prompt,
             response=content, parsed=script_data)

    _log(f"[ScriptAgent] 完成！标题: {script_data.get('title')}，"
         f"{len(script_data.get('characters', []))} 个角色，"
         f"{len(script_data.get('scenes', []))} 个场景")
    _log_stage_end()
    return {"script_data": script_data}


# ============================================================================
# Agent 2: CharacterPromptAgent — 角色 Tag 转换
# ============================================================================

def character_prompt_node(state: ComicState) -> dict:
    """
    将角色的自然语言外貌描述转换为 Danbooru 风格 Tag（不含画风/质量 Tag）。
    输出 core_tags 用于人设图和 IPAdapter 参考。
    """
    _log("\n" + "─" * 50)
    _log("[CharacterPromptAgent] 开始转换角色 Tag...")
    _log_stage_begin("character_prompt", "CharacterPromptAgent")

    characters = state["script_data"].get("characters", [])
    results = []

    system_prompt = """Convert character appearance from natural language into Danbooru-style tags.

Rules:
1. Start with "1girl" or "1boy" based on gender
2. Include age: "young adult", "teenager", etc.
3. Preserve exact colors: "silver hair", "golden eyes"
4. Include hair style, clothing with colors, accessories
5. Include skin tone if mentioned
6. NEVER include style tags (anime, realistic, masterpiece, quality, detailed, beautiful, etc.)
7. Output ONLY the comma-separated tag string — no explanations, no markdown

Example input:
"Young man with silver hair cut short, golden eyes, fair skin, wearing blue combat armor with silver trim and a blue cape"

Example output:
1boy, young adult, silver hair, short hair, golden eyes, fair skin, blue armor, silver trim, blue cape"""

    for char in characters:
        name = char.get("name", "Unknown")
        gender = char.get("gender", "male")
        _log(f"  → 转换: {name}")

        user_prompt = (
            f"Gender: {gender}, Age: {char.get('age_range', 'young_adult')}\n"
            f"Appearance: {char.get('appearance', '')}"
        )

        resp = _llm.chat(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.2, max_tokens=300,
        )
        core_tags = resp["choices"][0]["message"]["content"].strip()

        # 清理可能的 markdown 格式
        core_tags = re.sub(r"^[`'\"]+|[`'\"]+$", "", core_tags).strip()

        # 强制确保性别 Tag 正确：如果 LLM 没输出 1boy/1girl，根据 gender 补上
        if "1boy" not in core_tags and "1girl" not in core_tags:
            core_tags = ("1boy, " if gender == "male" else "1girl, ") + core_tags
        elif gender == "male" and "1girl" in core_tags:
            core_tags = core_tags.replace("1girl", "1boy")
        elif gender == "female" and "1boy" in core_tags:
            core_tags = core_tags.replace("1boy", "1girl")

        _log_llm(model="glm-4-flash", temperature=0.2, max_tokens=300,
                 system_prompt=system_prompt, user_prompt=user_prompt,
                 response=resp["choices"][0]["message"]["content"],
                 parsed={"name": name, "core_tags": core_tags})

        results.append({
            "name": name,
            "role": char.get("role", "supporting"),
            "gender": gender,
            "age_range": char.get("age_range", "young_adult"),
            "core_tags": core_tags,
            "appearance_desc": char.get("appearance", ""),
            "personality": char.get("personality", ""),
        })
        _log(f"  ✓ {name}: {core_tags}")

    _log(f"[CharacterPromptAgent] 完成！{len(results)} 个角色")
    _log_stage_end()
    return {"character_prompts": results}


# ============================================================================
# Agent 3: CharacterDesignAgent — 人设图生成
# ============================================================================

def character_design_node(state: ComicState) -> dict:
    """
    为每个角色生成人设图（半身像/胸像，确保头部完整出现在画面中）。
    使用 generate_character.json 工作流。
    """
    _log("\n" + "─" * 50)
    _log("[CharacterDesignAgent] 开始生成角色人设图...")
    _log_stage_begin("character_design", "CharacterDesignAgent")

    if not state["comfyui_available"] or _comfyui is None:
        _log("[CharacterDesignAgent] ComfyUI 未运行，跳过")
        _log_stage_end()
        return {"character_images": []}

    results = []
    for char in state["character_prompts"]:
        name = char["name"]
        gender = char.get("gender", "male")
        _log(f"  → 生成人设图: {name}")

        # 构建人设图 Prompt
        char_tags = char["core_tags"]
        # 根据性别加入强化 Tag
        if gender == "male":
            gender_boost = "male, masculine, flat chest"
        else:
            gender_boost = "female, feminine"
        composition = (
            "solo, portrait, head and shoulders, "
            "looking at viewer, entire head visible, full hair in frame, "
            "centered framing, ample headroom, "
            "simple background, white background, "
            "neutral expression, front view"
        )
        positive = f"{STYLE_PREFIX}, {char_tags}, {gender_boost}, {composition}"

        # 根据性别调整负向 Prompt
        if gender == "male":
            neg_prompt = NEGATIVE_CHARACTER + ", 1girl, female, breasts, feminine, girly"
        else:
            neg_prompt = NEGATIVE_CHARACTER

        _log(f"    Prompt: {positive[:150]}...")

        try:
            t0 = time.time()
            # 人设图使用随机种子（每次生成不同画风），但记录种子供分镜图复用
            char_seed = random.randint(0, 2**32 - 1)
            image_data, filename, saved_path, used_seed = _comfyui.generate(
                CHARACTER_WORKFLOW_PATH,
                overrides={
                    "6": {"text": positive},
                    "7": {"text": neg_prompt},
                    "5": {"width": 832, "height": 1216},
                },
                output_dir=CHARACTER_DIR,
                seed=char_seed,
            )
            dt = time.time() - t0

            # 上传到 ComfyUI input 目录供 IPAdapter 使用
            input_filename = f"char_{filename}"
            try:
                _comfyui.upload_image(image_data, input_filename, folder_type="input")
                _log(f"  ✓ 已上传: {input_filename}")
            except Exception as e:
                _log(f"  [WARN] 上传失败，使用原文件名: {e}")
                input_filename = filename

            _log_comfyui(character=name, workflow_path=CHARACTER_WORKFLOW_PATH,
                         overrides={"6": positive, "7": neg_prompt, "5": "832x1216",
                                    "seed": used_seed},
                         positive_prompt=positive, negative_prompt=neg_prompt,
                         output_file=saved_path, duration_seconds=dt, success=True)

            results.append({
                "name": name,
                "image_path": saved_path,
                "comfyui_input_filename": input_filename,
                "seed": used_seed,
            })
            _log(f"  ✓ 人设图保存: {saved_path}")

        except Exception as e:
            dt = time.time() - t0
            _log_comfyui(character=name, workflow_path=CHARACTER_WORKFLOW_PATH,
                         overrides={"6": positive, "7": neg_prompt, "5": "832x1216",
                                    "seed": char_seed},
                         positive_prompt=positive, negative_prompt=neg_prompt,
                         output_file="", duration_seconds=dt, success=False, error=str(e))
            _log(f"  ✗ 生成失败: {e}")
            results.append({
                "name": name,
                "image_path": "",
                "comfyui_input_filename": "",
                "seed": char_seed,
            })

    success = len([r for r in results if r["image_path"]])
    _log(f"[CharacterDesignAgent] 完成！{success}/{len(results)} 张")
    _log_stage_end()
    return {"character_images": results}


# ============================================================================
# Agent 4: StoryboardAgent — 分镜设计
# ============================================================================

def storyboard_node(state: ComicState) -> dict:
    """
    根据剧本设计分镜：
    - 镜头大小、机位角度、角色位置
    - 场景设置、氛围、光线
    - 画面中角色的动作
    输出每个分镜的结构化数据，供 ImageAgent 构建 Prompt。
    """
    _log("\n" + "─" * 50)
    _log("[StoryboardAgent] 开始设计分镜...")
    _log_stage_begin("storyboard", "StoryboardAgent")

    # 只读参考：角色名称和 Tag
    char_refs = [{"name": c["name"], "role": c["role"], "core_tags": c["core_tags"],
                  "gender": c.get("gender", "male")}
                 for c in state["character_prompts"]]

    system_prompt = """You are a professional manga storyboard director.

Design the cinematography for each panel.  Output ONLY valid JSON — no extra text.

For each panel provide ALL of:
- panel_id: "page.panel" (e.g., "1.1")
- page_id: integer
- shot_size: extreme_long | long | full | medium | close | extreme_close
- camera_angle: eye_level | high_angle | low_angle | dutch_angle
- characters_in_panel: [character names exactly matching provided names]
- primary_action: what characters DO (verbs only — e.g. "running forward", "drawing sword")
- pose_hint: character body pose description (e.g., "standing straight, right arm raised", "crouching, hands on ground")
- expression: character facial expression (e.g., "determined expression", "surprised, wide eyes")
- setting: background description in English tags (e.g., "ancient ruins, stone pillars, overgrown vines, moss")
- time_of_day: dawn|morning|afternoon|dusk|night
- weather: clear|cloudy|rainy|stormy|snowy|foggy
- atmosphere: (e.g., "tense", "peaceful", "dramatic", "mysterious")
- scene_lighting: specific lighting (e.g., "golden sunset backlight", "moonlight from window", "harsh overhead light")
- mood: tense|peaceful|dramatic|mysterious|joyful|sorrowful
- dialogue_hint: narrative hint (brief, 1 sentence)

Return: {"panels": [...]}"""

    user_prompt = (
        f"Title: {state['script_data'].get('title')}\n"
        f"Summary: {state['script_data'].get('summary', '')[:300]}\n"
        f"Characters: {json.dumps(char_refs, ensure_ascii=False)}\n"
        f"Scenes: {json.dumps(state['script_data'].get('scenes', []), ensure_ascii=False)}\n"
        f"Target: {state['target_pages']} pages, {state['target_panels_per_page']} panels/page"
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    resp = _llm.chat(messages=messages, temperature=0.7, max_tokens=3000)
    content = resp["choices"][0]["message"]["content"]

    retried = False
    try:
        data = extract_json(content)
    except (ValueError, json.JSONDecodeError) as e:
        _log(f"  [WARN] 首次 JSON 解析失败: {e}，发起重试...")
        retried = True
        # 追加纠错消息，要求 LLM 修复 JSON
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": (
                "Your JSON has a syntax error (likely a missing comma). "
                "Please fix the error and output ONLY the corrected valid JSON. "
                "Do NOT add any explanation or markdown — just the corrected JSON object."
            ),
        })
        resp = _llm.chat(messages=messages, temperature=0.3, max_tokens=3000)
        content = resp["choices"][0]["message"]["content"]
        data = extract_json(content)

    _log_llm(model="glm-4-flash", temperature=0.7 if not retried else 0.3,
             max_tokens=3000, system_prompt=system_prompt, user_prompt=user_prompt,
             response=content, parsed=data, retry=retried)

    panels = data.get("panels", data) if isinstance(data, dict) else data

    _log(f"[StoryboardAgent] 完成！{len(panels)} 个分镜")
    _log_stage_end()
    return {"panels": panels}


# ============================================================================
# Agent 5: ImageAgent — 分镜图生成
# ============================================================================

def _build_panel_positive(char_tags: str, panel: dict, chars_in_panel: list = None,
                         char_prompts_by_name: dict = None) -> str:
    """构建分镜图正向 Prompt，男性角色自动添加性别强化 Tag。"""
    # 镜头映射
    shot_map = {
        "extreme_long": "extreme wide shot, establishing shot",
        "long": "wide shot, full figure",
        "full": "full body shot",
        "medium": "medium shot, cowboy shot",
        "close": "close-up shot",
        "extreme_close": "extreme close-up, face focus",
    }
    angle_map = {
        "eye_level": "eye level",
        "high_angle": "from above, bird's eye view",
        "low_angle": "from below, low angle shot",
        "dutch_angle": "dutch angle",
    }
    mood_map = {
        "tense": "dynamic composition, motion lines",
        "peaceful": "serene composition, balanced framing",
        "dramatic": "dramatic composition, diagonal lines",
        "mysterious": "atmospheric composition, depth of field",
    }

    # 判断画面中角色性别：如果全是男性，加 male 强化 Tag
    gender_boost = ""
    if chars_in_panel and char_prompts_by_name:
        genders = []
        for n in chars_in_panel:
            info = char_prompts_by_name.get(n, {})
            if isinstance(info, dict):
                genders.append(info.get("gender", "male"))
            else:
                genders.append("male")
        if all(g == "male" for g in genders):
            gender_boost = "male, masculine, flat chest, "

    parts = [STYLE_PREFIX, gender_boost + char_tags]

    # 场景
    scene_parts = []
    for key in ("setting", "atmosphere", "scene_lighting"):
        v = panel.get(key, "")
        if v:
            scene_parts.append(v)
    if panel.get("weather", "") not in ("", "clear"):
        scene_parts.append(panel["weather"])
    if panel.get("time_of_day", ""):
        scene_parts.append(panel["time_of_day"])
    if scene_parts:
        parts.append(", ".join(scene_parts))

    # 构图
    comp_parts = [
        shot_map.get(panel.get("shot_size", "medium"), "medium shot"),
        angle_map.get(panel.get("camera_angle", "eye_level"), "eye level"),
    ]
    if panel.get("mood", "") in mood_map:
        comp_parts.append(mood_map[panel["mood"]])
    if panel.get("pose_hint", ""):
        comp_parts.append(panel["pose_hint"])
    if panel.get("expression", ""):
        comp_parts.append(panel["expression"])
    if panel.get("primary_action", ""):
        comp_parts.append(panel["primary_action"])
    parts.append(", ".join(comp_parts))

    return ", ".join(parts)


def image_node(state: ComicState) -> dict:
    """
    生成最终分镜图。
    使用 generate_picture.json 工作流 + IPAdapter（参考人设图）。
    """
    _log("\n" + "─" * 50)
    _log("[ImageAgent] 开始生成分镜图...")
    _log_stage_begin("image", "ImageAgent")

    if not state["comfyui_available"] or _comfyui is None:
        _log("[ImageAgent] ComfyUI 未运行，跳过")
        _log_stage_end()
        return {"panel_images": {}}

    # 筛选有人设图的角色（IPAdapter 参考源）+ 记录每个角色的种子
    chars_with_image = {}
    char_seeds = {}
    for c in state["character_images"]:
        if c["comfyui_input_filename"]:
            chars_with_image[c["name"]] = c["comfyui_input_filename"]
            char_seeds[c["name"]] = c.get("seed", None)

    if not chars_with_image:
        _log("[ImageAgent] 没有可用的人设图，跳过")
        _log_stage_end()
        return {"panel_images": {}}

    max_images = state["target_pages"] * state["target_panels_per_page"]
    panels = state["panels"][:max_images]
    _log(f"[ImageAgent] 将生成 {len(panels)} 张分镜图")

    panel_images = {}
    # 保留完整角色信息（含 gender）用于性别判断
    char_infos_by_name = {
        c["name"]: {"core_tags": c["core_tags"], "gender": c.get("gender", "male")}
        for c in state["character_prompts"]
    }
    char_prompts_by_name = {n: info["core_tags"] for n, info in char_infos_by_name.items()}

    # 轮转计数：多角色面板中轮流担任主体，让每个角色都有出场机会
    char_round_robin: dict = {}  # key: tuple(sorted(chars)), value: next index

    for i, panel in enumerate(panels):
        panel_id = panel.get("panel_id", f"panel_{i:03d}")

        # 每张图只选一个角色：多角色时轮流担任主体
        chars_in_panel = panel.get("characters_in_panel", [])
        chars_available = [n for n in chars_in_panel if n in char_prompts_by_name]
        if len(chars_available) > 1:
            key = tuple(sorted(chars_available))
            idx = char_round_robin.get(key, 0) % len(chars_available)
            char_round_robin[key] = idx + 1
            primary_char = chars_available[idx]
        elif len(chars_available) == 1:
            primary_char = chars_available[0]
        else:
            primary_char = None

        if primary_char and primary_char in char_prompts_by_name:
            char_tags = f"solo, {char_prompts_by_name[primary_char]}"
            primary_gender = (char_infos_by_name.get(primary_char, {}).get("gender", "male")
                              if isinstance(char_infos_by_name.get(primary_char), dict) else "male")
        else:
            char_tags = "solo, 1person"
            primary_gender = "male"

        # 判断主体角色是否为男性
        all_male = (primary_gender == "male")

        # IPAdapter 参考图：只使用主体角色的人设图
        ref_filename = chars_with_image.get(primary_char) if primary_char else None
        if not ref_filename:
            ref_filename = next(iter(chars_with_image.values()))

        # 使用该角色人设图时的种子，保证画风一致
        panel_seed = char_seeds.get(primary_char) if primary_char else None

        # 单角色构图（仅用一个角色的 Tag 避免冲突）
        single_char_list = [primary_char] if primary_char else []
        positive = _build_panel_positive(char_tags, panel,
                                         chars_in_panel=single_char_list,
                                         char_prompts_by_name=char_infos_by_name)
        neg_prompt = NEGATIVE_PANEL
        if all_male:
            neg_prompt += ", 1girl, female, breasts, feminine, girly, woman"

        multi_info = ""
        if len(chars_in_panel) > 1:
            multi_info = f" ({len(chars_in_panel)} roles, round-robin → {primary_char})"
        _log(f"\n  [{i+1}/{len(panels)}] {panel_id}{multi_info}")
        _log(f"    Prompt: {positive[:150]}...")
        _log(f"    参考图: {ref_filename}  |  种子: {panel_seed}")

        try:
            t0 = time.time()
            _, filename, saved_path, used_seed = _comfyui.generate(
                PANEL_WORKFLOW_PATH,
                overrides={
                    "6": {"text": positive},
                    "7": {"text": neg_prompt},
                    "5": {"width": 1024, "height": 1024},
                    "12": {"image": ref_filename},
                    "18": {"weight": 0.35, "weight_faceidv2": 0.4},
                },
                output_dir=PANEL_DIR,
                seed=panel_seed,
            )
            dt = time.time() - t0

            _log_comfyui(character=panel_id, workflow_path=PANEL_WORKFLOW_PATH,
                         overrides={"6": positive, "7": neg_prompt, "5": "1024x1024",
                                    "12": ref_filename, "18": "weight=0.35, weight_faceidv2=0.4",
                                    "seed": used_seed},
                         positive_prompt=positive, negative_prompt=neg_prompt,
                         output_file=saved_path, duration_seconds=dt, success=True)

            panel_images[panel_id] = saved_path
            _log(f"  ✓ 保存: {saved_path}")
        except Exception as e:
            dt = time.time() - t0
            _log_comfyui(character=panel_id, workflow_path=PANEL_WORKFLOW_PATH,
                         overrides={"6": positive, "7": neg_prompt, "5": "1024x1024",
                                    "12": ref_filename, "18": "weight=0.35, weight_faceidv2=0.4"},
                         positive_prompt=positive, negative_prompt=neg_prompt,
                         output_file="", duration_seconds=dt, success=False, error=str(e))
            _log(f"  ✗ 失败: {e}")
            panel_images[panel_id] = f"error: {e}"

    success = len([v for v in panel_images.values() if not str(v).startswith("error")])
    _log(f"\n[ImageAgent] 完成！{success}/{len(panels)} 张")
    _log_stage_end()
    return {"panel_images": panel_images}


# ============================================================================
# 工作流构建
# ============================================================================

def build_workflow():
    """构建 LangGraph 工作流。"""
    graph = StateGraph(ComicState)

    graph.add_node("script", script_node)
    graph.add_node("character_prompt", character_prompt_node)
    graph.add_node("character_design", character_design_node)
    graph.add_node("storyboard", storyboard_node)
    graph.add_node("image", image_node)

    graph.add_edge(START, "script")
    graph.add_edge("script", "character_prompt")
    graph.add_edge("character_prompt", "character_design")
    graph.add_edge("character_design", "storyboard")
    graph.add_edge("storyboard", "image")
    graph.add_edge("image", END)

    return graph.compile()


# ============================================================================
# 主入口
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("ComicWeaver - 简化版 Agent 工作流")
    print("=" * 60)

    user_input = (
        "一个关于魔法少女的奇幻故事。主角小樱是一个活泼的少女，粉色长发，碧绿眼眸，"
        "穿着华丽的粉色魔法裙，手持星星魔杖。"
        "她在月夜下遇到了神秘的黑发少女小夜（黑色长发，紫色瞳孔，穿着深蓝巫女服）。"
        "两人联手对抗黑暗势力，最终用友谊的力量守护了魔法世界。"
    )
    target_pages = 2
    target_panels_per_page = 3

    comfyui_ok = _check_comfyui()

    initial_state: ComicState = {
        "user_input": user_input,
        "target_pages": target_pages,
        "target_panels_per_page": target_panels_per_page,
        "comfyui_available": comfyui_ok,
        "script_data": {},
        "character_prompts": [],
        "character_images": [],
        "panels": [],
        "panel_images": {},
    }

    workflow = build_workflow()
    result = workflow.invoke(initial_state)

    # 输出摘要
    print("\n" + "=" * 60)
    print("执行完成")
    print("=" * 60)
    sd = result.get("script_data", {})
    print(f"标题: {sd.get('title', 'N/A')}")
    print(f"画风前缀: {STYLE_PREFIX}")
    print(f"角色:")
    for c in result.get("character_prompts", []):
        img = next((ci for ci in result.get("character_images", []) if ci["name"] == c["name"]), {})
        has_img = "✓" if img.get("image_path") else "✗"
        print(f"  {has_img} {c['name']}: {c['core_tags']}")
    panels = result.get("panels", [])
    panel_imgs = result.get("panel_images", {})
    print(f"分镜: {len(panels)} 个设计 / {len(panel_imgs)} 张图")

    _save_log()
    print(f"\n日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
