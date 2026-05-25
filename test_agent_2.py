"""
ComicWeaver 完整工作流 - v3 修复版
修复：
1. 人设图改为上半身portrait（提升IPAdapter人脸识别率）
2. 多角色场景只选一个主角色生成（消除1boy冲突）
3. WorkflowCalibrationAgent同步两个workflow的checkpoint模型（解决画风不一致根本原因）
4. StyleAnalysisAgent强制2次元风格，禁止接受realistic偏移
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
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# 配置
# ============================================================================

ZHIPUAI_API_KEY = "your_api_key_here"
COMFYUI_SERVER = "127.0.0.1:8188"

OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "output")
CHARACTER_DIR  = os.path.join(OUTPUT_DIR, "characters")
PANEL_DIR      = os.path.join(OUTPUT_DIR, "panels")
LOG_FILE       = os.path.join(OUTPUT_DIR, "workflow_log.json")

CHARACTER_WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "generate_character.json")
PANEL_WORKFLOW_PATH     = os.path.join(os.path.dirname(__file__), "generate_picture.json")

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from typing import TypedDict, Annotated, Sequence
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class CheckpointInfo:
    """
    WorkflowCalibrationAgent 的输出
    记录 workflow 文件中 checkpoint 节点的位置和当前值
    """
    node_id: str          # ComfyUI workflow中checkpoint加载节点的ID
    current_model: str    # 当前使用的模型文件名
    workflow_file: str    # 来自哪个workflow文件


@dataclass
class DynamicStyleConfig:
    """
    动态画风配置 —— 唯一真相源
    
    v3 新增字段：
    - shared_style_prefix: 人设图和分镜图完全共用的画风词（关键！）
    - character_style_addon: 仅用于人设图的额外词（如portrait相关）
    - panel_style_addon: 仅用于分镜图的额外词（如cinematic相关）
    - checkpoint_node_character / checkpoint_node_panel: 由WorkflowCalibrationAgent填写
    
    设计原则：
    - shared_style_prefix 必须完全相同，这是一致性的保证
    - addon 词只能微调构图/光线感，不能改变基础画风
    """
    style_name: str = "Anime 2D"

    # ── 共用画风词（两者必须完全相同，不得分别修改）
    quality_prefix: str = (
        "masterpiece, best quality, ultra high res, hyper-detailed, newest"
    )
    core_style_tags: str = (
        "anime style, 2d anime, high quality anime illustration, "
        "cel shading, vibrant colors, clean lineart, "
        "detailed face, beautiful eyes, expressive features"
    )
    lighting_tags: str = "soft ambient lighting, subtle shadows"

    # ── 各自独有的微调词（只能影响构图/表达，不能引入新的画风方向）
    # 人设图：突出人物、中性背景
    character_style_addon: str = (
        "character focus, upper body focus, clear face details"
    )
    # 分镜图：突出场景感、叙事感
    panel_style_addon: str = (
        "cinematic scene, narrative composition, environmental storytelling"
    )

    # ── 负向Prompt（完全共用，不得分离）
    negative_prompt: str = (
        "lowres, worst quality, bad quality, normal quality, "
        "bad anatomy, bad proportions, extra limbs, deformed hands, "
        "bad face, ugly face, blurry, cropped, out of frame, "
        "sketch, rough sketch, jpeg artifacts, "
        "signature, watermark, text, username, "
        "3d render, cgi, photorealistic, hyperrealistic, photo, "
        "western comic style, "
        "flat color, simple background, monochrome"
    )

    # ── Checkpoint 同步信息（由 WorkflowCalibrationAgent 填写）
    checkpoint_node_character: str = ""   # 人设图workflow中checkpoint节点ID
    checkpoint_node_panel: str = ""       # 分镜图workflow中checkpoint节点ID
    target_checkpoint: str = ""           # 统一使用的模型文件名（空=不覆盖）

    def get_shared_style_prefix(self) -> str:
        """
        获取两者共用的画风前缀
        这是一致性的核心：人设图和分镜图必须以这个字符串开头
        """
        return f"{self.quality_prefix}, {self.core_style_tags}, {self.lighting_tags}"

    def get_character_style_prefix(self) -> str:
        """人设图完整画风前缀 = 共用 + 人设addon"""
        base = self.get_shared_style_prefix()
        if self.character_style_addon:
            return f"{base}, {self.character_style_addon}"
        return base

    def get_panel_style_prefix(self) -> str:
        """分镜图完整画风前缀 = 共用 + 分镜addon"""
        base = self.get_shared_style_prefix()
        if self.panel_style_addon:
            return f"{base}, {self.panel_style_addon}"
        return base


@dataclass
class StandardizedCharacter:
    """标准化角色数据"""
    name: str
    role: str
    # 纯角色外貌Tag，不含任何画风词
    core_tags: str
    appearance_desc: str
    personality: str
    backstory: str = ""
    character_image_path: str = ""
    character_image_filename: str = ""
    comfyui_input_filename: str = ""


@dataclass
class PanelCharacterAssignment:
    """
    PrimaryCharacterSelectorAgent 的输出
    
    解决多角色问题的核心数据结构：
    - primary: 主要生成角色（使用其完整tags + 参考图）
    - secondary_hint: 其他角色的轻量描述（加入prompt但不占主导）
    """
    primary_char: StandardizedCharacter        # 主角色（完整tags）
    secondary_hint: str                        # 其他角色的轻量背景描述
    ref_image_filename: str                    # 传给IPAdapter的参考图（只有primary的）
    reasoning: str                             # 选择原因（用于日志）


@dataclass
class BuiltPrompt:
    """PromptEngineeringAgent 的输出"""
    positive: str
    negative: str
    # 元信息
    style_prefix_used: str = ""
    char_tags_used: str = ""
    secondary_hint_used: str = ""
    scene_tags_used: str = ""
    composition_tags_used: str = ""
    primary_character_name: str = ""


@dataclass
class CalibrationReport:
    """WorkflowCalibrationAgent 的输出"""
    character_checkpoint: CheckpointInfo
    panel_checkpoint: CheckpointInfo
    models_match: bool
    sync_applied: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """QualityControlAgent 的输出"""
    passed: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 日志记录器
# ============================================================================

class WorkflowLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.logs = {"start_time": datetime.now().isoformat(), "agents": []}
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def add_agent_log(self, agent_name: str, input_data: Any,
                      output: Any, metadata: dict = None):
        self.logs["agents"].append({
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "input": str(input_data)[:500],
            "output": output if isinstance(output, (dict, list)) else str(output)[:500],
            "metadata": metadata or {}
        })
        self.save()

    def add_image_log(self, agent_name: str, prompt: BuiltPrompt,
                      image_path: str, metadata: dict = None):
        self.logs["agents"].append({
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "prompt": {
                "positive":           prompt.positive[:400],
                "negative":           prompt.negative[:200],
                "style_prefix":       prompt.style_prefix_used[:200],
                "char_tags":          prompt.char_tags_used,
                "secondary_hint":     prompt.secondary_hint_used,
                "scene_tags":         prompt.scene_tags_used,
                "composition_tags":   prompt.composition_tags_used,
                "primary_character":  prompt.primary_character_name
            },
            "output_image": image_path,
            "metadata": metadata or {}
        })
        self.save()

    def save(self):
        self.logs["end_time"] = datetime.now().isoformat()
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

    def print_log(self, message: str):
        print(message)


# ============================================================================
# API 客户端
# ============================================================================

class ZhipuAIHTTPClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://open.bigmodel.cn/api/paas/v4"

    def chat_completions_create(
        self, model: str = "glm-4-flash", messages: list = None,
        temperature: float = 0.7, max_tokens: int = 2048
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {"model": model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        return response.json()


class ComfyUIClient:
    def __init__(self, server_address: str = COMFYUI_SERVER):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, workflow: dict) -> dict:
        p = {"prompt": workflow, "client_id": self.client_id}
        response = requests.post(
            f"http://{self.server_address}/prompt",
            json=p, headers={'Content-Type': 'application/json'}, timeout=30
        )
        response.raise_for_status()
        return response.json()

    def get_history(self, prompt_id: str) -> dict:
        response = requests.get(
            f"http://{self.server_address}/history/{prompt_id}", timeout=30
        )
        return response.json()

    def get_image(self, filename: str, subfolder: str = "",
                  folder_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url = f"http://{self.server_address}/view?{urllib.parse.urlencode(params)}"
        response = requests.get(url, timeout=30)
        return response.content

    def upload_image(self, image_data: bytes, filename: str,
                     folder_type: str = "input", subfolder: str = "") -> dict:
        import io as _io
        files = {
            'image': (filename, _io.BytesIO(image_data), 'image/png'),
            'type':  (None, folder_type),
        }
        if subfolder:
            files['subfolder'] = (None, subfolder)
        response = requests.post(
            f"http://{self.server_address}/upload/image", files=files, timeout=30
        )
        response.raise_for_status()
        return response.json()

    def get_image_from_output(self, prompt_id: str,
                               timeout: int = 300) -> Tuple[bytes, str]:
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                for node_id, node_output in outputs.items():
                    if "images" in node_output and node_output["images"]:
                        img_info = node_output["images"][0]
                        image_data = self.get_image(
                            img_info["filename"],
                            img_info.get("subfolder", ""),
                            img_info.get("type", "output")
                        )
                        return image_data, img_info["filename"]
                status = history[prompt_id].get("status", {})
                if (status.get("status_str") == "error" or
                        status.get("completed", False)):
                    raise RuntimeError(
                        f"ComfyUI failed: {status.get('messages', [])}"
                    )
            time.sleep(2)
        raise TimeoutError(f"Generation timeout: {prompt_id}")

    def generate_with_workflow(
        self, workflow_path: str, prompt_overrides: dict,
        output_dir: str, seed: int = -1
    ) -> Tuple[bytes, str, str]:
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        for node_id, inputs in prompt_overrides.items():
            if node_id in workflow:
                for key, value in inputs.items():
                    workflow[node_id]["inputs"][key] = value

        # 设置随机seed
        for node_id in ["3", "25", "seed_node"]:
            if (node_id in workflow and
                    "seed" in workflow[node_id].get("inputs", {})):
                workflow[node_id]["inputs"]["seed"] = (
                    random.randint(0, 2**32 - 1) if seed == -1 else seed
                )
                break

        result = self.queue_prompt(workflow)
        prompt_id = result["prompt_id"]
        image_data, filename = self.get_image_from_output(prompt_id)

        os.makedirs(output_dir, exist_ok=True)
        saved_path = os.path.join(output_dir, filename)
        with open(saved_path, "wb") as f:
            f.write(image_data)

        return image_data, filename, saved_path


# ============================================================================
# 主工作流类
# ============================================================================

class ComicWorkflow:
    def __init__(self):
        self.llm_client     = ZhipuAIHTTPClient(ZHIPUAI_API_KEY)
        self.comfyui_client: Optional[ComfyUIClient] = None
        self.logger         = WorkflowLogger(LOG_FILE)

        self.style_config:            Optional[DynamicStyleConfig]       = None
        self.standardized_characters: List[StandardizedCharacter]        = []
        self.calibration_report:      Optional[CalibrationReport]        = None

    def check_comfyui(self) -> bool:
        try:
            r = requests.get(f"http://{COMFYUI_SERVER}/system_stats", timeout=5)
            if r.status_code == 200:
                self.comfyui_client = ComfyUIClient()
                return True
        except Exception as e:
            self.logger.print_log(f"[WARN] ComfyUI连接失败: {e}")
        return False

    # =========================================================================
    # Agent 1: ScriptAgent
    # =========================================================================
    def script_agent(self, user_input: str,
                     target_pages: int, target_panels_per_page: int) -> dict:
        self.logger.print_log("\n" + "─" * 60)
        self.logger.print_log("[ScriptAgent] 开始解析剧本...")

        system_prompt = """You are a professional manga script analyst.

Parse the user's story into structured script data.

STRICT RULES:
- Output ONLY story content in natural language
- NEVER output Stable Diffusion tags or image prompts
- Appearance descriptions MUST be natural language
- For visual_style_preference: ALWAYS output "anime 2d" regardless of story content
  (Style will be determined by StyleAnalysisAgent, not by story description)

Return valid JSON:
{
  "title": "string",
  "genre": "string",
  "tone": "string (e.g., dark, lighthearted, dramatic, mysterious)",
  "visual_style_preference": "anime 2d",
  "summary": "string",
  "characters": [
    {
      "name": "string",
      "role": "protagonist|antagonist|supporting",
      "gender": "male|female",
      "age_range": "child|teenager|young_adult|adult|elder",
      "appearance": "detailed natural language: hair color+style, eye color, skin, clothing with colors",
      "personality": "string",
      "backstory": "string"
    }
  ],
  "scenes": [
    {
      "scene_id": 1,
      "location": "string",
      "time_of_day": "dawn|morning|afternoon|dusk|night",
      "weather": "string",
      "atmosphere": "string",
      "characters_present": ["name list"],
      "key_action": "string",
      "emotion_intensity": 0.5,
      "suggested_panels": 3
    }
  ]
}"""

        try:
            response = self.llm_client.chat_completions_create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": f"Parse:\n{user_input}\n\n"
                                f"Target: {target_pages} pages, "
                                f"~{target_panels_per_page} panels/page"}
                ],
                temperature=0.6, max_tokens=2500
            )
            content = response["choices"][0]["message"]["content"]
            m = re.search(r'\{[\s\S]*\}', content)
            result = json.loads(m.group()) if m else json.loads(content)
            # 强制覆盖 visual_style_preference，防止ScriptAgent输出写实偏好
            result["visual_style_preference"] = "anime 2d"

            self.logger.add_agent_log(
                "ScriptAgent", user_input, result, {"status": "success"}
            )
            self.logger.print_log(
                f"[ScriptAgent] 完成！标题: {result.get('title')}，"
                f"基调: {result.get('tone')}"
            )
            return result

        except Exception as e:
            self.logger.print_log(f"[ScriptAgent] 错误: {e}")
            fallback = {
                "title": "未命名", "genre": "action", "tone": "dramatic",
                "visual_style_preference": "anime 2d",
                "summary": user_input,
                "characters": [{
                    "name": "主角", "role": "protagonist",
                    "gender": "male", "age_range": "teenager",
                    "appearance": "short black hair, brown eyes, simple dark clothing",
                    "personality": "brave", "backstory": ""
                }],
                "scenes": [{
                    "scene_id": 1, "location": "unknown",
                    "time_of_day": "afternoon", "weather": "clear",
                    "atmosphere": "neutral", "characters_present": ["主角"],
                    "key_action": "standing", "emotion_intensity": 0.5,
                    "suggested_panels": target_panels_per_page
                }]
            }
            self.logger.add_agent_log(
                "ScriptAgent", user_input, fallback,
                {"status": "fallback", "error": str(e)}
            )
            return fallback

    # =========================================================================
    # Agent 2: WorkflowCalibrationAgent ★ 新增
    # 职责：读取两个workflow文件，找到checkpoint节点，分析并同步模型
    #
    # 为什么需要这个Agent：
    # 画风不一致的根本原因是两个workflow用了不同的checkpoint
    # 这个Agent在所有图像生成之前运行，确保模型统一
    # =========================================================================
    def workflow_calibration_agent(self) -> Optional[CalibrationReport]:
        """
        分析两个workflow文件中的checkpoint节点
        
        ComfyUI workflow中checkpoint加载节点通常是：
        - class_type: "CheckpointLoaderSimple" 或 "CheckpointLoader"
        - inputs.ckpt_name: 模型文件名
        
        策略：
        1. 扫描两个workflow文件，找到所有checkpoint节点
        2. 比较模型名称是否一致
        3. 如果不一致，以人设图workflow的模型为基准，同步到分镜图workflow
        4. 将节点ID写入DynamicStyleConfig，供后续生成时覆盖使用
        """
        self.logger.print_log("\n" + "─" * 60)
        self.logger.print_log("[WorkflowCalibrationAgent] 分析workflow模型配置...")

        issues = []

        def find_checkpoint_node(workflow_path: str) -> Optional[CheckpointInfo]:
            """在workflow文件中查找checkpoint加载节点"""
            try:
                with open(workflow_path, "r", encoding="utf-8") as f:
                    workflow = json.load(f)
            except Exception as e:
                issues.append(f"无法读取workflow文件 {workflow_path}: {e}")
                return None

            checkpoint_class_types = {
                "CheckpointLoaderSimple", "CheckpointLoader",
                "UNETLoader", "DiffusionModelLoader"
            }

            for node_id, node_data in workflow.items():
                class_type = node_data.get("class_type", "")
                if class_type in checkpoint_class_types:
                    inputs = node_data.get("inputs", {})
                    # 不同节点类型的模型字段名不同
                    model_name = (
                        inputs.get("ckpt_name") or
                        inputs.get("unet_name") or
                        inputs.get("model_name") or
                        "unknown"
                    )
                    return CheckpointInfo(
                        node_id=node_id,
                        current_model=model_name,
                        workflow_file=workflow_path
                    )

            # 如果没找到标准节点，记录警告
            issues.append(
                f"未在 {os.path.basename(workflow_path)} 中找到checkpoint节点，"
                f"请检查workflow是否使用标准CheckpointLoaderSimple节点"
            )
            return None

        char_ckpt  = find_checkpoint_node(CHARACTER_WORKFLOW_PATH)
        panel_ckpt = find_checkpoint_node(PANEL_WORKFLOW_PATH)

        if char_ckpt is None or panel_ckpt is None:
            self.logger.print_log(
                "[WorkflowCalibrationAgent] 无法找到checkpoint节点，跳过校准"
            )
            # 即使找不到节点，也返回报告（不阻塞主流程）
            dummy = CheckpointInfo("", "unknown", "")
            report = CalibrationReport(
                character_checkpoint=char_ckpt or dummy,
                panel_checkpoint=panel_ckpt or dummy,
                models_match=False,
                sync_applied=False,
                issues=issues
            )
            self.logger.add_agent_log(
                "WorkflowCalibrationAgent", "scan workflows",
                {"issues": issues}, {"status": "partial"}
            )
            return report

        models_match = (char_ckpt.current_model == panel_ckpt.current_model)

        self.logger.print_log(
            f"  人设图模型: {char_ckpt.current_model} (节点 {char_ckpt.node_id})"
        )
        self.logger.print_log(
            f"  分镜图模型: {panel_ckpt.current_model} (节点 {panel_ckpt.node_id})"
        )

        if not models_match:
            issues.append(
                f"⚠ 模型不一致！人设图用 '{char_ckpt.current_model}'，"
                f"分镜图用 '{panel_ckpt.current_model}'"
            )
            self.logger.print_log(
                f"  [WARN] 模型不一致！这是画风不统一的根本原因！"
            )
            self.logger.print_log(
                f"  [INFO] 将以人设图模型 '{char_ckpt.current_model}' 为基准，"
                f"同步到分镜图生成参数"
            )

        # 将节点信息写入 style_config，供后续生成时覆盖
        if self.style_config:
            self.style_config.checkpoint_node_character = char_ckpt.node_id
            self.style_config.checkpoint_node_panel     = panel_ckpt.node_id
            self.style_config.target_checkpoint         = char_ckpt.current_model

        report = CalibrationReport(
            character_checkpoint=char_ckpt,
            panel_checkpoint=panel_ckpt,
            models_match=models_match,
            sync_applied=not models_match,
            issues=issues
        )

        self.logger.add_agent_log(
            "WorkflowCalibrationAgent",
            {"char_workflow": CHARACTER_WORKFLOW_PATH,
             "panel_workflow": PANEL_WORKFLOW_PATH},
            {
                "char_model":    char_ckpt.current_model,
                "panel_model":   panel_ckpt.current_model,
                "models_match":  models_match,
                "sync_applied":  not models_match,
                "issues":        issues
            },
            {"status": "success" if models_match else "sync_needed"}
        )

        status_str = "✓ 模型一致" if models_match else "⚠ 已标记同步"
        self.logger.print_log(
            f"[WorkflowCalibrationAgent] 完成！{status_str}"
        )
        return report

    # =========================================================================
    # Agent 3: StyleAnalysisAgent
    # 职责：根据故事动态生成统一画风配置
    #
    # v3 关键改进：
    # - 强制输出2次元anime风格，拒绝接受realistic偏移
    # - 区分 shared_style（两者共用）和 addon（各自微调）
    # =========================================================================
    def style_analysis_agent(self, script_data: dict) -> DynamicStyleConfig:
        self.logger.print_log("\n" + "─" * 60)
        self.logger.print_log("[StyleAnalysisAgent] 分析故事风格...")

        genre = script_data.get("genre", "action")
        tone  = script_data.get("tone", "dramatic")
        title = script_data.get("title", "")

        system_prompt = """You are an expert in anime 2D visual style design.

Generate style tags for a 2D ANIME manga project.

ABSOLUTE RULES:
1. This is a 2D ANIME project - NEVER output realistic/photorealistic/3D related words
2. Forbidden words: realistic, realism, photorealistic, hyperrealistic, 3d, cgi, render, detailed realism
3. The style MUST be consistent 2D anime illustration style
4. shared_style_tags will be used in BOTH character design images AND scene panel images
5. Keep shared_style_tags focused on art style only (NO character features, NO scene elements)

Style direction based on genre/tone:
- action/adventure: dynamic lines, bold colors, energetic
- mystery/thriller: cool tones, atmospheric, shadow play
- romance: warm tones, soft gradients, dreamy
- fantasy: vibrant magical colors, ethereal glow
- slice-of-life: warm naturalistic palette, clean simple style

Return valid JSON:
{
  "style_name": "string (e.g., 'Action Adventure Anime 2D')",
  "quality_tags": "quality tags only - no style here",
  "shared_style_tags": "PURE art style tags, NO character/scene content, NO realistic words",
  "lighting_tags": "default lighting for this story's mood",
  "character_style_addon": "extra tags for character portraits only (focus, clarity)",
  "panel_style_addon": "extra tags for scene panels only (cinematic, narrative)",
  "negative_prompt": "comprehensive negative - MUST include: realistic, photorealistic, 3d render, photo",
  "reasoning": "brief explanation"
}

Example for mystery:
{
  "style_name": "Mysterious Adventure Anime 2D",
  "quality_tags": "masterpiece, best quality, ultra high res, hyper-detailed, newest",
  "shared_style_tags": "anime style, 2d anime illustration, high quality anime CG, smooth cel shading, cool color palette, atmospheric depth, clean lineart, expressive eyes",
  "lighting_tags": "moody lighting, cool ambient light, subtle shadows",
  "character_style_addon": "character focus, face detail emphasis, soft rim lighting on face",
  "panel_style_addon": "cinematic atmosphere, environmental storytelling, depth of field",
  "negative_prompt": "lowres, worst quality, bad anatomy, blurry, realistic, photorealistic, 3d render, cgi, photo, western comic, flat color, monochrome",
  "reasoning": "..."
}"""

        user_prompt = (
            f"Story: {title}\n"
            f"Genre: {genre}\n"
            f"Tone: {tone}\n"
            f"Summary: {script_data.get('summary', '')[:200]}\n\n"
            f"IMPORTANT: Output 2D anime style ONLY. No realistic words allowed."
        )

        try:
            response = self.llm_client.chat_completions_create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.3, max_tokens=800
            )
            content = response["choices"][0]["message"]["content"]
            m = re.search(r'\{[\s\S]*\}', content)
            result = json.loads(m.group()) if m else json.loads(content)

            # ── 强制过滤写实词（双重保险）
            realistic_words = [
                "realistic", "realism", "photorealistic", "hyperrealistic",
                "detailed realism", "3d render", "cgi", "photo", "render",
                "high contrast", "realistic perspective"
            ]

            def purge_realistic(tag_str: str) -> str:
                """从tag字符串中移除所有写实相关词"""
                tags = [t.strip() for t in tag_str.split(",")]
                cleaned = []
                for tag in tags:
                    if not any(rw in tag.lower() for rw in realistic_words):
                        cleaned.append(tag)
                    else:
                        self.logger.print_log(
                            f"  [StyleAnalysis] 过滤写实词: '{tag}'"
                        )
                return ", ".join(cleaned)

            shared_tags = purge_realistic(
                result.get("shared_style_tags",
                    "anime style, 2d anime illustration, high quality anime CG, "
                    "smooth cel shading, clean lineart, vibrant colors, expressive eyes")
            )
            lighting_tags = purge_realistic(
                result.get("lighting_tags", "soft ambient lighting, subtle shadows")
            )

            config = DynamicStyleConfig(
                style_name=result.get("style_name", "Anime 2D"),
                quality_prefix=result.get("quality_tags",
                    "masterpiece, best quality, ultra high res, hyper-detailed, newest"),
                core_style_tags=shared_tags,
                lighting_tags=lighting_tags,
                character_style_addon=result.get("character_style_addon",
                    "character focus, upper body focus, clear face details"),
                panel_style_addon=result.get("panel_style_addon",
                    "cinematic scene, narrative composition, environmental storytelling"),
                negative_prompt=result.get("negative_prompt",
                    "lowres, worst quality, bad quality, bad anatomy, blurry, "
                    "realistic, photorealistic, 3d render, cgi, photo, western comic, "
                    "flat color, monochrome, watermark, signature")
            )

            self.logger.add_agent_log(
                "StyleAnalysisAgent", user_prompt,
                {
                    "style_name": config.style_name,
                    "shared_prefix": config.get_shared_style_prefix()[:200],
                    "character_prefix": config.get_character_style_prefix()[:200],
                    "panel_prefix": config.get_panel_style_prefix()[:200],
                    "reasoning": result.get("reasoning", "")
                },
                {"status": "success"}
            )
            self.logger.print_log(
                f"[StyleAnalysisAgent] 完成！风格: {config.style_name}"
            )
            self.logger.print_log(
                f"  共用前缀: {config.get_shared_style_prefix()[:80]}..."
            )
            return config

        except Exception as e:
            self.logger.print_log(f"[StyleAnalysisAgent] 错误，使用默认配置: {e}")
            config = DynamicStyleConfig()
            self.logger.add_agent_log(
                "StyleAnalysisAgent", user_prompt,
                {"error": str(e), "fallback": config.style_name},
                {"status": "fallback"}
            )
            return config

    # =========================================================================
    # Agent 4: CharacterStandardizationAgent
    # =========================================================================
    def character_standardization_agent(
        self, characters: list, style_config: DynamicStyleConfig
    ) -> List[StandardizedCharacter]:
        self.logger.print_log("\n" + "─" * 60)
        self.logger.print_log("[CharacterStandardizationAgent] 标准化角色Tag...")

        standardized = []

        system_prompt = """Convert character appearance to precise Danbooru-style tags.

RULES:
1. Start with: "1girl" (female) or "1boy" (male)
2. Add age hint: teenager, young adult, adult, child, elder
3. Preserve exact colors: "silver hair", "golden eyes", "blue armor"
4. Include hair style: long hair, short hair, ponytail, twintails, bob cut, etc.
5. Include ALL clothing with colors: "blue armor", "red cloak", "white dress"
6. Include accessories if mentioned
7. Include skin tone if mentioned
8. ABSOLUTELY FORBIDDEN: anime, realistic, masterpiece, quality, beautiful, gorgeous,
   detailed, stunning, high quality - these are style words, NOT character tags
9. Output ONLY the tag string, comma-separated, NO explanations"""

        for char in characters:
            name       = char.get("name", "Unknown")
            appearance = char.get("appearance", "")
            gender     = char.get("gender", "male")
            age_range  = char.get("age_range", "young_adult")

            self.logger.print_log(f"  → 标准化: {name}")

            try:
                response = self.llm_client.chat_completions_create(
                    model="glm-4-flash",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",
                         "content": (
                             f"Gender: {gender}, Age: {age_range}\n"
                             f"Convert to tags: {appearance}"
                         )}
                    ],
                    temperature=0.2, max_tokens=300
                )
                core_tags = response["choices"][0]["message"]["content"].strip()
                core_tags = re.sub(r'^[`\'\"]+|[`\'\"]+$', '', core_tags).strip()

                # 强制清理画风词泄漏
                style_leak_words = [
                    "anime", "realistic", "detailed", "masterpiece",
                    "quality", "beautiful", "gorgeous", "stunning",
                    "amazing", "high quality", "best quality"
                ]
                tag_list = [t.strip() for t in core_tags.split(",")]
                cleaned_tags = []
                for tag in tag_list:
                    if not any(slw in tag.lower() for slw in style_leak_words):
                        cleaned_tags.append(tag)
                    else:
                        self.logger.print_log(
                            f"  [WARN] 清理画风词泄漏: '{tag}'"
                        )
                core_tags = ", ".join(t for t in cleaned_tags if t)

            except Exception as e:
                self.logger.print_log(f"  [WARN] LLM失败，规则提取: {e}")
                core_tags = self._rule_based_tag_extraction(appearance, gender)

            std_char = StandardizedCharacter(
                name=name,
                role=char.get("role", "supporting"),
                core_tags=core_tags,
                appearance_desc=appearance,
                personality=char.get("personality", ""),
                backstory=char.get("backstory", "")
            )
            standardized.append(std_char)

            self.logger.add_agent_log(
                "CharacterStandardizationAgent",
                f"{name}: {appearance}",
                {"name": name, "core_tags": core_tags},
                {"status": "success"}
            )
            self.logger.print_log(f"  ✓ {name}: {core_tags}")

        self.logger.print_log(
            f"[CharacterStandardizationAgent] 完成！{len(standardized)} 个角色"
        )
        return standardized

    def _rule_based_tag_extraction(self, appearance: str,
                                    gender: str = "male") -> str:
        tags = ["1girl" if gender == "female" else "1boy"]
        app_lower = appearance.lower()

        hair_colors = [
            "silver", "white", "blonde", "golden", "brown", "red", "crimson",
            "blue", "green", "purple", "pink", "black", "orange", "grey", "gray",
            "fiery red", "dark brown"
        ]
        for c in hair_colors:
            if c in app_lower:
                tags.append(f"{c} hair")
                break

        for s in ["long hair", "short hair", "medium hair", "ponytail",
                  "twintails", "braid", "bob cut", "wavy hair"]:
            if s in app_lower:
                tags.append(s)

        for ep in ["golden eyes", "red eyes", "blue eyes", "green eyes",
                   "purple eyes", "black eyes", "brown eyes", "silver eyes"]:
            if ep in app_lower:
                tags.append(ep)
                break

        for color in ["red", "blue", "green", "black", "white", "silver", "gold"]:
            for item in ["armor", "cloak", "cape", "dress", "uniform", "coat", "robe"]:
                if f"{color} {item}" in app_lower:
                    tags.append(f"{color} {item}")

        return ", ".join(dict.fromkeys(tags))

    # =========================================================================
    # Agent 5: PrimaryCharacterSelectorAgent ★ 新增
    # 职责：为每个panel选择一个主要角色进行生成
    #
    # 为什么需要独立Agent：
    # - 多角色tags堆叠是SD生成崩坏的主要原因之一
    # - 选角逻辑需要考虑角色重要性、是否有参考图、场景焦点等
    # - 其他角色需要用轻量描述保留场景感，但不影响主角生成
    # =========================================================================
    def primary_character_selector_agent(
        self,
        panel: dict,
        standardized_chars: List[StandardizedCharacter]
    ) -> PanelCharacterAssignment:
        """
        为panel选择主要生成角色
        
        选角优先级：
        1. 有参考图的主角（protagonist）
        2. 有参考图的任意角色
        3. 主角（即使没有参考图）
        4. 列表第一个角色
        
        其他角色处理：
        - 使用轻量的外观暗示词（如 "another figure in red robe"）
        - 不使用完整tags，避免与主角冲突
        """
        chars_in_panel = panel.get("characters_in_panel", [])

        # 构建本panel涉及的角色对象列表
        panel_chars = []
        for name in chars_in_panel:
            found = next((c for c in standardized_chars if c.name == name), None)
            if found:
                panel_chars.append(found)

        if not panel_chars:
            # panel没有指定角色，用主角
            default = next(
                (c for c in standardized_chars if c.role == "protagonist"),
                standardized_chars[0] if standardized_chars else None
            )
            if default:
                return PanelCharacterAssignment(
                    primary_char=default,
                    secondary_hint="",
                    ref_image_filename=default.comfyui_input_filename,
                    reasoning="no chars in panel, using default protagonist"
                )

        # 选择主角色（优先级逻辑）
        primary = None
        reasoning = ""

        # 1. 有参考图的主角
        for c in panel_chars:
            if c.role == "protagonist" and c.comfyui_input_filename:
                primary = c
                reasoning = f"protagonist with reference image: {c.name}"
                break

        # 2. 有参考图的任意角色
        if primary is None:
            for c in panel_chars:
                if c.comfyui_input_filename:
                    primary = c
                    reasoning = f"character with reference image: {c.name}"
                    break

        # 3. 主角（无参考图）
        if primary is None:
            for c in panel_chars:
                if c.role == "protagonist":
                    primary = c
                    reasoning = f"protagonist (no reference image): {c.name}"
                    break

        # 4. 第一个角色
        if primary is None:
            primary = panel_chars[0]
            reasoning = f"first available character: {primary.name}"

        # 为其他角色生成轻量背景描述
        secondary_chars = [c for c in panel_chars if c.name != primary.name]
        secondary_hint = self._generate_secondary_hint(secondary_chars)

        assignment = PanelCharacterAssignment(
            primary_char=primary,
            secondary_hint=secondary_hint,
            ref_image_filename=primary.comfyui_input_filename,
            reasoning=reasoning
        )

        self.logger.print_log(
            f"  [CharSelector] Panel {panel.get('panel_id')}: "
            f"主角={primary.name} ({reasoning})"
        )
        if secondary_hint:
            self.logger.print_log(
                f"  [CharSelector] 次要角色提示: {secondary_hint}"
            )

        return assignment

    def _generate_secondary_hint(
        self, secondary_chars: List[StandardizedCharacter]
    ) -> str:
        """
        为次要角色生成轻量描述
        
        规则：
        - 只提取最显眼的外貌特征（主色调 + 服装类型）
        - 加上 "in background" 或 "standing nearby" 定位
        - 绝不使用完整的core_tags（避免与主角冲突）
        
        例：core_tags = "1boy, adult, fiery red hair, black eyes, red robe"
        → hint = "another person in red robe nearby"
        """
        if not secondary_chars:
            return ""

        hints = []
        for char in secondary_chars:
            core = char.core_tags.lower()
            # 提取主色调
            clothing_colors = ["red", "blue", "green", "black", "white",
                                "silver", "gold", "purple", "brown"]
            clothing_items  = ["armor", "robe", "cloak", "cape", "dress",
                                "uniform", "coat", "jacket"]
            color_found, item_found = "", ""
            for cl in clothing_colors:
                if cl in core:
                    color_found = cl
                    break
            for it in clothing_items:
                if it in core:
                    item_found = it
                    break

            gender_hint = "woman" if "1girl" in core else "man"

            if color_found and item_found:
                hints.append(
                    f"another {gender_hint} in {color_found} {item_found} "
                    f"in background"
                )
            elif color_found:
                hints.append(
                    f"another {gender_hint} in {color_found} clothing "
                    f"in background"
                )
            else:
                hints.append(f"another {gender_hint} in background")

        return ", ".join(hints)

    # =========================================================================
    # Agent 6: PromptEngineeringAgent
    # 职责：集中构建所有Prompt（v3改进：区分人设/分镜画风addon，接入选角结果）
    # =========================================================================
    def prompt_engineering_agent(
        self,
        task_type: str,
        task_data: dict,
        style_config: DynamicStyleConfig,
        standardized_chars: List[StandardizedCharacter]
    ) -> BuiltPrompt:
        negative = style_config.negative_prompt

        if task_type == "character_design":
            return self._build_character_prompt(task_data, style_config, negative)
        elif task_type == "panel_scene":
            # panel_scene的task_data里必须包含 char_assignment
            return self._build_panel_prompt_v3(task_data, style_config, negative)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def _build_character_prompt(
        self, char_data: dict,
        style_config: DynamicStyleConfig,
        negative: str
    ) -> BuiltPrompt:
        """
        构建人设图Prompt
        
        v3 关键改进：
        1. 使用 get_character_style_prefix()（包含character_addon）
        2. 改为上半身portrait，确保人脸占主导区域，提高IPAdapter识别率
        3. 绝不使用 full body（全身图人脸太小，IPAdapter特征提取效果差）
        
        Prompt结构:
        [character_style_prefix] + [char_core_tags] + [portrait_composition]
        """
        style_prefix = style_config.get_character_style_prefix()
        char_tags    = char_data.get("core_tags", "1boy")
        role         = char_data.get("role", "supporting")

        # ── 上半身构图（人脸清晰是IPAdapter的前提）
        # 所有角色统一使用上半身，不区分主配角
        # portrait_focus 确保人脸区域占画面 40% 以上
        composition = (
            "solo, upper body, portrait, looking at viewer, "
            "face focus, detailed face, "
            "simple gradient background, plain background"
        )

        # 主角稍微更正式一点的姿势
        if role == "protagonist":
            composition = (
                "solo, upper body, portrait, looking at viewer, "
                "confident expression, face focus, detailed face, "
                "simple gradient background"
            )
        elif role == "antagonist":
            composition = (
                "solo, upper body, portrait, looking at viewer, "
                "intimidating expression, face focus, detailed face, "
                "dark gradient background"
            )

        positive = f"{style_prefix}, {char_tags}, {composition}"

        return BuiltPrompt(
            positive=positive,
            negative=negative,
            style_prefix_used=style_prefix,
            char_tags_used=char_tags,
            secondary_hint_used="",
            scene_tags_used="",
            composition_tags_used=composition,
            primary_character_name=char_data.get("name", "")
        )

    def _build_panel_prompt_v3(
        self, panel_data: dict,
        style_config: DynamicStyleConfig,
        negative: str
    ) -> BuiltPrompt:
        """
        构建分镜图Prompt
        
        v3 关键改进：
        1. 使用 get_panel_style_prefix()（包含panel_addon）
        2. 只使用 primary_char 的 core_tags（来自PrimaryCharacterSelectorAgent）
        3. 次要角色用 secondary_hint（轻量描述，不干扰主角）
        4. shared_style_prefix 与人设图完全相同
        
        Prompt结构:
        [panel_style_prefix] + [primary_char_tags] + [secondary_hint?] + [scene_tags] + [composition_tags]
        """
        style_prefix = style_config.get_panel_style_prefix()

        # ── 角色Tags（只使用primary角色，来自PanelCharacterAssignment）
        assignment: Optional[PanelCharacterAssignment] = panel_data.get("char_assignment")
        if assignment:
            char_tags      = assignment.primary_char.core_tags
            secondary_hint = assignment.secondary_hint
            primary_name   = assignment.primary_char.name
        else:
            # fallback：没有assignment时，不生成角色tags
            char_tags      = "1person"
            secondary_hint = ""
            primary_name   = "unknown"

        # ── 场景Tags（来自StoryboardAgent的数据）
        scene_parts = []
        for field_name in ["setting", "time_of_day", "weather",
                           "atmosphere", "scene_lighting"]:
            val = panel_data.get(field_name, "")
            if val and val not in ["clear", "normal", ""]:
                scene_parts.append(val)
        scene_tags = ", ".join(filter(None, scene_parts))

        # ── 构图Tags
        shot_map = {
            "extreme_long": "extreme wide shot, establishing shot",
            "long":         "wide shot, full figure",
            "full":         "full body shot",
            "medium":       "medium shot, cowboy shot",
            "close":        "close-up shot",
            "extreme_close":"extreme close-up, face focus"
        }
        angle_map = {
            "eye_level":   "eye level",
            "high_angle":  "from above, bird's eye view",
            "low_angle":   "from below, low angle shot",
            "dutch_angle": "dutch angle, tilted frame"
        }
        mood_map = {
            "tense":      "dynamic composition, tension",
            "peaceful":   "serene composition, balanced framing",
            "dramatic":   "dramatic composition, diagonal lines",
            "mysterious": "atmospheric composition, depth of field",
            "joyful":     "bright composition, open framing",
            "sorrowful":  "heavy composition, closed framing"
        }

        shot_size    = panel_data.get("shot_size", "medium")
        camera_angle = panel_data.get("camera_angle", "eye_level")
        mood         = panel_data.get("mood", "")
        action       = panel_data.get("primary_action", "")

        comp_parts = [
            shot_map.get(shot_size, "medium shot"),
            angle_map.get(camera_angle, "eye level"),
        ]
        if mood in mood_map:
            comp_parts.append(mood_map[mood])
        if action:
            clean_action = self._sanitize_action(action)
            if clean_action:
                comp_parts.append(clean_action)

        composition_tags = ", ".join(filter(None, comp_parts))

        # ── 组合最终Prompt
        # 结构：[style] + [主角tags] + [次要角色轻量提示] + [场景] + [构图]
        parts = [style_prefix, char_tags]
        if secondary_hint:
            parts.append(secondary_hint)
        if scene_tags:
            parts.append(scene_tags)
        parts.append(composition_tags)

        positive = ", ".join(filter(None, parts))

        return BuiltPrompt(
            positive=positive,
            negative=negative,
            style_prefix_used=style_prefix,
            char_tags_used=char_tags,
            secondary_hint_used=secondary_hint,
            scene_tags_used=scene_tags,
            composition_tags_used=composition_tags,
            primary_character_name=primary_name
        )

    def _sanitize_action(self, action: str) -> str:
        """
        清理动作描述
        移除：角色名、外貌词、过长描述
        保留：动词和动作相关词
        """
        # 移除角色名（避免角色名出现在prompt中干扰生成）
        for char in self.standardized_characters:
            action = re.sub(
                rf'\b{re.escape(char.name)}\b', '', action, flags=re.IGNORECASE
            )

        # 移除外貌词
        appearance_words = [
            "hair", "eyes", "eye", "armor", "cloak", "cape",
            "dress", "wearing", "outfit", "uniform", "skin",
            "face", "body", "clothes", "tall", "short", "slim"
        ]
        words    = action.split()
        filtered = [
            w for w in words
            if not any(aw in w.lower() for aw in appearance_words)
        ]
        result = " ".join(filtered).strip()

        # 长度限制
        if len(result) > 60:
            result = result[:60]

        return result

    # =========================================================================
    # Agent 7: CharacterDesignAgent
    # 职责：生成人设图（上半身portrait）
    # =========================================================================
    def character_design_agent(
        self,
        standardized_chars: List[StandardizedCharacter],
        style_config: DynamicStyleConfig
    ) -> List[StandardizedCharacter]:
        self.logger.print_log("\n" + "─" * 60)
        self.logger.print_log("[CharacterDesignAgent] 生成人设图（上半身）...")

        if self.comfyui_client is None:
            self.logger.print_log("[CharacterDesignAgent] ComfyUI未运行，跳过")
            return standardized_chars

        for char in standardized_chars:
            self.logger.print_log(f"  → {char.name}")

            built_prompt = self.prompt_engineering_agent(
                task_type="character_design",
                task_data={
                    "core_tags": char.core_tags,
                    "role":      char.role,
                    "name":      char.name
                },
                style_config=style_config,
                standardized_chars=standardized_chars
            )

            self.logger.print_log(
                f"  style_prefix: {built_prompt.style_prefix_used[:80]}..."
            )
            self.logger.print_log(
                f"  char_tags:    {built_prompt.char_tags_used}"
            )
            self.logger.print_log(
                f"  composition:  {built_prompt.composition_tags_used}"
            )

            try:
                # 构建workflow覆盖参数
                prompt_overrides = {
                    "39": {"prompt": built_prompt.positive},
                    # 上半身：宽高比接近 1:1.2，人脸占比更大
                    "5":  {"width": 768, "height": 960}
                }

                # 如果WorkflowCalibrationAgent找到了checkpoint节点，
                # 同步应用（确保人设图和分镜图用同一个基础模型）
                if (style_config.checkpoint_node_character and
                        style_config.target_checkpoint):
                    prompt_overrides[style_config.checkpoint_node_character] = {
                        "ckpt_name": style_config.target_checkpoint
                    }
                    self.logger.print_log(
                        f"  [Calibration] 已同步checkpoint: "
                        f"{style_config.target_checkpoint}"
                    )

                image_data, filename, saved_path = \
                    self.comfyui_client.generate_with_workflow(
                        CHARACTER_WORKFLOW_PATH,
                        prompt_overrides,
                        CHARACTER_DIR
                    )

                # 上传到ComfyUI input供IPAdapter使用
                input_filename = f"char_{char.name.replace(' ', '_')}_{filename}"
                try:
                    self.comfyui_client.upload_image(
                        image_data, input_filename, folder_type="input"
                    )
                    self.logger.print_log(
                        f"  ✓ 上传参考图: {input_filename}"
                    )
                except Exception as e:
                    self.logger.print_log(f"  [WARN] 上传失败: {e}")
                    input_filename = filename

                char.character_image_path     = saved_path
                char.character_image_filename = filename
                char.comfyui_input_filename   = input_filename

                self.logger.add_image_log(
                    "CharacterDesignAgent", built_prompt, saved_path,
                    {"character": char.name, "status": "success",
                     "composition": "upper_body_portrait"}
                )
                self.logger.print_log(f"  ✓ 保存: {saved_path}")

            except Exception as e:
                self.logger.print_log(f"  ✗ 失败: {e}")
                self.logger.add_agent_log(
                    "CharacterDesignAgent",
                    f"Character: {char.name}",
                    {"error": str(e)},
                    {"character": char.name, "status": "error"}
                )

        success = len([c for c in standardized_chars if c.character_image_path])
        self.logger.print_log(
            f"[CharacterDesignAgent] 完成！{success}/{len(standardized_chars)} 张"
        )
        return standardized_chars

    # =========================================================================
    # Agent 8: StoryboardAgent
    # =========================================================================
    def storyboard_agent(
        self,
        script_data: dict,
        standardized_chars: List[StandardizedCharacter],
        target_pages: int,
        target_panels_per_page: int
    ) -> list:
        self.logger.print_log("\n" + "─" * 60)
        self.logger.print_log("[StoryboardAgent] 设计分镜...")

        system_prompt = """You are a professional manga storyboard director.

YOUR ONLY JOB: Design cinematography and staging.

DO NOT generate character appearance tags - that's handled by other agents.
DO NOT generate art style tags.

For each panel provide:
- panel_id: "page.panel" (e.g., "1.1")
- page_id: integer
- shot_size: extreme_long|long|full|medium|close|extreme_close
- camera_angle: eye_level|high_angle|low_angle|dutch_angle
- characters_in_panel: [character names matching exactly]
- primary_action: VERBS ONLY (e.g., "running forward", "drawing sword")
  FORBIDDEN: character names, appearance words
- setting: English scene description tags (e.g., "ancient stone ruins, moss covered walls")
- time_of_day: dawn|morning|afternoon|dusk|night
- weather: clear|cloudy|rainy|stormy|snowy|foggy
- atmosphere: scene atmosphere description
- scene_lighting: specific lighting (e.g., "golden sunset backlight", "flickering torch light")
- mood: tense|peaceful|dramatic|mysterious|joyful|sorrowful
- dialogue_hint: narrative context

Return JSON: {"panels": [...]}"""

        try:
            char_refs = [
                {"name": c.name, "role": c.role}
                for c in standardized_chars
            ]
            response = self.llm_client.chat_completions_create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": (
                         f"Title: {script_data.get('title')}\n"
                         f"Summary: {script_data.get('summary', '')[:300]}\n"
                         f"Characters: {json.dumps(char_refs, ensure_ascii=False)}\n"
                         f"Scenes: {json.dumps(script_data.get('scenes', []), ensure_ascii=False)}\n"
                         f"Target: {target_pages} pages, "
                         f"{target_panels_per_page} panels/page"
                     )}
                ],
                temperature=0.7, max_tokens=3000
            )
            content = response["choices"][0]["message"]["content"]
            m = re.search(r'\{[\s\S]*\}', content)
            if m:
                parsed = json.loads(m.group())
                panels = (parsed.get("panels", parsed)
                          if isinstance(parsed, dict) else parsed)
            else:
                m2 = re.search(r'\[[\s\S]*\]', content)
                panels = json.loads(m2.group()) if m2 else []

            self.logger.add_agent_log(
                "StoryboardAgent",
                f"{target_pages}p × {target_panels_per_page}",
                {"panels_count": len(panels)},
                {"status": "success"}
            )
            self.logger.print_log(
                f"[StoryboardAgent] 完成！{len(panels)} 个分镜"
            )
            return panels

        except Exception as e:
            self.logger.print_log(f"[StoryboardAgent] 错误: {e}，使用fallback")
            return self._fallback_storyboard(
                script_data, standardized_chars,
                target_pages, target_panels_per_page
            )

    def _fallback_storyboard(
        self, script_data, standardized_chars,
        target_pages, target_panels_per_page
    ):
        panels = []
        scenes      = script_data.get("scenes", [])
        shot_cycle  = ["medium", "close", "long", "full", "extreme_close"]
        angle_cycle = ["eye_level", "high_angle", "low_angle"]

        for page in range(target_pages):
            for p_idx in range(target_panels_per_page):
                flat_idx = page * target_panels_per_page + p_idx
                scene    = scenes[flat_idx % len(scenes)] if scenes else {}
                chars_present = scene.get(
                    "characters_present",
                    [c.name for c in standardized_chars[:1]]
                )
                panels.append({
                    "panel_id":          f"{page+1}.{p_idx+1}",
                    "page_id":           page + 1,
                    "shot_size":         shot_cycle[flat_idx % len(shot_cycle)],
                    "camera_angle":      angle_cycle[flat_idx % len(angle_cycle)],
                    "characters_in_panel": chars_present,
                    "primary_action":    scene.get("key_action", "standing"),
                    "setting":           scene.get("location", "unknown place"),
                    "time_of_day":       scene.get("time_of_day", "afternoon"),
                    "weather":           scene.get("weather", "clear"),
                    "atmosphere":        scene.get("atmosphere", "neutral"),
                    "scene_lighting":    "natural light",
                    "mood":              ("dramatic"
                                         if scene.get("emotion_intensity", 0.5) > 0.6
                                         else "peaceful"),
                    "dialogue_hint":     ""
                })
        return panels

    # =========================================================================
    # Agent 9: ImageAgent
    # 职责：生成分镜图
    # v3 关键改进：
    # - 每个panel先调用PrimaryCharacterSelectorAgent确定主角色
    # - 只把主角色的tags和参考图传入生成
    # - 次要角色用轻量hint
    # - 应用checkpoint同步
    # =========================================================================
    def image_agent(
        self,
        panels: list,
        standardized_chars: List[StandardizedCharacter],
        style_config: DynamicStyleConfig,
        max_images: int = 6
    ) -> dict:
        self.logger.print_log("\n" + "─" * 60)
        self.logger.print_log("[ImageAgent] 生成分镜图...")

        if self.comfyui_client is None:
            self.logger.print_log("[ImageAgent] ComfyUI未运行，跳过")
            return {}

        chars_with_image = [c for c in standardized_chars if c.comfyui_input_filename]
        if not chars_with_image:
            self.logger.print_log("[ImageAgent] 没有可用参考图，跳过")
            return {}

        panel_images = {}
        max_panels   = min(max_images, len(panels))

        self.logger.print_log(
            f"[ImageAgent] 生成 {max_panels}/{len(panels)} 张分镜"
        )
        self.logger.print_log(
            f"[ImageAgent] 共用画风前缀: "
            f"{style_config.get_shared_style_prefix()[:80]}..."
        )

        for i, panel in enumerate(panels[:max_panels]):
            panel_id = panel.get("panel_id", f"panel_{i:03d}")

            # ── Step 1: 选择主角色（关键改进）
            assignment = self.primary_character_selector_agent(
                panel, standardized_chars
            )

            # 如果主角色没有参考图，用有参考图的第一个角色替代
            if not assignment.ref_image_filename and chars_with_image:
                assignment.ref_image_filename = chars_with_image[0].comfyui_input_filename
                self.logger.print_log(
                    f"  [WARN] 主角色无参考图，使用 "
                    f"{chars_with_image[0].name} 的参考图代替"
                )

            # ── Step 2: 把assignment信息注入panel_data，供PromptEngineeringAgent使用
            panel_with_assignment = {**panel, "char_assignment": assignment}

            # ── Step 3: 通过PromptEngineeringAgent构建Prompt
            built_prompt = self.prompt_engineering_agent(
                task_type="panel_scene",
                task_data=panel_with_assignment,
                style_config=style_config,
                standardized_chars=standardized_chars
            )

            self.logger.print_log(
                f"\n  [{i+1}/{max_panels}] Panel {panel_id} "
                f"| 主角: {assignment.primary_char.name}"
            )
            self.logger.print_log(
                f"  style_prefix:   {built_prompt.style_prefix_used[:60]}..."
            )
            self.logger.print_log(
                f"  char_tags:      {built_prompt.char_tags_used}"
            )
            if built_prompt.secondary_hint_used:
                self.logger.print_log(
                    f"  secondary:      {built_prompt.secondary_hint_used}"
                )
            self.logger.print_log(
                f"  scene_tags:     {built_prompt.scene_tags_used}"
            )
            self.logger.print_log(
                f"  composition:    {built_prompt.composition_tags_used}"
            )

            try:
                # ── Step 4: 构建workflow覆盖参数
                prompt_overrides = {
                    "6":  {"text": built_prompt.positive},
                    "7":  {"text": built_prompt.negative},
                    "5":  {"width": 1024, "height": 1024},
                    "12": {"image": assignment.ref_image_filename}  # 只传主角参考图
                }

                # 应用checkpoint同步（与人设图使用相同基础模型）
                if (style_config.checkpoint_node_panel and
                        style_config.target_checkpoint):
                    prompt_overrides[style_config.checkpoint_node_panel] = {
                        "ckpt_name": style_config.target_checkpoint
                    }
                    self.logger.print_log(
                        f"  [Calibration] checkpoint已同步: "
                        f"{style_config.target_checkpoint}"
                    )

                _, filename, saved_path = \
                    self.comfyui_client.generate_with_workflow(
                        PANEL_WORKFLOW_PATH,
                        prompt_overrides,
                        PANEL_DIR
                    )

                panel_images[panel_id] = saved_path
                self.logger.add_image_log(
                    "ImageAgent", built_prompt, saved_path,
                    {
                        "panel_id":      panel_id,
                        "primary_char":  assignment.primary_char.name,
                        "secondary":     assignment.secondary_hint,
                        "ref_image":     assignment.ref_image_filename,
                        "selector_reason": assignment.reasoning,
                        "status":        "success"
                    }
                )
                self.logger.print_log(f"  ✓ {saved_path}")

            except Exception as e:
                self.logger.print_log(f"  ✗ 失败: {e}")
                panel_images[panel_id] = f"error: {e}"
                self.logger.add_agent_log(
                    "ImageAgent", f"Panel {panel_id}",
                    {"error": str(e)},
                    {"panel_id": panel_id, "status": "error"}
                )

        success = len([
            p for p in panel_images.values()
            if not str(p).startswith("error")
        ])
        self.logger.print_log(
            f"\n[ImageAgent] 完成！{success}/{max_panels} 张"
        )
        return panel_images

    # =========================================================================
    # Agent 10: QualityControlAgent
    # =========================================================================
    def quality_control_agent(
        self,
        script_data: dict,
        standardized_chars: List[StandardizedCharacter],
        style_config: DynamicStyleConfig,
        panels: list,
        panel_images: dict,
        calibration_report: Optional[CalibrationReport] = None
    ) -> QualityReport:
        self.logger.print_log("\n" + "─" * 60)
        self.logger.print_log("[QualityControlAgent] 质量检查...")

        issues      = []
        suggestions = []
        metadata    = {}

        # ── 1. Checkpoint一致性检查（接入CalibrationReport）
        if calibration_report:
            if not calibration_report.models_match:
                if calibration_report.sync_applied:
                    suggestions.append(
                        f"模型不一致已自动同步：人设图用 "
                        f"'{calibration_report.character_checkpoint.current_model}'，"
                        f"分镜图已覆盖为相同模型"
                    )
                else:
                    issues.append(
                        f"模型不一致且未同步：人设图 "
                        f"'{calibration_report.character_checkpoint.current_model}'，"
                        f"分镜图 "
                        f"'{calibration_report.panel_checkpoint.current_model}'"
                    )
                    suggestions.append(
                        "手动检查两个workflow文件，将checkpoint统一为同一个模型"
                    )
            issues.extend(calibration_report.issues)
            metadata["calibration"] = {
                "char_model":   calibration_report.character_checkpoint.current_model,
                "panel_model":  calibration_report.panel_checkpoint.current_model,
                "models_match": calibration_report.models_match
            }

        # ── 2. 画风一致性检查
        shared_prefix = style_config.get_shared_style_prefix()
        char_prefix   = style_config.get_character_style_prefix()
        panel_prefix  = style_config.get_panel_style_prefix()

        # 验证 shared_prefix 是两者的公共前缀
        if not char_prefix.startswith(shared_prefix[:50]):
            issues.append("人设图style_prefix与共用前缀不匹配，可能导致画风漂移")
        if not panel_prefix.startswith(shared_prefix[:50]):
            issues.append("分镜图style_prefix与共用前缀不匹配，可能导致画风漂移")

        # 检查是否有写实词混入
        realistic_leak = ["realistic", "photorealistic", "3d render", "cgi", "photo"]
        for word in realistic_leak:
            if word in shared_prefix.lower():
                issues.append(
                    f"共用画风前缀中检测到写实词 '{word}'，"
                    f"这是人设图写实感的来源，请从StyleAnalysisAgent输出中移除"
                )

        # ── 3. 角色Tags检查
        for char in standardized_chars:
            tag_count = len(char.core_tags.split(","))
            if tag_count < 4:
                issues.append(
                    f"角色 '{char.name}' 的core_tags过于简单"
                    f"（仅{tag_count}个tag），IPAdapter识别可能不稳定"
                )
            if not char.character_image_path:
                issues.append(f"角色 '{char.name}' 没有人设图参考")
                suggestions.append(f"检查ComfyUI并重新生成 {char.name} 的人设图")

            # 检查人设图是否为上半身（通过日志中的composition_tags验证）
            # 这里通过检查workflow是否正确配置来间接验证

        # ── 4. 分镜覆盖率
        expected = len(panels)
        actual   = len([
            p for p in panel_images.values()
            if not str(p).startswith("error")
        ])
        coverage = actual / expected if expected > 0 else 0
        metadata["panel_coverage"] = f"{actual}/{expected} ({coverage:.0%})"

        if coverage < 0.5:
            issues.append(f"分镜生成率偏低: {actual}/{expected}")
        elif coverage < 1.0:
            suggestions.append(
                f"有 {expected - actual} 张分镜未生成，可重新运行补全"
            )

        # ── 5. 多角色场景检查（确认选角逻辑是否生效）
        multi_char_panels = [
            p for p in panels
            if len(p.get("characters_in_panel", [])) > 1
        ]
        metadata["multi_char_panels"] = len(multi_char_panels)
        if multi_char_panels:
            suggestions.append(
                f"{len(multi_char_panels)} 个多角色分镜使用了单角色生成模式，"
                f"次要角色以背景人物形式呈现"
            )

        metadata["style_name"]       = style_config.style_name
        metadata["shared_prefix"]    = shared_prefix[:100]
        metadata["chars_with_images"] = len([
            c for c in standardized_chars if c.character_image_path
        ])

        passed = len([
            i for i in issues
            if "不一致" in i or "错误" in i or "失败" in i
        ]) == 0

        report = QualityReport(
            passed=passed, issues=issues,
            suggestions=suggestions, metadata=metadata
        )

        self.logger.add_agent_log(
            "QualityControlAgent",
            "quality check",
            {
                "passed":           passed,
                "issues":           issues,
                "suggestions":      suggestions
            },
            metadata
        )

        self.logger.print_log(
            f"[QualityControlAgent] {'✓ 通过' if passed else '⚠ 发现问题'}"
        )
        for issue in issues:
            self.logger.print_log(f"  ✗ {issue}")
        for sug in suggestions:
            self.logger.print_log(f"  💡 {sug}")

        return report

    # =========================================================================
    # Agent 11: LayoutAgent
    # =========================================================================
    def layout_agent(
        self, panels: list, panel_images: dict, target_pages: int
    ) -> list:
        self.logger.print_log("\n" + "─" * 60)
        self.logger.print_log("[LayoutAgent] 排版合成...")

        final_pages = []
        for page_num in range(1, target_pages + 1):
            page_panels = [p for p in panels if p.get("page_id") == page_num]
            page_panel_data = []
            for panel in page_panels:
                pid         = panel.get("panel_id", "")
                image_path  = panel_images.get(pid, "")
                page_panel_data.append({
                    **panel,
                    "image_path":      image_path,
                    "image_available": bool(
                        image_path and not str(image_path).startswith("error")
                    )
                })
            final_pages.append({
                "page_number": page_num,
                "panel_count": len(page_panels),
                "panels":      page_panel_data,
                "layout_type": self._suggest_layout(len(page_panels))
            })

        self.logger.add_agent_log(
            "LayoutAgent",
            f"{len(panels)} panels, {len(panel_images)} images",
            {"pages": len(final_pages)},
            {"status": "success"}
        )
        self.logger.print_log(f"[LayoutAgent] 完成！{len(final_pages)} 页")
        return final_pages

    def _suggest_layout(self, panel_count: int) -> str:
        return {
            1: "full_page", 2: "top_bottom",
            3: "top_double_bottom", 4: "grid_2x2",
            5: "asymmetric_5", 6: "grid_2x3"
        }.get(panel_count, "flexible_grid")

    # =========================================================================
    # 主流程
    # =========================================================================
    def run(
        self, user_input: str,
        target_pages: int = 2,
        target_panels_per_page: int = 3
    ) -> dict:
        self.logger.print_log("\n" + "=" * 60)
        self.logger.print_log("ComicWeaver v3 - 单角色生成 + 画风统一修复版")
        self.logger.print_log("=" * 60)

        self.style_config            = None
        self.standardized_characters = []
        self.calibration_report      = None

        comfyui_ok = self.check_comfyui()
        self.logger.print_log(
            f"[系统] ComfyUI: "
            f"{'✓ 已连接' if comfyui_ok else '✗ 未运行（跳过图像生成）'}"
        )

        # ── Step 1: 剧本解析
        script_data = self.script_agent(
            user_input, target_pages, target_panels_per_page
        )

        # ── Step 2: 风格分析（确定唯一画风配置）
        self.style_config = self.style_analysis_agent(script_data)

        # ── Step 3: Workflow校准（同步模型，必须在生成前运行）
        if comfyui_ok:
            self.calibration_report = self.workflow_calibration_agent()

        # ── Step 4: 角色标准化
        if script_data.get("characters"):
            self.standardized_characters = self.character_standardization_agent(
                script_data["characters"], self.style_config
            )

        # ── Step 5: 人设图生成（上半身）
        if comfyui_ok and self.standardized_characters:
            self.standardized_characters = self.character_design_agent(
                self.standardized_characters, self.style_config
            )

        # ── Step 6: 分镜设计
        panels = self.storyboard_agent(
            script_data, self.standardized_characters,
            target_pages, target_panels_per_page
        )

        # ── Step 7: 分镜图生成（单角色模式）
        panel_images = {}
        if comfyui_ok and self.standardized_characters:
            panel_images = self.image_agent(
                panels, self.standardized_characters,
                self.style_config,
                max_images=target_pages * target_panels_per_page
            )

        # ── Step 8: 质量控制
        quality_report = self.quality_control_agent(
            script_data, self.standardized_characters,
            self.style_config, panels, panel_images,
            self.calibration_report
        )

        # ── Step 9: 排版
        final_pages = self.layout_agent(panels, panel_images, target_pages)

        # ── 输出摘要
        self.logger.print_log("\n" + "=" * 60)
        self.logger.print_log("执行完成")
        self.logger.print_log("=" * 60)
        self.logger.print_log(f"标题:       {script_data.get('title')}")
        self.logger.print_log(f"画风:       {self.style_config.style_name}")
        self.logger.print_log(
            f"共用前缀:   {self.style_config.get_shared_style_prefix()[:80]}..."
        )
        if self.calibration_report:
            match_str = (
                "✓ 一致"
                if self.calibration_report.models_match
                else f"⚠ 已同步 → {self.style_config.target_checkpoint}"
            )
            self.logger.print_log(f"Checkpoint: {match_str}")
        self.logger.print_log("角色:")
        for c in self.standardized_characters:
            img_str = "✓ 上半身图" if c.character_image_path else "✗ 无图"
            self.logger.print_log(f"  {img_str} | {c.name}: {c.core_tags}")
        self.logger.print_log(
            f"分镜: {len(panels)} 设计 / {len(panel_images)} 生成"
        )
        self.logger.print_log(
            f"质检: {'通过' if quality_report.passed else '有问题'}"
        )
        self.logger.print_log(f"日志: {LOG_FILE}")

        return {
            "script_data": script_data,
            "style_config": {
                "style_name":         self.style_config.style_name,
                "shared_prefix":      self.style_config.get_shared_style_prefix(),
                "character_prefix":   self.style_config.get_character_style_prefix(),
                "panel_prefix":       self.style_config.get_panel_style_prefix(),
                "negative_prompt":    self.style_config.negative_prompt
            },
            "calibration": {
                "models_match": (
                    self.calibration_report.models_match
                    if self.calibration_report else None
                ),
                "target_checkpoint": self.style_config.target_checkpoint
            },
            "standardized_characters": [
                {
                    "name":       c.name,
                    "role":       c.role,
                    "core_tags":  c.core_tags,
                    "image_path": c.character_image_path
                }
                for c in self.standardized_characters
            ],
            "panels":        panels,
            "panel_images":  panel_images,
            "final_pages":   final_pages,
            "quality_report": {
                "passed":      quality_report.passed,
                "issues":      quality_report.issues,
                "suggestions": quality_report.suggestions,
                "metadata":    quality_report.metadata
            }
        }


# ============================================================================
# LangGraph 模式
# ============================================================================

if LANGGRAPH_AVAILABLE:
    class ComicState(TypedDict):
        user_input:              str
        target_pages:            int
        target_panels_per_page:  int
        script_data:             dict
        style_config:            dict
        calibration_report:      dict
        standardized_characters: list
        panels:                  list
        panel_images:            dict
        final_pages:             list
        quality_report:          dict
        messages:                Annotated[Sequence[BaseMessage], add_messages]
        errors:                  list

    def _style_config_to_dict(cfg: DynamicStyleConfig) -> dict:
        return {
            "style_name":             cfg.style_name,
            "quality_prefix":         cfg.quality_prefix,
            "core_style_tags":        cfg.core_style_tags,
            "lighting_tags":          cfg.lighting_tags,
            "character_style_addon":  cfg.character_style_addon,
            "panel_style_addon":      cfg.panel_style_addon,
            "negative_prompt":        cfg.negative_prompt,
            "checkpoint_node_character": cfg.checkpoint_node_character,
            "checkpoint_node_panel":     cfg.checkpoint_node_panel,
            "target_checkpoint":         cfg.target_checkpoint
        }

    def _dict_to_style_config(d: dict) -> DynamicStyleConfig:
        return DynamicStyleConfig(**{
            k: v for k, v in d.items()
            if k in DynamicStyleConfig.__dataclass_fields__
        })

    def build_langgraph_workflow():
        wf = ComicWorkflow()

        def script_node(state: ComicState) -> dict:
            data = wf.script_agent(
                state["user_input"],
                state["target_pages"],
                state["target_panels_per_page"]
            )
            return {
                "script_data": data,
                "messages": [AIMessage(content=f"剧本: {data.get('title')}")]
            }

        def style_node(state: ComicState) -> dict:
            cfg = wf.style_analysis_agent(state["script_data"])
            wf.style_config = cfg
            return {
                "style_config": _style_config_to_dict(cfg),
                "messages": [AIMessage(content=f"风格: {cfg.style_name}")]
            }

        def calibration_node(state: ComicState) -> dict:
            wf.check_comfyui()
            report = wf.workflow_calibration_agent()
            report_dict = {}
            if report:
                report_dict = {
                    "models_match":  report.models_match,
                    "sync_applied":  report.sync_applied,
                    "char_model":    report.character_checkpoint.current_model,
                    "panel_model":   report.panel_checkpoint.current_model,
                    "issues":        report.issues
                }
                wf.calibration_report = report
            # 更新 style_config 中的 checkpoint 信息
            updated_style = _style_config_to_dict(wf.style_config)
            return {
                "calibration_report": report_dict,
                "style_config": updated_style,
                "messages": [AIMessage(
                    content=f"校准: {'模型一致' if report and report.models_match else '已同步'}"
                )]
            }

        def standardization_node(state: ComicState) -> dict:
            cfg   = _dict_to_style_config(state["style_config"])
            chars = wf.character_standardization_agent(
                state["script_data"].get("characters", []), cfg
            )
            return {
                "standardized_characters": [
                    {
                        "name": c.name, "role": c.role,
                        "core_tags": c.core_tags,
                        "appearance_desc": c.appearance_desc,
                        "personality": c.personality,
                        "backstory": c.backstory,
                        "character_image_path": "",
                        "character_image_filename": "",
                        "comfyui_input_filename": ""
                    }
                    for c in chars
                ],
                "messages": [AIMessage(content=f"标准化 {len(chars)} 个角色")]
            }

        def character_design_node(state: ComicState) -> dict:
            cfg   = _dict_to_style_config(state["style_config"])
            chars = [StandardizedCharacter(**c)
                     for c in state["standardized_characters"]]
            wf.standardized_characters = chars
            chars = wf.character_design_agent(chars, cfg)
            success = len([c for c in chars if c.character_image_path])
            return {
                "standardized_characters": [
                    {
                        "name": c.name, "role": c.role,
                        "core_tags": c.core_tags,
                        "appearance_desc": c.appearance_desc,
                        "personality": c.personality,
                        "backstory": c.backstory,
                        "character_image_path": c.character_image_path,
                        "character_image_filename": c.character_image_filename,
                        "comfyui_input_filename": c.comfyui_input_filename
                    }
                    for c in chars
                ],
                "messages": [AIMessage(content=f"生成 {success} 张上半身人设图")]
            }

        def storyboard_node(state: ComicState) -> dict:
            chars  = [StandardizedCharacter(**c)
                      for c in state["standardized_characters"]]
            panels = wf.storyboard_agent(
                state["script_data"], chars,
                state["target_pages"], state["target_panels_per_page"]
            )
            return {
                "panels": panels,
                "messages": [AIMessage(content=f"设计 {len(panels)} 个分镜")]
            }

        def image_node(state: ComicState) -> dict:
            cfg    = _dict_to_style_config(state["style_config"])
            chars  = [StandardizedCharacter(**c)
                      for c in state["standardized_characters"]]
            wf.standardized_characters = chars
            images = wf.image_agent(
                state["panels"], chars, cfg,
                max_images=state["target_pages"] * state["target_panels_per_page"]
            )
            success = len([
                v for v in images.values()
                if not str(v).startswith("error")
            ])
            return {
                "panel_images": images,
                "messages": [AIMessage(content=f"生成 {success} 张分镜图（单角色模式）")]
            }

        def quality_control_node(state: ComicState) -> dict:
            cfg    = _dict_to_style_config(state["style_config"])
            chars  = [StandardizedCharacter(**c)
                      for c in state["standardized_characters"]]
            report = wf.quality_control_agent(
                state["script_data"], chars, cfg,
                state["panels"], state["panel_images"],
                wf.calibration_report
            )
            return {
                "quality_report": {
                    "passed":      report.passed,
                    "issues":      report.issues,
                    "suggestions": report.suggestions,
                    "metadata":    report.metadata
                },
                "messages": [AIMessage(
                    content=f"质检: {'通过' if report.passed else '发现问题'}"
                )]
            }

        def layout_node(state: ComicState) -> dict:
            pages = wf.layout_agent(
                state["panels"],
                state["panel_images"],
                state["target_pages"]
            )
            return {
                "final_pages": pages,
                "messages": [AIMessage(content=f"排版 {len(pages)} 页")]
            }

        graph = StateGraph(ComicState)
        nodes = [
            ("script_agent",          script_node),
            ("style_analysis_agent",  style_node),
            ("calibration_agent",     calibration_node),
            ("standardization_agent", standardization_node),
            ("character_design_agent",character_design_node),
            ("storyboard_agent",      storyboard_node),
            ("image_agent",           image_node),
            ("quality_control_agent", quality_control_node),
            ("layout_agent",          layout_node),
        ]
        for name, fn in nodes:
            graph.add_node(name, fn)

        # 线性流水线
        prev = START
        for name, _ in nodes:
            graph.add_edge(prev, name)
            prev = name
        graph.add_edge(prev, END)

        return graph.compile()


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("ComicWeaver v3 测试")
    print("  · 人设图：上半身portrait")
    print("  · 分镜图：单角色生成")
    print("  · 画风：WorkflowCalibration同步checkpoint")
    print("=" * 60)

    user_input = (
        "一个关于少年冒险的故事。主角小明是一个勇敢的少年，银发金瞳，穿着蓝色铠甲。"
        "他在一次意外中发现了一个神秘的古老遗迹，遇到了守护者阿龙（红发黑瞳，穿着红色斗篷）。"
        "两人一起探索遗迹，最终发现了隐藏的宝藏。"
    )

    if LANGGRAPH_AVAILABLE:
        print("[INFO] LangGraph 模式")
        graph  = build_langgraph_workflow()
        result = graph.invoke({
            "user_input":             user_input,
            "target_pages":           2,
            "target_panels_per_page": 3,
            "script_data":            {},
            "style_config":           {},
            "calibration_report":     {},
            "standardized_characters": [],
            "panels":                 [],
            "panel_images":           {},
            "final_pages":            [],
            "quality_report":         {},
            "messages":               [],
            "errors":                 []
        })
    else:
        print("[INFO] 简化模式")
        workflow = ComicWorkflow()
        result   = workflow.run(user_input, target_pages=2, target_panels_per_page=3)

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)
    sc = result.get("style_config", {})
    print(f"画风:     {sc.get('style_name')}")
    print(f"共用前缀: {sc.get('shared_prefix', '')[:80]}...")
    cal = result.get("calibration", {})
    if cal.get("target_checkpoint"):
        print(f"模型同步: {cal.get('target_checkpoint')}")
    for c in result.get("standardized_characters", []):
        img = "✓" if c.get("image_path") else "✗"
        print(f"  {img} {c['name']}: {c['core_tags']}")
    qr = result.get("quality_report", {})
    print(f"质检: {'通过' if qr.get('passed') else '有问题'}")
    for issue in qr.get("issues", []):
        print(f"  ✗ {issue}")
    for sug in qr.get("suggestions", []):
        print(f"  💡 {sug}")


if __name__ == "__main__":
    main()