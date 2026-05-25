"""
ComicWeaver 完整工作流 - 架构重构版 v2
核心改进：
1. StyleAnalysisAgent 动态生成统一画风配置（替代硬编码UNIFIED_STYLE）
2. PromptEngineeringAgent 集中构建所有Prompt（人设图/分镜图使用完全相同前缀）
3. QualityControlAgent 对生成结果进行质量审查
4. 彻底消除prompt构建的分散逻辑
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

ZHIPUAI_API_KEY = "2487ed1dfc9f456fbd5630be90a18575.kPSrC0ZoSbjHBwgL"
COMFYUI_SERVER = "127.0.0.1:8188"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
CHARACTER_DIR = os.path.join(OUTPUT_DIR, "characters")
PANEL_DIR = os.path.join(OUTPUT_DIR, "panels")
LOG_FILE = os.path.join(OUTPUT_DIR, "workflow_log.json")

CHARACTER_WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "generate_character.json")
PANEL_WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "generate_picture.json")

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
class DynamicStyleConfig:
    """
    动态画风配置 —— 由 StyleAnalysisAgent 生成，是整个工作流的画风唯一真相源
    
    设计原则：
    - quality_prefix + core_style_tags 构成 style_prefix，人设图和分镜图必须使用完全相同的 style_prefix
    - 角色Tag（core_tags）永远不放入这里
    - 场景/构图Tag永远不放入这里
    - 只有画风相关内容在这里
    """
    # 风格名称（用于日志和调试）
    style_name: str = "anime realism"

    # 质量前缀（人设图和分镜图共用，必须完全相同）
    quality_prefix: str = (
        "masterpiece, best quality, ultra high res, hyper-detailed, 8K, newest"
    )

    # 核心风格Tag（人设图和分镜图共用，必须完全相同）
    # 这是画风一致性的核心：anime realism 意味着二次元写实风格
    core_style_tags: str = (
        "anime realism, semi-realistic anime, high quality anime CG, "
        "smooth cel shading, vibrant colors, soft rim lighting, "
        "detailed eyes, detailed face, sharp lineart, painterly"
    )

    # 光线/氛围Tag（由故事基调决定，人设图用中性光，分镜图按场景覆盖）
    lighting_tags: str = "soft ambient lighting, subtle shadows"

    # 负向Prompt（人设图和分镜图共用，必须完全相同）
    negative_prompt: str = (
        "lowres, worst quality, bad quality, normal quality, "
        "bad anatomy, bad proportions, extra limbs, deformed hands, "
        "bad face, ugly face, blurry, cropped, "
        "sketch, rough sketch, jpeg artifacts, "
        "signature, watermark, text, username, "
        "3d render, cgi, photorealistic, western comic style, "
        "flat color, simple background, monochrome"
    )

    def get_style_prefix(self) -> str:
        """
        获取统一的画风前缀字符串
        这个字符串必须被用于所有图像生成，包括人设图和分镜图
        只在 PromptEngineeringAgent 中调用，其他地方不得直接构建Prompt
        """
        return f"{self.quality_prefix}, {self.core_style_tags}, {self.lighting_tags}"


@dataclass
class StandardizedCharacter:
    """
    标准化角色数据 —— 所有下游Agent的唯一可信角色信息源
    
    不变性约束：
    - core_tags 一旦由 CharacterStandardizationAgent 设置，不得被任何其他Agent修改
    - core_tags 只包含角色外貌Tag，不包含任何画风/质量Tag
    - 画风Tag统一由 DynamicStyleConfig 管理
    """
    name: str
    role: str
    # 核心外貌Tag（纯角色描述，不含画风）
    # e.g., "1boy, silver hair, short hair, golden eyes, blue armor, silver trim, blue cape"
    core_tags: str
    # 自然语言外貌描述（供StoryboardAgent等文本Agent理解）
    appearance_desc: str
    personality: str
    backstory: str = ""
    # 人设图相关
    character_image_path: str = ""
    character_image_filename: str = ""
    comfyui_input_filename: str = ""


@dataclass
class BuiltPrompt:
    """
    PromptEngineeringAgent 的输出 —— 完整的图像生成Prompt
    """
    positive: str
    negative: str
    # 元信息用于日志和调试
    style_prefix_used: str = ""
    char_tags_used: str = ""
    scene_tags_used: str = ""
    composition_tags_used: str = ""


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

    def add_agent_log(self, agent_name: str, input_data: Any, output: Any, metadata: dict = None):
        self.logs["agents"].append({
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "input": str(input_data)[:500],
            "output": output if isinstance(output, (dict, list)) else str(output)[:500],
            "metadata": metadata or {}
        })
        self.save()

    def add_image_log(self, agent_name: str, prompt: BuiltPrompt, image_path: str, metadata: dict = None):
        self.logs["agents"].append({
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "prompt": {
                "positive": prompt.positive[:300],
                "negative": prompt.negative[:200],
                "style_prefix": prompt.style_prefix_used[:200],
                "char_tags": prompt.char_tags_used,
                "scene_tags": prompt.scene_tags_used,
                "composition_tags": prompt.composition_tags_used
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
        data = {
            "model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens
        }
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

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url = f"http://{self.server_address}/view?{urllib.parse.urlencode(params)}"
        response = requests.get(url, timeout=30)
        return response.content

    def upload_image(self, image_data: bytes, filename: str,
                     folder_type: str = "input", subfolder: str = "") -> dict:
        import io
        files = {
            'image': (filename, io.BytesIO(image_data), 'image/png'),
            'type': (None, folder_type),
        }
        if subfolder:
            files['subfolder'] = (None, subfolder)
        response = requests.post(
            f"http://{self.server_address}/upload/image", files=files, timeout=30
        )
        response.raise_for_status()
        return response.json()

    def get_image_from_output(self, prompt_id: str, timeout: int = 300) -> Tuple[bytes, str]:
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
                if status.get("status_str") == "error" or status.get("completed", False):
                    raise RuntimeError(f"ComfyUI failed: {status.get('messages', [])}")
            time.sleep(2)
        raise TimeoutError(f"Generation timeout: {prompt_id}")

    def generate_with_workflow(self, workflow_path: str, prompt_overrides: dict,
                                output_dir: str, seed: int = -1) -> Tuple[bytes, str, str]:
        """
        通用workflow执行方法
        prompt_overrides: {node_id: {input_key: value}} 格式的覆盖参数
        """
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        # 应用覆盖
        for node_id, inputs in prompt_overrides.items():
            if node_id in workflow:
                for key, value in inputs.items():
                    workflow[node_id]["inputs"][key] = value

        # 随机seed
        for node_id in ["3", "25", "seed_node"]:
            if node_id in workflow and "seed" in workflow[node_id].get("inputs", {}):
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
# Agent 实现
# ============================================================================

class ComicWorkflow:
    def __init__(self):
        self.llm_client = ZhipuAIHTTPClient(ZHIPUAI_API_KEY)
        self.comfyui_client: Optional[ComfyUIClient] = None
        self.logger = WorkflowLogger(LOG_FILE)

        # 工作流状态（这些是全局唯一真相源，下游只读）
        self.style_config: Optional[DynamicStyleConfig] = None
        self.standardized_characters: List[StandardizedCharacter] = []

    def check_comfyui(self) -> bool:
        try:
            response = requests.get(f"http://{COMFYUI_SERVER}/system_stats", timeout=5)
            if response.status_code == 200:
                self.comfyui_client = ComfyUIClient()
                return True
            return False
        except Exception as e:
            self.logger.print_log(f"[WARN] ComfyUI连接失败: {e}")
            return False

    # =========================================================================
    # Agent 1: ScriptAgent
    # 职责：纯故事内容解析，输出自然语言，禁止输出任何图像Tag
    # =========================================================================
    def script_agent(self, user_input: str, target_pages: int, target_panels_per_page: int) -> dict:
        self.logger.print_log("\n" + "─" * 50)
        self.logger.print_log("[ScriptAgent] 开始解析剧本...")

        system_prompt = """You are a professional manga script analyst.

Your job: Parse the user's story into structured script data.

STRICT RULES:
- Output ONLY story content: plot, characters (name, role, personality, appearance in natural language), scenes
- NEVER output any Stable Diffusion tags, Danbooru tags, or image generation prompts
- Appearance descriptions MUST be natural language (e.g., "He has silver hair cut short, with golden eyes and wears a blue combat armor")
- Tag conversion will be done by a specialized downstream agent

Return valid JSON:
{
  "title": "string",
  "genre": "string (e.g., action, romance, fantasy, sci-fi, slice-of-life)",
  "tone": "string (e.g., dark and gritty, lighthearted, dramatic, mysterious)",
  "visual_style_preference": "string (e.g., detailed realism, cute chibi, cinematic)",
  "summary": "string",
  "characters": [
    {
      "name": "string",
      "role": "protagonist|antagonist|supporting",
      "gender": "male|female",
      "age_range": "child|teenager|young_adult|adult|elder",
      "appearance": "detailed natural language description of hair, eyes, skin, clothing, accessories",
      "personality": "string",
      "backstory": "string (brief)"
    }
  ],
  "scenes": [
    {
      "scene_id": "integer",
      "location": "string",
      "time_of_day": "dawn|morning|afternoon|dusk|night",
      "weather": "string",
      "atmosphere": "string",
      "characters_present": ["name list"],
      "key_action": "string",
      "emotion_intensity": "0.0-1.0",
      "suggested_panels": "integer (1-4)"
    }
  ]
}"""

        user_prompt = (
            f"Parse this story into manga script:\n\n{user_input}\n\n"
            f"Target: {target_pages} pages, ~{target_panels_per_page} panels/page"
        )

        try:
            response = self.llm_client.chat_completions_create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6, max_tokens=2500
            )
            content = response["choices"][0]["message"]["content"]
            json_match = re.search(r'\{[\s\S]*\}', content)
            result = json.loads(json_match.group()) if json_match else json.loads(content)

            self.logger.add_agent_log("ScriptAgent", user_input, result, {"status": "success"})
            self.logger.print_log(
                f"[ScriptAgent] 完成！标题: {result.get('title')}，"
                f"风格偏好: {result.get('visual_style_preference')}，"
                f"基调: {result.get('tone')}"
            )
            return result

        except Exception as e:
            self.logger.print_log(f"[ScriptAgent] 错误: {e}")
            fallback = {
                "title": "未命名", "genre": "action", "tone": "dramatic",
                "visual_style_preference": "anime realism",
                "summary": user_input,
                "characters": [{"name": "主角", "role": "protagonist", "gender": "male",
                               "age_range": "teenager",
                               "appearance": "short black hair, brown eyes, simple dark clothing",
                               "personality": "brave", "backstory": ""}],
                "scenes": [{"scene_id": 1, "location": "unknown", "time_of_day": "afternoon",
                           "weather": "clear", "atmosphere": "neutral",
                           "characters_present": ["主角"], "key_action": "standing",
                           "emotion_intensity": 0.5, "suggested_panels": target_panels_per_page}]
            }
            self.logger.add_agent_log("ScriptAgent", user_input, fallback, {"status": "fallback", "error": str(e)})
            return fallback

    # =========================================================================
    # Agent 2: StyleAnalysisAgent ★ 新增核心Agent
    # 职责：根据故事内容动态生成统一画风配置
    # 这是解决人设图和分镜图画风不一致的根本方案
    # =========================================================================
    def style_analysis_agent(self, script_data: dict) -> DynamicStyleConfig:
        """
        分析故事内容，生成统一的 DynamicStyleConfig

        为什么需要这个Agent：
        1. 不同故事需要不同画风（奇幻 vs 现代 vs 科幻）
        2. 画风必须在第一时间确定，后续所有Agent都只读这个配置
        3. "anime realism" 等具体风格词必须出现在所有图像的prompt中
        4. 画风Tag一旦确定，PromptEngineeringAgent 会确保它出现在人设图和分镜图的相同位置
        """
        self.logger.print_log("\n" + "─" * 50)
        self.logger.print_log("[StyleAnalysisAgent] 分析故事风格...")

        genre = script_data.get("genre", "action")
        tone = script_data.get("tone", "dramatic")
        visual_pref = script_data.get("visual_style_preference", "anime realism")
        title = script_data.get("title", "")

        system_prompt = """You are an expert in anime/manga visual style design.

Based on the story's genre, tone, and visual preference, generate a comprehensive and specific set of Stable Diffusion style tags.

CRITICAL RULES:
1. The output style tags will be used as a PREFIX in EVERY image prompt (both character design and scene panels)
2. Include "anime realism" or equivalent for semi-realistic anime style
3. The style tags must be specific enough to ensure visual consistency across all images
4. Do NOT include character-specific tags (hair color, eye color, clothing) - only art style
5. Do NOT include scene-specific tags (forest, city, building) - only art style

Return valid JSON:
{
  "style_name": "descriptive name (e.g., 'Dark Fantasy Anime Realism')",
  "quality_tags": "quality and resolution tags only",
  "core_style_tags": "art style tags that define the overall look",
  "lighting_tags": "default lighting style (will be overridden per-scene)",
  "negative_prompt": "comprehensive negative prompt to avoid unwanted styles",
  "reasoning": "brief explanation of style choices"
}

Example for dark fantasy:
{
  "style_name": "Dark Fantasy Anime Realism",
  "quality_tags": "masterpiece, best quality, ultra high res, hyper-detailed, 8K, newest",
  "core_style_tags": "anime realism, semi-realistic anime, high quality anime CG, detailed cel shading, rich color palette, dramatic contrast, painterly, sharp lineart, detailed textures",
  "lighting_tags": "dramatic lighting, rim lighting, ambient occlusion, subtle shadows",
  "negative_prompt": "lowres, worst quality, bad quality, bad anatomy, blurry, cropped, extra limbs, deformed, bad face, sketch, jpeg artifacts, signature, watermark, 3d render, photorealistic, western comic, flat color, simple shading, monochrome"
}"""

        user_prompt = (
            f"Story: {title}\n"
            f"Genre: {genre}\n"
            f"Tone: {tone}\n"
            f"Visual preference: {visual_pref}\n"
            f"Summary: {script_data.get('summary', '')[:200]}"
        )

        try:
            response = self.llm_client.chat_completions_create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4, max_tokens=800
            )
            content = response["choices"][0]["message"]["content"]
            json_match = re.search(r'\{[\s\S]*\}', content)
            result = json.loads(json_match.group()) if json_match else json.loads(content)

            config = DynamicStyleConfig(
                style_name=result.get("style_name", "Anime Realism"),
                quality_prefix=result.get("quality_tags",
                    "masterpiece, best quality, ultra high res, hyper-detailed, 8K, newest"),
                core_style_tags=result.get("core_style_tags",
                    "anime realism, semi-realistic anime, high quality anime CG, "
                    "smooth cel shading, vibrant colors, sharp lineart, painterly"),
                lighting_tags=result.get("lighting_tags",
                    "soft ambient lighting, rim lighting, subtle shadows"),
                negative_prompt=result.get("negative_prompt",
                    "lowres, worst quality, bad quality, bad anatomy, blurry, cropped, "
                    "extra limbs, deformed, bad face, sketch, jpeg artifacts, signature, "
                    "watermark, 3d render, photorealistic, western comic, flat color, monochrome")
            )

            self.logger.add_agent_log("StyleAnalysisAgent", user_prompt,
                {
                    "style_name": config.style_name,
                    "style_prefix": config.get_style_prefix()[:200],
                    "reasoning": result.get("reasoning", "")
                }, {"status": "success"})

            self.logger.print_log(f"[StyleAnalysisAgent] 完成！风格: {config.style_name}")
            self.logger.print_log(f"[StyleAnalysisAgent] 风格前缀: {config.get_style_prefix()[:100]}...")
            return config

        except Exception as e:
            self.logger.print_log(f"[StyleAnalysisAgent] 错误，使用默认配置: {e}")
            config = DynamicStyleConfig()
            self.logger.add_agent_log("StyleAnalysisAgent", user_prompt,
                {"error": str(e), "fallback": config.style_name}, {"status": "fallback"})
            return config

    # =========================================================================
    # Agent 3: CharacterStandardizationAgent
    # 职责：将自然语言外貌描述转换为精确的角色Tag（不含画风Tag）
    # =========================================================================
    def character_standardization_agent(self, characters: list,
                                         style_config: DynamicStyleConfig) -> List[StandardizedCharacter]:
        """
        标准化角色Tag

        为什么只输出角色Tag，不含画风Tag：
        - 画风由 DynamicStyleConfig 统一管理
        - PromptEngineeringAgent 负责组合画风前缀 + 角色Tag
        - 角色Tag保持纯净，便于IPAdapter等技术精确匹配角色特征
        """
        self.logger.print_log("\n" + "─" * 50)
        self.logger.print_log("[CharacterStandardizationAgent] 开始标准化角色Tag...")

        standardized = []

        for char in characters:
            name = char.get("name", "Unknown")
            appearance = char.get("appearance", "")
            role = char.get("role", "supporting")
            gender = char.get("gender", "male")
            age_range = char.get("age_range", "young_adult")

            self.logger.print_log(f"  → 标准化: {name}")

            system_prompt = """You are an expert in Stable Diffusion/Danbooru tag conversion for anime characters.

Convert the character's natural language appearance description into precise Danbooru-style tags.

RULES:
1. Start with gender tag: "1girl" (female) or "1boy" (male)
2. Add age hint: "young adult", "teenager", "child" etc.
3. ALWAYS preserve exact colors: "silver hair" not just "hair", "golden eyes" not just "eyes"
4. Include hair style: long hair, short hair, ponytail, twintails, etc.
5. Include ALL clothing items with colors: "blue armor", "red cloak", "white dress"
6. Include accessories: "earrings", "necklace", "sword", etc.
7. Include skin tone if mentioned: "fair skin", "dark skin", "tan skin"
8. NEVER include style tags like "anime", "realistic", "detailed" - those are handled separately
9. Output ONLY the tag string, comma-separated, no explanations

Example:
Input: "Young man with silver hair in a short neat cut, golden eyes, fair skin, wearing blue combat armor with silver trim and a blue cape"
Output: 1boy, young adult, silver hair, short hair, neat hair, golden eyes, fair skin, blue armor, silver trim, blue cape"""

            user_prompt = (
                f"Gender: {gender}, Age: {age_range}\n"
                f"Convert this appearance to tags: {appearance}"
            )

            try:
                response = self.llm_client.chat_completions_create(
                    model="glm-4-flash",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2, max_tokens=300
                )
                core_tags = response["choices"][0]["message"]["content"].strip()
                # 清理markdown格式
                core_tags = re.sub(r'^[`\'\"]+|[`\'\"]+$', '', core_tags).strip()
                # 确保不含画风Tag（二次检查）
                style_leak_words = ["anime", "realistic", "detailed", "masterpiece", "quality",
                                    "beautiful", "gorgeous", "stunning", "amazing"]
                for word in style_leak_words:
                    if word in core_tags.lower():
                        self.logger.print_log(f"  [WARN] 检测到画风Tag泄漏: '{word}'，已清理")
                        core_tags = re.sub(rf'\b{word}\b[^,]*,?\s*', '', core_tags, flags=re.IGNORECASE)
                core_tags = re.sub(r',\s*,', ',', core_tags).strip(', ')

            except Exception as e:
                self.logger.print_log(f"  [WARN] LLM失败，使用规则提取: {e}")
                core_tags = self._rule_based_tag_extraction(appearance, gender)

            std_char = StandardizedCharacter(
                name=name,
                role=role,
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

        self.logger.print_log(f"[CharacterStandardizationAgent] 完成！{len(standardized)} 个角色")
        return standardized

    def _rule_based_tag_extraction(self, appearance: str, gender: str = "male") -> str:
        """规则提取（LLM失败时的fallback）"""
        tags = ["1girl" if gender == "female" else "1boy"]
        app_lower = appearance.lower()

        hair_colors = ["silver", "white", "blonde", "golden", "brown", "red", "crimson",
                       "blue", "green", "purple", "pink", "black", "orange", "grey", "gray"]
        for c in hair_colors:
            if c in app_lower:
                tags.append(f"{c} hair")
                break

        hair_styles = ["long hair", "short hair", "medium hair", "ponytail",
                       "twintails", "braid", "bob cut", "wavy hair", "curly hair"]
        for s in hair_styles:
            if s in app_lower:
                tags.append(s)

        eye_patterns = ["golden eyes", "red eyes", "blue eyes", "green eyes",
                        "purple eyes", "black eyes", "brown eyes", "silver eyes",
                        "crimson eyes", "yellow eyes"]
        for ep in eye_patterns:
            if ep in app_lower:
                tags.append(ep)
                break

        for color in ["red", "blue", "green", "black", "white", "silver", "gold"]:
            for item in ["armor", "cloak", "cape", "dress", "uniform", "coat", "jacket", "robe"]:
                if f"{color} {item}" in app_lower:
                    tags.append(f"{color} {item}")

        return ", ".join(dict.fromkeys(tags))

    # =========================================================================
    # Agent 4: PromptEngineeringAgent ★ 新增核心Agent
    # 职责：集中构建所有图像Prompt，确保画风前缀完全一致
    # 这是解决人设图/分镜图风格不一致的技术核心
    # =========================================================================
    def prompt_engineering_agent(self, task_type: str, task_data: dict,
                                  style_config: DynamicStyleConfig,
                                  standardized_chars: List[StandardizedCharacter]) -> BuiltPrompt:
        """
        统一Prompt构建Agent

        为什么需要独立的Agent：
        - 之前人设图prompt在CharacterDesignAgent里，分镜图prompt在ImageAgent._build_panel_prompt里
        - 两处逻辑不同，导致画风Tag不一致
        - 现在所有Prompt都在这里构建，style_prefix完全相同

        task_type: "character_design" | "panel_scene"
        task_data: 角色或分镜的具体信息

        Prompt结构（对所有图像完全一致）：
        [style_prefix] + [character_core_tags] + [task_specific_tags]
                ↑                    ↑                     ↑
           完全相同              标准化且不变           场景/构图独有
        """
        style_prefix = style_config.get_style_prefix()
        negative = style_config.negative_prompt

        if task_type == "character_design":
            return self._build_character_prompt(task_data, style_prefix, negative, standardized_chars)
        elif task_type == "panel_scene":
            return self._build_panel_prompt_v2(task_data, style_prefix, negative, standardized_chars)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def _build_character_prompt(self, char_data: dict, style_prefix: str,
                                  negative: str, _: List[StandardizedCharacter]) -> BuiltPrompt:
        """
        构建人设图Prompt

        结构: [统一画风前缀] + [角色核心Tag] + [人设图专用构图]
        """
        char_tags = char_data.get("core_tags", "1boy")
        role = char_data.get("role", "protagonist")

        # 人设图专用构图Tag
        if role == "protagonist":
            composition = "solo, full body, standing, looking at viewer, neutral pose, plain background, character sheet"
        elif role == "antagonist":
            composition = "solo, full body, standing, looking at viewer, intimidating pose, dramatic background"
        else:
            composition = "solo, upper body, looking at viewer, neutral expression"

        positive = f"{style_prefix}, {char_tags}, {composition}"

        return BuiltPrompt(
            positive=positive,
            negative=negative,
            style_prefix_used=style_prefix,
            char_tags_used=char_tags,
            scene_tags_used="",
            composition_tags_used=composition
        )

    def _build_panel_prompt_v2(self, panel_data: dict, style_prefix: str,
                                negative: str, standardized_chars: List[StandardizedCharacter]) -> BuiltPrompt:
        """
        构建分镜图Prompt

        结构: [统一画风前缀] + [角色核心Tag] + [场景Tag] + [构图Tag]

        关键：style_prefix 与人设图完全相同！！
        """
        # 1. 获取角色核心Tag（从标准化角色列表查找，不自行生成）
        chars_in_panel = panel_data.get("characters_in_panel", [])
        char_tags_parts = []

        for char_name in chars_in_panel:
            found = next((c for c in standardized_chars if c.name == char_name), None)
            if found:
                char_tags_parts.append(found.core_tags)
            else:
                self.logger.print_log(f"  [WARN] 未找到角色 '{char_name}' 的标准化Tag")

        char_tags = ", ".join(char_tags_parts) if char_tags_parts else "1person"

        # 2. 场景Tag（由StoryboardAgent提供，PromptEngineeringAgent只做格式化）
        scene_elements = []
        setting = panel_data.get("setting", "")
        time_of_day = panel_data.get("time_of_day", "")
        weather = panel_data.get("weather", "")
        atmosphere = panel_data.get("atmosphere", "")
        scene_lighting = panel_data.get("scene_lighting", "")

        if setting:
            scene_elements.append(setting)
        if time_of_day:
            scene_elements.append(time_of_day)
        if weather and weather not in ["clear", "normal"]:
            scene_elements.append(weather)
        if atmosphere:
            scene_elements.append(atmosphere)
        if scene_lighting:
            scene_elements.append(scene_lighting)

        scene_tags = ", ".join(filter(None, scene_elements))

        # 3. 构图Tag
        shot_map = {
            "extreme_long": "extreme wide shot, establishing shot",
            "long": "wide shot, full figure",
            "full": "full body shot",
            "medium": "medium shot, cowboy shot",
            "close": "close-up shot",
            "extreme_close": "extreme close-up, face focus"
        }
        angle_map = {
            "eye_level": "eye level",
            "high_angle": "from above, bird's eye",
            "low_angle": "from below, low angle shot",
            "dutch_angle": "dutch angle"
        }
        mood_map = {
            "tense": "dynamic composition, motion lines",
            "peaceful": "serene composition, balanced framing",
            "dramatic": "dramatic composition, diagonal lines",
            "mysterious": "atmospheric composition, depth of field"
        }

        shot_size = panel_data.get("shot_size", "medium")
        camera_angle = panel_data.get("camera_angle", "eye_level")
        mood = panel_data.get("mood", "")
        action = panel_data.get("primary_action", "")

        composition_parts = [
            shot_map.get(shot_size, "medium shot"),
            angle_map.get(camera_angle, "eye level"),
        ]
        if mood in mood_map:
            composition_parts.append(mood_map[mood])
        if action:
            # 只保留动作动词，剔除外貌词
            action_clean = self._sanitize_action(action)
            if action_clean:
                composition_parts.append(action_clean)

        composition_tags = ", ".join(filter(None, composition_parts))

        # 4. 组合最终Prompt
        # 结构严格一致：[style_prefix], [char_tags], [scene_tags], [composition_tags]
        parts = [style_prefix, char_tags]
        if scene_tags:
            parts.append(scene_tags)
        parts.append(composition_tags)

        positive = ", ".join(filter(None, parts))

        return BuiltPrompt(
            positive=positive,
            negative=negative,
            style_prefix_used=style_prefix,
            char_tags_used=char_tags,
            scene_tags_used=scene_tags,
            composition_tags_used=composition_tags
        )

    def _sanitize_action(self, action: str) -> str:
        """清理动作描述中的外貌词汇（这些词在core_tags里，重复会干扰生成）"""
        appearance_words = [
            "hair", "eyes", "eye", "armor", "cloak", "cape", "dress",
            "wearing", "outfit", "uniform", "skin", "face", "body",
            "clothes", "clothing", "tall", "short", "slim", "muscular"
        ]
        words = action.split()
        filtered = [w for w in words if not any(aw in w.lower() for aw in appearance_words)]
        return " ".join(filtered).strip()

    # =========================================================================
    # Agent 5: CharacterDesignAgent
    # 职责：生成人设图（调用PromptEngineeringAgent获取Prompt）
    # =========================================================================
    def character_design_agent(self, standardized_chars: List[StandardizedCharacter],
                                style_config: DynamicStyleConfig) -> List[StandardizedCharacter]:
        self.logger.print_log("\n" + "─" * 50)
        self.logger.print_log("[CharacterDesignAgent] 开始生成角色人设图...")

        if self.comfyui_client is None:
            self.logger.print_log("[CharacterDesignAgent] ComfyUI未运行，跳过")
            return standardized_chars

        for char in standardized_chars:
            self.logger.print_log(f"  → 生成人设图: {char.name}")

            # 通过 PromptEngineeringAgent 构建Prompt（不在这里直接拼接）
            built_prompt = self.prompt_engineering_agent(
                task_type="character_design",
                task_data={"core_tags": char.core_tags, "role": char.role, "name": char.name},
                style_config=style_config,
                standardized_chars=standardized_chars
            )

            self.logger.print_log(f"  Prompt结构:")
            self.logger.print_log(f"    style_prefix: {built_prompt.style_prefix_used[:80]}...")
            self.logger.print_log(f"    char_tags:    {built_prompt.char_tags_used}")
            self.logger.print_log(f"    composition:  {built_prompt.composition_tags_used}")

            try:
                # 使用通用workflow执行
                prompt_overrides = {
                    "39": {"prompt": built_prompt.positive},
                    "5": {"width": 832, "height": 1216}
                }
                image_data, filename, saved_path = self.comfyui_client.generate_with_workflow(
                    CHARACTER_WORKFLOW_PATH, prompt_overrides, CHARACTER_DIR
                )

                # 上传到ComfyUI input目录（供IPAdapter使用）
                input_filename = f"char_{filename}"
                try:
                    self.comfyui_client.upload_image(image_data, input_filename, folder_type="input")
                    self.logger.print_log(f"  ✓ 已上传到ComfyUI input: {input_filename}")
                except Exception as e:
                    self.logger.print_log(f"  [WARN] 上传失败: {e}")
                    input_filename = filename

                char.character_image_path = saved_path
                char.character_image_filename = filename
                char.comfyui_input_filename = input_filename

                self.logger.add_image_log(
                    "CharacterDesignAgent", built_prompt, saved_path,
                    {"character": char.name, "status": "success"}
                )
                self.logger.print_log(f"  ✓ 人设图保存: {saved_path}")

            except Exception as e:
                self.logger.print_log(f"  ✗ 失败: {e}")
                self.logger.add_agent_log(
                    "CharacterDesignAgent",
                    f"Character: {char.name}",
                    {"error": str(e)},
                    {"character": char.name, "status": "error"}
                )

        success = len([c for c in standardized_chars if c.character_image_path])
        self.logger.print_log(f"[CharacterDesignAgent] 完成！{success}/{len(standardized_chars)} 张")
        return standardized_chars

    # =========================================================================
    # Agent 6: StoryboardAgent
    # 职责：设计镜头语言（构图/动作/场景），绝不生成或修改角色Tag
    # =========================================================================
    def storyboard_agent(self, script_data: dict, standardized_chars: List[StandardizedCharacter],
                         target_pages: int, target_panels_per_page: int) -> list:
        self.logger.print_log("\n" + "─" * 50)
        self.logger.print_log("[StoryboardAgent] 开始设计分镜...")

        # 提供给LLM的角色参考（只读，明确声明不能修改Tag）
        char_refs = [
            {
                "name": c.name,
                "role": c.role,
                "core_tags": c.core_tags,  # 只读参考
                "personality": c.personality
            }
            for c in standardized_chars
        ]

        system_prompt = """You are a professional manga storyboard director.

YOUR ONLY JOB: Design the cinematography and staging of each panel.
- Shot size, camera angle, character positions, actions, scene setting
- You MUST NOT generate or modify any character appearance tags
- You MUST NOT generate any art style tags

For each panel, provide:
- panel_id: "page.panel" format (e.g., "1.1", "1.2")
- page_id: integer
- shot_size: extreme_long | long | full | medium | close | extreme_close
- camera_angle: eye_level | high_angle | low_angle | dutch_angle
- characters_in_panel: list of character NAMES (must match provided character names exactly)
- primary_action: what characters are DOING (verbs only, no appearance descriptions)
  Good: "running forward", "drawing sword", "embracing"
  Bad: "silver-haired boy running" (appearance is NOT your job)
- setting: location description in English tags (e.g., "ancient ruins, stone pillars, overgrown vines")
- time_of_day: dawn|morning|afternoon|dusk|night
- weather: clear|cloudy|rainy|stormy|snowy|foggy
- atmosphere: emotional/visual atmosphere (e.g., "tense", "peaceful", "dramatic", "mysterious")
- scene_lighting: specific lighting for this scene (e.g., "golden sunset backlight", "harsh rain lighting", "moonlight")
- mood: tense | peaceful | dramatic | mysterious | joyful | sorrowful
- dialogue_hint: brief hint of what's happening narratively

Return JSON with "panels" array."""

        user_prompt = (
            f"Design storyboard for:\n"
            f"Title: {script_data.get('title')}\n"
            f"Summary: {script_data.get('summary', '')[:300]}\n"
            f"Characters (READ-ONLY, names and roles only): {json.dumps([{'name': c['name'], 'role': c['role']} for c in char_refs], ensure_ascii=False)}\n"
            f"Scenes: {json.dumps(script_data.get('scenes', []), ensure_ascii=False)}\n"
            f"Target: {target_pages} pages, {target_panels_per_page} panels per page"
        )

        try:
            response = self.llm_client.chat_completions_create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7, max_tokens=3000
            )
            content = response["choices"][0]["message"]["content"]

            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                parsed = json.loads(json_match.group())
                panels = parsed.get("panels", parsed) if isinstance(parsed, dict) else parsed
            else:
                json_match = re.search(r'\[[\s\S]*\]', content)
                panels = json.loads(json_match.group()) if json_match else []

            self.logger.add_agent_log("StoryboardAgent", user_prompt,
                {"panels_count": len(panels)}, {"status": "success"})
            self.logger.print_log(f"[StoryboardAgent] 完成！{len(panels)} 个分镜")
            return panels

        except Exception as e:
            self.logger.print_log(f"[StoryboardAgent] 错误: {e}，使用fallback")
            panels = self._fallback_storyboard(script_data, standardized_chars, target_pages, target_panels_per_page)
            self.logger.add_agent_log("StoryboardAgent", user_prompt,
                {"panels_count": len(panels), "fallback": True}, {"status": "fallback", "error": str(e)})
            return panels

    def _fallback_storyboard(self, script_data, standardized_chars, target_pages, target_panels_per_page):
        panels = []
        scenes = script_data.get("scenes", [])
        shot_cycle = ["medium", "close", "long", "full", "extreme_close"]
        angle_cycle = ["eye_level", "high_angle", "low_angle"]

        for page in range(target_pages):
            for panel_idx in range(target_panels_per_page):
                flat_idx = page * target_panels_per_page + panel_idx
                scene = scenes[flat_idx % len(scenes)] if scenes else {}
                chars_present = scene.get("characters_present", [c.name for c in standardized_chars[:1]])

                panels.append({
                    "panel_id": f"{page+1}.{panel_idx+1}",
                    "page_id": page + 1,
                    "shot_size": shot_cycle[flat_idx % len(shot_cycle)],
                    "camera_angle": angle_cycle[flat_idx % len(angle_cycle)],
                    "characters_in_panel": chars_present,
                    "primary_action": scene.get("key_action", "standing"),
                    "setting": scene.get("location", "unknown place"),
                    "time_of_day": scene.get("time_of_day", "afternoon"),
                    "weather": scene.get("weather", "clear"),
                    "atmosphere": scene.get("atmosphere", "neutral"),
                    "scene_lighting": "natural light",
                    "mood": "dramatic" if scene.get("emotion_intensity", 0.5) > 0.6 else "peaceful",
                    "dialogue_hint": ""
                })
        return panels

    # =========================================================================
    # Agent 7: ImageAgent
    # 职责：生成分镜图（调用PromptEngineeringAgent获取Prompt）
    # =========================================================================
    def image_agent(self, panels: list, standardized_chars: List[StandardizedCharacter],
                    style_config: DynamicStyleConfig, max_images: int = 6) -> dict:
        self.logger.print_log("\n" + "─" * 50)
        self.logger.print_log("[ImageAgent] 开始生成分镜图...")

        if self.comfyui_client is None:
            self.logger.print_log("[ImageAgent] ComfyUI未运行，跳过")
            return {}

        # 找到有人设图的角色（用于IPAdapter参考）
        chars_with_image = [c for c in standardized_chars if c.comfyui_input_filename]
        if not chars_with_image:
            self.logger.print_log("[ImageAgent] 没有可用的人设图，跳过")
            return {}

        panel_images = {}
        max_panels = min(max_images, len(panels))

        self.logger.print_log(f"[ImageAgent] 将生成 {max_panels}/{len(panels)} 张分镜图")
        self.logger.print_log(f"[ImageAgent] 使用风格前缀: {style_config.get_style_prefix()[:80]}...")

        for i, panel in enumerate(panels[:max_panels]):
            panel_id = panel.get("panel_id", f"panel_{i:03d}")
            chars_in_panel = panel.get("characters_in_panel", [])

            # 选择参考图（优先使用主角）
            ref_char = None
            for name in chars_in_panel:
                found = next((c for c in chars_with_image if c.name == name), None)
                if found:
                    ref_char = found
                    break
            if not ref_char:
                ref_char = chars_with_image[0]

            # 通过 PromptEngineeringAgent 构建Prompt
            built_prompt = self.prompt_engineering_agent(
                task_type="panel_scene",
                task_data=panel,
                style_config=style_config,
                standardized_chars=standardized_chars
            )

            self.logger.print_log(f"\n  [{i+1}/{max_panels}] 生成分镜 {panel_id}")
            self.logger.print_log(f"  style_prefix: {built_prompt.style_prefix_used[:60]}...")
            self.logger.print_log(f"  char_tags:    {built_prompt.char_tags_used}")
            self.logger.print_log(f"  scene_tags:   {built_prompt.scene_tags_used}")
            self.logger.print_log(f"  composition:  {built_prompt.composition_tags_used}")

            try:
                prompt_overrides = {
                    "6": {"text": built_prompt.positive},
                    "7": {"text": built_prompt.negative},
                    "5": {"width": 1024, "height": 1024},
                    "12": {"image": ref_char.comfyui_input_filename}
                }
                _, filename, saved_path = self.comfyui_client.generate_with_workflow(
                    PANEL_WORKFLOW_PATH, prompt_overrides, PANEL_DIR
                )

                panel_images[panel_id] = saved_path

                self.logger.add_image_log(
                    "ImageAgent", built_prompt, saved_path,
                    {"panel_id": panel_id, "ref_character": ref_char.name, "status": "success"}
                )
                self.logger.print_log(f"  ✓ 保存: {saved_path}")

            except Exception as e:
                self.logger.print_log(f"  ✗ 失败: {e}")
                panel_images[panel_id] = f"error: {e}"
                self.logger.add_agent_log(
                    "ImageAgent", f"Panel {panel_id}",
                    {"error": str(e)}, {"panel_id": panel_id, "status": "error"}
                )

        success = len([p for p in panel_images.values() if not str(p).startswith("error")])
        self.logger.print_log(f"\n[ImageAgent] 完成！成功 {success}/{max_panels} 张")
        return panel_images

    # =========================================================================
    # Agent 8: QualityControlAgent ★ 新增Agent
    # 职责：审查整个工作流的输出质量，发现问题并记录建议
    # =========================================================================
    def quality_control_agent(self, script_data: dict, standardized_chars: List[StandardizedCharacter],
                               style_config: DynamicStyleConfig, panels: list,
                               panel_images: dict) -> QualityReport:
        """
        质量控制Agent

        为什么需要这个Agent：
        1. 各步骤可能失败，需要统一汇总问题
        2. 检查Tag一致性（分镜图是否都用了相同的画风前缀）
        3. 检查角色覆盖率（是否所有角色都有人设图）
        4. 检查分镜图数量是否达标
        5. 为用户提供改进建议
        """
        self.logger.print_log("\n" + "─" * 50)
        self.logger.print_log("[QualityControlAgent] 开始质量检查...")

        issues = []
        suggestions = []
        metadata = {}

        # 1. 检查角色标准化质量
        for char in standardized_chars:
            if not char.core_tags or len(char.core_tags.split(",")) < 3:
                issues.append(f"角色 '{char.name}' 的core_tags过于简单: '{char.core_tags}'")
                suggestions.append(f"考虑为 '{char.name}' 手动补充更详细的外貌Tag")

            # 检查是否有画风Tag泄漏到core_tags
            style_leak = ["anime", "realistic", "masterpiece", "quality", "detailed"]
            for leak in style_leak:
                if leak in char.core_tags.lower():
                    issues.append(f"角色 '{char.name}' 的core_tags包含画风词 '{leak}'，可能影响一致性")

            if not char.character_image_path:
                issues.append(f"角色 '{char.name}' 没有生成人设图")
                suggestions.append(f"请确保ComfyUI正常运行并重新生成")

        # 2. 检查分镜覆盖率
        expected_panels = len(panels)
        actual_panels = len([p for p in panel_images.values() if not str(p).startswith("error")])
        coverage = actual_panels / expected_panels if expected_panels > 0 else 0
        metadata["panel_coverage"] = f"{actual_panels}/{expected_panels} ({coverage:.0%})"

        if coverage < 0.5:
            issues.append(f"分镜图生成率偏低: {actual_panels}/{expected_panels}")
            suggestions.append("检查ComfyUI稳定性，考虑减少max_images参数")
        elif coverage < 0.8:
            suggestions.append(f"部分分镜图生成失败 ({actual_panels}/{expected_panels})，可重新运行")

        # 3. 检查风格配置完整性
        style_prefix = style_config.get_style_prefix()
        if "anime realism" not in style_prefix.lower() and "anime" not in style_prefix.lower():
            issues.append("风格配置中未检测到anime相关词，图像风格可能不一致")
            suggestions.append("考虑在DynamicStyleConfig中添加'anime realism'")

        # 4. 检查角色Tag在分镜Prompt中的一致性（通过日志验证）
        metadata["style_prefix"] = style_prefix[:100]
        metadata["characters_with_images"] = len([c for c in standardized_chars if c.character_image_path])
        metadata["characters_total"] = len(standardized_chars)

        # 5. 使用LLM进行智能质量评估
        try:
            qa_prompt = f"""Review this manga generation workflow output and identify any issues:

Script: {script_data.get('title')} ({script_data.get('genre')})
Characters generated: {len([c for c in standardized_chars if c.character_image_path])}/{len(standardized_chars)}
Panels generated: {actual_panels}/{expected_panels}
Art style: {style_config.style_name}
Style prefix: {style_prefix[:150]}

Character tags:
{chr(10).join([f"- {c.name}: {c.core_tags}" for c in standardized_chars])}

Identify: 1) Any obvious tag conflicts 2) Missing important visual elements 3) Suggestions for improvement
Return JSON: {{"issues": [], "suggestions": []}}"""

            response = self.llm_client.chat_completions_create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": qa_prompt}],
                temperature=0.3, max_tokens=600
            )
            content = response["choices"][0]["message"]["content"]
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                qa_result = json.loads(json_match.group())
                issues.extend(qa_result.get("issues", []))
                suggestions.extend(qa_result.get("suggestions", []))

        except Exception as e:
            self.logger.print_log(f"  [WARN] LLM质检失败: {e}")

        passed = len([i for i in issues if "error" in i.lower() or "失败" in i]) == 0
        report = QualityReport(passed=passed, issues=issues, suggestions=suggestions, metadata=metadata)

        self.logger.add_agent_log("QualityControlAgent",
            f"Checking {len(standardized_chars)} chars, {len(panels)} panels",
            {"passed": passed, "issues_count": len(issues),
             "issues": issues, "suggestions": suggestions},
            metadata)

        self.logger.print_log(f"[QualityControlAgent] 质检{'通过' if passed else '发现问题'}！")
        for issue in issues:
            self.logger.print_log(f"  ⚠ {issue}")
        for suggestion in suggestions:
            self.logger.print_log(f"  💡 {suggestion}")
        self.logger.print_log(f"  分镜覆盖率: {metadata['panel_coverage']}")

        return report

    # =========================================================================
    # Agent 9: LayoutAgent
    # =========================================================================
    def layout_agent(self, panels: list, panel_images: dict, target_pages: int) -> list:
        self.logger.print_log("\n" + "─" * 50)
        self.logger.print_log("[LayoutAgent] 开始排版合成...")

        final_pages = []
        for page_num in range(1, target_pages + 1):
            page_panels = [p for p in panels if p.get("page_id") == page_num]
            page_panel_data = []

            for panel in page_panels:
                pid = panel.get("panel_id", "")
                image_path = panel_images.get(pid, "")
                page_panel_data.append({
                    **panel,
                    "image_path": image_path,
                    "image_available": bool(image_path and not str(image_path).startswith("error"))
                })

            final_pages.append({
                "page_number": page_num,
                "panel_count": len(page_panels),
                "panels": page_panel_data,
                "layout_type": self._suggest_layout(len(page_panels))
            })

        self.logger.add_agent_log("LayoutAgent",
            f"{len(panels)} panels, {len(panel_images)} images",
            {"pages": len(final_pages)}, {"status": "success"})
        self.logger.print_log(f"[LayoutAgent] 完成！{len(final_pages)} 页")
        return final_pages

    def _suggest_layout(self, panel_count: int) -> str:
        layout_map = {1: "full_page", 2: "top_bottom", 3: "top_double_bottom",
                      4: "grid_2x2", 5: "asymmetric_5", 6: "grid_2x3"}
        return layout_map.get(panel_count, "flexible_grid")

    # =========================================================================
    # 主流程
    # =========================================================================
    def run(self, user_input: str, target_pages: int = 2, target_panels_per_page: int = 3) -> dict:
        self.logger.print_log("\n" + "=" * 60)
        self.logger.print_log("ComicWeaver v2 - 架构重构版")
        self.logger.print_log("=" * 60)

        self.style_config = None
        self.standardized_characters = []

        comfyui_ok = self.check_comfyui()
        self.logger.print_log(f"[系统] ComfyUI: {'✓ 已连接' if comfyui_ok else '✗ 未运行（跳过图像生成）'}")

        # ── Step 1: 剧本解析
        script_data = self.script_agent(user_input, target_pages, target_panels_per_page)

        # ── Step 2: 风格分析（新增核心步骤）
        self.style_config = self.style_analysis_agent(script_data)

        # ── Step 3: 角色标准化
        if script_data.get("characters"):
            self.standardized_characters = self.character_standardization_agent(
                script_data["characters"], self.style_config
            )

        # ── Step 4: 人设图生成
        if comfyui_ok and self.standardized_characters:
            self.standardized_characters = self.character_design_agent(
                self.standardized_characters, self.style_config
            )

        # ── Step 5: 分镜设计
        panels = self.storyboard_agent(
            script_data, self.standardized_characters, target_pages, target_panels_per_page
        )

        # ── Step 6: 分镜图生成
        panel_images = {}
        if comfyui_ok and self.standardized_characters:
            panel_images = self.image_agent(
                panels, self.standardized_characters, self.style_config,
                max_images=target_pages * target_panels_per_page
            )

        # ── Step 7: 质量控制（新增）
        quality_report = self.quality_control_agent(
            script_data, self.standardized_characters, self.style_config, panels, panel_images
        )

        # ── Step 8: 排版
        final_pages = self.layout_agent(panels, panel_images, target_pages)

        # 输出摘要
        self.logger.print_log("\n" + "=" * 60)
        self.logger.print_log("执行完成")
        self.logger.print_log("=" * 60)
        self.logger.print_log(f"标题:     {script_data.get('title')}")
        self.logger.print_log(f"画风:     {self.style_config.style_name}")
        self.logger.print_log(f"画风前缀: {self.style_config.get_style_prefix()[:100]}...")
        self.logger.print_log(f"角色:")
        for c in self.standardized_characters:
            img_status = "✓" if c.character_image_path else "✗"
            self.logger.print_log(f"  {img_status} {c.name}: {c.core_tags}")
        self.logger.print_log(f"分镜: {len(panels)} 个设计 / {len(panel_images)} 张图")
        self.logger.print_log(f"质检: {'通过' if quality_report.passed else '有问题'}")
        self.logger.print_log(f"日志: {LOG_FILE}")

        return {
            "script_data": script_data,
            "style_config": {
                "style_name": self.style_config.style_name,
                "style_prefix": self.style_config.get_style_prefix(),
                "negative_prompt": self.style_config.negative_prompt
            },
            "standardized_characters": [
                {"name": c.name, "role": c.role, "core_tags": c.core_tags,
                 "image_path": c.character_image_path}
                for c in self.standardized_characters
            ],
            "panels": panels,
            "panel_images": panel_images,
            "final_pages": final_pages,
            "quality_report": {
                "passed": quality_report.passed,
                "issues": quality_report.issues,
                "suggestions": quality_report.suggestions,
                "metadata": quality_report.metadata
            }
        }


# ============================================================================
# LangGraph 模式
# ============================================================================

if LANGGRAPH_AVAILABLE:
    class ComicState(TypedDict):
        user_input: str
        target_pages: int
        target_panels_per_page: int
        script_data: dict
        style_config: dict          # DynamicStyleConfig 序列化后的dict
        standardized_characters: list
        panels: list
        panel_images: dict
        final_pages: list
        quality_report: dict
        messages: Annotated[Sequence[BaseMessage], add_messages]
        errors: list

    def _style_config_to_dict(cfg: DynamicStyleConfig) -> dict:
        return {
            "style_name": cfg.style_name, "quality_prefix": cfg.quality_prefix,
            "core_style_tags": cfg.core_style_tags, "lighting_tags": cfg.lighting_tags,
            "negative_prompt": cfg.negative_prompt
        }

    def _dict_to_style_config(d: dict) -> DynamicStyleConfig:
        return DynamicStyleConfig(**d)

    def build_langgraph_workflow():
        wf = ComicWorkflow()

        def script_node(state: ComicState) -> dict:
            data = wf.script_agent(state["user_input"], state["target_pages"], state["target_panels_per_page"])
            return {"script_data": data,
                    "messages": [AIMessage(content=f"剧本解析完成: {data.get('title')}")]}

        def style_node(state: ComicState) -> dict:
            cfg = wf.style_analysis_agent(state["script_data"])
            wf.style_config = cfg
            return {"style_config": _style_config_to_dict(cfg),
                    "messages": [AIMessage(content=f"风格确定: {cfg.style_name}")]}

        def standardization_node(state: ComicState) -> dict:
            cfg = _dict_to_style_config(state["style_config"])
            chars = wf.character_standardization_agent(state["script_data"].get("characters", []), cfg)
            return {
                "standardized_characters": [
                    {"name": c.name, "role": c.role, "core_tags": c.core_tags,
                     "appearance_desc": c.appearance_desc, "personality": c.personality,
                     "backstory": c.backstory, "character_image_path": "",
                     "character_image_filename": "", "comfyui_input_filename": ""}
                    for c in chars
                ],
                "messages": [AIMessage(content=f"标准化了 {len(chars)} 个角色")]
            }

        def character_design_node(state: ComicState) -> dict:
            wf.check_comfyui()
            cfg = _dict_to_style_config(state["style_config"])
            chars = [StandardizedCharacter(**c) for c in state["standardized_characters"]]
            chars = wf.character_design_agent(chars, cfg)
            return {
                "standardized_characters": [
                    {"name": c.name, "role": c.role, "core_tags": c.core_tags,
                     "appearance_desc": c.appearance_desc, "personality": c.personality,
                     "backstory": c.backstory, "character_image_path": c.character_image_path,
                     "character_image_filename": c.character_image_filename,
                     "comfyui_input_filename": c.comfyui_input_filename}
                    for c in chars
                ],
                "messages": [AIMessage(content=f"生成了 {len([c for c in chars if c.character_image_path])} 张人设图")]
            }

        def storyboard_node(state: ComicState) -> dict:
            chars = [StandardizedCharacter(**c) for c in state["standardized_characters"]]
            panels = wf.storyboard_agent(state["script_data"], chars,
                                          state["target_pages"], state["target_panels_per_page"])
            return {"panels": panels, "messages": [AIMessage(content=f"设计了 {len(panels)} 个分镜")]}

        def image_node(state: ComicState) -> dict:
            cfg = _dict_to_style_config(state["style_config"])
            chars = [StandardizedCharacter(**c) for c in state["standardized_characters"]]
            images = wf.image_agent(state["panels"], chars, cfg,
                                     max_images=state["target_pages"] * state["target_panels_per_page"])
            return {"panel_images": images,
                    "messages": [AIMessage(content=f"生成了 {len(images)} 张分镜图")]}

        def quality_control_node(state: ComicState) -> dict:
            cfg = _dict_to_style_config(state["style_config"])
            chars = [StandardizedCharacter(**c) for c in state["standardized_characters"]]
            report = wf.quality_control_agent(state["script_data"], chars, cfg,
                                               state["panels"], state["panel_images"])
            return {
                "quality_report": {
                    "passed": report.passed, "issues": report.issues,
                    "suggestions": report.suggestions, "metadata": report.metadata
                },
                "messages": [AIMessage(content=f"质检{'通过' if report.passed else '发现问题'}")]
            }

        def layout_node(state: ComicState) -> dict:
            pages = wf.layout_agent(state["panels"], state["panel_images"], state["target_pages"])
            return {"final_pages": pages, "messages": [AIMessage(content=f"排版完成: {len(pages)} 页")]}

        graph = StateGraph(ComicState)
        for name, fn in [
            ("script_agent", script_node),
            ("style_analysis_agent", style_node),
            ("standardization_agent", standardization_node),
            ("character_design_agent", character_design_node),
            ("storyboard_agent", storyboard_node),
            ("image_agent", image_node),
            ("quality_control_agent", quality_control_node),
            ("layout_agent", layout_node),
        ]:
            graph.add_node(name, fn)

        graph.add_edge(START, "script_agent")
        graph.add_edge("script_agent", "style_analysis_agent")
        graph.add_edge("style_analysis_agent", "standardization_agent")
        graph.add_edge("standardization_agent", "character_design_agent")
        graph.add_edge("character_design_agent", "storyboard_agent")
        graph.add_edge("storyboard_agent", "image_agent")
        graph.add_edge("image_agent", "quality_control_agent")
        graph.add_edge("quality_control_agent", "layout_agent")
        graph.add_edge("layout_agent", END)

        return graph.compile()


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("ComicWeaver v2 工作流测试")
    print("=" * 60)

    user_input = (
        "一个关于少年冒险的故事。主角小明是一个勇敢的少年，银发金瞳，穿着蓝色铠甲。"
        "他在一次意外中发现了一个神秘的古老遗迹，遇到了守护者阿龙（红发黑瞳，穿着红色斗篷）。"
        "两人一起探索遗迹，最终发现了隐藏的宝藏。"
    )
    target_pages = 2
    target_panels_per_page = 3

    if LANGGRAPH_AVAILABLE:
        print("[INFO] LangGraph 模式")
        graph = build_langgraph_workflow()
        result = graph.invoke({
            "user_input": user_input,
            "target_pages": target_pages,
            "target_panels_per_page": target_panels_per_page,
            "script_data": {}, "style_config": {},
            "standardized_characters": [], "panels": [],
            "panel_images": {}, "final_pages": {},
            "quality_report": {}, "messages": [], "errors": []
        })
    else:
        print("[INFO] 简化模式")
        workflow = ComicWorkflow()
        result = workflow.run(user_input, target_pages, target_panels_per_page)

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)
    print(f"标题: {result['script_data'].get('title')}")
    if result.get('style_config'):
        print(f"画风: {result['style_config'].get('style_name')}")
        print(f"画风前缀: {result['style_config'].get('style_prefix', '')[:100]}...")
    for c in result.get('standardized_characters', []):
        print(f"  角色 {c['name']}: {c['core_tags']}")


if __name__ == "__main__":
    main()