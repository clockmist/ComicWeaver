"""
ComicWeaver 完整工作流 - 整合版
使用 LangGraph 框架连接各个 Agent，集成真实 ComfyUI 图像生成：
- LLM API: 智谱 AI GLM-4 (HTTP 直接调用)
- 文生图 API: ComfyUI (人设生成 + IPAdapter分镜图生成)

特性：
1. 记录每个 Agent 的输入 prompt 和输出到日志文件
2. 图像保存到 output 文件夹
3. 支持两种模式：LangGraph 模式和简化模式
4. LLM生成高质量英文SDXL prompt，符合tag规范
"""
import os
import sys
import json
import time
import uuid
import urllib.parse
import requests
import random
from typing import Dict, Any, List, Optional
from datetime import datetime

# 设置输出编码为 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# 配置
# ============================================================================

ZHIPUAI_API_KEY = "2487ed1dfc9f456fbd5630be90a18575.kPSrC0ZoSbjHBwgL"
COMFYUI_SERVER = "127.0.0.1:8188"

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
CHARACTER_DIR = os.path.join(OUTPUT_DIR, "characters")  # 人设图保存目录
PANEL_DIR = os.path.join(OUTPUT_DIR, "panels")  # 分镜图保存目录
LOG_FILE = os.path.join(OUTPUT_DIR, "workflow_log.json")

# 工作流模板路径
CHARACTER_WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "generate_character.json")
PANEL_WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "generate_picture.json")

# 检查是否安装了 LangGraph
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from typing import TypedDict, Annotated, Sequence
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# ============================================================================
# 日志记录器
# ============================================================================

class WorkflowLogger:
    """工作流日志记录器"""

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.logs = {
            "start_time": datetime.now().isoformat(),
            "agents": []
        }
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def add_agent_log(self, agent_name: str, input_prompt: str, output: Any, metadata: dict = None):
        """添加 Agent 日志"""
        agent_log = {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "input_prompt": input_prompt,
            "output": output,
            "metadata": metadata or {}
        }
        self.logs["agents"].append(agent_log)
        self.save()

    def add_image_log(self, agent_name: str, input_prompt: str, image_path: str, metadata: dict = None):
        """添加图像生成日志"""
        agent_log = {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "input_prompt": input_prompt,
            "output_image": image_path,
            "metadata": metadata or {}
        }
        self.logs["agents"].append(agent_log)
        self.save()

    def save(self):
        """保存日志到文件"""
        self.logs["end_time"] = datetime.now().isoformat()
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

    def print_log(self, message: str):
        """打印并记录日志"""
        print(message)


# ============================================================================
# API 客户端
# ============================================================================

class ZhipuAIHTTPClient:
    """智谱 AI HTTP 客户端"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://open.bigmodel.cn/api/paas/v4"

    def chat_completions_create(
        self,
        model: str = "glm-4-flash",
        messages: list = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """调用聊天补全 API"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        return response.json()


class ComfyUIClient:
    """ComfyUI 文生图客户端 - 支持人设生成和分镜图生成"""

    def __init__(self, server_address: str = COMFYUI_SERVER):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, workflow: dict) -> dict:
        """提交工作流到 ComfyUI"""
        p = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            f"http://{self.server_address}/prompt",
            json=p,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def get_history(self, prompt_id: str) -> dict:
        """获取生成历史"""
        response = requests.get(
            f"http://{self.server_address}/history/{prompt_id}",
            timeout=30
        )
        return response.json()

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """获取生成的图片"""
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }
        url = f"http://{self.server_address}/view?{urllib.parse.urlencode(params)}"
        response = requests.get(url, timeout=30)
        return response.content

    def upload_image(self, image_data: bytes, filename: str, folder_type: str = "input", subfolder: str = "") -> dict:
        """
        上传图片到ComfyUI的input目录
        这是关键：让LoadImage节点能访问到人设图
        """
        import io
        files = {
            'image': (filename, io.BytesIO(image_data), 'image/png'),
            'type': (None, folder_type),
        }
        if subfolder:
            files['subfolder'] = (None, subfolder)

        url = f"http://{self.server_address}/upload/image"
        response = requests.post(url, files=files, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_image_from_output(self, prompt_id: str, timeout: int = 300) -> tuple:
        """
        等待生成完成并从输出节点获取图片
        返回: (image_data, filename)
        """
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})

                # 查找包含images的输出节点（PreviewImage或SaveImage）
                for node_id, node_output in outputs.items():
                    if "images" in node_output and node_output["images"]:
                        image_info = node_output["images"][0]
                        filename = image_info["filename"]
                        subfolder = image_info.get("subfolder", "")
                        folder_type = image_info.get("type", "output")

                        image_data = self.get_image(filename, subfolder, folder_type)
                        return image_data, filename

                # 检查是否执行完成但没有图片输出
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "error" or status.get("completed", False):
                    errors = history[prompt_id].get("status", {}).get("messages", [])
                    raise RuntimeError(f"ComfyUI execution failed: {errors}")

            time.sleep(2)

        raise TimeoutError(f"Generation timeout for prompt_id {prompt_id}")

    def generate_character(
        self,
        prompt_text: str,
        negative_prompt: str = "",
        seed: int = -1,
        width: int = 832,
        height: int = 1216
    ) -> tuple:
        """
        生成角色人设图
        使用 generate_character.json 工作流
        返回: (saved_path, filename, comfyui_input_filename)
        """
        # 加载人设生成工作流模板
        with open(CHARACTER_WORKFLOW_PATH, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        # 修改节点参数
        # 节点39是角色描述prompt，节点38是质量前缀
        if "39" in workflow:
            workflow["39"]["inputs"]["prompt"] = prompt_text

        if "5" in workflow:
            workflow["5"]["inputs"]["width"] = width
            workflow["5"]["inputs"]["height"] = height

        if "3" in workflow:
            if seed == -1:
                workflow["3"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
            else:
                workflow["3"]["inputs"]["seed"] = seed

        # 提交生成任务
        result = self.queue_prompt(workflow)
        prompt_id = result["prompt_id"]

        # 等待并获取图片
        image_data, filename = self.get_image_from_output(prompt_id)

        # 保存到本地角色目录
        os.makedirs(CHARACTER_DIR, exist_ok=True)
        saved_path = os.path.join(CHARACTER_DIR, filename)
        with open(saved_path, "wb") as f:
            f.write(image_data)

        # 上传到ComfyUI的input目录，这样LoadImage节点才能访问
        # 使用固定前缀方便后续查找
        input_filename = f"char_{filename}"
        try:
            self.upload_image(image_data, input_filename, folder_type="input")
            print(f"[ComfyUI] 人设图已上传到input目录: {input_filename}")
        except Exception as e:
            print(f"[WARN] 上传图片到ComfyUI失败: {e}")
            # 尝试复制到ComfyUI的input目录（如果知道路径）
            input_filename = filename  # fallback

        return saved_path, filename, input_filename

    def generate_panel(
        self,
        prompt_text: str,
        character_image_filename: str,
        negative_prompt: str = "",
        seed: int = -1,
        width: int = 1024,
        height: int = 1024
    ) -> tuple:
        """
        生成分镜图（使用IPAdapter保持角色一致性）
        使用 generate_picture.json 工作流
        character_image_filename: 已上传到ComfyUI input目录的文件名
        返回: (saved_path, filename)
        """
        # 加载分镜图生成工作流模板
        with open(PANEL_WORKFLOW_PATH, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        # 修改节点参数
        # 节点6是正向prompt
        if "6" in workflow:
            workflow["6"]["inputs"]["text"] = prompt_text

        # 节点7是负向prompt
        if "7" in workflow and negative_prompt:
            workflow["7"]["inputs"]["text"] = negative_prompt

        # 节点5设置尺寸
        if "5" in workflow:
            workflow["5"]["inputs"]["width"] = width
            workflow["5"]["inputs"]["height"] = height

        # 节点3设置seed
        if "3" in workflow:
            if seed == -1:
                workflow["3"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
            else:
                workflow["3"]["inputs"]["seed"] = seed

        # 节点12加载人设参考图 - 使用已上传到input目录的文件名
        if "12" in workflow:
            workflow["12"]["inputs"]["image"] = character_image_filename

        # 提交生成任务
        result = self.queue_prompt(workflow)
        prompt_id = result["prompt_id"]

        # 等待并获取图片
        image_data, filename = self.get_image_from_output(prompt_id)

        # 保存到分镜目录
        os.makedirs(PANEL_DIR, exist_ok=True)
        saved_path = os.path.join(PANEL_DIR, filename)
        with open(saved_path, "wb") as f:
            f.write(image_data)

        return saved_path, filename


# ============================================================================
# Prompt 构建器 - 生成高质量英文SDXL prompt
# ============================================================================

class PromptBuilder:
    """构建符合SDXL规范的高质量英文prompt"""

    # SDXL质量前缀（用于人设生成）
    CHARACTER_QUALITY_PREFIX = (
        "masterpiece, best quality, ultra high res, hyper-detailed, "
        "anime realism, 8K, newest, official art, "
    )

    # SDXL质量前缀（用于分镜图生成）
    PANEL_QUALITY_PREFIX = (
        "masterpiece, best quality, ultra high res, hyper-detailed, "
        "anime style, cinematic composition, 8K, newest, "
    )

    # 标准负向prompt
    DEFAULT_NEGATIVE = (
        "lowres, worst quality, bad quality, bad anatomy, simple background, "
        "blurry, lowres, cropped, extra limbs, bad anatomy, deformed hands, "
        "bad face, sketch, jpeg artifacts, signature, watermark, old, oldest, "
        "3d render, photorealistic, western comic style, realistic"
    )

    @classmethod
    def build_character_prompt(cls, character_info: dict) -> str:
        """
        根据角色信息构建人设生成prompt
        遵循Danbooru tag风格，适合SDXL anime模型
        """
        name = character_info.get("name", "character")
        appearance = character_info.get("appearance", "")
        personality = character_info.get("personality", "")

        # 构建基础角色描述
        tags = []

        # 角色数量
        tags.append("1girl" if "female" in appearance.lower() or "girl" in appearance.lower() else "1boy")
        tags.append("solo")

        # 外貌特征转换为tags
        appearance_tags = cls._appearance_to_tags(appearance)
        tags.extend(appearance_tags)

        # 姿势和构图（人设图用标准肖像）
        tags.extend([
            "portrait",
            "looking at viewer",
            "simple background",
            "white background",
            "upper body"
        ])

        # 风格tag
        tags.extend([
            "anime style",
            "detailed eyes",
            "detailed face",
            "sharp focus"
        ])

        # 组合prompt
        character_desc = ", ".join(tags)
        full_prompt = cls.CHARACTER_QUALITY_PREFIX + character_desc

        return full_prompt

    @classmethod
    def build_panel_prompt(
        cls,
        character_info: dict,
        scene_info: dict,
        panel_info: dict
    ) -> str:
        """
        构建分镜图生成prompt
        包含角色一致性tag + 场景描述 + 动作/构图
        """
        tags = []

        # 角色基础tag（保持与人设一致的核心特征）
        appearance = character_info.get("appearance", "")
        appearance_tags = cls._appearance_to_tags(appearance)
        tags.extend(appearance_tags)

        # 角色数量
        tags.append("1girl" if "female" in appearance.lower() or "girl" in appearance.lower() else "1boy")

        # 场景描述
        location = scene_info.get("location", "")
        time_of_day = scene_info.get("time_of_day", "")
        atmosphere = scene_info.get("atmosphere", "")

        if location:
            tags.append(location)
        if time_of_day:
            tags.append(time_of_day)
        if atmosphere:
            tags.append(atmosphere)

        # 分镜特定信息
        shot_size = panel_info.get("shot_size", "medium")
        camera_angle = panel_info.get("camera_angle", "eye_level")
        primary_action = panel_info.get("primary_action", "")
        mood = panel_info.get("mood", "")

        # 景别tag
        shot_tags = {
            "extreme_long": "extreme wide shot",
            "long": "wide shot",
            "full": "full body",
            "medium": "medium shot",
            "close": "close-up",
            "extreme_close": "extreme close-up"
        }
        tags.append(shot_tags.get(shot_size, "medium shot"))

        # 角度tag
        angle_tags = {
            "eye_level": "eye-level shot",
            "high_angle": "high angle",
            "low_angle": "low angle"
        }
        tags.append(angle_tags.get(camera_angle, "eye-level shot"))

        # 动作
        if primary_action:
            tags.append(primary_action)

        # 情绪
        if mood:
            tags.append(mood)

        # 画面质量
        tags.extend([
            "detailed background",
            "dynamic angle",
            "cinematic lighting",
            "anime style"
        ])

        # 组合prompt
        scene_desc = ", ".join(tags)
        full_prompt = cls.PANEL_QUALITY_PREFIX + scene_desc

        return full_prompt

    @classmethod
    def _appearance_to_tags(cls, appearance: str) -> List[str]:
        """将外貌描述转换为Danbooru风格的tags"""
        tags = []
        appearance_lower = appearance.lower()

        # 发色
        hair_colors = {
            "black": "black hair", "white": "white hair", "silver": "silver hair",
            "blonde": "blonde hair", "yellow": "blonde hair", "brown": "brown hair",
            "red": "red hair", "blue": "blue hair", "green": "green hair",
            "purple": "purple hair", "pink": "pink hair", "orange": "orange hair",
            "gray": "grey hair", "grey": "grey hair"
        }
        for color, tag in hair_colors.items():
            if color in appearance_lower:
                tags.append(tag)
                break

        # 发型
        hair_styles = {
            "long": "long hair", "short": "short hair", "medium": "medium hair",
            "ponytail": "ponytail", "twintail": "twintails", "braid": "braid",
            "bob": "bob cut", "ahoge": "ahoge", "bangs": "bangs"
        }
        for style, tag in hair_styles.items():
            if style in appearance_lower:
                tags.append(tag)

        # 瞳色
        eye_colors = {
            "red eye": "red eyes", "blue eye": "blue eyes", "green eye": "green eyes",
            "yellow eye": "yellow eyes", "purple eye": "purple eyes", "golden eye": "golden eyes",
            "amber eye": "amber eyes", "brown eye": "brown eyes", "black eye": "black eyes"
        }
        for desc, tag in eye_colors.items():
            if desc in appearance_lower:
                tags.append(tag)
                break

        # 服装元素
        clothing = {
            "school uniform": "school uniform", "armor": "armor", "dress": "dress",
            "kimono": "kimono", "suit": "suit", "jacket": "jacket",
            "cape": "cape", "hood": "hood", "glasses": "glasses"
        }
        for item, tag in clothing.items():
            if item in appearance_lower:
                tags.append(tag)

        # 其他特征
        features = {
            "dark skin": "dark skin", "tattoo": "tattoo", "scar": "scar",
            "earring": "earrings", "necklace": "necklace", "ribbon": "ribbon"
        }
        for feat, tag in features.items():
            if feat in appearance_lower:
                tags.append(tag)

        return tags

    @classmethod
    def get_negative_prompt(cls) -> str:
        """获取标准负向prompt"""
        return cls.DEFAULT_NEGATIVE


# ============================================================================
# Agent 实现
# ============================================================================

class ComicWorkflow:
    """漫画创作工作流 - 集成真实图像生成"""

    def __init__(self):
        self.llm_client = ZhipuAIHTTPClient(ZHIPUAI_API_KEY)
        self.comfyui_client = None
        self.logger = WorkflowLogger(LOG_FILE)

    def check_comfyui(self) -> bool:
        """检查 ComfyUI 是否运行"""
        try:
            response = requests.get(
                f"http://{COMFYUI_SERVER}/system_stats",
                timeout=5
            )
            if response.status_code == 200:
                self.comfyui_client = ComfyUIClient()
                return True
            return False
        except Exception as e:
            self.logger.print_log(f"[WARN] ComfyUI连接失败: {e}")
            return False

    def script_agent(self, user_input: str, target_pages: int, target_panels_per_page: int) -> dict:
        """
        剧本理解 Agent - 使用真实 LLM API
        解析用户输入，生成剧本、角色、场景
        """
        self.logger.print_log("\n[ScriptAgent] 开始解析剧本...")

        system_prompt = """You are a professional manga script analyst. Analyze the user's plot description and generate:
1. Manga title
2. Plot summary  
3. Main characters (2-3, with name, role, detailed appearance description, personality)
4. Scene list (each with location, time_of_day, atmosphere, characters_present, dialogues, emotion_intensity 0-1)

Return ONLY valid JSON with these exact fields:
- title: string
- summary: string  
- characters: array of {name, role(protagonist/antagonist/supporting), appearance (detailed physical description including hair color/style, eye color, clothing, distinguishing features), personality}
- scenes: array of {location, time_of_day, atmosphere, characters_present(array), dialogues(array), emotion_intensity(number)}

Write appearance descriptions in English, detailed enough for image generation (e.g., "long silver hair, golden eyes, dark skin, wearing a blue combat armor with red cape")."""

        user_prompt = f"Analyze this plot and generate manga script data:\n\n{user_input}\n\nTarget: {target_pages} pages, {target_panels_per_page} panels per page"

        full_input = f"System: {system_prompt}\n\nUser: {user_prompt}"

        try:
            response = self.llm_client.chat_completions_create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            content = response["choices"][0]["message"]["content"]

            # 提取JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(content)

            script_data = {
                "title": result.get("title", "Untitled Manga"),
                "summary": result.get("summary", ""),
                "characters": result.get("characters", []),
                "scenes": result.get("scenes", []),
                "emotion_curve": [s.get("emotion_intensity", 0.5) for s in result.get("scenes", [])]
            }

            self.logger.add_agent_log(
                agent_name="ScriptAgent",
                input_prompt=full_input,
                output=script_data,
                metadata={"model": "glm-4-flash", "status": "success"}
            )

            self.logger.print_log(f"[ScriptAgent] 完成！标题：{script_data['title']}")
            self.logger.print_log(f"[ScriptAgent] 角色：{len(script_data['characters'])}，场景：{len(script_data['scenes'])}")

            return script_data

        except Exception as e:
            self.logger.print_log(f"[ScriptAgent] 错误：{e}")
            default_data = {
                "title": "Untitled Manga",
                "summary": user_input,
                "characters": [
                    {
                        "name": "Protagonist",
                        "role": "protagonist",
                        "appearance": "short black hair, brown eyes, wearing a simple tunic",
                        "personality": "brave and curious"
                    }
                ],
                "scenes": [
                    {
                        "location": "mysterious ruins",
                        "time_of_day": "day",
                        "atmosphere": "adventurous",
                        "characters_present": ["Protagonist"],
                        "emotion_intensity": 0.5
                    }
                ],
                "emotion_curve": [0.5]
            }

            self.logger.add_agent_log(
                agent_name="ScriptAgent",
                input_prompt=full_input,
                output={"error": str(e), "fallback": default_data},
                metadata={"status": "error"}
            )

            return default_data

    def storyboard_agent(self, script_data: dict, target_pages: int, target_panels_per_page: int) -> list:
        """
        分镜设计 Agent - 使用真实 LLM API
        生成详细分镜计划，包含SDXL-ready的英文prompt
        """
        self.logger.print_log("\n[StoryboardAgent] 开始设计分镜...")

        system_prompt = """You are a professional manga storyboard artist. Design detailed panel plans based on the script.

For each panel, generate:
- panel_id: unique ID
- page_id: page number
- shot_size: one of [extreme_long, long, full, medium, close, extreme_close]
- camera_angle: one of [eye_level, high_angle, low_angle]
- characters_in_panel: list of character names present
- primary_action: detailed action description (English, suitable for prompt)
- setting: scene setting description (English)
- mood: emotional mood (English)
- positive_prompt: COMPLETE English SDXL prompt following this format:
  "masterpiece, best quality, [character tags], [shot size], [camera angle], [action], [setting], [mood], detailed background, cinematic lighting, anime style"
  Use Danbooru-style tags for characters (1girl/1boy, hair color, eye color, clothing).
- negative_prompt: standard negative prompt

Return JSON with "panels" array."""

        user_prompt = f"Design storyboard for:\nTitle: {script_data['title']}\nSummary: {script_data['summary']}\nCharacters: {json.dumps(script_data['characters'], ensure_ascii=False)}\nScenes: {json.dumps(script_data['scenes'], ensure_ascii=False)}\n\nTarget: {target_pages} pages, {target_panels_per_page} panels per page"

        full_input = f"System: {system_prompt}\n\nUser: {user_prompt}"

        try:
            response = self.llm_client.chat_completions_create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=3000
            )

            content = response["choices"][0]["message"]["content"]

            # 提取JSON
            import re
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                panels = json.loads(json_match.group())
            else:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    result = json.loads(json_match.group())
                    panels = result.get("panels", [])
                else:
                    panels = []

            self.logger.add_agent_log(
                agent_name="StoryboardAgent",
                input_prompt=full_input,
                output={"panels": panels, "count": len(panels)},
                metadata={"model": "glm-4-flash", "status": "success"}
            )

            self.logger.print_log(f"[StoryboardAgent] 完成！{len(panels)} 个分镜")

            return panels

        except Exception as e:
            self.logger.print_log(f"[StoryboardAgent] 错误：{e}")
            # 规则生成fallback
            panels = []
            panel_id = 0
            for page in range(target_pages):
                for panel in range(target_panels_per_page):
                    scene_idx = panel_id % len(script_data['scenes']) if script_data['scenes'] else 0
                    scene = script_data['scenes'][scene_idx] if script_data['scenes'] else {}
                    char = script_data['characters'][0] if script_data['characters'] else {"name": "Unknown", "appearance": "unknown character"}

                    panels.append({
                        "panel_id": f"panel_{panel_id:03d}",
                        "page_id": f"page_{page:03d}",
                        "shot_size": "medium",
                        "camera_angle": "eye_level",
                        "characters_in_panel": scene.get("characters_present", [char["name"]]),
                        "primary_action": "standing",
                        "setting": scene.get("location", "unknown location"),
                        "mood": scene.get("atmosphere", "neutral"),
                        "positive_prompt": PromptBuilder.build_panel_prompt(
                            char, scene, {"shot_size": "medium", "camera_angle": "eye_level", "primary_action": "standing", "mood": scene.get("atmosphere", "neutral")}
                        ),
                        "negative_prompt": PromptBuilder.get_negative_prompt()
                    })
                    panel_id += 1

            self.logger.add_agent_log(
                agent_name="StoryboardAgent",
                input_prompt=full_input,
                output={"panels": panels, "count": len(panels), "fallback": True},
                metadata={"status": "fallback", "error": str(e)}
            )

            self.logger.print_log(f"[StoryboardAgent] Fallback生成了 {len(panels)} 个分镜")
            return panels

    def character_design_agent(self, characters: list) -> dict:
        """
        角色设计 Agent - 生成角色人设图
        为每个主要角色生成一张人设参考图
        """
        self.logger.print_log("\n[CharacterDesignAgent] 开始生成角色人设图...")

        if self.comfyui_client is None:
            self.logger.print_log("[CharacterDesignAgent] ComfyUI未运行，跳过")
            self.logger.add_agent_log(
                agent_name="CharacterDesignAgent",
                input_prompt="ComfyUI not running",
                output={"status": "skipped", "reason": "ComfyUI not available"}
            )
            return {}

        character_images = {}

        for i, char in enumerate(characters):
            char_name = char.get("name", f"character_{i}")
            self.logger.print_log(f"[CharacterDesignAgent] 生成角色 {i+1}/{len(characters)}: {char_name}")

            try:
                # 使用PromptBuilder构建高质量prompt
                prompt = PromptBuilder.build_character_prompt(char)
                negative = PromptBuilder.get_negative_prompt()

                self.logger.print_log(f"[CharacterDesignAgent] Prompt: {prompt[:120]}...")

                # 生成人设图
                saved_path, filename, input_filename = self.comfyui_client.generate_character(
                    prompt_text=prompt,
                    negative_prompt=negative,
                    seed=-1,
                    width=832,
                    height=1216
                )

                character_images[char_name] = {
                    "path": saved_path,
                    "filename": filename,
                    "input_filename": input_filename,  # 用于IPAdapter的input目录文件名
                    "prompt": prompt
                }

                self.logger.add_image_log(
                    agent_name="CharacterDesignAgent",
                    input_prompt=prompt,
                    image_path=saved_path,
                    metadata={
                        "character": char_name,
                        "type": "character_design",
                        "status": "success"
                    }
                )

                self.logger.print_log(f"[CharacterDesignAgent] 完成：{saved_path}")

            except Exception as e:
                self.logger.print_log(f"[CharacterDesignAgent] 生成失败：{e}")
                character_images[char_name] = {"error": str(e)}

                self.logger.add_agent_log(
                    agent_name="CharacterDesignAgent",
                    input_prompt=f"Character: {char_name}",
                    output={"error": str(e)},
                    metadata={"character": char_name, "status": "error"}
                )

        self.logger.print_log(f"[CharacterDesignAgent] 完成！生成了 {len([v for v in character_images.values() if 'path' in v])} 张人设图")
        return character_images

    def image_agent(self, panels: list, character_images: dict, max_images: int = 6) -> dict:
        """
        图像生成 Agent - 使用ComfyUI + IPAdapter生成分镜图
        """
        self.logger.print_log("\n[ImageAgent] 开始生成分镜图...")

        if self.comfyui_client is None:
            self.logger.print_log("[ImageAgent] ComfyUI未运行，跳过图像生成")
            self.logger.add_agent_log(
                agent_name="ImageAgent",
                input_prompt="ComfyUI not running",
                output={"status": "skipped", "reason": "ComfyUI not available"}
            )
            return {}

        panel_images = {}
        max_panels = min(max_images, len(panels))

        self.logger.print_log(f"[ImageAgent] 将生成 {max_panels} 张分镜图")

        for i, panel in enumerate(panels[:max_panels]):
            panel_id = panel.get('panel_id', f'panel_{i:03d}')

            # 获取角色参考图
            chars_in_panel = panel.get('characters_in_panel', [])
            ref_char = chars_in_panel[0] if chars_in_panel else None

            if not ref_char or ref_char not in character_images or 'path' not in character_images.get(ref_char, {}):
                self.logger.print_log(f"[ImageAgent] 跳过 {panel_id}: 无角色参考图")
                continue

            char_image_path = character_images[ref_char]['path']

            # 获取或构建prompt
            if 'positive_prompt' in panel and panel['positive_prompt']:
                prompt = panel['positive_prompt']
            else:
                # fallback: 使用PromptBuilder
                char_info = {"name": ref_char, "appearance": "unknown"}
                scene_info = {"location": panel.get('setting', ''), "atmosphere": panel.get('mood', '')}
                prompt = PromptBuilder.build_panel_prompt(char_info, scene_info, panel)

            negative = panel.get('negative_prompt', PromptBuilder.get_negative_prompt())

            self.logger.print_log(f"[ImageAgent] 生成 {i+1}/{max_panels}: {panel_id}")
            self.logger.print_log(f"[ImageAgent] Prompt: {prompt[:100]}...")

            try:
                # 使用IPAdapter生成，保持角色一致性
                # 使用上传到ComfyUI input目录的文件名
                char_input_filename = character_images[ref_char].get("input_filename", os.path.basename(char_image_path))
                saved_path, filename = self.comfyui_client.generate_panel(
                    prompt_text=prompt,
                    character_image_filename=char_input_filename,
                    negative_prompt=negative,
                    seed=-1,
                    width=1024,
                    height=1024
                )

                panel_images[panel_id] = saved_path

                self.logger.add_image_log(
                    agent_name="ImageAgent",
                    input_prompt=prompt,
                    image_path=saved_path,
                    metadata={
                        "panel_id": panel_id,
                        "character": ref_char,
                        "type": "panel",
                        "status": "success"
                    }
                )

                self.logger.print_log(f"[ImageAgent] 完成：{saved_path}")

            except Exception as e:
                self.logger.print_log(f"[ImageAgent] 生成失败：{e}")
                panel_images[panel_id] = f"error: {str(e)}"

                self.logger.add_image_log(
                    agent_name="ImageAgent",
                    input_prompt=prompt,
                    image_path="",
                    metadata={
                        "panel_id": panel_id,
                        "status": "error",
                        "error": str(e)
                    }
                )

        success_count = len([p for p in panel_images.values() if not str(p).startswith('error')])
        self.logger.print_log(f"[ImageAgent] 完成！成功 {success_count}/{max_panels} 张")

        return panel_images

    def layout_agent(self, panels: list, panel_images: dict, target_pages: int) -> list:
        """
        排版合成 Agent
        """
        self.logger.print_log("\n[LayoutAgent] 开始排版合成...")

        final_pages = []
        for page in range(target_pages):
            page_panels = [
                p for p in panels
                if p.get('page_id') == f"page_{page:03d}"
            ]

            final_pages.append({
                "page_id": f"page_{page:03d}",
                "page_number": page + 1,
                "panel_count": len(page_panels),
                "panels": page_panels
            })

        self.logger.add_agent_log(
            agent_name="LayoutAgent",
            input_prompt=f"Panels: {len(panels)}, Images: {len(panel_images)}, Pages: {target_pages}",
            output={"pages": final_pages, "total_pages": len(final_pages)},
            metadata={"status": "success"}
        )

        self.logger.print_log(f"[LayoutAgent] 完成！{len(final_pages)} 页")
        return final_pages

    def run(self, user_input: str, target_pages: int = 2, target_panels_per_page: int = 3) -> dict:
        """运行完整工作流"""

        self.logger.print_log("\n" + "=" * 60)
        self.logger.print_log("ComicWeaver 工作流")
        self.logger.print_log("=" * 60)

        # 检查 ComfyUI
        comfyui_ok = self.check_comfyui()
        if comfyui_ok:
            self.logger.print_log("[OK] ComfyUI 连接成功")
        else:
            self.logger.print_log("[WARN] ComfyUI 未运行，将跳过图像生成")

        # 步骤 1: 剧本理解
        script_data = self.script_agent(user_input, target_pages, target_panels_per_page)

        # 步骤 2: 角色设计（生成人设图）
        character_images = {}
        if comfyui_ok and script_data.get('characters'):
            character_images = self.character_design_agent(script_data['characters'])

        # 步骤 3: 分镜设计
        panels = self.storyboard_agent(script_data, target_pages, target_panels_per_page)

        # 步骤 4: 图像生成（分镜图）
        panel_images = {}
        if comfyui_ok and character_images:
            panel_images = self.image_agent(panels, character_images, max_images=target_pages * target_panels_per_page)

        # 步骤 5: 排版合成
        final_pages = self.layout_agent(panels, panel_images, target_pages)

        # 返回结果
        result = {
            "script_data": script_data,
            "character_images": character_images,
            "panels": panels,
            "panel_images": panel_images,
            "final_pages": final_pages
        }

        # 输出摘要
        self.logger.print_log("\n" + "=" * 60)
        self.logger.print_log("工作流执行完成！")
        self.logger.print_log("=" * 60)
        self.logger.print_log(f"剧本标题：{script_data['title']}")
        self.logger.print_log(f"角色数量：{len(script_data['characters'])}")
        self.logger.print_log(f"场景数量：{len(script_data['scenes'])}")
        self.logger.print_log(f"分镜数量：{len(panels)}")
        self.logger.print_log(f"人设图：{len([v for v in character_images.values() if 'path' in v])} 张")
        self.logger.print_log(f"分镜图：{len([p for p in panel_images.values() if not str(p).startswith('error')])} 张")
        self.logger.print_log(f"最终页面：{len(final_pages)} 页")

        if script_data['characters']:
            self.logger.print_log("\n角色列表：")
            for char in script_data['characters']:
                self.logger.print_log(f"  - {char.get('name', 'Unknown')} ({char.get('role', 'unknown')})")
                if char.get('name') in character_images and 'path' in character_images[char['name']]:
                    self.logger.print_log(f"    人设图：{character_images[char['name']]['path']}")

        if panel_images:
            self.logger.print_log("\n生成的分镜图：")
            for panel_id, path in panel_images.items():
                if not str(path).startswith('error'):
                    self.logger.print_log(f"  - {panel_id}: {path}")

        self.logger.print_log(f"\n日志已保存到：{LOG_FILE}")

        return result


# ============================================================================
# LangGraph 模式（如果可用）
# ============================================================================

if LANGGRAPH_AVAILABLE:
    class ComicState(TypedDict):
        """漫画创作工作流状态"""
        user_input: str
        target_pages: int
        target_panels_per_page: int
        script_title: str
        script_summary: str
        characters: list
        scenes: list
        emotion_curve: list
        character_images: dict
        panels: list
        panel_images: dict
        final_pages: list
        messages: Annotated[Sequence[BaseMessage], add_messages]
        errors: list
        logs: list

    def build_langgraph_workflow():
        """构建 LangGraph 工作流"""

        workflow_client = ComicWorkflow()

        def script_node(state: ComicState) -> dict:
            script_data = workflow_client.script_agent(
                state['user_input'],
                state['target_pages'],
                state['target_panels_per_page']
            )
            return {
                "script_title": script_data['title'],
                "script_summary": script_data['summary'],
                "characters": script_data['characters'],
                "scenes": script_data['scenes'],
                "emotion_curve": script_data['emotion_curve'],
                "messages": [AIMessage(content=f"剧本解析完成：{script_data['title']}")],
                "logs": ["ScriptAgent: 剧本解析完成"]
            }

        def character_design_node(state: ComicState) -> dict:
            if workflow_client.comfyui_client is None:
                workflow_client.check_comfyui()

            if state['characters'] and workflow_client.comfyui_client:
                char_images = workflow_client.character_design_agent(state['characters'])
                return {
                    "character_images": char_images,
                    "messages": [AIMessage(content=f"角色设计完成：{len([v for v in char_images.values() if 'path' in v])} 张人设图")],
                    "logs": [f"CharacterDesignAgent: 生成了 {len(char_images)} 张人设图"]
                }
            return {
                "character_images": {},
                "messages": [AIMessage(content="跳过角色设计")],
                "logs": ["CharacterDesignAgent: 跳过"]
            }

        def storyboard_node(state: ComicState) -> dict:
            script_data = {
                "title": state['script_title'],
                "summary": state['script_summary'],
                "characters": state['characters'],
                "scenes": state['scenes']
            }
            panels = workflow_client.storyboard_agent(
                script_data,
                state['target_pages'],
                state['target_panels_per_page']
            )
            return {
                "panels": panels,
                "messages": [AIMessage(content=f"分镜设计完成：{len(panels)} 个分镜")],
                "logs": [f"StoryboardAgent: 生成了 {len(panels)} 个分镜"]
            }

        def image_node(state: ComicState) -> dict:
            if workflow_client.comfyui_client is None:
                workflow_client.check_comfyui()

            if state['panels'] and state.get('character_images') and workflow_client.comfyui_client:
                panel_images = workflow_client.image_agent(
                    state['panels'],
                    state['character_images'],
                    max_images=state['target_pages'] * state['target_panels_per_page']
                )
                return {
                    "panel_images": panel_images,
                    "messages": [AIMessage(content=f"图像生成完成：{len([p for p in panel_images.values() if not str(p).startswith('error')])} 张")],
                    "logs": [f"ImageAgent: 生成了 {len(panel_images)} 张图像"]
                }
            return {
                "panel_images": {},
                "messages": [AIMessage(content="跳过图像生成")],
                "logs": ["ImageAgent: 跳过"]
            }

        def layout_node(state: ComicState) -> dict:
            final_pages = workflow_client.layout_agent(
                state['panels'],
                state.get('panel_images', {}),
                state['target_pages']
            )
            return {
                "final_pages": final_pages,
                "messages": [AIMessage(content=f"排版合成完成：{len(final_pages)} 页")],
                "logs": [f"LayoutAgent: 合成了 {len(final_pages)} 页"]
            }

        # 创建状态图
        graph = StateGraph(ComicState)

        # 添加节点
        graph.add_node("script_agent", script_node)
        graph.add_node("character_design_agent", character_design_node)
        graph.add_node("storyboard_agent", storyboard_node)
        graph.add_node("image_agent", image_node)
        graph.add_node("layout_agent", layout_node)

        # 定义边
        graph.add_edge(START, "script_agent")
        graph.add_edge("script_agent", "character_design_agent")
        graph.add_edge("character_design_agent", "storyboard_agent")
        graph.add_edge("storyboard_agent", "image_agent")
        graph.add_edge("image_agent", "layout_agent")
        graph.add_edge("layout_agent", END)

        return graph.compile()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""

    print("\n" + "=" * 60)
    print("ComicWeaver 完整工作流测试")
    print("=" * 60)

    # 测试输入
    user_input = (
        "一个关于少年冒险的故事。主角小明是一个勇敢的少年，银发金瞳，穿着蓝色铠甲。"
        "他在一次意外中发现了一个神秘的古老遗迹，遇到了守护者阿龙（红发黑瞳，穿着红色斗篷）。"
        "两人一起探索遗迹，最终发现了隐藏的宝藏。"
    )
    target_pages = 2
    target_panels_per_page = 3

    print(f"\n输入剧情：{user_input}")
    print(f"目标：{target_pages} 页，每页 {target_panels_per_page} 个分镜")

    # 选择模式
    if LANGGRAPH_AVAILABLE:
        print("\n[INFO] 使用 LangGraph 模式")

        graph = build_langgraph_workflow()

        initial_state = {
            "user_input": user_input,
            "target_pages": target_pages,
            "target_panels_per_page": target_panels_per_page,
            "script_title": "",
            "script_summary": "",
            "characters": [],
            "scenes": [],
            "emotion_curve": [],
            "character_images": {},
            "panels": [],
            "panel_images": {},
            "final_pages": [],
            "messages": [],
            "errors": [],
            "logs": []
        }

        result = graph.invoke(initial_state)

        print("\n" + "=" * 60)
        print("工作流执行完成！")
        print("=" * 60)
        print(f"剧本标题：{result['script_title']}")
        print(f"人设图：{len([v for v in result.get('character_images', {}).values() if 'path' in v])} 张")
        print(f"分镜数量：{len(result['panels'])}")
        print(f"生成分镜图：{len([p for p in result.get('panel_images', {}).values() if not str(p).startswith('error')])} 张")

    else:
        print("\n[INFO] 使用简化模式")

        workflow = ComicWorkflow()
        result = workflow.run(
            user_input=user_input,
            target_pages=target_pages,
            target_panels_per_page=target_panels_per_page
        )

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()