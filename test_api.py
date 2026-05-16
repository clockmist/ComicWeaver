"""
ComicWeaver 完整工作流测试

使用 LangGraph 框架连接各个 Agent，将 Mock 数据替换为真实的 API 调用：
- LLM API: 智谱 AI GLM-4 (HTTP 直接调用)
- 文生图 API: ComfyUI

特性：
1. 记录每个 Agent 的输入 prompt 和输出到日志文件
2. 图像保存到 output 文件夹
3. 支持两种模式：LangGraph 模式和简化模式
"""
import os
import sys
import json
import time
import uuid
import urllib.parse
import requests
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
LOG_FILE = os.path.join(OUTPUT_DIR, "workflow_log.json")

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
        
        # 确保输出目录存在
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
        
        # 实时写入文件
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
        
        # 实时写入文件
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
        max_tokens: int = 1024
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
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        return response.json()


class ComfyUIClient:
    """ComfyUI 文生图客户端"""
    
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
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_history(self, prompt_id: str) -> dict:
        """获取生成历史"""
        response = requests.get(f"http://{self.server_address}/history/{prompt_id}")
        return response.json()
    
    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """获取生成的图片"""
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }
        url = f"http://{self.server_address}/view?{urllib.parse.urlencode(params)}"
        response = requests.get(url)
        return response.content
    
    def wait_and_get_image(self, prompt_id: str, timeout: int = 180) -> bytes:
        """等待生成完成并获取图片数据（增加超时时间到3分钟）"""
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                if "76" in outputs and "images" in outputs["76"]:
                    image = outputs["76"]["images"][0]
                    return self.get_image(
                        image["filename"],
                        image.get("subfolder", "")
                    )
            time.sleep(2)  # 增加轮询间隔
        raise TimeoutError("Generation timeout")
    
    def generate(
        self,
        prompt_text: str,
        output_path: str,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1
    ) -> str:
        """生成图片并保存到指定位置"""
        # 加载工作流模板
        workflow_path = os.path.join(os.path.dirname(__file__), "workflow_api.json")
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        
        # 修改节点参数
        if "30:45" in workflow:
            workflow["30:45"]["inputs"]["text"] = prompt_text
        
        if "30:41" in workflow:
            workflow["30:41"]["inputs"]["width"] = width
            workflow["30:41"]["inputs"]["height"] = height
        
        if "30:44" in workflow:
            if seed == -1:
                import random
                workflow["30:44"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
            else:
                workflow["30:44"]["inputs"]["seed"] = seed
        
        if "76" in workflow:
            workflow["76"]["inputs"]["filename_prefix"] = "comic_panel"
        
        # 提交生成任务
        result = self.queue_prompt(workflow)
        prompt_id = result["prompt_id"]
        
        # 等待并获取图片
        image_data = self.wait_and_get_image(prompt_id)
        
        # 保存到指定路径
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        return output_path


# ============================================================================
# Agent 实现
# ============================================================================

class ComicWorkflow:
    """漫画创作工作流"""
    
    def __init__(self):
        self.llm_client = ZhipuAIHTTPClient(ZHIPUAI_API_KEY)
        self.comfyui_client = None
        self.logger = WorkflowLogger(LOG_FILE)
    
    def check_comfyui(self) -> bool:
        """检查 ComfyUI 是否运行"""
        try:
            response = requests.get(f"http://{COMFYUI_SERVER}/system_stats", timeout=3)
            self.comfyui_client = ComfyUIClient()
            return True
        except:
            return False
    
    def script_agent(self, user_input: str, target_pages: int, target_panels_per_page: int) -> dict:
        """
        剧本理解 Agent - 使用真实 LLM API
        解析用户输入，生成剧本、角色、场景
        """
        self.logger.print_log("\n[ScriptAgent] 开始解析剧本...")
        
        # 构造提示词
        system_prompt = """你是一个专业的漫画剧本分析师。请分析用户提供的剧情描述，生成：
1. 漫画标题
2. 剧情摘要
3. 主要角色（2-3个，包含姓名、角色类型、外貌、性格）
4. 场景列表（每个场景包含地点、时间、氛围、在场角色、对话、情感强度）

请以 JSON 格式返回结果，包含以下字段：
- title: 漫画标题
- summary: 剧情摘要
- characters: 角色列表，每个角色包含 name, role(protagonist/antagonist/supporting), appearance, personality
- scenes: 场景列表，每个场景包含 location, time_of_day, atmosphere, characters_present, dialogues, emotion_intensity(0-1)"""
        
        user_prompt = f"""请分析以下剧情描述，生成漫画剧本：

{user_input}

目标页数：{target_pages}
每页分镜数：{target_panels_per_page}"""
        
        # 记录输入
        full_input = f"System: {system_prompt}\n\nUser: {user_prompt}"
        
        try:
            # 调用 LLM
            response = self.llm_client.chat_completions_create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500
            )
            
            content = response["choices"][0]["message"]["content"]
            
            # 尝试解析 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(content)
            
            # 构造返回结果
            script_data = {
                "title": result.get("title", "未命名漫画"),
                "summary": result.get("summary", ""),
                "characters": result.get("characters", []),
                "scenes": result.get("scenes", []),
                "emotion_curve": [s.get("emotion_intensity", 0.5) for s in result.get("scenes", [])]
            }
            
            # 记录日志
            self.logger.add_agent_log(
                agent_name="ScriptAgent",
                input_prompt=full_input,
                output=script_data,
                metadata={"model": "glm-4-flash", "tokens": len(content)}
            )
            
            self.logger.print_log(f"[ScriptAgent] 完成！标题：{script_data['title']}")
            self.logger.print_log(f"[ScriptAgent] 角色数：{len(script_data['characters'])}，场景数：{len(script_data['scenes'])}")
            
            return script_data
            
        except Exception as e:
            self.logger.print_log(f"[ScriptAgent] 错误：{e}")
            # 返回默认数据
            default_data = {
                "title": "未命名漫画",
                "summary": user_input,
                "characters": [
                    {"name": "主角", "role": "protagonist", "appearance": "普通", "personality": "勇敢"}
                ],
                "scenes": [
                    {"location": "未知", "time_of_day": "白天", "atmosphere": "普通", "characters_present": ["主角"], "emotion_intensity": 0.5}
                ],
                "emotion_curve": [0.5]
            }
            
            # 记录错误日志
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
        根据剧本生成详细的分镜计划
        """
        self.logger.print_log("\n[StoryboardAgent] 开始设计分镜...")
        
        # 构造提示词
        system_prompt = """你是一个专业的漫画分镜师。根据剧本和场景，设计详细的分镜计划。
每个分镜应包含：
- panel_id: 分镜ID
- page_id: 所在页
- shot_size: 景别(extreme_long/long/full/medium/close/extreme_close)
- camera_angle: 镜头角度(eye_level/high_angle/low_angle)
- characters_in_panel: 场内角色
- primary_action: 主要动作
- setting: 场景描述
- mood: 情绪氛围
- positive_prompt: 图像生成正向提示词
- negative_prompt: 图像生成负向提示词

请以 JSON 格式返回 panels 列表。"""
        
        user_prompt = f"""请为以下剧本设计分镜：

标题：{script_data['title']}
摘要：{script_data['summary']}
角色：{json.dumps(script_data['characters'], ensure_ascii=False)}
场景：{json.dumps(script_data['scenes'], ensure_ascii=False)}

目标页数：{target_pages}
每页分镜数：{target_panels_per_page}"""
        
        # 记录输入
        full_input = f"System: {system_prompt}\n\nUser: {user_prompt}"
        
        try:
            # 调用 LLM
            response = self.llm_client.chat_completions_create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000
            )
            
            content = response["choices"][0]["message"]["content"]
            
            # 解析 JSON
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
            
            # 记录日志
            self.logger.add_agent_log(
                agent_name="StoryboardAgent",
                input_prompt=full_input,
                output={"panels": panels, "count": len(panels)},
                metadata={"model": "glm-4-flash", "tokens": len(content)}
            )
            
            self.logger.print_log(f"[StoryboardAgent] 完成！生成了 {len(panels)} 个分镜")
            
            return panels
            
        except Exception as e:
            self.logger.print_log(f"[StoryboardAgent] 错误：{e}")
            # 使用简单的规则生成
            panels = []
            panel_id = 0
            for page in range(target_pages):
                for panel in range(target_panels_per_page):
                    scene_idx = panel_id % len(script_data['scenes']) if script_data['scenes'] else 0
                    scene = script_data['scenes'][scene_idx] if script_data['scenes'] else {}
                    
                    panels.append({
                        "panel_id": f"panel_{panel_id:03d}",
                        "page_id": f"page_{page:03d}",
                        "shot_size": "medium",
                        "camera_angle": "eye_level",
                        "characters_in_panel": scene.get("characters_present", []),
                        "primary_action": scene.get("location", "未知场景"),
                        "setting": scene.get("location", ""),
                        "mood": scene.get("atmosphere", ""),
                        "positive_prompt": f"manga style, {scene.get('location', '')}, {scene.get('atmosphere', '')}",
                        "negative_prompt": "low quality, blurry"
                    })
                    panel_id += 1
            
            # 记录日志
            self.logger.add_agent_log(
                agent_name="StoryboardAgent",
                input_prompt=full_input,
                output={"panels": panels, "count": len(panels), "fallback": True},
                metadata={"status": "fallback", "error": str(e)}
            )
            
            self.logger.print_log(f"[StoryboardAgent] 使用规则生成了 {len(panels)} 个分镜")
            return panels
    
    def image_agent(self, panels: list, max_images: int = 3) -> dict:
        """
        图像生成 Agent - 使用真实 ComfyUI API
        为每个分镜生成图像
        """
        self.logger.print_log("\n[ImageAgent] 开始生成图像...")
        
        if self.comfyui_client is None:
            self.logger.print_log("[ImageAgent] ComfyUI 未运行，跳过图像生成")
            self.logger.add_agent_log(
                agent_name="ImageAgent",
                input_prompt="ComfyUI not running",
                output={"status": "skipped", "reason": "ComfyUI not available"}
            )
            return {}
        
        panel_images = {}
        output_dir = os.path.join(OUTPUT_DIR, "panels")
        
        # 只生成指定数量的分镜
        max_panels = min(max_images, len(panels))
        
        self.logger.print_log(f"[ImageAgent] 将生成 {max_panels} 张图像")
        
        for i, panel in enumerate(panels[:max_panels]):
            panel_id = panel['panel_id']
            prompt = panel.get('positive_prompt', 'manga style, beautiful scene')
            
            self.logger.print_log(f"[ImageAgent] 生成图像 {i+1}/{max_panels}: {panel_id}")
            self.logger.print_log(f"[ImageAgent] Prompt: {prompt[:100]}...")
            
            try:
                output_path = os.path.join(output_dir, f"{panel_id}.png")
                saved_path = self.comfyui_client.generate(
                    prompt_text=prompt,
                    output_path=output_path,
                    width=768,
                    height=1024,
                    seed=-1
                )
                panel_images[panel_id] = saved_path
                
                # 记录图像生成日志
                self.logger.add_image_log(
                    agent_name="ImageAgent",
                    input_prompt=prompt,
                    image_path=saved_path,
                    metadata={
                        "panel_id": panel_id,
                        "width": 768,
                        "height": 1024,
                        "status": "success"
                    }
                )
                
                self.logger.print_log(f"[ImageAgent] 完成：{saved_path}")
                
            except Exception as e:
                self.logger.print_log(f"[ImageAgent] 生成失败：{e}")
                panel_images[panel_id] = f"error: {str(e)}"
                
                # 记录错误日志
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
        
        self.logger.print_log(f"[ImageAgent] 完成！生成了 {len([p for p in panel_images.values() if not p.startswith('error')])} 张图像")
        
        return panel_images
    
    def layout_agent(self, panels: list, panel_images: dict, target_pages: int) -> list:
        """
        排版合成 Agent - 简化实现
        将生成的图像合成为最终页面
        """
        self.logger.print_log("\n[LayoutAgent] 开始排版合成...")
        
        # 简化实现：只记录结果
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
        
        # 记录日志
        self.logger.add_agent_log(
            agent_name="LayoutAgent",
            input_prompt=f"Panels: {len(panels)}, Images: {len(panel_images)}, Pages: {target_pages}",
            output={"pages": final_pages, "total_pages": len(final_pages)},
            metadata={"status": "success"}
        )
        
        self.logger.print_log(f"[LayoutAgent] 完成！合成了 {len(final_pages)} 页")
        
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
        
        # 步骤 2: 分镜设计
        panels = self.storyboard_agent(script_data, target_pages, target_panels_per_page)
        
        # 步骤 3: 图像生成
        panel_images = self.image_agent(panels, max_images=3) if comfyui_ok else {}
        
        # 步骤 4: 排版合成
        final_pages = self.layout_agent(panels, panel_images, target_pages)
        
        # 返回结果
        result = {
            "script_data": script_data,
            "panels": panels,
            "panel_images": panel_images,
            "final_pages": final_pages
        }
        
        # 输出摘要
        self.logger.print_log("\n" + "=" * 60)
        self.logger.print_log("工作流执行完成！")
        self.logger.print_log("=" * 60)
        self.logger.print_log(f"剧本标题：{script_data['title']}")
        self.logger.print_log(f"剧本摘要：{script_data['summary'][:100]}...")
        self.logger.print_log(f"角色数量：{len(script_data['characters'])}")
        self.logger.print_log(f"场景数量：{len(script_data['scenes'])}")
        self.logger.print_log(f"分镜数量：{len(panels)}")
        self.logger.print_log(f"生成图像：{len([p for p in panel_images.values() if not str(p).startswith('error')])} 张")
        self.logger.print_log(f"最终页面：{len(final_pages)} 页")
        
        # 输出角色信息
        if script_data['characters']:
            self.logger.print_log("\n角色列表：")
            for char in script_data['characters']:
                self.logger.print_log(f"  - {char.get('name', '未知')} ({char.get('role', '未知')}): {char.get('appearance', '')}")
        
        # 输出图像路径
        if panel_images:
            self.logger.print_log("\n生成的图像：")
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
            panel_images = workflow_client.image_agent(state['panels'], max_images=3)
            return {
                "panel_images": panel_images,
                "messages": [AIMessage(content=f"图像生成完成：{len(panel_images)} 张")],
                "logs": [f"ImageAgent: 生成了 {len(panel_images)} 张图像"]
            }
        
        def layout_node(state: ComicState) -> dict:
            final_pages = workflow_client.layout_agent(
                state['panels'],
                state['panel_images'],
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
        graph.add_node("storyboard_agent", storyboard_node)
        graph.add_node("image_agent", image_node)
        graph.add_node("layout_agent", layout_node)
        
        # 定义边
        graph.add_edge(START, "script_agent")
        graph.add_edge("script_agent", "storyboard_agent")
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
    user_input = "一个关于少年冒险的故事。主角小明是一个勇敢的少年，他在一次意外中发现了一个神秘的古老遗迹，遇到了守护者阿龙。两人一起探索遗迹，最终发现了隐藏的宝藏。"
    target_pages = 2
    target_panels_per_page = 3
    
    print(f"\n输入剧情：{user_input}")
    print(f"目标：{target_pages} 页，每页 {target_panels_per_page} 个分镜")
    
    # 选择模式
    if LANGGRAPH_AVAILABLE:
        print("\n[INFO] 使用 LangGraph 模式")
        
        # 构建工作流
        graph = build_langgraph_workflow()
        
        # 初始化状态
        initial_state = {
            "user_input": user_input,
            "target_pages": target_pages,
            "target_panels_per_page": target_panels_per_page,
            "script_title": "",
            "script_summary": "",
            "characters": [],
            "scenes": [],
            "emotion_curve": [],
            "panels": [],
            "panel_images": {},
            "final_pages": [],
            "messages": [],
            "errors": [],
            "logs": []
        }
        
        # 执行工作流
        result = graph.invoke(initial_state)
        
        # 输出结果
        print("\n" + "=" * 60)
        print("工作流执行完成！")
        print("=" * 60)
        print(f"剧本标题：{result['script_title']}")
        print(f"分镜数量：{len(result['panels'])}")
        print(f"生成图像：{len(result['panel_images'])} 张")
        
    else:
        print("\n[INFO] 使用简化模式")
        
        # 创建工作流实例
        workflow = ComicWorkflow()
        
        # 运行工作流
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
