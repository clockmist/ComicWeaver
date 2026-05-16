import json
import requests
import uuid
import os
import time
import urllib.parse


class ComfyUIClient:
    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
    
    def queue_prompt(self, workflow):
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
        if response.status_code != 200:
            print(f"Error response: {response.text[:500]}")
        response.raise_for_status()
        return response.json()
    
    def get_history(self, prompt_id):
        """获取生成历史"""
        response = requests.get(f"http://{self.server_address}/history/{prompt_id}")
        return response.json()
    
    def get_image(self, filename, subfolder="", folder_type="output"):
        """获取生成的图片"""
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }
        url = f"http://{self.server_address}/view?{urllib.parse.urlencode(params)}"
        response = requests.get(url)
        return response.content
    
    def wait_and_get_image(self, prompt_id, timeout=300):
        """等待生成完成并获取图片数据"""
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                # 找 SaveImage 节点 (ID: 76)
                if "76" in outputs and "images" in outputs["76"]:
                    image = outputs["76"]["images"][0]
                    return self.get_image(
                        image["filename"],
                        image.get("subfolder", "")
                    )
            time.sleep(1)
        raise TimeoutError("Generation timeout")
    
    def generate(self, prompt_text, output_path, width=1024, height=1024, seed=-1):
        """
        生成图片并保存到指定位置
        
        Args:
            prompt_text: 正向提示词
            output_path: 保存路径
            width/height: 图片尺寸
            seed: 随机种子，-1 表示随机
        """
        # 加载工作流模板
        workflow_path = os.path.join(os.path.dirname(__file__), "workflow_api.json")
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        
        # ========================================
        # 修改节点参数
        # ========================================
        
        # 1. 修改 CLIPTextEncode (30:45) - prompt
        if "30:45" in workflow:
            workflow["30:45"]["inputs"]["text"] = prompt_text
            print(f"✅ Modified prompt: {prompt_text[:50]}...")
        
        # 2. 修改 EmptySD3LatentImage (30:41) - 尺寸
        if "30:41" in workflow:
            workflow["30:41"]["inputs"]["width"] = width
            workflow["30:41"]["inputs"]["height"] = height
            print(f"✅ Modified size: {width}x{height}")
        
        # 3. 修改 KSampler (30:44) - 种子
        if "30:44" in workflow:
            if seed == -1:
                # 随机种子：使用默认值 + randomize 控制
                import random
                workflow["30:44"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
            else:
                workflow["30:44"]["inputs"]["seed"] = seed
            print(f"✅ Modified seed: {workflow['30:44']['inputs']['seed']}")
        
        # 4. 修改 SaveImage (76) - 文件名前缀
        if "76" in workflow:
            workflow["76"]["inputs"]["filename_prefix"] = "agent_generated"
        
        # 提交生成任务
        print(f"\n📤 Submitting to ComfyUI...")
        result = self.queue_prompt(workflow)
        prompt_id = result["prompt_id"]
        print(f"🆔 Prompt ID: {prompt_id}")
        
        # 等待并获取图片
        print("⏳ Generating... (this may take 1-3 minutes)")
        image_data = self.wait_and_get_image(prompt_id)
        
        # 保存到指定路径
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        print(f"✅ Image saved to: {output_path}")
        return output_path


# ============ 使用示例 ============

if __name__ == "__main__":
    # 检查 ComfyUI 是否运行
    try:
        requests.get("http://127.0.0.1:8188/system_stats", timeout=3)
        print("✅ ComfyUI is running\n")
    except:
        print("❌ ComfyUI is not running!")
        print("   Please start it first:")
        print("   conda activate comfyui")
        print("   cd D:\\ComfyUI")
        print("   python main.py")
        exit(1)
    
    # 初始化客户端
    client = ComfyUIClient()
    
    # 定义 prompt
    manga_prompt = (
        "1girl, solo, pink hair, short pink hair, hair between eyes, pink eyes, beautiful detailed eyes, mechanical armor, mecha suit, white and pink color scheme, sci-fi bodysuit, armored gauntlets, thigh armor, shoulder pauldrons, glowing pink accents on armor, cockpit interior background, holographic HUD, anime style, cel shading, crisp lineart, vibrant colors, masterpiece, best quality, highly detailed, Wuthering Waves style, Ames inspired"
    )
    
    # 指定保存路径
    output_path = r"D:\教材\大三下\计算机图形学\lab3\ComicWeaver\output_manga1.png"
    
    # 生成
    try:
        saved_path = client.generate(
            prompt_text=manga_prompt,
            output_path=output_path,
            width=1024,
            height=1024,
            seed=-1  # 随机种子
        )
        print(f"\n🎉 Done! Image at: {saved_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()