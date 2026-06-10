# ComicWeaver

> 基于多Agent协作的自动化漫画创作系统 — 复旦大学2026计算机图形学课程项目

ComicWeaver 是一个 Agent + 图形学的实验性系统，从一句话主题或完整剧本自动生成 2D 漫画。系统采用 LangGraph 编排多个 Agent（故事、角色、剧本、分镜、图像、气泡、排版），通过 ComfyUI 扩散模型生成漫画图片，并提供 Gradio 前端供用户交互式创作。

## 当前状态

**v0.4.0 — 全流程可用**

支持从故事输入到最终排版漫画的完整流水线。Agent 默认使用本地规则/占位图（无需 GPU 即可跑通框架），配置 LLM 与 ComfyUI 后可获得真实生成效果。YOLO 人脸检测与 VLM 辅助确保气泡排版不遮挡角色面部。

## 环境要求

- Python >= 3.10
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)（可选，用于真实图像生成）
- YOLO 模型文件（可选，用于人脸检测排版）

## 安装

```bash
# 1. 克隆仓库（含 ComfyUI 子模块）
git clone --recurse-submodules <repo-url>
cd ComicWeaver

# 2. 创建虚拟环境
python -m venv .venv
.venv/Scripts/activate   # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. 安装 ComicWeaver
pip install -e .

# 4. 安装可选依赖
pip install -e ".[agents]"    # LangGraph / LangChain（使用真实 LLM Agent 需要）
pip install -e ".[image]"     # diffusers / torch（本地生图备选，需 GPU）
pip install -e ".[dev]"       # pytest / ruff / mypy（开发用）
```

## 配置

### 快速开始（无需配置）

默认不依赖任何外部 API 即可运行，Agent 使用本地规则生成内容，图像使用占位图。直接跳到 [启动](#启动) 即可体验框架。

### 配置文件方式

```bash
# 复制示例配置并编辑
cp configs/comicweaver.example.yaml configs/comicweaver.yaml
```

配置文件 `configs/comicweaver.yaml` 包含以下段：

```yaml
llm:        # 文本 LLM（剧本、分镜、角色标签等）
vlm:        # 视觉 LLM（人脸检测、气泡定位、图像问答）
image:      # 图像生成（ComfyUI）
storage:    # 存储路径
runtime:    # 运行时参数
yolo:       # YOLO 人脸检测模型
```

### 环境变量覆盖

所有配置项均可通过环境变量覆盖，无需修改配置文件：

**LLM（文本模型）**
```bash
export COMICWEAVER_LLM_ENABLED=true
export COMICWEAVER_LLM_PROVIDER=openai_compatible
export COMICWEAVER_LLM_BASE_URL=https://api.deepseek.com/v1
export COMICWEAVER_LLM_API_KEY=your_api_key
export COMICWEAVER_LLM_MODEL=deepseek-v4-flash
export COMICWEAVER_LLM_TEMPERATURE=0.3
export COMICWEAVER_LLM_MAX_TOKENS=8192
export COMICWEAVER_LLM_TIMEOUT_SECONDS=300
```

**图像生成（ComfyUI）**
```bash
export COMICWEAVER_IMAGE_ENABLED=true
export COMICWEAVER_IMAGE_PROVIDER=comfyui
export COMICWEAVER_IMAGE_SERVER_URL=http://127.0.0.1:8188
export COMICWEAVER_IMAGE_MODEL=animagine-xl-4.0-opt.safetensors
export COMICWEAVER_IMAGE_CHARACTER_WORKFLOW=anim_gen_char.json
export COMICWEAVER_IMAGE_PANEL_WORKFLOW=anim_gen_pic.json
export COMICWEAVER_IMAGE_FALLBACK=false
export COMICWEAVER_IMAGE_TIMEOUT_SECONDS=300
```

**存储路径**
```bash
export COMICWEAVER_PROJECTS_DIR=projects
export COMICWEAVER_CACHE_DIR=cache
export COMICWEAVER_OUTPUTS_DIR=outputs
```

**YOLO 人脸检测**
```bash
export COMICWEAVER_YOLO_MODEL_PATH=yolo/face_yolov8n.pt
export COMICWEAVER_YOLO_CONFIDENCE=0.3
```

### .env 文件

也可以在项目根目录创建 `.env` 文件写入以上环境变量（已由 `.gitignore` 排除）。

## ComfyUI 部署

ComfyUI 已作为 Git 子模块包含在 `external/ComfyUI/`。首次使用需初始化：

```bash
git submodule update --init --recursive
```

### 安装 ComfyUI 依赖

```bash
cd external/ComfyUI
pip install -r requirements.txt
```

### 下载模型

根据工作流配置，需要将模型放入 `external/ComfyUI/models/` 对应子目录。以 Animagine XL 为例：

- **Checkpoint**: `animagine-xl-4.0-opt.safetensors` → `models/checkpoints/`
- **VAE**: `ae.safetensors`（通常已内置）


### 启动 ComfyUI

```bash
cd external/ComfyUI
python main.py --listen 127.0.0.1 --port 8188
```

### 工作流 JSON 文件

ComicWeaver 通过 ComfyUI API 提交工作流 JSON 并自动填入提示词与种子。项目根目录提供的工作流文件：

| 文件 | 用途 |
|------|------|
| `anim_gen_char.json` | 角色参考图生成（Animagine XL + KSampler） |
| `anim_gen_pic.json` | 漫画面板生成（IP-Adapter 工作流） |

在配置中通过 `workflow_character_path` 和 `workflow_panel_path` 指定使用哪个工作流文件。

## 启动

```bash
# 方式一：模块启动
python -m comicweaver

# 方式二：控制台脚本
comicweaver

# 方式三：直接运行入口
python run.py
```

打开浏览器访问 **http://localhost:7860**

## 使用流程

### 1. 创建项目（📋 项目 Tab）

- 输入故事主题或完整剧本
- 选择创作模式：简单 / 专业 / 改编
- 选择画风：日漫 / 美漫 / 条漫 / 写实
- 选择交互模式：半自动 / 全自动 / 全交互
- 设置目标页数（1-20）
- 点击「🚀 创建项目」

### 2. 运行工作流（⚡ 创作 Tab）

- 选择起始阶段（可从任意阶段开始或重跑）
- 点击「▶ 启动工作流」
- 在半自动模式下，每个 Agent 完成后会暂停等待用户确认
  - ✅ 接受并继续：接受当前结果，进入下一阶段
  - ↻ 重新生成：重新执行当前 Agent
  - 可在指导框中输入自然语言建议（如 "让角色更年轻"）
- 全自动模式下一键生成全部内容

### 3. 预览结果（📖 预览 Tab）

- 翻页浏览生成的漫画页面
- 切换显示/隐藏对话气泡边界
- 查看剧本、分镜、角色档案等中间产物
- 一键下载所有页面

## 项目结构

```
ComicWeaver/
├── src/comicweaver/          # 应用代码
│   ├── core/                 # 共享类型、状态、基类、异常
│   ├── agents/               # 7 个 Agent + 排版引擎
│   │   ├── story_agent.py       # 故事生成 Agent
│   │   ├── character_agent.py   # 角色设计 Agent
│   │   ├── script_agent.py      # 剧本生成 Agent
│   │   ├── storyboard_agent.py  # 分镜设计 Agent
│   │   ├── image_agent.py       # 图像生成 Agent
│   │   ├── bubble_agent.py      # 台词气泡 Agent
│   │   ├── layout_agent.py      # 排版 Agent
│   │   └── layout/              # 排版引擎（气泡/合成/人脸检测/求解器/模板）
│   ├── orchestrator/         # LangGraph 工作流编排
│   ├── storage/              # 项目持久化与路径管理
│   ├── ui/                   # Gradio 前端
│   │   ├── app_new.py           # 主界面（三层架构）
│   │   ├── callbacks/           # 业务回调
│   │   ├── renderers/           # HTML 渲染
│   │   ├── composables/         # 跨组件组合
│   │   └── session.py           # 会话状态管理
│   ├── interaction/          # 交互模式适配
│   ├── config.py             # 配置加载（YAML + 环境变量）
│   ├── api.py                # LLM / VLM / ComfyUI API 客户端
│   └── utils/                # 工具函数
├── configs/                  # 配置文件
│   ├── comicweaver.example.yaml  # 示例配置（提交到 Git）
│   └── comicweaver.yaml          # 本地配置（不提交）
├── external/ComfyUI/         # ComfyUI 子模块
├── projects/                 # 项目输出目录
├── tests/
│   ├── unit/                 # 单元测试
│   └── integration/          # 集成测试
├── docs/                     # 设计文档
└── yolo/                     # YOLO 模型文件
```

## 测试

```bash
# 运行所有测试
pytest

# 仅单元测试
pytest tests/unit/

# 仅集成测试
pytest tests/integration/

# 代码检查
ruff check src tests
mypy src
```

测试会自动隔离本地配置，避免依赖开发者的 API 密钥。

## 扩展点

### 新增 Agent

1. 继承 `core/base.py` 中的 `BaseAgent[TIn, TOut]`
2. 实现 `run()` 与 `astream()` 方法
3. 在 `agents/__init__.py` 注册
4. 在 `orchestrator/graph.py` 中编排进工作流

### 切换 LLM 后端

任何兼容 OpenAI `/chat/completions` 格式的 API 均可使用，只需修改 `base_url`。已测试过：
- DeepSeek（deepseek-v4-flash 等）
- 智谱 GLM（glm-4-flash 等）
- 其他 OpenAI 兼容服务

### 切换图像生成后端

目前内置 ComfyUI 适配器。如需其他后端：
1. 继承 `ImageGenerationRequest/Response` 模型
2. 实现提交 + 轮询 + 下载逻辑
3. 在 `api.py` 添加新 Client 类

## 常见问题

**Q: 没有 GPU 能运行吗？**
可以。默认不启用 LLM 和图像生成时，系统使用本地规则和占位图，能完整跑通整个流水线框架。

**Q: ComfyUI 连不上？**
检查 ComfyUI 是否在 `http://127.0.0.1:8188` 运行，确认配置中 `image.enabled=true` 且 `image.server_url` 正确。

**Q: LLM 返回格式错误？**
部分模型对 JSON 输出支持有限。代码会自动重试（先尝试 `json_object` 格式，失败后回退到无格式约束），并自动剥离 markdown 代码块包装。

**Q: 人物一致性不好？**
这是当前的主要挑战。系统目前通过固定种子和角色特征prompt保持一致性。可尝试切换模型、调整工作流、或将风格设为黑白漫画降低视觉一致性要求。

## 设计文档

- [开发思路框架](docs/开发思路框架.md)
- [Agent 接口规范](docs/agents/README.md)
- [课程项目方案](docs/漫画创作Agent系统：期末课程项目方案.html)

## 许可证

MIT License
