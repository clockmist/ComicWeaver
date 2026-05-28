# ComicWeaver

> 基于多Agent协作的自动化漫画创作系统 - 复旦大学2026计算机图形学课程项目

ComicWeaver 是一个 Agent + 图形学的实验性系统,从一句话主题或完整剧本自动生成 2D 漫画。

## 当前状态

**v0.2.0 - API-ready 框架**

本版本提供完整的可扩展骨架与可演示前端。Agent 会优先使用配置的 LLM / 图像 API,
未配置时自动退回本地规则与占位图后端,因此默认运行不需要 GPU。

## 安装

```bash
# 从仓库根目录
python -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -e .
```

## 配置

默认不需要配置即可运行。需要接入真实后端时,复制示例配置:

```bash
mkdir -p configs
cp configs/comicweaver.example.yaml configs/comicweaver.yaml
```

也可以通过环境变量覆盖:

```bash
export COMICWEAVER_LLM_ENABLED=true
export COMICWEAVER_LLM_PROVIDER=openai_compatible
export COMICWEAVER_LLM_BASE_URL=https://api.example.com/v1
export COMICWEAVER_LLM_API_KEY=your_key
export COMICWEAVER_LLM_MODEL=your_json_model

export COMICWEAVER_IMAGE_ENABLED=true
export COMICWEAVER_IMAGE_PROVIDER=comfyui
export COMICWEAVER_IMAGE_SERVER_URL=http://127.0.0.1:8188
```

## 启动

```bash
python -m comicweaver
# 或
comicweaver
```

打开浏览器访问 http://localhost:7860

## 项目结构

```
ComicWeaver/
├── src/comicweaver/
│   ├── core/           # 共享类型与基类
│   ├── agents/         # 5个生产Agent + 1个审查Agent
│   ├── orchestrator/   # LangGraph编排
│   ├── review/         # 审查Rubric
│   ├── interaction/    # 交互模式与流式适配
│   ├── models/         # UMI图像生成接口
│   ├── storage/        # 项目持久化
│   └── ui/             # Gradio前端
├── tests/              # 测试
└── docs/               # 设计文档
```

## 设计文档

- [开发思路框架](docs/开发思路框架.md)
- [Agent 接口规范](docs/agents/README.md)

## 扩展点

每个 Agent 通过 `BaseAgent` 接口扩展。要替换或新增 API 后端:

1. 继承 `BaseAgent[TIn, TOut]`
2. 实现 `run()` 与 `astream()`
3. 在 `agents/__init__.py` 注册

详见 [docs/agents/00-base.md](docs/agents/00-base.md)
