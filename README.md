# ComicWeaver

> 基于多Agent协作的自动化漫画创作系统 - 复旦大学2026计算机图形学课程项目

ComicWeaver 是一个 Agent + 图形学的实验性系统,从一句话主题或完整剧本自动生成 2D 漫画。

## 当前状态

**v0.1.0 - 初始框架**

本版本提供完整的可扩展骨架与可演示前端,但 Agent 内部使用 Mock 实现,不调用真实的 LLM 与图像生成模型。运行不需要 GPU。

## 安装

```bash
# 从仓库根目录
python -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -e .
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

每个 Agent 通过 `BaseAgent` 接口扩展。要替换 Mock 实现:

1. 继承 `BaseAgent[TIn, TOut]`
2. 实现 `run()` 与 `astream()`
3. 在 `agents/__init__.py` 注册

详见 [docs/agents/00-base.md](docs/agents/00-base.md)
