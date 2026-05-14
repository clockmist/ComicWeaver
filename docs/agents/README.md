# Agent 接口文档索引

> ComicWeaver 系统所有Agent的接口规范集合。所有Agent必须先遵循 `00-base.md` 的基础契约。

## 1. 文档清单

| # | 文档 | Agent | 负责成员 | 类型 |
|---|------|-------|---------|------|
| 00 | [基类与共享类型](./00-base.md) | BaseAgent / Types | 全员共建 | 基础契约 |
| 01 | [剧本理解Agent](./01-script-agent.md) | ScriptAgent | A | 生产 |
| 02 | [角色管理Agent](./02-character-agent.md) | CharacterAgent | B | 生产 |
| 03 | [分镜设计Agent](./03-storyboard-agent.md) | StoryboardAgent | C | 生产 |
| 04 | [图像生成Agent](./04-image-agent.md) | ImageAgent | C | 生产 |
| 05 | [排版合成Agent](./05-layout-agent.md) | LayoutAgent | D | 生产 |
| 06 | [审查Agent](./06-reviewer-agent.md) | ReviewerAgent | A | 横向 |

## 2. Agent 协作关系

```
┌──────────────────────────────────────────────────────────┐
│  用户输入                                                 │
│      ↓                                                   │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │ ScriptAgent  │ →→ │ ReviewerAgent│ ─→ PASS/REVISE   │
│  └──────────────┘    └──────────────┘                   │
│      ↓ ScriptOutput                                     │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │CharacterAgent│ →→ │ ReviewerAgent│                   │
│  └──────────────┘    └──────────────┘                   │
│      ↓ CharacterDB + ReferenceWindow                    │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │StoryboardAgt │ →→ │ ReviewerAgent│                   │
│  └──────────────┘    └──────────────┘                   │
│      ↓ PageLayout + PromptPack                          │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │ ImageAgent   │ →→ │ ReviewerAgent│                   │
│  └──────────────┘    └──────────────┘                   │
│      ↓ PanelImage (循环N次)            ↓ 高质量帧入档    │
│      │                          → CharacterAgent.update │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │ LayoutAgent  │ →→ │ ReviewerAgent│                   │
│  └──────────────┘    └──────────────┘                   │
│      ↓ FinalPage + Exports                              │
│  最终输出                                                │
└──────────────────────────────────────────────────────────┘
```

## 3. 数据契约速查

| 上游产出 | 下游消费 | 关键模型 |
|---------|---------|---------|
| ScriptOutput | CharacterAgent / StoryboardAgent | `scenes` `characters` `emotion_curve` |
| CharacterOutput | ImageAgent | `ReferenceWindow` `IdentityEmbedding` |
| StoryboardOutput | ImageAgent / LayoutAgent | `PanelPlan.prompt_pack` `bbox` |
| ImageOutput | LayoutAgent / CharacterAgent | `PanelImage` `consistency_scores` |
| LayoutOutput | 用户 | `FinalPage` `ExportArtifact` |
| ReviewOutput | 编排层 | `ReviewFeedback` `EscalationSummary` |

## 4. 通用约定

所有Agent均遵循 `00-base.md` 中定义的:

- `BaseAgent[TIn, TOut]` 抽象类
- `AgentContext` 运行时上下文
- `StreamEvent` 流式事件协议
- `ReviewFeedback` 审查反馈协议
- `AgentOutputMeta` 输出元信息
- 异常分类与处理(`AgentError` 体系)

## 5. 开发顺序建议

**第一周(架构搭建)**:
1. 先冻结 `00-base.md` 的所有共享类型
2. 各成员基于Mock实现自己Agent的`run()`基础版
3. ReviewerAgent提供格式校验框架(质量审查可Mock)

**第二周(管线打通)**:
1. 各Agent实现真实逻辑替换Mock
2. ReviewerAgent接入5个Rubric
3. 端到端跑通最简单路径

**第三周(系统集成)**:
1. 流式事件接入前端
2. 返工循环完整测试
3. ESCALATE路径与用户决策

**第四周(交付)**:
1. 性能调优,达到各Agent性能指标
2. 测试覆盖各文档"测试要点"章节

## 6. 接口变更流程

1. 提议变更需更新对应Agent文档,版本号递增
2. 若涉及共享类型(`00-base.md`),需团队评审
3. 上下游Agent的兼容性检查
4. 更新Rubric(若行为变化影响审查)
5. 在文档CHANGELOG记录变更原因

## 7. 文档维护

- 每个Agent文档独立版本号,与代码版本同步
- 重大变更必须更新本索引的"文档清单"
- 接口契约最终以Pydantic模型代码为准,文档仅作设计说明

---

**文档版本**: v1.0
**创建日期**: 2026-05-14
**适用阶段**: 项目启动期(W0)
