# 角色管理 Agent (CharacterAgent)

> 系统的"演员经纪人"。负责角色视觉档案的构建、维护与跨面板一致性保障。

## 1. 职责边界

**做什么**:
- 基于剧本中的角色草稿生成视觉参考图
- 构建并维护角色档案库(CharacterDB)
- 计算CLIP特征嵌入并建立索引
- 为每个分镜动态生成"参考窗口"(链式参考核心)
- 角色档案的版本管理与回溯

**不做什么**:
- 不进行最终漫画生成(交给图像Agent)
- 不决定角色在分镜中的位置(交给分镜Agent)
- 不修改剧本中的角色描述(只读)

## 2. 接口定义

```python
class CharacterAgent(BaseAgent[CharacterInput, CharacterOutput]):
    name = "character_agent"
    version = "1.0.0"
    rubric_id = "rubric_character_v1"
```

## 3. 输入模型

```python
class CharacterInput(BaseModel):
    operation: Literal["init", "build_reference_window", "update_archive"]

    # init: 初始化角色档案
    character_drafts: list[CharacterDraft] | None = None
    style_preset: str | None = None

    # build_reference_window: 为某个分镜构建参考
    panel_id: str | None = None
    target_characters: list[str] | None = None  # char_id列表
    recent_panels: list[PanelImage] | None = None  # 近期面板
    window_size: int = 3            # 参考窗口大小

    # update_archive: 把高质量帧加入档案
    quality_panel: PanelImage | None = None
    quality_score: float | None = None

    feedback: ReviewFeedback | None = None
```

## 4. 输出模型

```python
class CharacterOutput(BaseModel):
    operation: str

    # init输出
    character_db: CharacterDB | None = None

    # build_reference_window输出
    reference_window: ReferenceWindow | None = None
    identity_embedding: IdentityEmbedding | None = None

    # update_archive输出
    archive_updated: bool = False
    new_archive_entry_id: str | None = None

    meta: AgentOutputMeta


class CharacterDB(BaseModel):
    """角色档案库。"""
    project_id: str
    characters: dict[str, CharacterProfile]  # char_id → profile
    version: int
    created_at: float
    updated_at: float


class CharacterProfile(BaseModel):
    char_id: str
    name: str
    base_reference: ReferenceImage   # 用户确认的基础参考图
    archive: list[ArchiveEntry]      # 历史高质量出现
    visual_traits: VisualTraits      # 关键视觉特征(用于一致性)
    clip_embedding_id: str           # FAISS索引中的ID
    style_preset: str
    locked: bool = False             # 锁定后不再自动更新


class ReferenceImage(BaseModel):
    image_id: str
    image_path: str                  # 本地路径或URL
    source: Literal["generated", "uploaded", "panel"]
    generation_prompt: str | None = None
    confidence: float                # 0.0-1.0
    created_at: float


class ArchiveEntry(BaseModel):
    entry_id: str
    panel_id: str                    # 来源面板
    image_path: str
    pose: str                        # 姿态描述
    expression: str                  # 表情描述
    quality_score: float
    timestamp: float                 # 创建时间(用于时间衰减)


class VisualTraits(BaseModel):
    """从参考图中提取的关键视觉特征,作为强约束。"""
    hair: str                        # "短黑发,刘海"
    eyes: str                        # "蓝色,圆眼"
    body: str                        # "瘦高"
    clothing: str                    # "白衬衫黑外套"
    distinctive: list[str]           # 显著标志(疤痕/纹身/配饰)


class ReferenceWindow(BaseModel):
    """传给图像Agent的参考集合。"""
    panel_id: str
    char_id: str
    references: list[WeightedReference]
    window_strategy: Literal["chain", "base_only", "fallback"]


class WeightedReference(BaseModel):
    image_path: str
    weight: float                    # 时间衰减权重
    role: Literal["base", "recent_n-1", "recent_n-2", "archive"]
    embedding_id: str | None = None


class IdentityEmbedding(BaseModel):
    """动态身份嵌入向量,供图像Agent注入IP-Adapter。"""
    char_id: str
    embedding: list[float]           # CLIP特征向量
    dim: int
    weights_used: dict[str, float]   # 各参考图的权重
```

## 5. 处理流程

### 5.1 init - 初始化角色档案

```
输入: CharacterDraft列表
   ↓
[图像Agent调用] 为每个角色生成3张候选参考图
   → 提示词 = appearance + style_preset
   ↓
🔔 等待用户确认(半自动模式)
   - 选择最佳参考 / 上传自定义图 / 调整描述重生
   ↓
[CLIP特征提取]
   → 计算每个角色的embedding
   ↓
[FAISS索引]
   → 建立角色embedding检索
   ↓
[VisualTraits提取]
   → LLM从参考图描述+生成prompt中抽取关键特征
   ↓
输出: CharacterDB
```

### 5.2 build_reference_window - 构建参考窗口

**核心算法(链式参考)**:

```python
def build_window(char_id, recent_panels, window_size=3):
    profile = character_db[char_id]
    refs = [
        WeightedReference(
            image_path=profile.base_reference.image_path,
            weight=1.0,                    # 基础参考权重最高
            role="base",
        )
    ]

    # 取最近N-1个出现该角色的面板
    relevant = [p for p in recent_panels
                if char_id in p.characters_present][-window_size+1:]

    for i, panel in enumerate(relevant):
        # 时间衰减权重: 越近权重越大
        decay = 0.7 ** (len(relevant) - i - 1)
        refs.append(WeightedReference(
            image_path=panel.image_path,
            weight=0.6 * decay,            # 近期参考权重低于base
            role=f"recent_n-{len(relevant)-i}",
        ))

    return ReferenceWindow(
        char_id=char_id,
        references=refs,
        window_strategy="chain",
    )
```

**动态身份嵌入计算**:

```python
def compute_identity_embedding(window):
    embeddings = []
    weights = []
    for ref in window.references:
        emb = clip_extract(ref.image_path)
        embeddings.append(emb)
        weights.append(ref.weight)

    weights = normalize(weights)
    fused = sum(w * e for w, e in zip(weights, embeddings))
    return IdentityEmbedding(
        embedding=fused.tolist(),
        weights_used={r.role: r.weight for r in window.references},
    )
```

### 5.3 update_archive - 更新档案库

触发条件: 图像Agent生成的某帧 + 审查Agent评分 ≥ 8.0

```
输入: 高质量PanelImage
   ↓
[CLIP特征提取]
   → 计算embedding
   ↓
[相似度检查]
   → 与base_reference相似度 ≥ 0.7 才入档
   ↓
[姿态/表情识别]
   → LLM描述pose/expression
   ↓
[档案库更新]
   → 限制每角色档案 ≤ 20条,FIFO淘汰
   ↓
输出: archive_updated=True
```

## 6. 一致性保障策略

### 6.1 三层参考机制

| 层级 | 来源 | 权重 | 作用 |
|------|------|------|------|
| L1 基础参考 | 用户确认 | 1.0 | 角色身份锚点(必须) |
| L2 近期面板 | 上几格输出 | 0.3-0.6 | 平滑过渡,避免突变 |
| L3 档案库 | 高质量历史 | 0.2 | 丰富姿态/表情(可选) |

### 6.2 降级策略

| 场景 | 降级方案 |
|------|---------|
| 角色首次出场 | 仅用base_reference |
| 近期面板不可用 | 跳过L2,使用L1+L3 |
| CLIP提取失败 | 仅使用文本VisualTraits注入prompt |
| FAISS索引损坏 | 重建索引,使用base_reference |

## 7. 流式事件

| 阶段 | 事件类型 | 内容 |
|------|---------|------|
| init启动 | LOG | "开始构建角色档案..." |
| 候选图生成中 | PROGRESS | 0→1.0(每个角色) |
| 候选图就绪 | PARTIAL_OUTPUT | 候选图Gallery |
| 等待用户确认 | LOG | "请选择基础参考图" |
| CLIP计算 | PROGRESS | |
| build_window启动 | LOG | "为panel_X构建参考..." |
| 完成 | DONE | CharacterOutput |

## 8. 返工策略

| 审查问题 | 返工动作 |
|---------|---------|
| 描述具象度不足 | 重新生成候选参考图,prompt加入"详细外观描述" |
| 视觉特征不明确 | 重新提取VisualTraits,使用更强的LLM抽取prompt |
| 角色辨识度低 | 加入distinctive标志(配饰/发型),重生候选 |
| 风格不一致 | 强制对齐style_preset,重生 |

## 9. 数据持久化

```
projects/{project_id}/
├── character_db.json              # CharacterDB主表
├── characters/
│   ├── {char_id}/
│   │   ├── base_reference.png
│   │   ├── archive/
│   │   │   ├── {entry_id}.png
│   │   │   └── ...
│   │   └── embeddings.npy         # 缓存的CLIP特征
└── faiss_index.bin                # FAISS索引文件
```

## 10. 性能指标

| 指标 | 目标 |
|------|------|
| init P50 (3角色) | ≤ 60s (含图像生成) |
| build_window P50 | ≤ 200ms |
| CLIP特征提取 | ≤ 100ms/张 |
| FAISS查询 P99 | ≤ 50ms |
| 角色一致性CLIP相似度 | ≥ 0.75 |

## 11. 测试要点

- 单角色 / 多角色 / 角色重出场场景
- base_reference不存在时的降级
- 链式参考权重计算正确性
- 档案库容量上限触发淘汰
- FAISS索引并发读写

---

**文档版本**: v1.0
**负责成员**: B
**依赖**: BaseAgent, ScriptAgent输出, ImageAgent(用于生成参考图)
**下游**: 图像生成Agent(消费ReferenceWindow)
