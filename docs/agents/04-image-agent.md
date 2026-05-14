# 图像生成 Agent (ImageAgent)

> 系统的"执行画师"。将分镜提示词与角色参考转换为像素图像,屏蔽多种后端模型差异。

## 1. 职责边界

**做什么**:
- 接收分镜提示词包 + 角色参考窗口,生成单帧图像
- 通过UMI(Unified Model Interface)抽象多种后端
- 自动质量检查与参数优化迭代
- 多后端能力协商与降级
- 生成元信息记录(seed/参数/耗时)

**不做什么**:
- 不规划构图(交给分镜Agent)
- 不维护角色档案(交给角色Agent)
- 不渲染对话气泡(交给排版Agent)

## 2. 接口定义

```python
class ImageAgent(BaseAgent[ImageInput, ImageOutput]):
    name = "image_agent"
    version = "1.0.0"
    rubric_id = "rubric_image_v1"
```

## 3. 输入模型

```python
class ImageInput(BaseModel):
    panel_plan: PanelPlan            # 来自StoryboardOutput
    reference_windows: dict[str, ReferenceWindow]  # char_id → window
    identity_embeddings: dict[str, IdentityEmbedding]
    style_preset: str

    # 后端选择
    backend_preference: BackendPreference = BackendPreference.AUTO
    quality_target: Literal["draft", "standard", "high"] = "standard"

    # 控制参数
    seed: int | None = None
    width: int = 768
    height: int = 1024

    # 返工时携带前次结果
    previous_output: PanelImage | None = None
    feedback: ReviewFeedback | None = None
```

## 4. 输出模型

```python
class ImageOutput(BaseModel):
    panel_image: PanelImage
    backend_used: str                # "sdxl" | "diffsensei" | "api_xx"
    fallback_chain: list[str]        # 降级历史(若发生)
    self_check: SelfCheckResult      # Agent自评

    meta: AgentOutputMeta


class PanelImage(BaseModel):
    panel_id: str
    image_path: str                  # 本地路径
    image_format: Literal["png", "webp"]
    width: int
    height: int

    # 生成元信息
    backend: str
    model_version: str
    seed: int
    steps: int
    cfg_scale: float
    sampler: str
    generation_time_ms: int

    # 内容描述
    characters_present: list[str]
    prompt_used: str
    negative_prompt_used: str

    # 一致性指标
    consistency_scores: dict[str, float]  # char_id → CLIP相似度
    timestamp: float


class SelfCheckResult(BaseModel):
    """Agent生成后的自评(供审查Agent参考)。"""
    matches_prompt: float            # 0-1,主观估计
    character_consistency: float
    artifact_warnings: list[str]     # 检测到的瑕疵
    suggested_retry: bool


class BackendPreference(str, Enum):
    AUTO = "auto"                    # 根据quality_target与panel_plan自动选
    PREFER_SDXL = "prefer_sdxl"
    PREFER_DIFFSENSEI = "prefer_diffsensei"
    PREFER_API = "prefer_api"
    LOCAL_ONLY = "local_only"        # 禁用云端API
```

## 5. UMI 后端接口

所有后端实现以下抽象类:

```python
class ImageBackend(ABC):
    backend_id: str
    capabilities: BackendCapabilities

    @abstractmethod
    async def generate(
        self,
        request: BackendRequest,
    ) -> BackendResponse: ...

    @abstractmethod
    async def health_check(self) -> bool: ...


class BackendCapabilities(BaseModel):
    supports_ip_adapter: bool        # 角色参考
    supports_controlnet: bool        # 构图控制
    supports_multi_character: bool   # 多角色掩码
    max_resolution: tuple[int, int]
    typical_latency_ms: int
    cost_per_generation: float       # 估算成本(¥)
    quality_tier: Literal["basic", "good", "best"]


class BackendRequest(BaseModel):
    """标准化的后端请求(由ImageAgent构造)。"""
    prompt: str
    negative_prompt: str
    reference_images: list[WeightedReference]
    identity_embeddings: dict[str, list[float]]
    width: int
    height: int
    steps: int
    cfg_scale: float
    seed: int
    sampler: str
    layout_mask: list[CharacterMask] | None = None  # 多角色掩码


class CharacterMask(BaseModel):
    char_id: str
    bbox: BoundingBox                # 该角色应出现的区域
    weight: float


class BackendResponse(BaseModel):
    image_data: bytes                # PNG字节流
    metadata: dict
    warnings: list[str]
    used_seed: int
```

## 6. 处理流程

```
输入: PanelPlan + ReferenceWindow
   ↓
[Step 1] 后端选择
   → 根据quality_target/角色数/能力协商
   ↓
[Step 2] 请求构造
   → 拼装BackendRequest
   → 注入identity_embeddings
   ↓
[Step 3] 缓存查询
   → inputs_hash命中 → 直接返回缓存
   ↓
[Step 4] 调用后端
   → backend.generate(request)
   → 失败 → 触发降级链
   ↓
[Step 5] 自检
   → CLIP相似度计算(与各角色base_reference比对)
   → 瑕疵检测(分辨率/黑屏/纯色)
   ↓
[Step 6] 元信息记录
   → 落盘image + metadata.json
   ↓
[Step 7] 缓存写入
   → diskcache,key=inputs_hash
   ↓
输出: ImageOutput
```

## 7. 后端选择策略

```python
def select_backend(panel_plan, quality_target, preference):
    char_count = len(panel_plan.characters_in_panel)

    # 多角色高质量 → DiffSensei
    if char_count >= 2 and quality_target == "high":
        if available("diffsensei"):
            return "diffsensei"

    # 高质量单角色 → SDXL
    if quality_target in ("standard", "high"):
        if available("sdxl"):
            return "sdxl"

    # 草稿 → SD1.5(快速)
    if quality_target == "draft":
        return "sd15"

    # GPU不可用 → 云端API
    if not gpu_available():
        return "api_default"

    return "sdxl"  # fallback
```

## 8. 降级链

```
主选后端
   ↓ 失败
备选1: 同等级不同后端
   ↓ 失败
备选2: 降级到SDXL基础版
   ↓ 失败
备选3: 云端API
   ↓ 失败
ResourceError → 触发用户决策
```

降级时的参数调整:
- 切换到更小模型: 降低steps与分辨率
- 切换到API: 移除IP-Adapter,改用文本VisualTraits
- 全部失败: 返回占位图 + ESCALATE

## 9. 质量检查规则

### 9.1 硬性检查(失败必返工)

```python
def hard_check(image, panel_plan):
    if image.width < 512 or image.height < 512:
        return False, "分辨率过低"
    if is_solid_color(image, threshold=0.9):
        return False, "近纯色图像"
    if is_black_or_white(image):
        return False, "黑屏/白屏"
    return True, ""
```

### 9.2 软性检查(供审查Agent参考)

```python
def soft_check(image, references, panel_plan):
    consistency = {}
    for char_id, ref in references.items():
        clip_sim = compute_clip_similarity(image, ref.base_reference)
        consistency[char_id] = clip_sim

    return SelfCheckResult(
        matches_prompt=estimate_prompt_match(image, panel_plan),
        character_consistency=mean(consistency.values()),
        artifact_warnings=detect_artifacts(image),
        suggested_retry=any(s < 0.6 for s in consistency.values()),
    )
```

## 10. 缓存策略

```python
def cache_key(inputs: ImageInput) -> str:
    return hash({
        "prompt": inputs.panel_plan.prompt_pack.positive_prompt,
        "negative": inputs.panel_plan.prompt_pack.negative_prompt,
        "refs": [r.image_path for w in inputs.reference_windows.values()
                              for r in w.references],
        "seed": inputs.seed,
        "size": (inputs.width, inputs.height),
        "backend": inputs.backend_preference,
    })
```

- 缓存目录: `cache/images/{cache_key}.png` + `.json`
- 容量上限: 5GB,LRU淘汰
- 返工时绕过缓存(强制重新生成)

## 11. 流式事件

| 阶段 | 事件类型 | 内容 |
|------|---------|------|
| 后端选择 | LOG | "使用 SDXL 生成" |
| 缓存命中 | LOG | "缓存命中,直接返回" |
| 生成中 | PROGRESS | 0.0→1.0(基于steps) |
| 中间步数 | PARTIAL_OUTPUT | 中间latent解码图(可选) |
| 自检 | LOG | "一致性: 0.82, 通过" |
| 降级触发 | LOG | "SDXL失败,降级到API" |
| 完成 | DONE | ImageOutput |

## 12. 返工策略

| 审查问题 | 返工动作 |
|---------|---------|
| 角色一致性低 | 提高IP-Adapter权重(+0.2),换seed |
| 与提示词不匹配 | 提高cfg_scale(+1.0),强化关键词 |
| 构图问题 | 加入构图负向提示,换seed |
| 风格漂移 | 强化style_preset_tags,降低temperature |
| 多角色串脸 | 切换到DiffSensei(若可用),启用layout_mask |
| 瑕疵明显 | 提高steps,使用质量更高sampler |

返工时seed策略:
- 第1次返工: 保持seed,只调参
- 第2次返工: 换seed
- 第3次返工: 切换后端

## 13. 性能指标

| 指标 | 目标 |
|------|------|
| SDXL P50 (768x1024) | ≤ 12s |
| DiffSensei P50 | ≤ 25s |
| API P50 | ≤ 8s |
| 缓存命中率 | ≥ 30% |
| 硬性检查通过率 | ≥ 98% |
| 角色一致性平均CLIP | ≥ 0.75 |

## 14. 测试要点

- 各后端单元测试(可Mock)
- 降级链触发测试
- 缓存命中/失效测试
- 多角色掩码生成正确性
- 极端参数边界(超大分辨率/极端seed)
- 自检算法的稳定性

---

**文档版本**: v1.0
**负责成员**: C
**依赖**: BaseAgent, StoryboardOutput, CharacterAgent输出
**下游**: 排版合成Agent(消费PanelImage), 角色管理Agent(高质量帧入档)
