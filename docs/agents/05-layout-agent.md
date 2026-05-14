# 排版合成 Agent (LayoutAgent)

> 系统的"最终装配线"。将分散的单帧图像拼装为完整漫画页面,处理对话气泡、文字与多格式导出。

## 1. 职责边界

**做什么**:
- 按分镜布局拼接单帧图像为完整页面
- 智能放置对话气泡,避让关键画面
- 文字渲染(自动换行/字号自适应/字体管理)
- 跨页与出血处理
- 多格式导出(PNG/PDF/PSD)

**不做什么**:
- 不修改单帧图像内容(交给图像Agent返工)
- 不重新规划布局(分镜Agent已确定bbox)
- 不生成新对话(剧本Agent已提供)

## 2. 接口定义

```python
class LayoutAgent(BaseAgent[LayoutInput, LayoutOutput]):
    name = "layout_agent"
    version = "1.0.0"
    rubric_id = "rubric_layout_v1"
```

## 3. 输入模型

```python
class LayoutInput(BaseModel):
    pages: list[PageLayout]          # 来自StoryboardOutput
    panel_images: dict[str, PanelImage]  # panel_id → PanelImage
    style_preset: str

    # 排版偏好
    page_size: PageSize = PageSize.A4
    dpi: int = 300
    gutter_width: int = 8            # 格间距(px)
    border_width: int = 2
    background_color: str = "#FFFFFF"

    # 字体配置
    font_config: FontConfig

    # 输出格式
    export_formats: list[ExportFormat] = [ExportFormat.PNG]

    feedback: ReviewFeedback | None = None


class FontConfig(BaseModel):
    speech_font: str = "fonts/SourceHanSansCN.otf"
    thought_font: str = "fonts/SourceHanSerifCN.otf"
    narration_font: str = "fonts/SourceHanSerifCN-Bold.otf"
    sfx_font: str = "fonts/MangaSFX.ttf"
    base_size_pt: int = 12
    min_size_pt: int = 8
    max_size_pt: int = 20


class PageSize(BaseModel):
    name: str                        # "A4" | "B5" | "Web"
    width_mm: float
    height_mm: float


class ExportFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    PDF = "pdf"
    PSD = "psd"
    WEBP = "webp"
```

## 4. 输出模型

```python
class LayoutOutput(BaseModel):
    final_pages: list[FinalPage]
    exports: list[ExportArtifact]
    overall_metrics: LayoutMetrics

    meta: AgentOutputMeta


class FinalPage(BaseModel):
    page_id: str
    page_number: int
    image_path: str                  # 合成后的页面PNG
    width_px: int
    height_px: int
    bubbles: list[PlacedBubble]      # 已放置的气泡记录
    layout_warnings: list[str]


class PlacedBubble(BaseModel):
    bubble_id: str
    panel_id: str
    dialogue_index: int
    bubble_type: Literal["speech", "thought", "shout", "whisper", "narration"]
    bbox: BoundingBox                # 在页面中的位置
    tail_target: tuple[float, float] | None  # 气泡尾部指向(说话角色)
    text: str
    font_size_pt: int
    line_count: int
    occlusion_score: float           # 遮挡评分(越低越好)


class ExportArtifact(BaseModel):
    format: ExportFormat
    file_path: str
    file_size_bytes: int
    page_count: int


class LayoutMetrics(BaseModel):
    total_pages: int
    total_bubbles: int
    avg_occlusion_score: float
    bubbles_overflow_count: int      # 文字溢出次数
    auto_resize_count: int           # 自动缩字次数
```

## 5. 处理流程

```
输入: PageLayout + PanelImage字典
   ↓
[Step 1] 画布初始化
   → 按page_size/dpi创建空白页
   ↓
[Step 2] 单帧拼接
   → 按bbox将PanelImage缩放/裁剪/粘贴
   → 应用PanelShape(矩形/斜切/出血/异形)
   → 绘制边框与gutter
   ↓
[Step 3] 显著性检测
   → 检测每个panel的关键区域(角色脸/动作焦点)
   → 输出avoid_zones映射
   ↓
[Step 4] 气泡布局优化
   → 候选位置生成
   → 多目标优化(避让/阅读流/美观)
   ↓
[Step 5] 文字渲染
   → 自动换行
   → 字号自适应
   → 应用气泡样式(类型决定形状/边框)
   ↓
[Step 6] 后处理
   → 全局色调统一
   → 边缘锐化
   → 元信息嵌入(EXIF)
   ↓
[Step 7] 导出
   → 多格式批量导出
   ↓
输出: LayoutOutput
```

## 6. 核心算法

### 6.1 单帧拼接

对不同PanelShape的处理:

| Shape | 处理方式 |
|-------|---------|
| RECTANGLE | 直接缩放粘贴,绘制边框 |
| DIAGONAL | 应用斜切mask,边框跟随 |
| BLEED | 突破gutter边界,延伸到页面边缘 |
| SPLASH | 占据整页或大部分,无边框或细边框 |
| CIRCULAR | 圆形mask,常用于回忆/特写 |
| JAGGED | 锯齿边框,常用于冲击/呐喊 |

### 6.2 气泡位置优化

**优化目标函数**:

```python
def bubble_score(candidate_pos, panel, dialogue, prev_bubbles):
    # 越低越好
    occlusion = compute_occlusion(candidate_pos, panel.avoid_zones)
    flow_penalty = compute_flow_violation(candidate_pos, prev_bubbles)
    overflow_risk = check_text_overflow(candidate_pos, dialogue.text)
    aesthetic_penalty = -compute_aesthetic(candidate_pos, panel)

    return (
        occlusion * 3.0 +
        flow_penalty * 2.0 +
        overflow_risk * 1.5 +
        aesthetic_penalty * 1.0
    )
```

**搜索策略**:
1. 生成8-12个候选位置(围绕panel四角与边缘)
2. 对每个候选计算score
3. 选择score最低的位置
4. 若全部score超过阈值,触发"溢出告警"

### 6.3 自动换行与字号

```python
def fit_text(text, bbox, font_config, max_lines=4):
    for size in range(font_config.base_size_pt,
                      font_config.min_size_pt - 1, -1):
        wrapped = wrap_text(text, bbox.width, size)
        if len(wrapped) <= max_lines:
            if total_height(wrapped, size) <= bbox.height:
                return wrapped, size
    # 仍放不下 → 截断 + ...
    return truncate_with_ellipsis(text, bbox, font_config.min_size_pt)
```

### 6.4 气泡形状渲染

| bubble_type | 形状 | 字体 |
|-------------|------|------|
| speech | 椭圆 + 尾部 | speech_font |
| thought | 云朵 + 小圆点尾部 | thought_font (italic) |
| shout | 锯齿星形 | sfx_font (bold) |
| whisper | 虚线椭圆 | speech_font (small) |
| narration | 矩形框,无尾部 | narration_font |

### 6.5 显著性检测

```python
def detect_avoid_zones(panel_image):
    # 方法1: 简单启发式
    #   - 通过CLIP获取关键区域
    #   - 通过边缘密度估计动作焦点
    # 方法2(可选): 调用YOLO/Mediapipe检测人脸

    zones = []
    # 角色面部区域
    faces = detect_faces(panel_image)  # 简单haar或mediapipe
    for face in faces:
        zones.append(BoundingBox(
            x=face.x - 0.1, y=face.y - 0.1,
            width=face.width + 0.2, height=face.height + 0.2,
        ))
    return zones
```

## 7. 流式事件

| 阶段 | 事件类型 | 内容 |
|------|---------|------|
| Step 1-2 | PROGRESS | 0→0.4(每页拼接) |
| 单页拼接完成 | PARTIAL_OUTPUT | 无气泡的页面预览 |
| Step 3-4 | PROGRESS | →0.7 |
| 气泡放置 | LOG | "Page 3: 4气泡放置完成" |
| Step 5 | PROGRESS | →0.85 |
| 溢出告警 | LOG | "对话过长,字号缩至8pt" |
| Step 7 | PROGRESS | →1.0 |
| 完成 | DONE | LayoutOutput |

## 8. 返工策略

| 审查问题 | 返工动作 |
|---------|---------|
| 气泡遮挡角色脸 | 重跑Step 4,加强avoid_zones权重 |
| 阅读顺序混乱 | 重跑Step 4,启用严格阅读流约束 |
| 文字溢出 | 切换到双气泡分割长对话 |
| 气泡风格不统一 | 强制使用同一bubble样式集 |
| 边框过粗/过细 | 调整border_width,重跑Step 2 |
| 配色与画面冲突 | 切换气泡背景到半透明 |

## 9. 导出格式细节

### 9.1 PNG

- 单页一个文件,命名 `page_{N:02d}.png`
- 8bit RGB,无alpha
- 默认300dpi

### 9.2 PDF

- 单文件包含所有页
- 嵌入字体(避免依赖系统字体)
- 元信息: 标题/作者/创建时间

### 9.3 PSD

- 分层导出(便于二次编辑):
  - L1: 背景
  - L2: 单帧图像组(每个panel一层)
  - L3: 边框层
  - L4: 气泡层(每个bubble一层)
  - L5: 文字层
- 使用 `psd-tools` 或 `pytoshop` 库

## 10. 性能指标

| 指标 | 目标 |
|------|------|
| 单页合成 P50 | ≤ 5s |
| 8页全流程 P50 | ≤ 60s |
| 气泡溢出率 | ≤ 5% |
| 平均遮挡评分 | ≤ 0.15 |
| PNG文件大小(A4 300dpi) | 2-5MB |

## 11. 测试要点

- 各PanelShape的渲染正确性
- 极端文本(超长/超短/全标点)的换行
- 气泡位置优化的稳定性
- 多语言文本(中英混合)
- 显著性检测在低质量图像上的鲁棒性
- 多格式导出的一致性

---

**文档版本**: v1.0
**负责成员**: D
**依赖**: BaseAgent, StoryboardOutput, ImageAgent输出
**下游**: 用户(导出文件)、审查Agent(最终审核)
