## 项目框架

### 1. 文件说明

#### 1.1 generate_character.json

​	ComfyUI中生成角色的工作流配置。简单的checkpoint加上正面，负面提示词再加上K采样器。



#### 1.2 generate_picture.json

​	ComfyUI中生成最终图片的工作流。目前只使用了ip adapter。但是效果不是很好，可能是工作流也可能是prompt的问题



#### 1.3 comic_agent.py

​	目前新写的agent工作流，虽然简陋但是能够跑通。输入是一段故事描述（由于模型限制最好将性别固定为女，更改模型后可能会好一点），输出是人设图和多张人物相关的图片。

​	整体框架采用langgraph, 生成图片是通过http发送给ComfyUI进行生图任务。目前只实现了5个agent：剧本生成agent, 角色人设生成agent，角色人设图生成agent，分镜prompt生成agent和漫画图片生成agent。





### 2. 项目配置

#### 1. LLM api

采用了智谱的glm-4-flash，因为有免费额度。

#### 2. ComfyUI

**后续ComfyUI处可能需要大量地修改模型和配置**

​	目前运行项目需要下载模型，照理说只要是checkpoints类型的就能够跑起来，但是不同的模型超参数一般是不同的，目前使用的文生图模型是`IL-novaOrangeXL_v100.safetensors`。

​	除此之外，需要安装插件ComfyUI_IPAdapter_plus，以及相对应的CLIP，IPAdapter模型，目前采用的是`CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`和`ip-adapter-faceid-plusv2_sdxl.bin`。然后根据`https://github.com/cubiq/ComfyUI_IPAdapter_plus`中的说明配置`insightface`。



### 3. 项目难点和后续工作

1. 人物一致性：

   目前我们保持任务一致性的方法是使用ipadapter加上固定种子。调研到为了保持人物一致性一般是采用三种方法：IPAdapter, ControlNet和LoRA。LoRA训练起来比较麻烦，感觉不太适合agent这种偏轻量级的任务；而尝试IPAdapter多次发现效果不是很好；还未尝试过 ControlNet。

   ​	后续工作：可以将生成图片风格改成黑白漫画风，这样子能在视觉层面降低一致性的要求，也能试试不同模型看看效果。更进一步可以探索更稳定的模型以及主流的工作流以及探索ControlNet效果。

   

2. 排版agent：

​	目前完全没有对于排版的实现，初步思路是由剧本生成agent，或者新建一个agent先安排好一页漫画里面所有图片的位置，待漫画图	片生成agent生成结果之后组装图片。而组装可以调用ComfyUI中的`CR Comic Panel Templates`实现，也可以调python库尝试手动	实现。



	3. 台词agent：

​	有两种想法：一种是在生成图片的时候用prompt控制画面的重心在什么位置，后续就把台词放在重心的另一面；另一种方法是利用动	漫人脸识别模型，例如`yolov5-anime` 或 `yolov8-animeface`（但是它们的训练集应该是彩色图片），只要台词不遮挡住人脸就行了	。以及如果和排版agent配合得好的话，在画框外新生成一个框放台词也是可行的。



### 4. 项目重点

​	经过调研，现在一些模型加上LoRA或者一些全新的模型（例如diffsensei）能够很好地生成漫画了。那我认为我们项目的重点要放在生成过程整体的可控性，就像`src`文件夹里面那个框架一样，要让用户在整个生成过程中能及时介入并提供生成的建议，并驳回不满意的生成。

​	除此之外，漫画生成这一块，之前大多是关于人物一致性的研究。如果把排版加上台词生成处理好的话这个项目应该也能有所创新性。在agent层面上，可以加入一些外接知识库，让LLM生成更优质的prompt指导生图。
