## 项目框架

### 1. 文件说明

#### 1.1 generate_character.json

ComfyUI中生成角色的工作流配置，由于是直接复制别人的工作流，所以存在一些多余的节点，后续可修改

#### 1.2 generate_picture.json

ComfyUI中生成最终图片的工作流。目前只使用了ip adapter。但是效果不是很好，可能是工作流也可能是prompt的问题

#### 1.3 test.py/test_api.py

测试各种接口的连接，无其他作用

#### 1.4 test_agent.py/test_agent_2.py

由ai生成的整个agent框架，使用了langgraph，并调用了comfyUI。由于是ai生成，效果方面不是很好，后续需要人为写一些代码，从而对prompt进行更加精细的控制。

#### 1.5 comic_agent.py

目前新写的agent工作流，虽然简陋但是能够跑通。输入是一段故事描述（由于模型限制最好将性别固定为女，更改模型后可能会好一点），输出是人设图和多张人物相关的图片

### 2. 项目配置

#### 1. LLM api

采用了智谱的glm-4-flash，因为有免费额度。

#### 2. ComfyUI

需要下载模型，照理说只要是checkpoints类型的就能够跑起来，但是不同的模型超参数一般是不同的，目前使用的文生图模型是IL-novaOrangeXL_v100.safetensors。

除此之外，需要安装插件ComfyUI_IPAdapter_plus，以及相对应的CLIP，IPAdapter模型，目前采用的是CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors和ip-adapter-faceid-plusv2_sdxl.bin。然后根据`https://github.com/cubiq/ComfyUI_IPAdapter_plus`中的说明配置`insightface`。

后续ComfyUI处可能需要大量地修改模型和配置

### 3. 修改过程（已修改完成）

目前test_agent中存在的问题：

1. 代码长度太长了，有些地方根本没必要这么长，例如我们假定一定使用的是langgraph框架，不用考虑没有langgraph的情况，这样子我们就能用langgraph的state在节点间传递信息，从而节省很多代码(一些非必要的分支语句也能删了)
2. 如果一些地方报错，目前代码的逻辑是加入一些默认的输出，但是真实agent这样子做会违反客户的需求，如果输出格式不符合，现阶段直接报错就行了（后续可能会使用其他方式进行管理）
3. agent逻辑不清晰，目前处在开发初期，我们完全不需要最后两个agent，即质量控制agent和排版agent。风格agent我们也不需要，先固定成generate_character.json中38号节点里面的masterpiece, best quality, ultra high res, hyper-detailed, anime realism, 8K, newest即可。其他agent也能简化，我们需要的基本上就是剧本解析agent（将用户输入扩充成剧本，自然语言描述人物，以及最后生成几张图，和图里面的内容），角色prompt生成agent（将角色的自然语言描述改成角色prompt），角色人设生成prompt(人设图是半身像或者胸像，需要保证整个头部出现在画面里面)，分镜生成prompt(根据自然语言描述，生成背景，人物姿势，镜头视角等相关prompt),画面生成agent（根据prompt，和人设图用ipadapter生成最终的画面）

### 4. 后续工作

1. 寻找更加稳定地保持人物一致性的方法，现在只是简单的使用了ipadapter和固定种子
2. 或许大家可以各自试试各种模型看看哪一种效果最好
3. 优化整体agent工作流，即尝试在prompt层面更好地控制画面，以及加入排版agent和加入台词agent
