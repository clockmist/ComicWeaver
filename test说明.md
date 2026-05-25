## 项目框架
### 1. 文件说明
#### 1.1 generate_character.json
ComfyUI中生成角色的工作流配置，由于是直接复制别人的工作流，所以存在一些多余的节点，后续可修改

#### 1.2 generate_picture.json
ComfyUI中生成最终图片的工作流。目前只使用了ip adapter。但是效果不是很好，可能是工作流也可能是prompt的问题

#### 1.2 test.py/test_api.py
测试各种接口的连接，无其他作用

#### 1. test_agent.py/test_agent_2.py
由ai生成的整个agent框架，使用了langgraph，并调用了comfyUI。由于是ai生成，效果方面不是很好，后续需要人为写一些代码，从而对prompt进行更加精细的控制。


### 2. 项目配置
#### 1. LLM api
采用了智谱的glm-4-flash，因为有免费额度。

#### 2. ComfyUI
需要下载模型，照理说只要是checkpoints类型的就能够跑起来，但是不同的模型超参数一般是不同的，目前使用的文生图模型是IL-novaOrangeXL_v100.safetensors。

除此之外，需要安装插件ComfyUI_IPAdapter_plus，以及相对应的CLIP，IPAdapter模型，目前采用的是CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors和ip-adapter-faceid-plusv2_sdxl.bin。然后根据`https://github.com/cubiq/ComfyUI_IPAdapter_plus`中的说明配置`insightface`。

后续ComfyUI处可能需要大量地修改模型和配置
