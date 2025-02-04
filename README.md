# MMSegmentation 训练脚本

本项目包含多个语义分割模型的训练脚本，基于 MMSegmentation 框架。

## 项目结构准备

1. 创建必要的目录:
```bash
mkdir data pretrain work_dirs
```

2. 下载预训练权重到 pretrain/ 目录:
```bash
# ResNet-50 预训练权重 (基础卷积模型使用)
wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth -P pretrain/

# Swin-Transformer 预训练权重 (Transformer类模型使用)
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth -P pretrain/

# ViT 预训练权重 (ViT-Adapter等模型使用)
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth -P pretrain/

# SegFormer 预训练权重
wget https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth -P pretrain/

# PoolFormer 预训练权重
wget https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/poolformer/poolformer_s12_8xb256_in1k_20221230-a59e4e30.pth -P pretrain/
```

3. 下载并解压 ADE20K 数据集:
```bash
# 下载数据集
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

# 解压到data目录
unzip ADEChallengeData2016.zip -d data/ade/
```

## 环境配置

1. 安装基础依赖:
```bash
pip install torch torchvision
pip install -U openmim
pip install mmengine
# 安装MMCV (如果在线安装失败，可以尝试以下方法)
# 1. 从 https://pypi.org/project/mmcv/2.1.0/#files 下载源文件 mmcv-2.1.0.tar.gz
# 2. pip install mmcv-2.1.0.tar.gz
pip install mmcv>=2.0.0
pip install mmsegmentation
```

2. 安装额外依赖(针对特定模型):
```bash
# 对于 Mask2Former 等模型
pip install mmdet

# 对于 PoolFormer 等模型
pip install mmpretrain
```

## 目录结构

确保项目结构如下:
```
data/
└── ade/
    └── ADEChallengeData2016/
        ├── annotations/
        │   ├── training/
        │   └── validation/
        └── images/
            ├── training/
            └── validation/

pretrain/
├── resnet50-11ad3fa6.pth
├── swin_base_patch4_window12_384_22k.pth
├── jx_vit_base_p16_384-83fb41ba.pth
├── mit_b0_20220624-7e0fe6dd.pth
└── poolformer_s12_8xb256_in1k_20221230-a59e4e30.pth
```

## 可用模型
| **类别**       | **模型**                   | **训练脚本**                     | **说明**                                                                 |
|:---------------|:---------------------------|:---------------------------------|:-------------------------------------------------------------------------|
| **基础卷积**   | FCN                        | [train_fcn.py](train_fcn.py)     | 全卷积网络，将全连接层替换为卷积层，输出与输入图像相同大小的分割结果。   |
|                | PSPNet                     | [train_pspnet.py](train_pspnet.py) | 金字塔场景解析网络，通过金字塔池化模块捕获不同尺度的上下文信息。         |
|                | DeepLabV3+                 | [train_deeplabv3plus.py](train_deeplabv3plus.py) | 基于空洞卷积的编解码器结构，捕捉多尺度信息，提高分割精度。               |
|                | UPerNet                    | [train_upernet.py](train_upernet.py) | 统一感知解析网络，处理多种视觉任务，具有较好的通用性和扩展性。           |
|                | OCRNet                     | [train_ocrnet.py](train_ocrnet.py) | 引入目标上下文信息，提高分割的准确性。                                   |
| **注意力机制** | GCNet                      | [train_gcnet.py](train_gcnet.py) | 全局上下文网络，捕获长距离依赖关系，增强特征表示能力。                   |
|                | ISANet                     | [train_isanet.py](train_isanet.py) | 交互式自注意力网络，增强特征之间的交互信息。                             |
|                | NonLocal                   | [train_nonlocal.py](train_nonlocal.py) | 非局部注意力网络，通过非局部操作捕获长距离依赖关系。                     |
|                | DNLNet                     | [train_dnlnet.py](train_dnlnet.py) | 双重注意力网络，同时捕捉空间和通道维度的信息。                           |
|                | PSANet                     | [train_psanet.py](train_psanet.py) | 点采样注意力网络，提高分割效率。                                         |
| **Transformer**| SegFormer                  | [train_segformer.py](train_segformer.py) | 分层 Transformer 网络，采用轻量级解码器，高效进行语义分割。               |
|                | Segmenter                  | [train_segmenter.py](train_segmenter.py) | 纯 Transformer 分割网络，利用强大的特征提取能力实现高精度分割。           |
|                | SegNext                    | [train_segnext.py](train_segnext.py) | 结合卷积和 Transformer 优点，提升分割性能。                               |
|                | SETR                       | [train_setr.py](train_setr.py) | Transformer 编码器网络，捕捉全局信息，提升分割精度。                       |
|                | ViT-Adapter                | [train_vit_deit.py](train_vit_deit.py) | 通过适配器将预训练的视觉 Transformer 应用于分割任务。                   |
| **实例分割**   | Mask2Former                | [train_mask2former.py](train_mask2former.py) | 掩码注意力网络，处理实例边界，提升实例分割性能。                         |
|                | MaskFormer                 | [train_maskformer.py](train_maskformer.py) | 生成掩码进行实例分割，实现高效分割。                                     |
|                | PointRend                  | [train_pointrend.py](train_pointrend.py) | 点渲染策略，关键位置精细分割，提升实例分割精度。                         |
|                | KNet                       | [train_knet.py](train_knet.py) | 核网络，捕捉实例形状信息，提升实例分割效果。                             |
| **轻量级**     | MobileNetV2                | [train_mobilenetv2.py](train_mobilenetv2.py) | 移动端优化的卷积网络，低计算复杂度，适用于资源受限设备。                 |
|                | PoolFormer                 | [train_poolformer.py](train_poolformer.py) | 池化 Transformer 网络，通过池化操作实现高效特征提取。                     |
| **其他**       | FastFCN                    | [train_fastfcn.py](train_fastfcn.py) | 快速全卷积网络，快速卷积操作，提高分割效率。                             |
|                | ResNeSt                    | [train_resnest.py](train_resnest.py) | 改进的残差结构，增强特征表示能力，提升分割性能。                         |
|                | SeMask                     | [train_sem_fpn.py](train_sem_fpn.py) | 语义掩码特征金字塔网络，捕捉不同尺度语义信息，提升分割效果。             |
|                | STDC                       | [train_stdc.py](train_stdc.py) | 短时程动态卷积网络，减少计算量，提高分割效率。                           |
|                | Twins                      | [train_twins.py](train_twins.py) | 孪生 Transformer 网络，捕捉不同尺度特征信息，提升分割性能。               |
|                | SWIN                       | [train_swin.py](train_swin.py) | 层次化窗口 Transformer 网络，减少计算量，提高分割效率。                   |



## 训练说明

1. 所有模型使用统一的训练参数:
- 批次大小: 2
- 训练轮次: 80k/160k iterations (根据模型配置)
- 验证间隔: 4000 iterations
- 图像尺寸: 512x512

2. 启动训练:
```bash
# 训练单个模型
python train_fcn.py
```

3. 训练输出:
- 模型权重保存在: `work_dirs/{model_name}_ade20k/`
- 训练日志保存在: `work_dirs/{model_name}_ade20k/`

