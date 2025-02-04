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

### MMSegmentation 模型列表

| 类别 | 模型 | 训练脚本 | 说明 |
| :---: | :--- | :--- | :--- |
| **基础卷积** | **FCN** <br> 全卷积网络，是最早的端到端全卷积网络，用于图像语义分割，将传统卷积网络中的全连接层替换为卷积层，能够输出与输入图像相同大小的分割结果。 | `train_fcn.py` | 该模型采用全卷积结构，能够直接处理任意大小的输入图像，适用于各种语义分割任务。 |
|  | **PSPNet** <br> 金字塔场景解析网络，通过金字塔池化模块捕获不同尺度的上下文信息，提升分割性能。 | `train_pspnet.py` | 引入金字塔池化模块，能够有效地融合不同尺度的特征，对复杂场景的分割效果较好。 |
|  | **DeepLabV3+** <br> 基于空洞卷积的语义分割模型，采用编解码器结构，能够捕捉多尺度信息，提高分割精度。 | `train_deeplabv3plus.py` | 利用空洞卷积扩大感受野，同时通过编解码器结构细化分割结果，在多个数据集上取得了优异的成绩。 |
|  | **UPerNet** <br> 统一感知解析网络，通过统一的架构处理不同类型的视觉任务，如语义分割、实例分割等。 | `train_upernet.py` | 采用统一的架构，能够同时处理多种视觉任务，具有较好的通用性和扩展性。 |
|  | **OCRNet** <br> 目标上下文表示网络，通过引入目标上下文信息，提高分割的准确性。 | `train_ocrnet.py` | 强调目标上下文信息的重要性，能够更好地处理目标之间的关系，提升分割效果。 |
| **注意力机制** | **GCNet** <br> 全局上下文网络，通过全局上下文模块捕获长距离依赖关系，增强特征表示能力。 | `train_gcnet.py` | 引入全局上下文模块，能够有效地捕捉图像中的长距离依赖关系，提高分割性能。 |
|  | **ISANet** <br> 交互式自注意力网络，通过交互式自注意力机制增强特征表示。 | `train_isanet.py` | 采用交互式自注意力机制，能够更好地捕捉特征之间的交互信息，提升分割精度。 |
|  | **NonLocal** <br> 非局部注意力网络，通过非局部操作捕获长距离依赖关系。 | `train_nonlocal.py` | 利用非局部操作，能够有效地捕捉图像中的长距离依赖关系，对复杂场景的分割效果较好。 |
|  | **DNLNet** <br> 双重注意力网络，通过双重注意力机制增强特征表示。 | `train_dnlnet.py` | 采用双重注意力机制，能够同时捕捉空间和通道维度的信息，提升分割性能。 |
|  | **PSANet** <br> 点采样注意力网络，通过点采样注意力机制提高分割效率。 | `train_psanet.py` | 引入点采样注意力机制，能够在保证分割精度的前提下，提高分割效率。 |
| **Transformer** | **SegFormer** <br> 分层 Transformer 网络，采用分层结构和轻量级解码器，能够高效地进行语义分割。 | `train_segformer.py` | 采用分层结构和轻量级解码器，具有较高的计算效率和分割性能。 |
|  | **Segmenter** <br> 纯 Transformer 分割网络，直接使用 Transformer 进行图像分割。 | `train_segmenter.py` | 纯 Transformer 架构，能够充分利用 Transformer 的强大特征提取能力，实现高精度的分割。 |
|  | **SegNext** <br> 下一代分割器，结合了卷积和 Transformer 的优点，提升分割性能。 | `train_segnext.py` | 融合了卷积和 Transformer 的优势，能够在不同数据集上取得较好的分割效果。 |
|  | **SETR** <br> Transformer 编码器网络，使用 Transformer 作为编码器，提高分割精度。 | `train_setr.py` | 采用 Transformer 作为编码器，能够捕捉图像中的全局信息，提升分割精度。 |
|  | **ViT - Adapter** <br> 视觉 Transformer 适配器，通过适配器将预训练的视觉 Transformer 应用于分割任务。 | `train_vit_deit.py` | 利用适配器将预训练的视觉 Transformer 迁移到分割任务中，提高模型的泛化能力。 |
| **实例分割** | **Mask2Former** <br> 掩码注意力网络，通过掩码注意力机制进行实例分割。 | `train_mask2former.py` | 引入掩码注意力机制，能够更好地处理实例之间的边界，提升实例分割性能。 |
|  | **MaskFormer** <br> 掩码生成网络，通过生成掩码进行实例分割。 | `train_maskformer.py` | 采用掩码生成策略，能够直接生成实例掩码，实现高效的实例分割。 |
|  | **PointRend** <br> 点渲染网络，通过点渲染策略提高实例分割的精度。 | `train_pointrend.py` | 利用点渲染策略，能够在关键位置进行精细分割，提升实例分割的精度。 |
|  | **KNet** <br> 核网络，通过核网络进行实例分割。 | `train_knet.py` | 采用核网络结构，能够有效地捕捉实例的形状信息，提升实例分割效果。 |
| **轻量级** | **MobileNetV2** <br> 移动端优化的卷积网络，适用于资源受限的设备。 | `train_mobilenetv2.py` | 专门为移动端设备优化，具有较低的计算复杂度和内存占用，能够在移动端实现实时分割。 |
|  | **PoolFormer** <br> 池化 Transformer 网络，通过池化操作实现高效的特征提取。 | `train_poolformer.py` | 采用池化操作代替传统的注意力机制，具有较高的计算效率和分割性能。 |
| **其他** | **FastFCN** <br> 快速全卷积网络，通过快速的卷积操作提高分割效率。 | `train_fastfcn.py` | 采用快速的卷积操作，能够在保证分割精度的前提下，提高分割效率。 |
|  | **ResNeSt** <br> 分割增强残差网络，通过改进的残差结构提升分割性能。 | `train_resnest.py` | 引入改进的残差结构，能够增强特征表示能力，提高分割性能。 |
|  | **SeMask** <br> 语义掩码特征金字塔网络，通过语义掩码和特征金字塔结构进行分割。 | `train_sem_fpn.py` | 结合语义掩码和特征金字塔结构，能够有效地捕捉不同尺度的语义信息，提升分割效果。 |
|  | **STDC** <br> 短时程动态卷积网络，通过短时程动态卷积提高分割效率。 | `train_stdc.py` | 采用短时程动态卷积，能够在保证分割精度的前提下，减少计算量，提高分割效率。 |
|  | **Twins** <br> 孪生 Transformer 网络，通过孪生结构提高分割性能。 | `train_twins.py` | 采用孪生结构，能够同时捕捉不同尺度的特征信息，提升分割性能。 |
|  | **SWIN** <br> 层次化窗口 Transformer 网络，通过层次化窗口结构提高分割效率。 | `train_swin.py` | 引入层次化窗口结构，能够有效地减少计算量，提高分割效率。 |
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

