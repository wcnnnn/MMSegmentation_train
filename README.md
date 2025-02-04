# MMSegmentation 训练脚本

本项目包含多个语义分割模型的训练脚本，基于 MMSegmentation 框架。

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

## 数据集准备

1. 下载 ADE20K 数据集并解压到指定目录:
```
data/ade/ADEChallengeData2016/
├── annotations/  # 标注文件
│   ├── training/
│   └── validation/
├── images/      # 图像文件
│   ├── training/
│   └── validation/
```

## 可用模型

| 类别 | 模型 | 训练脚本 | 说明 |
|:---:|:---|:---|:---|
| **基础卷积** | FCN | train_fcn.py | 全卷积网络 |
| | PSPNet | train_pspnet.py | 金字塔场景解析 |
| | DeepLabV3+ | train_deeplabv3plus.py | 带编解码器的空洞卷积 |
| | UPerNet | train_upernet.py | 统一感知解析 |
| | OCRNet | train_ocrnet.py | 目标上下文表示 |
| **注意力机制** | GCNet | train_gcnet.py | 全局上下文网络 |
| | ISANet | train_isanet.py | 交互式自注意力 |
| | NonLocal | train_nonlocal.py | 非局部注意力 |
| | DNLNet | train_dnlnet.py | 双重注意力 |
| | PSANet | train_psanet.py | 点采样注意力 |
| **Transformer** | SegFormer | train_segformer.py | 分层Transformer |
| | Segmenter | train_segmenter.py | 纯Transformer分割 |
| | SegNext | train_segnext.py | 下一代分割器 |
| | SETR | train_setr.py | Transformer编码器 |
| | ViT-Adapter | train_vit_deit.py | 视觉Transformer适配器 |
| **实例分割** | Mask2Former | train_mask2former.py | 掩码注意力 |
| | MaskFormer | train_maskformer.py | 掩码生成 |
| | PointRend | train_pointrend.py | 点渲染 |
| | KNet | train_knet.py | 核网络 |
| **轻量级** | MobileNetV2 | train_mobilenetv2.py | 移动端优化 |
| | PoolFormer | train_poolformer.py | 池化Transformer |
| **其他** | FastFCN | train_fastfcn.py | 快速全卷积 |
| | ResNeSt | train_resnest.py | 分割增强残差网络 |
| | SeMask | train_sem_fpn.py | 语义掩码特征金字塔 |
| | STDC | train_stdc.py | 短时程动态卷积 |
| | Twins | train_twins.py | 孪生Transformer |
| | SWIN | train_swin.py | 层次化窗口Transformer | 

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
- 每个模型保留最新的3个检查点
