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

| 模型类别 | 模型名称 | 训练脚本 |
|---------|---------|----------|
| **基础卷积模型** | FCN | train_fcn.py |
|  | PSPNet | train_pspnet.py |
|  | DeepLabV3+ | train_deeplabv3plus.py |
|  | UPerNet | train_upernet.py |
|  | OCRNet | train_ocrnet.py |
| **注意力机制模型** | GCNet | train_gcnet.py |
|  | ISANet | train_isanet.py |
|  | NonLocal | train_nonlocal.py |
|  | DNLNet | train_dnlnet.py |
|  | PSANet | train_psanet.py |
|  | DMNet | train_dmnet.py |
|  | ENCNet | train_encnet.py |
| **Transformer系列** | SegFormer | train_segformer.py |
|  | Segmenter | train_segmenter.py |
|  | SegNext | train_segnext.py |
|  | SETR | train_setr.py |
|  | ViT-Adapter | train_vit_deit.py |
| **实例分割模型** | Mask2Former | train_mask2former.py |
|  | MaskFormer | train_maskformer.py |
|  | PointRend | train_pointrend.py |
|  | KNet | train_knet.py |
| **轻量级模型** | MobileNetV2 | train_mobilenetv2.py |
|  | PoolFormer | train_poolformer.py |
| **其他特色模型** | FastFCN | train_fastfcn.py |
|  | ResNeSt | train_resnest.py |
|  | SeMask | train_sem_fpn.py |
|  | STDC | train_stdc.py |
|  | Twins | train_twins.py |
|  | SWIN | train_swin.py |

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

## 注意事项

1. 显存需求:
- 基础模型(FCN等): 8GB显存足够
- 高级模型(KNet等): 建议11GB以上显存

2. 训练时间:
- 40k iterations: 约12-15小时
- 80k iterations: 约24-30小时

3. 常见问题:
- 如果出现显存不足，可以尝试减小batch_size
- 如果训练速度慢，可以减小num_workers的值