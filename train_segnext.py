import sys
import os

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

def main():
    register_all_modules()
    
    # 使用SegNext配置
    cfg = Config.fromfile('configs/segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py')
    
    # 修改配置
    cfg.work_dir = './work_dirs/segnext_ade20k'
    
    # 添加数据预处理器配置
    crop_size = (512, 512)
    cfg.model.data_preprocessor = dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size)
    
    # 训练参数
    cfg.train_dataloader.batch_size = 2
    cfg.train_dataloader.num_workers = 1
    cfg.val_dataloader.batch_size = 2
    cfg.val_dataloader.num_workers = 1
    
    # 训练轮次
    cfg.train_cfg.max_iters = 160000
    cfg.train_cfg.val_interval = 5000
    
    # 检查点设置
    cfg.default_hooks.checkpoint.interval = 5000
    cfg.default_hooks.checkpoint.max_keep_ckpts = 3
    cfg.default_hooks.logger.interval = 10
    
    # 数据集路径
    cfg.data_root = 'data/ade/ADEChallengeData2016'
    
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main() 