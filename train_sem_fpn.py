import sys
import os

from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

def main():
    register_all_modules()
    
    # 使用FPN配置
    cfg = Config.fromfile('configs/sem_fpn/fpn_r50_4xb4-160k_ade20k-512x512.py')
    
    # 修改配置
    cfg.work_dir = './work_dirs/sem_fpn_ade20k'
    
    # 训练参数
    cfg.train_dataloader.batch_size = 4
    cfg.train_dataloader.num_workers = 4
    cfg.val_dataloader.batch_size = 4
    cfg.val_dataloader.num_workers = 4
    
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