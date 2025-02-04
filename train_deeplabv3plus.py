from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

def main():
    # 注册所有模块
    register_all_modules()
    
    # 加载配置文件 - 使用ADE20K的配置
    cfg = Config.fromfile('configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-160k_ade20k-512x512.py')
    
    # 修改配置
    # 1. 设置工作目录
    cfg.work_dir = './work_dirs/deeplabv3plus_ade20k'
    
    # 2. 设置训练参数
    cfg.train_dataloader.batch_size = 4  # 根据您的GPU显存调整
    cfg.train_dataloader.num_workers = 4
    
    # 3. 设置验证参数
    cfg.val_dataloader.batch_size = 4
    cfg.val_dataloader.num_workers = 4
    
    # 4. 设置评估参数
    cfg.test_evaluator.pop('metric_items', None)
    
    # 5. 设置训练轮次和验证间隔
    cfg.train_cfg.max_iters = 160000  # 增加迭代次数
    cfg.train_cfg.val_interval = 5000  # 相应调整验证间隔
    
    # 6. 设置检查点保存参数
    cfg.default_hooks.checkpoint.interval = 5000
    cfg.default_hooks.checkpoint.max_keep_ckpts = 3
    
    # 7. 设置日志参数
    cfg.default_hooks.logger.interval = 10
    
    # 8. 修改数据集路径
    cfg.data_root = 'data/ade/ADEChallengeData2016'  # 使用默认路径
    
    # 确保数据集目录结构正确
    cfg.train_dataloader.dataset.data_prefix = dict(
        img_path='images/training',
        seg_map_path='annotations/training'
    )
    cfg.val_dataloader.dataset.data_prefix = dict(
        img_path='images/validation',
        seg_map_path='annotations/validation'
    )
    
    # 创建训练器
    runner = Runner.from_cfg(cfg)
    
    # 开始训练
    runner.train()

if __name__ == '__main__':
    main() 