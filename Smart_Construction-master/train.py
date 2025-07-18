import argparse
import os
import sys
import yaml
import math
import random
import shutil
import time
import glob
import numpy as np
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# 导入YOLOv5自定义模块
import test  # 导入测试模块计算mAP
from models.yolo import Model
from utils import google_utils
from utils.datasets import *
from utils.utils import *

# 设置混合精度训练
mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

# 超参数配置
hyp = {'optimizer': 'SGD',  # ['adam', 'SGD', None] if none, default is SGD
       'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
       'momentum': 0.937,  # SGD momentum/Adam beta1
       'weight_decay': 5e-4,  # optimizer weight decay
       'giou': 0.05,  # giou loss gain
       'cls': 0.5,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'anchor_t': 4.0,  # anchor-multiple threshold
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
       'degrees': 0.0,  # image rotation (+/- deg)
       'translate': 0.0,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)


# 创建唯一目录函数
def increment_dir(dir, name=''):
    # 生成唯一目录名 runs/exp --> runs/exp1, runs/exp2 etc.
    n = 0  # 初始编号
    dir = str(Path(dir))  # 转换为Path对象
    d = dir + name  # 基础目录
    while os.path.exists(d):
        n += 1  # 递增编号
        d = f'{dir}{name}{n}'  # 新目录名
    os.makedirs(d, exist_ok=True)  # 创建目录
    return d


def train(hyp, tb_writer, opt, device):
    print(f'Hyperparameters {hyp}')
    log_dir = tb_writer.log_dir if tb_writer else 'runs/evolution'  # 运行目录
    wdir = str(Path(log_dir) / 'weights') + os.sep  # 权重目录
    os.makedirs(wdir, exist_ok=True)  # 确保目录存在
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    results_file = log_dir + os.sep + 'results.txt'
    epochs, batch_size, total_batch_size, weights, rank = \
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.local_rank

    # 保存运行设置
    with open(Path(log_dir) / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(Path(log_dir) / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # 配置
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # 加载数据配置
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # 类别数和名称
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # 检查类别数

    # 移除之前的结果
    if rank in [-1, 0]:
        for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
            os.remove(f)

    # 创建模型
    model = Model(opt.cfg, nc=nc).to(device)

    # 图像尺寸
    gs = int(max(model.stride))  # 网格大小（最大步长）
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # 验证图像尺寸是gs的倍数

    # 优化器
    nbs = 64  # 标称批量大小
    accumulate = max(round(nbs / total_batch_size), 1)  # 在优化前累积损失
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # 缩放权重衰减

    # 优化器参数分组
    pg0, pg1, pg2 = [], [], []  # 偏置、卷积权重、其他参数
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                pg2.append(v)  # 偏置
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v)  # 应用权重衰减
            else:
                pg0.append(v)  # 其他参数

    # 选择优化器
    if hyp['optimizer'] == 'adam':
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # 添加权重衰减组
    optimizer.add_param_group({'params': pg2})  # 添加偏置组
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # 加载模型
    with torch_distributed_zero_first(rank):
        google_utils.attempt_download(weights)
    start_epoch, best_fitness = 0, 0.0
    if weights.endswith('.pt'):  # PyTorch格式
        try:
            ckpt = torch.load(weights, map_location=device)  # 加载检查点

            # 加载模型权重
            exclude = ['anchor']  # 排除的键
            ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                             if k in model.state_dict() and not any(x in k for x in exclude)
                             and model.state_dict()[k].shape == v.shape}
            model.load_state_dict(ckpt['model'], strict=False)
            print('Transferred %g/%g items from %s' % (len(ckpt['model']), len(model.state_dict()), weights))

            # 加载优化器
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # 加载训练结果
            if ckpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(ckpt['training_results'])  # 写入结果

            # 恢复训练的epoch
            start_epoch = ckpt['epoch'] + 1
            if epochs < start_epoch:
                print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                      (weights, ckpt['epoch'], epochs))
                epochs += ckpt['epoch']  # 微调额外的epochs

            del ckpt
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Training from scratch.")
            weights = ''  # 加载失败，从头开始训练

    # 混合精度训练
    if mixed_precision:
        try:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
        except:
            print("Apex not installed, running without mixed precision.")
            mixed_precision = False

    # 学习率调度器 - 余弦退火
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # 余弦退火
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 数据并行模式
    if device.type != 'cpu' and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # 同步批归一化
    if opt.sync_bn and device.type != 'cpu' and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print('Using SyncBatchNorm()')

    # 指数移动平均
    ema = torch_utils.ModelEMA(model) if rank in [-1, 0] else None

    # DDP模式
    if device.type != 'cpu' and rank != -1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # 创建训练数据加载器
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt, hyp=hyp, augment=True,
                                            cache=opt.cache_images, rect=opt.rect, local_rank=rank,
                                            world_size=opt.world_size)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # 最大标签类别
    nb = len(dataloader)  # 批次数
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # 创建测试数据加载器
    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt, hyp=hyp, augment=False,
                                       cache=opt.cache_images, rect=True, local_rank=-1, world_size=opt.world_size)[0]

    # 设置模型参数
    hyp['cls'] *= nc / 80.  # 缩放COCO调优的分类损失权重
    model.nc = nc  # 附加类别数到模型
    model.hyp = hyp  # 附加超参数到模型
    model.gr = 1.0  # giou损失比率
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # 附加类别权重
    model.names = names

    # 类别频率分析
    if rank in [-1, 0]:
        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # 类别
        plot_labels(labels, save_dir=log_dir)
        if tb_writer:
            tb_writer.add_histogram('classes', c, 0)

        # 检查锚点
        if not opt.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # 开始训练
    t0 = time.time()
    nw = max(3 * nb, 1e3)  # 热身迭代次数，max(3 epochs, 1k iterations)
    maps = np.zeros(nc)  # 每类mAP
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # 不移动
    if rank in [0, -1]:
        print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
        print('Using %g dataloader workers' % dataloader.num_workers)
        print('Starting training for %g epochs...' % epochs)

    # 训练主循环
    for epoch in range(start_epoch, epochs):  # epoch 循环
        model.train()

        # 更新图像权重（可选）
        if dataset.image_weights:
            if rank in [-1, 0]:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # 类别权重
                image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # 随机加权索引
            # 广播索引
            if rank != -1:
                indices = torch.zeros([dataset.n], dtype=torch.int)
                if rank == 0:
                    indices[:] = torch.from_tensor(dataset.indices, dtype=torch.int)
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        mloss = torch.zeros(4, device=device)  # 平均损失
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(pbar, total=nb)  # 进度条
        optimizer.zero_grad()

        # 批次循环
        for i, (imgs, targets, paths, _) in pbar:  # batch 循环
            ni = i + nb * epoch  # 累计批次（从训练开始）
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # 热身训练
            if ni <= nw:
                xi = [0, nw]  # x插值点
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # 偏置学习率从0.1降到lr0，其他学习率从0.0升到lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # 多尺度训练
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # 随机尺寸
                sf = sz / max(imgs.shape[2:])  # 缩放因子
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # 新形状
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # 前向传播
            pred = model(imgs)

            # 计算损失
            loss, loss_items = compute_loss(pred, targets.to(device), model)  # 按批次大小缩放
            if rank != -1:
                loss *= opt.world_size  # 在DDP模式下平均梯度
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # 反向传播
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # 优化
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            # 打印训练信息
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # 更新平均损失
                mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # GPU内存
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # 绘制训练批次图像
                if ni < 3:
                    f = str(Path(log_dir) / ('train_batch%g.jpg' % ni))  # 文件名
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    if tb_writer and result is not None:
                        tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)

        # 更新学习率
        scheduler.step()

        # 验证和保存模型（仅主进程）
        if rank in [-1, 0]:
            # 更新EMA模型属性
            if ema is not None:
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])

            # 计算mAP
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # 计算mAP
                results, maps, times = test.test(opt.data,
                                                 batch_size=total_batch_size,
                                                 imgsz=imgsz_test,
                                                 save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                                 model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=log_dir)

                # 写入结果文件
                with open(results_file, 'a') as f:
                    f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

                # 更新Tensorboard
                if tb_writer:
                    tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                            'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                    for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                        tb_writer.add_scalar(tag, x, epoch)

                # 更新最佳mAP
                fi = fitness(np.array(results).reshape(1, -1))  # 适应度 = [P, R, mAP, F1]的加权组合
                if fi > best_fitness:
                    best_fitness = fi

            # 保存模型
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, 'r') as f:  # 创建检查点
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema.module if hasattr(ema, 'module') else ema.ema,
                            'optimizer': None if final_epoch else optimizer.state_dict()}

                # 保存last.pt和best.pt
                torch.save(ckpt, last)
                if (best_fitness == fi) and not final_epoch:
                    torch.save(ckpt, best)
                del ckpt

    # 训练结束
    if rank in [-1, 0]:
        # 优化模型（去除优化器信息）
        n = ('_' if len(opt.name) and not opt.name.isnumeric() else '') + opt.name
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # 重命名
                ispt = f2.endswith('.pt')  # 是否为.pt文件
                strip_optimizer(f2) if ispt else None  # 去除优化器信息

        # 完成训练
        if not opt.evolve:
            plot_results(save_dir=log_dir)  # 保存结果图
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    # 清理资源
    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/custom_yolov5.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/custom_data.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyp.yaml path (optional)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help="Total batch size for all gpus.")
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const='get_last', default=False,
                        help='resume from given path/to/last.pt, or most recent run if blank.')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov5m.pt', help='initial weights path')  # 修正Windows路径
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_args()

    # 修正权重路径中的反斜杠（Windows兼容性）
    if opt.weights and '\\' in opt.weights:
        opt.weights = opt.weights.replace('\\', '/')

    # 恢复训练
    last = get_latest_run() if opt.resume == 'get_last' else opt.resume  # 从最近的运行恢复
    if last and not opt.weights:
        print(f'Resuming training from {last}')
    opt.weights = last if opt.resume and not opt.weights else opt.weights

    # 检查文件和目录
    if opt.local_rank in [-1, 0]:
        check_git_status()
    opt.cfg = check_file(opt.cfg)  # 检查模型配置文件
    opt.data = check_file(opt.data)  # 检查数据配置文件

    # 更新超参数
    if opt.hyp:
        opt.hyp = check_file(opt.hyp)  # 检查超参数文件
        with open(opt.hyp) as f:
            hyp.update(yaml.load(f, Loader=yaml.FullLoader))  # 更新超参数

    # 扩展图像尺寸参数
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # 扩展为2个尺寸（训练和测试）

    # 选择设备
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    opt.total_batch_size = opt.batch_size
    opt.world_size = 1

    # 设置混合精度和分布式训练
    if device.type == 'cpu':
        mixed_precision = False
    elif opt.local_rank != -1:
        # DDP模式
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # 分布式后端

        opt.world_size = dist.get_world_size()
        assert opt.batch_size % opt.world_size == 0, "Batch size is not a multiple of the number of devices given!"
        opt.batch_size = opt.total_batch_size // opt.world_size
    print(opt)

    # 训练模型
    if not opt.evolve:
        if opt.local_rank in [-1, 0]:
            print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
            # 创建唯一的日志目录，避免冲突
            log_dir = increment_dir('runs/exp', opt.name)
            tb_writer = SummaryWriter(log_dir=log_dir)
        else:
            tb_writer = None
        train(hyp, tb_writer, opt, device)

    # 超参数进化（可选）
    else:
        assert opt.local_rank == -1, "DDP mode currently not implemented for Evolve!"

        tb_writer = None
        opt.notest, opt.nosave = True, True  # 仅测试/保存最终epoch
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # 下载evolve.txt（如果存在）

        # 超参数进化循环
        for _ in range(10):  # 进化10代
            if os.path.exists('evolve.txt'):  # 如果evolve.txt存在：选择最佳超参数并变异
                # 选择父代
                parent = 'single'  # 父代选择方法：'single'或'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # 考虑的先前结果数量
                x = x[np.argsort(-fitness(x))][:n]  # 前n个最佳变异
                w = fitness(x) - fitness(x).min()  # 权重
                if parent == 'single' or len(x) == 1:
                    x = x[random.choices(range(n), weights=w)[0]]  # 加权选择
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # 加权组合

                # 变异
                mp, s = 0.9, 0.2  # 变异概率，标准差
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # 增益
                ng = len(g)
                v = np.ones(ng)
                while all(v == 1):  # 变异直到发生变化（防止重复）
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = x[i + 7] * v[i]  # 变异

            # 裁剪超参数到限制范围
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # 训练变异后的模型
            results = train(hyp.copy(), tb_writer, opt, device)

            # 写入变异结果
            print_mutation(hyp, results, opt.bucket)