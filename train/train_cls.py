from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import OrderedDict
import argparse
import pandas as pd
import torch
from loss_cls import CombinedLoss
import torch.optim as optim
import yaml
from archs.unimedxcls import ImprovedClassifier
from torch.optim import lr_scheduler
from tqdm import tqdm
from chest_datasets import get_chest_dataloader
from colon_datasets import get_colon_dataloader
from cls_datasets import get_dataloader
from endo_datasets import get_train_dataloader
from utils import AverageMeter, str2bool
from tensorboardX import SummaryWriter
import os
import torch.nn as nn


def calculate_metrics(outputs, targets, task_type='binary'):
    """
    计算分类指标，支持二分类和多标签分类

    Args:
        outputs: 模型输出
            - 二分类: shape为(batch_size, 2)的logits输出
            - 多标签: shape为(batch_size, num_classes)的sigmoid输出
        targets: 真实标签
            - 二分类: shape为(batch_size,)的0/1标签
            - 多标签: shape为(batch_size, num_classes)的0/1标签
        task_type: 任务类型，'binary'表示二分类，'multilabel'表示多标签分类

    Returns:
        accuracy, precision, recall, f1
    """
    if task_type == 'binary':
        # 二分类任务
        # 将logits转换为预测类别
        if outputs.dim() == 2:
            if outputs.shape[1] == 2:  # (batch_size, 2)的logits输出
                _, predictions = torch.max(outputs, dim=1)
            else:  # (batch_size, 1)的sigmoid输出
                predictions = (outputs.squeeze() >= 0.5).float()
        else:  # (batch_size,)的单一输出
            predictions = (outputs >= 0.5).float()

        # 转换为numpy数组
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        # 计算二分类指标
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='binary', zero_division=0)
        recall = recall_score(targets, predictions, average='binary', zero_division=0)
        f1 = f1_score(targets, predictions, average='binary', zero_division=0)

    elif task_type == 'multilabel':
        # 多标签分类任务
        # 将输出转换为二值预测（0/1）
        predictions = (outputs >= 0.5).float()

        # 转换为numpy数组
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        # 计算多标签分类指标
        accuracy = accuracy_score(targets.ravel(), predictions.ravel())
        precision = precision_score(targets, predictions, average='macro', zero_division=0)
        recall = recall_score(targets, predictions, average='macro', zero_division=0)
        f1 = f1_score(targets, predictions, average='macro', zero_division=0)

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return accuracy, precision, recall, f1


def train(train_loader, model, criterion, optimizer,task_type='binary'):
    avg_meters = {
        'loss': AverageMeter(),
        'acc': AverageMeter(),
        'precision': AverageMeter(),
        'recall': AverageMeter(),
        'f1': AverageMeter()
    }

    model.train()
    pbar = tqdm(total=len(train_loader))

    for images, labels, _ in train_loader:  # _ 是 img_id，在训练时不需要使用
        images = images.float().cuda()  # 确保图像是float类型
        labels = labels.long().cuda()  # 确保标签是long类型

        # 计算输出
        outputs = model(images)
        loss = criterion(outputs, labels)


        _, predictions = torch.max(outputs, dim=1)
        # 计算指标
        acc, prec, rec, f1 = calculate_metrics(predictions, labels, task_type=task_type)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), images.size(0))
        avg_meters['acc'].update(acc, images.size(0))
        avg_meters['precision'].update(prec, images.size(0))
        avg_meters['recall'].update(rec, images.size(0))
        avg_meters['f1'].update(f1, images.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('acc', avg_meters['acc'].avg),
            ('prec', avg_meters['precision'].avg),
            ('rec', avg_meters['recall'].avg),
            ('f1', avg_meters['f1'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('acc', avg_meters['acc'].avg),
        ('precision', avg_meters['precision'].avg),
        ('recall', avg_meters['recall'].avg),
        ('f1', avg_meters['f1'].avg)
    ])



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=2400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=20, type=int,


                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--dataseed', default=42, type=int,
                        help='')

    # dataset
    parser.add_argument('--dataset', default='chest_xray', help='dataset name')
    parser.add_argument('--data_dir', default='/root/autodl-tmp/data/classification', help='dataset dir')
    parser.add_argument('--output_dir', default='/root/autodl-tmp/result', help='output dir')

    # model
    parser.add_argument('--model_name', default='cls',choices=['seg', 'cls'],)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')

    # loss
    parser.add_argument('--loss', default='')

    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])

    parser.add_argument('--lr', '--learning_rate', default=5e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,



                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=300, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config


def save_checkpoint(state, save_path):
    """保存checkpoint"""
    torch.save(state, save_path)
    print(f"Checkpoint saved: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = vars(parse_args())

    # 基础设置
    config['num_classes'] = config['num_classes']
    config['name'] = f"{config['dataset']}_{config['model_name']}"
    model_dir = os.path.join('/root/autodl-tmp/result/models', config['name'])
    os.makedirs(model_dir, exist_ok=True)

    # 设置tensorboard
    writer = SummaryWriter(os.path.join(model_dir, 'tf_logs'))

    # 保存配置和打印信息
    print('-' * 20)
    for key, value in config.items():
        print(f'{key}: {value}')
    print('-' * 20)

    with open(os.path.join(model_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    # 创建模型
    print(f"=> creating model {config['name']}")
    model = ImprovedClassifier(
        num_classes=config['num_classes'],
    ).to(device)

    # 设置损失函数、优化器和学习率调度器
    # criterion = nn.BCEWithLogitsLoss().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # 在训练代码中替换原有的BCEWithLogitsLoss:
    # criterion = CombinedLoss(
    #     auto_weight=True,  # 设置是否使用自动权重
    #     focal_weight=3.0,  # 手动设置focal loss权重
    #     dice_weight=1.0,  # 手动设置dice loss权重
    #     smooth_weight=0.5  # 手动设置label smoothing权重
    # ).cuda()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    ) if config['optimizer'] == 'Adam' else optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        momentum=config['momentum'],
        nesterov=config['nesterov'],
        weight_decay=config['weight_decay']
    )

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['min_lr']
    ) if config['scheduler'] == 'CosineAnnealingLR' else None

    # 加载数据
    # train_loader = get_colon_dataloader(
    #     csv_path=os.path.join(config['data_dir'], 'colon_train.csv'),
    #     img_dir=os.path.join(config['data_dir'], 'images'),
    #     batch_size=config['batch_size'],
    #     num_workers=config['num_workers']
    # )

    train_loader = get_dataloader(
        dataset_name=config['dataset'],
        root_dir=config['data_dir'],
        split='train',
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    # 初始化日志
    log = {
        'epoch': [],
        'loss': [],
        'acc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    # 加载checkpoint
    start_epoch = 0
    best_f1 = 0
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pth')

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint['best_f1']
        log = checkpoint['log']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")

    # 训练循环
    print("Starting training...")
    for epoch in range(start_epoch, config['epochs']):
        print(f'Epoch [{epoch + 1}/{config["epochs"]}]')

        # 训练
        train_log = train(train_loader, model, criterion, optimizer,task_type='binary')

        # 学习率调整
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('learning_rate', current_lr, epoch)

        # 更新和保存日志
        log['epoch'].append(epoch)
        for key, value in train_log.items():
            log[key].append(value)
            writer.add_scalar(f'train/{key}', value, epoch)

        # 打印当前epoch的训练结果
        print(
            'Loss: {:.4f} - Acc: {:.4f} - Precision: {:.4f} - Recall: {:.4f} - F1: {:.4f}'.format(
                train_log['loss'], train_log['acc'], train_log['precision'],
                train_log['recall'], train_log['f1']
            )
        )

        # 保存日志到CSV
        pd.DataFrame(log).to_csv(os.path.join(model_dir, 'log.csv'), index=False)

        # 定期保存checkpoint
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'best_f1': best_f1,
                'log': log
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")

        # 如果当前epoch的F1分数更好，保存最佳模型
        if train_log['f1'] > best_f1:
            best_f1 = train_log['f1']
            torch.save(
                model.state_dict(),
                os.path.join(model_dir, 'best_model.pth')
            )
            print(f"New best model saved with F1: {best_f1:.4f}")

        torch.cuda.empty_cache()

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pth'))
    print("Training completed. Final model saved.")

    writer.close()


if __name__ == '__main__':
    main()