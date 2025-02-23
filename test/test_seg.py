import argparse
import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import matplotlib.pyplot as plt
from archs.unimedxseg import UniMedXSeg
from train.datasets import get_test_dataloaders
from train.metrics import iou_score
from train.utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default='/root/autodl-tmp/best_iou_model.pth',help='trained model path')
    parser.add_argument('--model_name', default='UniMedXSeg', help='model architecture')
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--data_dir', default='/root/autodl-fs/Kvasir-SEG-Split', help='dataset directory')
    parser.add_argument('--output_dir', default='/root/autodl-tmp/test-final', help='output directory')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--deep_supervision', default=False, type=bool)

    return parser.parse_args()


def save_visualization(image, mask, pred, save_path, index):
    """保存原图、GT和预测结果的对比图"""
    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(3, 1, 1)
    image = image * std + mean

    # 转换张量到numpy数组并调整通道顺序
    image = image.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC

    # 将图像从[0,1]转换到[0,255]并确保在有效范围内
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    mask = mask.cpu().numpy()
    pred = pred.cpu().numpy()

    # 创建subplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 显示原图
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 显示Ground Truth
    # 显示Ground Truth - 使用squeeze()去掉多余的维度
    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # 显示预测结果 - 使用squeeze()去掉多余的维度
    axes[2].imshow(pred.squeeze(), cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{index}.png'),
                bbox_inches='tight',
                dpi=300)
    plt.close()


def test(test_loader, model, output_dir):
    avg_meters = {
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'SE': AverageMeter(),
        'PC': AverageMeter(),
        'F1': AverageMeter(),
        'SP': AverageMeter(),
        'ACC': AverageMeter()
    }

    # 创建可视化结果保存目录
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    model.eval()

    # 用于存储每个样本的指标和对应的数据
    sample_metrics = []
    sample_data = []

    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for i, batch in enumerate(test_loader):
            input = batch['image'].cuda()
            target = batch['mask'].cuda()
            image_paths = batch['image_path']  # 这里获取的是完整路径

            if args.deep_supervision:
                outputs = model(input)
                output = outputs[-1]
            else:
                output = model(input)

            iou, dice, SE, PC, F1, SP, ACC = iou_score(output, target)

            # 获取预测掩码
            pred_masks = (output > 0.5).float()

            # 记录每个样本的指标和数据
            for b in range(input.size(0)):
                image_name = os.path.basename(image_paths[b])  # 先获取文件名（含扩展名）
                image_name = os.path.splitext(image_name)[0]  # 移除扩展名
                sample_metrics.append({
                    'image_name': image_name,
                    'iou': iou,
                    'dice': dice,
                    'SE': SE,
                    'PC': PC,
                    'F1': F1,
                    'SP': SP,
                    'ACC': ACC
                })

                sample_data.append({
                    'image_name': image_name,  # 存储不含扩展名的文件名
                    'image': input[b],
                    'mask': target[b],
                    'pred': pred_masks[b, 0],
                    'metrics': sample_metrics[-1]
                })

            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['SE'].update(SE, input.size(0))
            avg_meters['PC'].update(PC, input.size(0))
            avg_meters['F1'].update(F1, input.size(0))
            avg_meters['SP'].update(SP, input.size(0))
            avg_meters['ACC'].update(ACC, input.size(0))

            postfix = OrderedDict([
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('SE', avg_meters['SE'].avg),
                ('PC', avg_meters['PC'].avg),
                ('F1', avg_meters['F1'].avg),
                ('SP', avg_meters['SP'].avg),
                ('ACC', avg_meters['ACC'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()


    # 可视化
    print("\nSaving all segmentation results...")
    for data in sample_data:
        # 直接使用已经处理好的image_name
        save_visualization(
            data['image'],
            data['mask'],
            data['pred'],
            vis_dir,
            data['image_name']  # 已经是干净的文件名了，不需要再处理
        )

    # 保存详细的指标数据
    sample_df = pd.DataFrame(sample_metrics)
    sample_df.to_csv(os.path.join(output_dir, 'sample_metrics.csv'), index=False)

    # 计算并保存统计指标
    stats = {
        'metric': ['IoU', 'Dice', 'Sensitivity', 'Precision', 'F1', 'Specificity', 'Accuracy'],
        'mean': [
            avg_meters['iou'].avg,
            avg_meters['dice'].avg,
            avg_meters['SE'].avg,
            avg_meters['PC'].avg,
            avg_meters['F1'].avg,
            avg_meters['SP'].avg,
            avg_meters['ACC'].avg
        ],
        'std': [
            np.std(sample_df['iou']),
            np.std(sample_df['dice']),
            np.std(sample_df['SE']),
            np.std(sample_df['PC']),
            np.std(sample_df['F1']),
            np.std(sample_df['SP']),
            np.std(sample_df['ACC'])
        ]
    }

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)

    return OrderedDict([
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg),
        ('SE', avg_meters['SE'].avg),
        ('PC', avg_meters['PC'].avg),
        ('F1', avg_meters['F1'].avg),
        ('SP', avg_meters['SP'].avg),
        ('ACC', avg_meters['ACC'].avg)
    ])


def main():
    global args
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # 创建模型
    model = UniMedXSeg(
        n_channels=args.input_channels,
        n_classes=args.num_classes,
        device=device
    ).to(device)

    # 加载预训练权重
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()

    # 获取测试数据加载器
    test_loader = get_test_dataloaders(args.data_dir, batch_size=args.batch_size, img_size=(256, 256))

    # 测试模型并获取结果
    test_log = test(test_loader, model, args.output_dir)

    # 打印最终结果
    print('\nTest Results:')
    print('IoU: %.4f' % test_log['iou'])
    print('Dice: %.4f' % test_log['dice'])
    print('Sensitivity: %.4f' % test_log['SE'])
    print('Precision: %.4f' % test_log['PC'])
    print('F1: %.4f' % test_log['F1'])
    print('Specificity: %.4f' % test_log['SP'])
    print('Accuracy: %.4f' % test_log['ACC'])

    print(f'\nDetailed results have been saved to {args.output_dir}')
    print(f'Visualizations have been saved to {os.path.join(args.output_dir, "visualizations")}')


if __name__ == '__main__':
    main()

