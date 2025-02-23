import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from archs.unimedxcls import ImprovedClassifier
from train.chest_datasets import get_test_chest_dataloader
from train.colon_datasets import get_test_colon_dataloader
from train.endo_datasets import get_test_dataloader
from train.cls_datasets import get_dataloader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import label_binarize


def plot_roc_pr_curves(y_true, y_score, label_names, output_dir):
    """绘制ROC和PR曲线

    Args:
        y_true: 真实标签 shape(n_samples,)
        y_score: 预测概率 shape(n_samples, n_classes)或(n_samples,)用于二分类
        label_names: 标签名称列表
        output_dir: 输出目录
    """
    # 使用默认样式
    plt.style.use('default')

    # 设置全局字体大小和样式
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })

    # 确定是二分类还是多分类
    is_binary = len(y_score.shape) == 1 or y_score.shape[1] == 1

    if is_binary:
        # 二分类情况
        if len(y_score.shape) == 2:
            y_score = y_score.ravel()

        # 绘制ROC曲线
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, 'b-', lw=2, label=f'ROC curve')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)

        # 绘制PR曲线
        plt.subplot(122)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, 'r-', lw=2, label=f'PR curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall (PR) Curve')
        plt.legend(loc="lower left")
        plt.grid(True, linestyle='--', alpha=0.7)

    else:
        # 多分类情况
        n_classes = y_score.shape[1]

        # 将真实标签转换为one-hot编码
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # 绘制ROC曲线
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{label_names[i]}')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)

        # 绘制PR曲线
        plt.subplot(122)
        for i, color in zip(range(n_classes), colors):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
            plt.plot(recall, precision, color=color, lw=2, label=f'{label_names[i]}')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, label_names, output_dir):
    """绘制混淆矩阵热力图

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        label_names: 标签名称列表
        output_dir: 输出目录
    """
    # 处理二分类情况下的标签
    if len(label_names) == 1:
        # 如果只提供了一个标签(tumor)，自动添加另一个标签(non-tumor)
        label_names = ['non-tumor', 'tumor']
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 使用更好的颜色映射
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)

    # 添加标题和标签
    plt.title('Confusion Matrix', pad=20, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # 设置刻度标签
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45, ha='right')
    plt.yticks(tick_marks, label_names)

    # 调整字体大小
    plt.tick_params(labelsize=10)

    # 在每个单元格中添加数值和百分比
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        # 添加数值
        text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
        plt.text(j, i, text,
                 horizontalalignment='center',
                 verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=10)

    # 添加网格线使边界更清晰
    plt.grid(False)

    # 确保图形边界完整显示
    plt.tight_layout()

    # 保存图形
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default='/root/autodl-tmp/result/models/chest_xray_cls/best_model.pth',
                        help='trained model path')
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--dataset', default='chest_xray', help='dataset name')
    parser.add_argument('--data_dir', default='/root/autodl-tmp/data/classification',
                        help='dataset directory')
    parser.add_argument('--csv_path', default='/root/autodl-tmp/data/classification/MedFMC_val/colon/colon_val_updated.csv',
                        help='test csv file path')
    parser.add_argument('--output_dir', default='/root/autodl-tmp/cls-results',
                        help='output directory')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='batch size')

    return parser.parse_args()


def calculate_metrics(y_true, y_score):
    """计算各种评估指标，支持二分类和多分类任务
    Args:
        y_true: 真实标签，shape为(n_samples,)
        y_score: 预测分数
            - 二分类: shape为(n_samples, 2)或(n_samples, 1)
            - 多分类: shape为(n_samples, n_classes)
    Returns:
        metrics: 包含各项评估指标的字典
    """
    try:
        # 打印输入形状，用于调试
        print(f"Input shapes - y_true: {y_true.shape}, y_score: {y_score.shape}")

        # 确保y_true是一维数组
        y_true = y_true.reshape(-1)

        # 处理不同形状的输出
        if len(y_score.shape) == 2:
            if y_score.shape[1] == 2:  # (n_samples, 2) 的标准二分类输出
                print("Processing binary classification with two columns")
                y_score_proba = y_score[:, 1]  # 取第二列作为正类概率
                y_pred = np.argmax(y_score, axis=1)
                n_classes = 2
            elif y_score.shape[1] == 1:  # (n_samples, 1) 的单列输出
                print("Processing binary classification with single column")
                y_score_proba = y_score.reshape(-1)
                y_pred = (y_score_proba >= 0.5).astype(int)
                n_classes = 2
            else:  # 多分类情况
                print(f"Processing multi-class classification with {y_score.shape[1]} classes")
                y_score_proba = y_score
                y_pred = np.argmax(y_score, axis=1)
                n_classes = y_score.shape[1]
        else:  # 处理 (n_samples,) 的输出
            print("Processing binary classification with flattened output")
            y_score_proba = y_score
            y_pred = (y_score >= 0.5).astype(int)
            n_classes = 2

        metrics = {}

        # 计算每个类别的指标
        for i in range(n_classes):
            true_binary = (y_true == i)
            pred_binary = (y_pred == i)

            if n_classes == 2:
                # 二分类情况下的预测概率
                pred_proba = y_score_proba if i == 1 else 1 - y_score_proba
            else:
                # 多分类情况下的预测概率
                pred_proba = y_score[:, i]

            metrics[f'class_{i}'] = {
                'accuracy': accuracy_score(true_binary, pred_binary),
                'precision': precision_score(true_binary, pred_binary, zero_division=0),
                'recall': recall_score(true_binary, pred_binary, zero_division=0),
                'f1': f1_score(true_binary, pred_binary, zero_division=0),
                'auc': roc_auc_score(true_binary, pred_proba)
            }

        # 计算宏平均指标
        metrics['macro_avg'] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': np.mean([metrics[f'class_{i}']['precision'] for i in range(n_classes)]),
            'recall': np.mean([metrics[f'class_{i}']['recall'] for i in range(n_classes)]),
            'f1': np.mean([metrics[f'class_{i}']['f1'] for i in range(n_classes)]),
            'auc': np.mean([metrics[f'class_{i}']['auc'] for i in range(n_classes)])
        }

        return metrics

    except Exception as e:
        print(f"Error in calculate_metrics: {str(e)}")
        print(f"y_true shape: {y_true.shape}")
        print(f"y_score shape: {y_score.shape}")
        raise e

def test(test_loader, model, output_dir, label_names):
    """测试模型并保存结果
    Args:
        test_loader: 数据加载器
        model: 模型
        output_dir: 输出目录
        label_names: 标签名称列表
    """
    if 'study_id' in label_names:
        label_names = [label for label in label_names if label != 'study_id']

    print(f"Using label names: {label_names}")  # 调试信息

    model.eval()

    # 存储所有预测和真实标签
    all_outputs = []
    all_labels = []
    all_img_ids = []
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for i, (images, labels, img_ids) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()

            # 前向传播
            outputs = model(images)  # 模型已经在forward中应用了sigmoid

            # 存储结果
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_img_ids.extend(img_ids)

            pbar.update(1)
        pbar.close()

        # 合并所有批次的结果
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # 计算评估指标
        metrics = calculate_metrics(all_labels, all_outputs)

        # 创建结果字典
        results_dict = {
            'img_id': all_img_ids,
            'true_label': all_labels
        }

        # 根据输出形状添加预测概率
        if all_outputs.shape[1] == 2:  # 标准二分类输出
            results_dict.update({
                'pred_prob_0': all_outputs[:, 0],
                'pred_prob_1': all_outputs[:, 1],
                'predicted_label': np.argmax(all_outputs, axis=1)
            })
        else:  # 单列输出或多分类输出
            for i in range(all_outputs.shape[1]):
                results_dict[f'pred_prob_{i}'] = all_outputs[:, i]
            results_dict['predicted_label'] = np.argmax(all_outputs, axis=1) if all_outputs.shape[1] > 1 else (
                        all_outputs >= 0.5).astype(int)

        # 保存详细结果
        results_df = pd.DataFrame(results_dict)
        results_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)


        # 创建新格式的汇总DataFrame
        summary_rows = []

        # 添加宏平均行
        summary_rows.append({
            'Class': 'Macro Average',
            'Accuracy': metrics['macro_avg']['accuracy'],
            'Precision': metrics['macro_avg']['precision'],
            'Recall': metrics['macro_avg']['recall'],
            'F1-Score': metrics['macro_avg']['f1'],
            'AUC': metrics['macro_avg']['auc']
        })

        # 动态创建类别顺序字典
        class_order = {name: idx for idx, name in enumerate(label_names)}

        # 添加每个类别的行
        for label in label_names:
            i = class_order[label]
            true_binary = (all_labels == i)
            pred_binary = (results_dict['predicted_label'] == i)
            pred_proba = all_outputs[:, i] if all_outputs.shape[1] > 1 else (
                all_outputs[:, 1] if i == 1 else 1 - all_outputs[:, 1])

            summary_rows.append({
                'Class': label,
                'Accuracy': accuracy_score(true_binary, pred_binary),
                'Precision': precision_score(true_binary, pred_binary, zero_division=0),
                'Recall': recall_score(true_binary, pred_binary, zero_division=0),
                'F1-Score': f1_score(true_binary, pred_binary, zero_division=0),
                'AUC': roc_auc_score(true_binary, pred_proba)
            })

        # 创建并保存汇总DataFrame
        summary_df = pd.DataFrame(summary_rows)

        # 设置列的顺序
        columns_order = ['Class', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        summary_df = summary_df[columns_order]

        # 保存汇总指标，确保6位小数
        summary_df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'),
                          float_format='%.6f',
                          index=False)


        # 绘制ROC和PR曲线
        if all_outputs.shape[1] == 2:  # 二分类情况
            plot_roc_pr_curves(all_labels, all_outputs[:, 1], label_names, output_dir)
        else:
            plot_roc_pr_curves(all_labels, all_outputs, label_names, output_dir)

        # 绘制混淆矩阵
        plot_confusion_matrix(all_labels, results_dict['predicted_label'], label_names, output_dir)

        return metrics


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # 创建模型
    model = ImprovedClassifier(
        num_classes=args.num_classes,
    ).to(device)

    # 加载预训练权重
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()

    # 获取测试数据加载器
    # test_loader = get_test_chest_dataloader(
    #     args.csv_path,
    #     args.data_dir,
    #     batch_size=args.batch_size
    # )

    # test_loader = get_test_colon_dataloader(
    #         args.csv_path,
    #         args.data_dir,
    #         batch_size=args.batch_size
    # )

    # test_loader = get_test_dataloader(
    #     args.csv_path,
    #     args.data_dir,
    #     batch_size=args.batch_size
    # )
    test_loader = get_dataloader(args.dataset, args.data_dir, 'test', batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 获取标签名称
    label_names = test_loader.dataset.get_label_names()

    # 测试模型并获取结果
    metrics = test(test_loader, model, args.output_dir, label_names)

    # 打印最终结果
    print('\nTest Results:')
    print(f'Average Accuracy: {metrics["macro_avg"]["accuracy"]:.4f}')
    print(f'Average Precision: {metrics["macro_avg"]["precision"]:.4f}')
    print(f'Average Recall: {metrics["macro_avg"]["recall"]:.4f}')
    print(f'Average F1-Score: {metrics["macro_avg"]["f1"]:.4f}')
    print(f'Average AUC: {metrics["macro_avg"]["auc"]:.4f}')

    print(f'\nDetailed results have been saved to {args.output_dir}')


if __name__ == '__main__':
    main()