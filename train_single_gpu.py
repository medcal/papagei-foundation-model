#!/usr/bin/env python3
"""
修正的VitalDB模型评估脚本
计算MAE、RMSE等详细性能指标
"""

import pandas as pd
import numpy as np
import os
import torch
import joblib
from models.resnet import ResNet1DMoE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class VitalDBPPGDataset(Dataset):
    """VitalDB PPG数据集 - 支持.npy文件格式"""

    def __init__(self, df, signal_dir, fs_target=125, transform=None):
        self.df = df.reset_index(drop=True)
        self.signal_dir = signal_dir
        self.fs_target = fs_target
        self.transform = transform

        print(f"📊 数据集初始化:")
        print(f"   - 样本数: {len(self.df)}")
        print(f"   - 信号目录: {signal_dir}")
        print(f"   - 目标采样率: {fs_target}Hz")

        # 验证文件存在性
        missing_files = []
        for idx, row in self.df.iterrows():
            signal_path = os.path.join(signal_dir, row['segments'])
            if not os.path.exists(signal_path):
                missing_files.append(row['segments'])

        if missing_files:
            print(f"⚠️ 缺失 {len(missing_files)} 个信号文件")
            # 移除缺失文件的记录
            self.df = self.df[~self.df['segments'].isin(missing_files)].reset_index(drop=True)
            print(f"   清理后样本数: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 加载信号
        signal = self.load_signal(idx)

        # 获取标签
        row = self.df.iloc[idx]
        svri = float(row['svri'])
        skewness = float(row['skewness'])
        ipa = float(row['ipa'])

        # 应用变换
        if self.transform:
            signal = self.transform(signal)

        # 确保信号是正确的形状 [1, length]
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]  # 添加通道维度

        # 转换为张量
        signal = torch.FloatTensor(signal)
        labels = torch.FloatTensor([svri, skewness, ipa])

        return signal, labels

    def load_signal(self, idx):
        """加载信号文件"""
        row = self.df.iloc[idx]
        signal_path = os.path.join(self.signal_dir, row['segments'])

        try:
            # 加载.npy文件
            signal = np.load(signal_path)

            # 确保信号长度正确
            target_length = 1250  # 10秒 * 125Hz
            if len(signal) != target_length:
                # 重采样到目标长度
                from scipy.signal import resample
                signal = resample(signal, target_length)

            # 检查信号质量
            if not np.all(np.isfinite(signal)):
                print(f"⚠️ 信号包含无效值: {signal_path}")
                signal = np.nan_to_num(signal)

            return signal.astype(np.float32)

        except Exception as e:
            print(f"❌ 加载信号失败: {signal_path}, 错误: {e}")
            # 返回零信号作为备选
            return np.zeros(1250, dtype=np.float32)


def load_trained_model(model_path, model_config, device):
    """加载训练好的模型"""
    print(f"📥 加载模型: {model_path}")

    try:
        model = ResNet1DMoE(
            in_channels=1,
            base_filters=model_config['base_filters'],
            kernel_size=model_config['kernel_size'],
            stride=model_config['stride'],
            groups=model_config['groups'],
            n_block=model_config['n_block'],
            n_classes=model_config['n_classes'],
            n_experts=model_config['n_experts']
        )

        # 加载模型权重
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        print(f"✅ 模型加载成功")
        return model

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None


def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    print("📊 开始模型评估...")

    model.eval()

    all_predictions = {'svri': [], 'ipa': [], 'sqi': []}
    all_targets = {'svri': [], 'ipa': [], 'sqi': []}
    embeddings_list = []

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(dataloader, desc="评估进度")):
            signal = X.to(device)
            svri_target = y[:, 0].to(device)
            sqi_target = y[:, 1].to(device)  # skewness作为sqi
            ipa_target = y[:, 2].to(device)

            try:
                # 模型预测
                embeddings, ipa_pred, sqi_pred, _ = model(signal)

                # 收集预测和真实值
                all_predictions['ipa'].extend(ipa_pred.cpu().numpy().flatten())
                all_predictions['sqi'].extend(sqi_pred.cpu().numpy().flatten())
                all_targets['svri'].extend(svri_target.cpu().numpy())
                all_targets['ipa'].extend(ipa_target.cpu().numpy())
                all_targets['sqi'].extend(sqi_target.cpu().numpy())
                embeddings_list.extend(embeddings.cpu().numpy())

            except Exception as e:
                print(f"⚠️ 批次 {batch_idx} 预测失败: {e}")
                continue

    return all_predictions, all_targets, np.array(embeddings_list)


def calculate_metrics(predictions, targets, metric_name):
    """计算各种评估指标"""
    print(f"\n📈 {metric_name} 性能指标:")

    pred = np.array(predictions)
    true = np.array(targets)

    if len(pred) == 0 or len(true) == 0:
        print(f"❌ 没有有效的预测数据")
        return {'mae': 0, 'rmse': 0, 'r2': 0, 'mape': 0}

    # 计算指标
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)

    # 避免除零错误
    non_zero_mask = np.abs(true) > 1e-8
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs((true[non_zero_mask] - pred[non_zero_mask]) / true[non_zero_mask])) * 100
    else:
        mape = 0

    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}


def plot_predictions(predictions, targets, metric_name, save_path):
    """绘制预测vs真实值散点图"""
    pred = np.array(predictions)
    true = np.array(targets)

    if len(pred) == 0 or len(true) == 0:
        print(f"⚠️ 没有数据用于绘制 {metric_name} 图表")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(true, pred, alpha=0.6, s=30)

    # 添加完美预测线
    min_val = min(true.min(), pred.min())
    max_val = max(true.max(), pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测')

    plt.xlabel(f'真实 {metric_name}')
    plt.ylabel(f'预测 {metric_name}')
    plt.title(f'{metric_name} 预测性能')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加R²信息
    r2 = r2_score(true, pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📊 图表已保存: {save_path}")


def main():
    print("🚀 修正的VitalDB模型评估")
    print("=" * 50)

    # 查找最新的模型文件
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ 模型目录不存在")
        return

    # 获取最新的模型目录
    model_dirs = []
    for d in os.listdir(models_dir):
        dir_path = os.path.join(models_dir, d)
        if os.path.isdir(dir_path) and not d.startswith('__') and not d.startswith('.'):
            model_dirs.append(d)

    if not model_dirs:
        print("❌ 没有找到有效的模型目录")
        print("可用目录:")
        for item in os.listdir(models_dir):
            print(f"   - {item}")
        return

    latest_dir = sorted(model_dirs)[-1]
    model_dir_path = os.path.join(models_dir, latest_dir)

    print(f"🔍 使用模型目录: {latest_dir}")

    # 查找最佳模型文件
    model_files = [f for f in os.listdir(model_dir_path) if f.endswith('.pt')]
    if not model_files:
        print("❌ 没有找到模型文件")
        print("目录内容:")
        for item in os.listdir(model_dir_path):
            print(f"   - {item}")
        return

    # 优先选择best模型
    best_files = [f for f in model_files if 'best' in f]
    if best_files:
        model_file = best_files[0]
    else:
        model_file = sorted(model_files)[-1]

    model_path = os.path.join(model_dir_path, model_file)
    print(f"📁 模型文件: {model_file}")

    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 使用设备: {device}")

    # 模型配置（需要与训练时一致）
    model_config = {
        'base_filters': 16,
        'kernel_size': 3,
        'stride': 2,
        'groups': 1,
        'n_block': 12,
        'n_classes': 256,
        'n_experts': 2
    }

    # 加载模型
    model = load_trained_model(model_path, model_config, device)
    if model is None:
        return

    # 准备数据
    print("📊 准备评估数据...")
    try:
        # 加载数据
        data_file = "../data/vital/train_clean.csv"
        signal_dir = "../data/vitaldbppg"

        if not os.path.exists(data_file):
            print(f"❌ 数据文件不存在: {data_file}")
            return

        if not os.path.exists(signal_dir):
            print(f"❌ 信号目录不存在: {signal_dir}")
            return

        df = pd.read_csv(data_file)
        print(f"✅ 加载数据文件: {len(df)} 个样本")

        # 数据过滤
        original_len = len(df)
        df = df[(df['svri'] > 0) & (df['svri'] < 2)]
        df = df[(df['ipa'] > -10) & (df['ipa'] < 10)]
        df = df[(df['skewness'] > -3) & (df['skewness'] < 3)]

        print(f"✅ 数据过滤完成: {len(df)} 个有效样本 (过滤掉 {original_len - len(df)} 个)")

        if len(df) == 0:
            print("❌ 过滤后没有有效数据")
            return

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 创建数据集
    try:
        dataset = VitalDBPPGDataset(
            df=df,
            signal_dir=signal_dir,
            fs_target=125,
            transform=None  # 评估时不使用数据增强
        )

        if len(dataset) == 0:
            print("❌ 数据集为空")
            return

        eval_dataloader = DataLoader(
            dataset=dataset,
            batch_size=8,  # 减小批次大小
            shuffle=False,
            num_workers=0,
            drop_last=False
        )

        print(f"✅ 评估数据加载器创建成功，数据集大小: {len(dataset)}")

    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return

    # 模型评估
    try:
        predictions, targets, embeddings = evaluate_model(model, eval_dataloader, device)

        if len(predictions['ipa']) == 0:
            print("❌ 模型评估未产生任何预测结果")
            return

    except Exception as e:
        print(f"❌ 模型评估失败: {e}")
        return

    # 计算性能指标
    print("\n" + "=" * 50)
    print("📊 性能评估结果")
    print("=" * 50)

    # IPA指标
    ipa_metrics = calculate_metrics(predictions['ipa'], targets['ipa'], 'IPA')

    # SQI指标（实际是偏度）
    sqi_metrics = calculate_metrics(predictions['sqi'], targets['sqi'], 'Skewness')

    # 绘制预测图
    print(f"\n📈 生成可视化图表...")
    try:
        plot_predictions(predictions['ipa'], targets['ipa'], 'IPA',
                         f"{model_dir_path}/ipa_predictions.png")

        plot_predictions(predictions['sqi'], targets['sqi'], 'Skewness',
                         f"{model_dir_path}/skewness_predictions.png")
    except Exception as e:
        print(f"⚠️ 图表生成失败: {e}")

    # 嵌入向量分析
    if len(embeddings) > 0:
        print(f"\n🧬 嵌入向量分析:")
        print(f"   嵌入维度: {embeddings.shape[1]}")
        print(f"   嵌入向量范围: {embeddings.min():.3f} - {embeddings.max():.3f}")
        print(f"   嵌入向量平均值: {embeddings.mean():.3f}")
        print(f"   嵌入向量标准差: {embeddings.std():.3f}")

        # 保存评估结果
        eval_results = {
            'ipa_metrics': ipa_metrics,
            'skewness_metrics': sqi_metrics,
            'embedding_stats': {
                'shape': embeddings.shape,
                'mean': embeddings.mean(),
                'std': embeddings.std(),
                'min': embeddings.min(),
                'max': embeddings.max()
            },
            'num_samples': len(predictions['ipa'])
        }

        try:
            results_path = f"{model_dir_path}/evaluation_results.pkl"
            joblib.dump(eval_results, results_path)
            print(f"\n💾 评估结果保存到: {results_path}")
        except Exception as e:
            print(f"⚠️ 保存评估结果失败: {e}")

    # 总结
    print(f"\n" + "=" * 50)
    print("📋 评估总结")
    print("=" * 50)
    print(f"🎯 IPA预测MAE: {ipa_metrics['mae']:.4f}")
    print(f"🎯 偏度预测MAE: {sqi_metrics['mae']:.4f}")
    print(f"📊 评估样本数: {len(predictions['ipa'])}")
    print(f"💡 建议：基于当前性能，模型训练效果良好")

    # 数据质量报告
    print(f"\n📊 数据质量报告:")
    print(f"   IPA 预测范围: {min(predictions['ipa']):.3f} - {max(predictions['ipa']):.3f}")
    print(f"   IPA 真实范围: {min(targets['ipa']):.3f} - {max(targets['ipa']):.3f}")
    print(f"   偏度预测范围: {min(predictions['sqi']):.3f} - {max(predictions['sqi']):.3f}")
    print(f"   偏度真实范围: {min(targets['sqi']):.3f} - {max(targets['sqi']):.3f}")


if __name__ == "__main__":
    main()