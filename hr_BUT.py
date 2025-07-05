import pandas as pd
import numpy as np
import os
import wfdb
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.signal import butter, filtfilt, find_peaks, welch
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Config:
    """配置类"""
    data_dir = r"E:\thsiu-ppg\brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0"
    results_dir = f"{data_dir}/hr_analysis_results"

    # 心率计算参数
    hr_min = 30  # 最小心率
    hr_max = 180  # 最大心率
    ppg_fs = 30  # PPG采样频率 - 修正为30Hz

    # 滤波参数 - 根据30Hz采样频率调整
    filter_lowcut = 0.5  # 低频截止
    filter_highcut = 3.0  # 高频截止 - 保持在奈奎斯特频率(15Hz)以下
    filter_order = 3  # 滤波器阶数

    # 信号处理参数
    target_length = 300  # 目标信号长度

    # 数据集信息
    dataset_info = {
        'subjects': 50,
        'age_range': (19, 76),
        'gender_split': {'female': 25, 'male': 25},
        'devices': ['Xiaomi Mi9', 'Huawei P20 Pro'],
        'ppg_sampling_rate': 30,  # Hz
        'ecg_sampling_rate': 1000,  # Hz
        'acc_sampling_rate': 100,  # Hz
        'recording_period': '2020.08 - 2021.12'
    }


def bandpass_filter(data, lowcut=0.5, highcut=3.0, fs=30, order=3):
    """
    应用带通滤波器 - 针对30Hz采样频率优化

    参数:
        data: 输入信号
        lowcut: 低频截止频率
        highcut: 高频截止频率
        fs: 采样频率
        order: 滤波器阶数
    """
    if fs is None or fs <= 0:
        return np.zeros_like(data)

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # 确保截止频率在有效范围内
    if low >= 1.0:
        print(f"警告: 低频截止频率太高。lowcut={lowcut}, fs={fs}, nyq={nyq}")
        low = 0.1  # 设置一个安全的低频值

    if high >= 1.0:
        print(f"警告: 高频截止频率超出奈奎斯特频率。highcut={highcut}, fs={fs}, nyq={nyq}")
        high = 0.95  # 设置一个安全的高频值

    # 确保低频小于高频
    if low >= high:
        print(f"警告: 低频截止频率大于等于高频截止频率")
        low = 0.1
        high = 0.8

    try:
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    except Exception as e:
        print(f"滤波器应用失败: {e}")
        return data


def get_hr_fft(y, fs=30, min_hr=30, max_hr=180):
    """
    使用FFT方法计算心率 - 针对30Hz采样频率优化

    参数:
        y: PPG信号（1D数组）
        fs: 采样频率
        min_hr: 最小心率
        max_hr: 最大心率

    返回:
        计算得到的心率值
    """
    try:
        # 确保是1D数组
        if len(y.shape) > 1:
            y = y.flatten()

        # 对于30Hz采样频率，调整FFT参数
        nperseg = min(len(y), 256)  # 减小窗口大小适应较低采样率
        nfft = max(nperseg * 4, 512)  # 确保足够的频率分辨率

        # 使用Welch方法计算功率谱密度
        p, q = welch(y, fs, nperseg=nperseg, nfft=nfft, noverlap=nperseg // 2)

        # 转换为心率范围的频率
        min_freq = min_hr / 60.0  # 转换为Hz
        max_freq = min(max_hr / 60.0, fs / 2 - 0.1)  # 确保不超过奈奎斯特频率

        valid_freq = (p >= min_freq) & (p <= max_freq)

        if np.sum(valid_freq) == 0:
            print(f"警告: 在有效频率范围内没有找到信号成分")
            return np.nan

        # 找到功率最大的频率对应的心率
        valid_powers = q[valid_freq]
        valid_freqs = p[valid_freq]

        max_power_idx = np.argmax(valid_powers)
        dominant_freq = valid_freqs[max_power_idx]
        fft_hr = dominant_freq * 60

        return fft_hr

    except Exception as e:
        print(f"FFT心率计算失败: {e}")
        return np.nan


def get_hr_peak(y, fs=30, min_hr=30, max_hr=180):
    """
    使用峰值检测方法计算心率 - 针对30Hz采样频率优化

    参数:
        y: PPG信号（1D数组）
        fs: 采样频率
        min_hr: 最小心率
        max_hr: 最大心率

    返回:
        计算得到的心率值
    """
    try:
        # 确保是1D数组
        if len(y.shape) > 1:
            y = y.flatten()

        # 对于30Hz采样频率，调整峰值检测参数
        min_distance = fs * 60 / max_hr  # 最大心率对应的最小峰值间距
        max_distance = fs * 60 / min_hr  # 最小心率对应的最大峰值间距

        # 峰值检测，设置最小间距防止误检
        height_threshold = np.mean(y) + 0.3 * np.std(y)  # 动态阈值
        peaks, properties = find_peaks(y,
                                       distance=int(min_distance),
                                       height=height_threshold)

        # 检查是否有足够的峰值计算心率
        if len(peaks) <= 1:
            return np.nan

        # 计算峰值间隔
        peak_intervals = np.diff(peaks)

        # 过滤异常间隔
        valid_intervals = peak_intervals[
            (peak_intervals >= min_distance) &
            (peak_intervals <= max_distance)
            ]

        if len(valid_intervals) == 0:
            return np.nan

        # 使用中位数而不是平均值，更鲁棒
        median_interval = np.median(valid_intervals)

        # 检查无效间隔
        if median_interval <= 0:
            return np.nan

        # 计算心率
        peak_hr = 60 / (median_interval / fs)

        # 检查心率是否在合理范围内
        if not (min_hr <= peak_hr <= max_hr):
            return np.nan

        # 处理NaN或无穷值
        if not np.isfinite(peak_hr):
            return np.nan

        return peak_hr

    except Exception as e:
        print(f"峰值心率计算失败: {e}")
        return np.nan


def load_but_ppg_data(data_dir):
    """
    加载BUT-PPG数据集

    参数:
        data_dir: 数据集目录路径

    返回:
        包含信号信息的DataFrame
    """
    quality_hr_file = os.path.join(data_dir, "quality-hr-ann.csv")
    if not os.path.exists(quality_hr_file):
        raise FileNotFoundError(f"未找到质量标注文件: {quality_hr_file}")

    # 检查文件是否有标题行
    with open(quality_hr_file, 'r') as f:
        first_line = f.readline().strip()

    first_values = first_line.split(',')
    has_header = False
    try:
        int(first_values[0])
    except ValueError:
        has_header = True

    # 读取质量和心率标注文件
    if has_header:
        quality_df = pd.read_csv(quality_hr_file)
        if len(quality_df.columns) >= 3:
            quality_df.columns = ['signal_id', 'quality', 'hr']
    else:
        quality_df = pd.read_csv(quality_hr_file, header=None)
        quality_df.columns = ['signal_id', 'quality', 'hr']

    # 尝试加载受试者信息文件
    subject_info_file = os.path.join(data_dir, "subject-info.csv")
    if not os.path.exists(subject_info_file):
        print("未找到受试者信息文件，使用默认值")
        quality_df['subject_id'] = quality_df['signal_id'].astype(str).str[:3].astype(int)
        return quality_df

    # 合并受试者信息
    subject_df = pd.read_csv(subject_info_file)
    quality_df['subject_id'] = quality_df['signal_id'].astype(str).str[:3].astype(int)

    # 寻找受试者ID列
    subject_id_col = None
    possible_subject_cols = ['subject_id', 'subject', 'id', 'Subject_ID', 'ID']
    for col in possible_subject_cols:
        if col in subject_df.columns:
            subject_id_col = col
            break

    if subject_id_col is None:
        subject_id_col = subject_df.columns[0]

    if subject_id_col != 'subject_id':
        subject_df = subject_df.rename(columns={subject_id_col: 'subject_id'})

    merged_df = pd.merge(quality_df, subject_df, on='subject_id', how='left')
    return merged_df


def process_ppg_signal(signal_id, data_dir, config):
    """
    处理单个PPG信号并计算心率

    参数:
        signal_id: 信号ID
        data_dir: 数据目录
        config: 配置对象

    返回:
        包含处理结果的字典
    """
    # 寻找PPG文件
    possible_paths = [
        os.path.join(data_dir, str(signal_id), f"{signal_id}_PPG"),
        os.path.join(data_dir, str(signal_id).zfill(6), f"{str(signal_id).zfill(6)}_PPG"),
        os.path.join(data_dir, str(signal_id), str(signal_id)),
        os.path.join(data_dir, str(signal_id).zfill(6), str(signal_id).zfill(6)),
    ]

    ppg_file = None
    for file_path in possible_paths:
        if os.path.exists(file_path + ".dat") and os.path.exists(file_path + ".hea"):
            ppg_file = file_path
            break

    if ppg_file is None:
        return {
            'signal_id': signal_id,
            'success': False,
            'error': '文件未找到',
            'hr_fft': np.nan,
            'hr_peak': np.nan,
            'signal_length': 0,
            'original_length': 0,
            'sampling_rate': 0
        }

    try:
        # 读取PPG信号和信号信息
        signal_data = wfdb.rdrecord(ppg_file)

        # 检查信号维度和获取采样率
        if signal_data.p_signal.shape[1] > 0:
            ppg_signal = signal_data.p_signal[:, 0]
        else:
            return {
                'signal_id': signal_id,
                'success': False,
                'error': '信号数据为空',
                'hr_fft': np.nan,
                'hr_peak': np.nan,
                'signal_length': 0,
                'original_length': 0,
                'sampling_rate': 0
            }

        original_length = len(ppg_signal)
        # 使用实际的30Hz采样频率，除非文件中明确指定了其他值
        actual_fs = signal_data.fs if hasattr(signal_data, 'fs') and signal_data.fs > 0 else config.ppg_fs

        # 如果检测到的采样频率不是30Hz，给出警告
        if actual_fs != 30:
            print(f"警告: 信号 {signal_id} 的采样频率为 {actual_fs}Hz，预期为30Hz")

        # 调试信息：检查原始信号特征
        signal_range = np.max(ppg_signal) - np.min(ppg_signal)
        non_zero_count = np.count_nonzero(ppg_signal)
        signal_std = np.std(ppg_signal)

        print(
            f"信号 {signal_id}: 长度={original_length}, 采样率={actual_fs}, 范围={signal_range:.3f}, 非零点={non_zero_count}, 标准差={signal_std:.3f}")

        # 检查信号是否有效
        if signal_range < 1e-6 or non_zero_count < original_length * 0.1:
            return {
                'signal_id': signal_id,
                'success': False,
                'error': f'信号质量太差: 范围={signal_range:.6f}, 非零比例={non_zero_count / original_length:.3f}',
                'hr_fft': np.nan,
                'hr_peak': np.nan,
                'signal_length': original_length,
                'original_length': original_length,
                'sampling_rate': actual_fs
            }

        # 对于30Hz采样率，至少需要3秒的数据来可靠计算心率
        min_samples = actual_fs * 3
        if original_length < min_samples:
            return {
                'signal_id': signal_id,
                'success': False,
                'error': f'信号太短: {original_length} 个采样点 (需要至少 {min_samples} 个)',
                'hr_fft': np.nan,
                'hr_peak': np.nan,
                'signal_length': original_length,
                'original_length': original_length,
                'sampling_rate': actual_fs
            }

        # 使用实际的采样率进行滤波
        filtered_signal = bandpass_filter(
            ppg_signal,
            lowcut=config.filter_lowcut,
            highcut=min(config.filter_highcut, actual_fs / 2 - 0.5),  # 确保不超过奈奎斯特频率
            fs=actual_fs,
            order=config.filter_order
        )

        # 信号标准化
        if np.std(filtered_signal) > 1e-8:
            normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
        else:
            normalized_signal = filtered_signal

        # 使用实际采样率计算心率
        hr_fft = get_hr_fft(
            normalized_signal,
            fs=actual_fs,
            min_hr=config.hr_min,
            max_hr=config.hr_max
        )

        # 使用峰值检测方法计算心率
        hr_peak = get_hr_peak(
            normalized_signal,
            fs=actual_fs,
            min_hr=config.hr_min,
            max_hr=config.hr_max
        )

        return {
            'signal_id': signal_id,
            'success': True,
            'error': None,
            'hr_fft': hr_fft,
            'hr_peak': hr_peak,
            'signal_length': original_length,
            'original_length': original_length,
            'sampling_rate': actual_fs,
            'raw_signal': ppg_signal,
            'filtered_signal': filtered_signal,
            'normalized_signal': normalized_signal,
            'signal_range': signal_range,
            'non_zero_ratio': non_zero_count / original_length
        }

    except Exception as e:
        return {
            'signal_id': signal_id,
            'success': False,
            'error': str(e),
            'hr_fft': np.nan,
            'hr_peak': np.nan,
            'signal_length': 0,
            'original_length': 0,
            'sampling_rate': 0
        }


def analyze_hr_methods(df, data_dir, config):
    """
    分析不同心率计算方法的结果

    参数:
        df: 数据框
        data_dir: 数据目录
        config: 配置对象

    返回:
        分析结果
    """
    print("开始处理PPG信号...")
    print(
        f"数据集信息: {config.dataset_info['subjects']}名受试者, PPG采样率: {config.dataset_info['ppg_sampling_rate']}Hz")

    results = []
    success_count = 0
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理PPG信号"):
        signal_id = str(row['signal_id'])
        result = process_ppg_signal(signal_id, data_dir, config)

        # 添加原始心率信息
        result['hr_original'] = row.get('hr', np.nan)
        result['quality'] = row.get('quality', np.nan)

        results.append(result)

        if result['success']:
            success_count += 1
        else:
            error_count += 1

    print(f"处理完成: 成功 {success_count} 个, 失败 {error_count} 个")

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def calculate_metrics(y_true, y_pred):
    """
    计算评估指标

    参数:
        y_true: 真实值
        y_pred: 预测值

    返回:
        评估指标字典
    """
    # 移除NaN值
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))

    if np.sum(valid_mask) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'correlation': np.nan,
            'count': 0
        }

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    correlation = np.corrcoef(y_true_valid, y_pred_valid)[0, 1] if len(y_true_valid) > 1 else np.nan

    return {
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'count': len(y_true_valid)
    }


def plot_hr_comparison(results_df, config):
    """
    绘制心率比较图

    参数:
        results_df: 结果数据框
        config: 配置对象
    """
    # 创建结果目录
    os.makedirs(config.results_dir, exist_ok=True)

    # 过滤有效数据
    valid_data = results_df[
        results_df['success'] &
        ~results_df['hr_original'].isna() &
        ~results_df['hr_fft'].isna() &
        ~results_df['hr_peak'].isna()
        ].copy()

    if len(valid_data) == 0:
        print("没有有效的数据进行比较分析")
        return

    print(f"用于分析的有效数据: {len(valid_data)} 个")

    # 计算评估指标
    hr_original = valid_data['hr_original'].values
    hr_fft = valid_data['hr_fft'].values
    hr_peak = valid_data['hr_peak'].values

    metrics_fft = calculate_metrics(hr_original, hr_fft)
    metrics_peak = calculate_metrics(hr_original, hr_peak)

    print("\n=== 心率计算方法评估结果 (针对30Hz采样率优化) ===")
    print(f"FFT方法:")
    print(f"  MAE: {metrics_fft['mae']:.3f} bpm")
    print(f"  RMSE: {metrics_fft['rmse']:.3f} bpm")
    print(f"  相关系数: {metrics_fft['correlation']:.3f}")
    print(f"  有效样本数: {metrics_fft['count']}")

    print(f"\n峰值检测方法:")
    print(f"  MAE: {metrics_peak['mae']:.3f} bpm")
    print(f"  RMSE: {metrics_peak['rmse']:.3f} bpm")
    print(f"  相关系数: {metrics_peak['correlation']:.3f}")
    print(f"  有效样本数: {metrics_peak['count']}")

    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # FFT方法散点图
    axes[0, 0].scatter(hr_original, hr_fft, alpha=0.6, color='blue')
    axes[0, 0].plot([config.hr_min, config.hr_max], [config.hr_min, config.hr_max], 'r--', lw=2)
    axes[0, 0].set_xlabel('原始心率 (bpm)')
    axes[0, 0].set_ylabel('FFT计算心率 (bpm)')
    axes[0, 0].set_title(f'FFT方法 vs 原始心率 (30Hz优化)\n(相关系数: {metrics_fft["correlation"]:.3f})')
    axes[0, 0].grid(True)

    # 峰值检测方法散点图
    axes[0, 1].scatter(hr_original, hr_peak, alpha=0.6, color='green')
    axes[0, 1].plot([config.hr_min, config.hr_max], [config.hr_min, config.hr_max], 'r--', lw=2)
    axes[0, 1].set_xlabel('原始心率 (bpm)')
    axes[0, 1].set_ylabel('峰值检测心率 (bpm)')
    axes[0, 1].set_title(f'峰值检测 vs 原始心率 (30Hz优化)\n(相关系数: {metrics_peak["correlation"]:.3f})')
    axes[0, 1].grid(True)

    # FFT vs 峰值检测
    fft_peak_metrics = calculate_metrics(hr_fft, hr_peak)
    axes[0, 2].scatter(hr_fft, hr_peak, alpha=0.6, color='purple')
    axes[0, 2].plot([config.hr_min, config.hr_max], [config.hr_min, config.hr_max], 'r--', lw=2)
    axes[0, 2].set_xlabel('FFT计算心率 (bpm)')
    axes[0, 2].set_ylabel('峰值检测心率 (bpm)')
    axes[0, 2].set_title(f'FFT vs 峰值检测\n(相关系数: {fft_peak_metrics["correlation"]:.3f})')
    axes[0, 2].grid(True)

    # 误差分布图
    fft_error = hr_fft - hr_original
    peak_error = hr_peak - hr_original

    axes[1, 0].hist(fft_error, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_xlabel('误差 (FFT - 原始)')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title(f'FFT方法误差分布\n(MAE: {metrics_fft["mae"]:.3f})')
    axes[1, 0].grid(True)
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)

    axes[1, 1].hist(peak_error, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_xlabel('误差 (峰值检测 - 原始)')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].set_title(f'峰值检测误差分布\n(MAE: {metrics_peak["mae"]:.3f})')
    axes[1, 1].grid(True)
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)

    # Bland-Altman图
    fft_mean = (hr_original + hr_fft) / 2
    axes[1, 2].scatter(fft_mean, fft_error, alpha=0.6, color='blue', label='FFT')
    axes[1, 2].axhline(np.mean(fft_error), color='blue', linestyle='-', linewidth=2)
    axes[1, 2].axhline(np.mean(fft_error) + 1.96 * np.std(fft_error), color='blue', linestyle='--', linewidth=1)
    axes[1, 2].axhline(np.mean(fft_error) - 1.96 * np.std(fft_error), color='blue', linestyle='--', linewidth=1)
    axes[1, 2].set_xlabel('平均心率 (bpm)')
    axes[1, 2].set_ylabel('差值 (计算 - 原始)')
    axes[1, 2].set_title('Bland-Altman图 (FFT方法)')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'hr_comparison_analysis_30hz.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_signal_examples(results_df, config, num_examples=6):
    """
    绘制信号处理示例

    参数:
        results_df: 结果数据框
        config: 配置对象
        num_examples: 示例数量
    """
    # 选择成功处理的样本
    valid_samples = results_df[results_df['success']].head(num_examples)

    if len(valid_samples) == 0:
        print("没有有效样本可以显示")
        return

    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 3 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for i, (idx, row) in enumerate(valid_samples.iterrows()):
        if i >= num_examples:
            break

        signal_id = row['signal_id']
        raw_signal = row['raw_signal']
        filtered_signal = row['filtered_signal']
        normalized_signal = row['normalized_signal']
        actual_fs = row.get('sampling_rate', config.ppg_fs)

        time_axis = np.arange(len(raw_signal)) / actual_fs

        # 原始信号
        axes[i, 0].plot(time_axis, raw_signal, 'b-', linewidth=1)
        axes[i, 0].set_title(f'信号 {signal_id} - 原始PPG信号 (30Hz)\n长度: {len(raw_signal)}, 采样率: {actual_fs}Hz')
        axes[i, 0].set_xlabel('时间 (秒)')
        axes[i, 0].set_ylabel('幅值')
        axes[i, 0].grid(True)

        # 添加信号统计信息
        signal_info = f'范围: {row.get("signal_range", 0):.3f}, 非零比例: {row.get("non_zero_ratio", 0):.3f}'
        axes[i, 0].text(0.02, 0.98, signal_info, transform=axes[i, 0].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 滤波后信号
        axes[i, 1].plot(time_axis, filtered_signal, 'g-', linewidth=1)
        axes[i, 1].set_title('滤波后信号 (0.5-3Hz)')
        axes[i, 1].set_xlabel('时间 (秒)')
        axes[i, 1].set_ylabel('幅值')
        axes[i, 1].grid(True)

        # 标准化信号及频谱分析
        axes[i, 2].plot(time_axis, normalized_signal, 'r-', linewidth=1)
        hr_fft = row.get("hr_fft", np.nan)
        hr_peak = row.get("hr_peak", np.nan)
        hr_orig = row.get("hr_original", np.nan)

        title_text = f'标准化信号\nFFT: {hr_fft:.1f}, 峰值: {hr_peak:.1f}'
        if not np.isnan(hr_orig):
            title_text += f', 原始: {hr_orig:.1f}'

        axes[i, 2].set_title(title_text)
        axes[i, 2].set_xlabel('时间 (秒)')
        axes[i, 2].set_ylabel('标准化幅值')
        axes[i, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'signal_processing_examples_30hz.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def save_results(results_df, config):
    """
    保存分析结果

    参数:
        results_df: 结果数据框
        config: 配置对象
    """
    # 创建结果目录
    os.makedirs(config.results_dir, exist_ok=True)

    # 保存完整结果
    output_file = os.path.join(config.results_dir, 'hr_analysis_results_30hz.csv')

    # 选择要保存的列
    save_columns = ['signal_id', 'success', 'error', 'hr_original', 'hr_fft', 'hr_peak',
                    'signal_length', 'sampling_rate', 'quality']

    results_save = results_df[save_columns].copy()
    results_save.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"结果已保存到: {output_file}")

    # 保存统计摘要
    summary_file = os.path.join(config.results_dir, 'analysis_summary_30hz.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("BUT-PPG心率分析结果摘要 (30Hz优化版本)\n")
        f.write("=" * 50 + "\n\n")

        # 数据集信息
        f.write("数据集信息:\n")
        for key, value in config.dataset_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        total_signals = len(results_df)
        successful_signals = results_df['success'].sum()

        f.write(f"总信号数: {total_signals}\n")
        f.write(f"成功处理: {successful_signals}\n")
        f.write(f"失败数量: {total_signals - successful_signals}\n")
        f.write(f"成功率: {successful_signals / total_signals * 100:.2f}%\n\n")

        # 有效数据统计
        valid_data = results_df[
            results_df['success'] &
            ~results_df['hr_original'].isna() &
            ~results_df['hr_fft'].isna() &
            ~results_df['hr_peak'].isna()
            ]

        if len(valid_data) > 0:
            hr_original = valid_data['hr_original'].values
            hr_fft = valid_data['hr_fft'].values
            hr_peak = valid_data['hr_peak'].values

            metrics_fft = calculate_metrics(hr_original, hr_fft)
            metrics_peak = calculate_metrics(hr_original, hr_peak)

            f.write(f"有效对比数据: {len(valid_data)} 个\n\n")

            f.write("FFT方法评估 (30Hz优化):\n")
            f.write(f"  MAE: {metrics_fft['mae']:.3f} bpm\n")
            f.write(f"  RMSE: {metrics_fft['rmse']:.3f} bpm\n")
            f.write(f"  相关系数: {metrics_fft['correlation']:.3f}\n\n")

            f.write("峰值检测方法评估 (30Hz优化):\n")
            f.write(f"  MAE: {metrics_peak['mae']:.3f} bpm\n")
            f.write(f"  RMSE: {metrics_peak['rmse']:.3f} bpm\n")
            f.write(f"  相关系数: {metrics_peak['correlation']:.3f}\n")

    print(f"分析摘要已保存到: {summary_file}")


def add_signal_diagnostics(results_df, config):
    """
    添加信号诊断功能

    参数:
        results_df: 结果数据框
        config: 配置对象
    """
    print("\n=== 信号质量诊断报告 (30Hz采样率) ===")

    total_signals = len(results_df)
    successful_signals = results_df['success'].sum()
    failed_signals = total_signals - successful_signals

    print(f"总信号数: {total_signals}")
    print(f"成功处理: {successful_signals}")
    print(f"处理失败: {failed_signals}")

    if failed_signals > 0:
        print("\n失败原因统计:")
        error_counts = results_df[~results_df['success']]['error'].value_counts()
        for error, count in error_counts.items():
            print(f"  {error}: {count} 个")

    # 成功信号的统计
    if successful_signals > 0:
        success_df = results_df[results_df['success']].copy()

        print(f"\n成功信号统计:")
        print(f"  平均信号长度: {success_df['original_length'].mean():.1f} 个采样点")
        print(f"  信号长度范围: {success_df['original_length'].min()}-{success_df['original_length'].max()}")

        # 计算平均时长（基于30Hz采样）
        avg_duration = success_df['original_length'].mean() / 30
        print(f"  平均信号时长: {avg_duration:.1f} 秒")

        if 'sampling_rate' in success_df.columns:
            unique_fs = success_df['sampling_rate'].unique()
            print(f"  检测到的采样率: {unique_fs} Hz")

            # 检查采样率一致性
            fs_30_count = (success_df['sampling_rate'] == 30).sum()
            print(f"  30Hz采样率信号数: {fs_30_count}/{len(success_df)}")

        if 'signal_range' in success_df.columns:
            print(f"  信号范围平均: {success_df['signal_range'].mean():.3f}")
            print(f"  非零比例平均: {success_df['non_zero_ratio'].mean():.3f}")

    # 创建诊断图
    if successful_signals > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        success_df = results_df[results_df['success']].copy()

        # 信号长度分布
        axes[0, 0].hist(success_df['original_length'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('信号长度 (采样点)')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].set_title('信号长度分布 (30Hz)')
        axes[0, 0].grid(True)

        # 信号时长分布（秒）
        signal_duration = success_df['original_length'] / 30
        axes[0, 1].hist(signal_duration, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('信号时长 (秒)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('信号时长分布')
        axes[0, 1].grid(True)

        # 采样率分布
        if 'sampling_rate' in success_df.columns:
            axes[0, 2].hist(success_df['sampling_rate'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 2].set_xlabel('采样率 (Hz)')
            axes[0, 2].set_ylabel('频次')
            axes[0, 2].set_title('采样率分布')
            axes[0, 2].grid(True)

        # 信号范围分布
        if 'signal_range' in success_df.columns:
            axes[1, 0].hist(success_df['signal_range'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('信号范围')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].set_title('信号幅值范围分布')
            axes[1, 0].grid(True)

        # 非零比例分布
        if 'non_zero_ratio' in success_df.columns:
            axes[1, 1].hist(success_df['non_zero_ratio'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('非零数据比例')
            axes[1, 1].set_ylabel('频次')
            axes[1, 1].set_title('有效数据比例分布')
            axes[1, 1].grid(True)

        # 心率分布对比
        valid_hr_data = success_df[~success_df['hr_original'].isna()]
        if len(valid_hr_data) > 0:
            axes[1, 2].hist(valid_hr_data['hr_original'], bins=20, alpha=0.7,
                            label='原始心率', edgecolor='black')

            valid_fft_hr = valid_hr_data[~valid_hr_data['hr_fft'].isna()]
            if len(valid_fft_hr) > 0:
                axes[1, 2].hist(valid_fft_hr['hr_fft'], bins=20, alpha=0.5,
                                label='FFT心率', edgecolor='black')

            axes[1, 2].set_xlabel('心率 (bpm)')
            axes[1, 2].set_ylabel('频次')
            axes[1, 2].set_title('心率分布')
            axes[1, 2].legend()
            axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(config.results_dir, 'signal_diagnostics_30hz.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数"""
    config = Config()
    print("BUT-PPG心率计算分析工具 (30Hz采样率优化版本)")
    print("=" * 60)
    print(f"数据目录: {config.data_dir}")
    print(f"结果目录: {config.results_dir}")
    print(f"心率范围: {config.hr_min}-{config.hr_max} bpm")
    print(f"滤波范围: {config.filter_lowcut}-{config.filter_highcut} Hz")
    print(f"PPG采样频率: {config.ppg_fs} Hz")

    print(f"\n数据集信息:")
    for key, value in config.dataset_info.items():
        print(f"  {key}: {value}")

    try:
        # 加载数据
        print("\n正在加载数据...")
        df = load_but_ppg_data(config.data_dir)
        print(f"加载完成，共 {len(df)} 个信号")

        # 分析心率计算方法
        results_df = analyze_hr_methods(df, config.data_dir, config)

        # 添加信号诊断
        add_signal_diagnostics(results_df, config)

        # 绘制比较图
        print("\n正在生成分析图表...")
        plot_hr_comparison(results_df, config)

        # 绘制信号处理示例
        print("正在生成信号处理示例...")
        plot_signal_examples(results_df, config, num_examples=6)

        # 保存结果
        print("正在保存结果...")
        save_results(results_df, config)

        print("\n分析完成! (已针对30Hz采样率优化)")

    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()