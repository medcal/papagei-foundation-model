import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class PPGHeartRateAnalyzer:
    """PPG心率分析器"""

    def __init__(self, fs=125, min_hr=30, max_hr=180):
        """
        初始化心率分析器

        参数:
            fs: 采样频率 (Hz)
            min_hr: 最小心率 (bpm)
            max_hr: 最大心率 (bpm)
        """
        self.fs = fs
        self.min_hr = min_hr
        self.max_hr = max_hr
        self.min_freq = min_hr / 60.0  # 转换为Hz
        self.max_freq = max_hr / 60.0  # 转换为Hz

    def bandpass_filter(self, data, lowcut=0.5, highcut=3.5, order=4):
        """
        应用带通滤波器

        参数:
            data: 输入信号
            lowcut: 低频截止频率
            highcut: 高频截止频率
            order: 滤波器阶数
        """
        if len(data) < 4:
            return data

        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = min(highcut / nyq, 0.95)  # 确保不超过奈奎斯特频率

        if low >= high:
            return data

        try:
            b, a = butter(order, [low, high], btype='band')
            filtered_data = filtfilt(b, a, data)
            return filtered_data
        except Exception as e:
            print(f"滤波失败: {e}")
            return data

    def preprocess_signal(self, signal):
        """
        信号预处理

        参数:
            signal: 原始PPG信号

        返回:
            处理后的信号
        """
        # 确保是1D数组
        if len(signal.shape) > 1:
            signal = signal.flatten()

        # 去除异常值
        q1, q3 = np.percentile(signal, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        signal = np.clip(signal, lower_bound, upper_bound)

        # 去趋势
        signal = detrend(signal)

        # 带通滤波
        signal = self.bandpass_filter(signal)

        # 标准化
        if np.std(signal) > 1e-8:
            signal = (signal - np.mean(signal)) / np.std(signal)

        return signal

    def get_hr_fft_improved(self, signal):
        """
        改进的FFT心率计算方法

        参数:
            signal: 预处理后的PPG信号

        返回:
            计算得到的心率值
        """
        try:
            if len(signal) < 4:
                return np.nan

            # 使用Welch方法计算功率谱密度
            window_length = min(len(signal) // 4, 256)
            overlap = window_length // 2

            freqs, psd = welch(signal, fs=self.fs,
                               nperseg=window_length,
                               noverlap=overlap,
                               nfft=max(512, len(signal)))

            # 查找有效频率范围
            valid_freq_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)

            if not np.any(valid_freq_mask):
                return np.nan

            valid_freqs = freqs[valid_freq_mask]
            valid_psd = psd[valid_freq_mask]

            # 查找主导频率
            max_power_idx = np.argmax(valid_psd)
            dominant_freq = valid_freqs[max_power_idx]

            # 计算心率
            hr = dominant_freq * 60

            # 验证结果合理性
            if self.min_hr <= hr <= self.max_hr:
                return hr
            else:
                return np.nan

        except Exception as e:
            print(f"FFT心率计算失败: {e}")
            return np.nan

    def get_hr_peak_improved(self, signal):
        """
        改进的峰值检测心率计算方法

        参数:
            signal: 预处理后的PPG信号

        返回:
            计算得到的心率值
        """
        try:
            if len(signal) < self.fs:  # 至少需要1秒的数据
                return np.nan

            # 动态阈值峰值检测
            signal_std = np.std(signal)
            signal_mean = np.mean(signal)
            height_threshold = signal_mean + 0.3 * signal_std

            # 设置最小峰值间距（基于最大心率）
            min_distance = int(self.fs * 60 / self.max_hr)

            # 查找峰值
            peaks, properties = find_peaks(signal,
                                           height=height_threshold,
                                           distance=min_distance,
                                           prominence=0.1 * signal_std)

            if len(peaks) < 2:
                return np.nan

            # 计算峰值间隔
            peak_intervals = np.diff(peaks)

            # 过滤异常间隔
            max_interval = self.fs * 60 / self.min_hr
            min_interval = self.fs * 60 / self.max_hr

            valid_intervals = peak_intervals[
                (peak_intervals >= min_interval) &
                (peak_intervals <= max_interval)
                ]

            if len(valid_intervals) == 0:
                return np.nan

            # 使用中位数计算平均间隔（更鲁棒）
            median_interval = np.median(valid_intervals)

            # 计算心率
            hr = 60 * self.fs / median_interval

            # 验证结果合理性
            if self.min_hr <= hr <= self.max_hr:
                return hr
            else:
                return np.nan

        except Exception as e:
            print(f"峰值心率计算失败: {e}")
            return np.nan


class PPGDatasetAnalyzer:
    """PPG数据集分析器"""

    def __init__(self, data_dir, fs=125):
        """
        初始化数据集分析器

        参数:
            data_dir: 数据集根目录
            fs: 采样频率
        """
        self.data_dir = data_dir
        self.fs = fs
        self.hr_analyzer = PPGHeartRateAnalyzer(fs=fs)

    def load_ppg_signal(self, subject_id, segment_idx):
        """加载PPG信号"""
        try:
            # 格式化subject_id为4位数字
            subject_str = str(subject_id).zfill(4)
            signal_file = os.path.join(self.data_dir, f"ppg/{subject_str}/segment_{segment_idx}.npy")

            if os.path.exists(signal_file):
                signal = np.load(signal_file)
                return signal
            else:
                # 尝试备用文件结构
                main_dir = os.path.join(self.data_dir, "Data File/0_subject/")
                signal_file = os.path.join(main_dir, f"{subject_id}_{segment_idx}.txt")
                if os.path.exists(signal_file):
                    signal = pd.read_csv(signal_file, sep='\t', header=None)
                    return signal.values.squeeze()[:-1]
                else:
                    return None
        except Exception as e:
            print(f"加载信号失败 (subject {subject_id}, segment {segment_idx}): {e}")
            return None

    def calculate_subject_hr(self, subject_id, true_hr, methods=['fft', 'peak']):
        """
        计算单个受试者的心率

        参数:
            subject_id: 受试者ID
            true_hr: 真实心率
            methods: 使用的心率计算方法列表

        返回:
            包含各种方法结果的字典
        """
        result = {
            'subject_id': subject_id,
            'true_hr': true_hr,
            'valid_segments': 0,
            'signal_quality': 'unknown'
        }

        # 初始化方法结果
        for method in methods:
            result[f'hr_{method}'] = []
            result[f'hr_{method}_mean'] = np.nan

        segment_hrs = {method: [] for method in methods}
        valid_signals = []

        # 处理3个片段
        for segment_idx in range(1, 4):
            signal = self.load_ppg_signal(subject_id, segment_idx)

            if signal is not None and len(signal) > 0:
                try:
                    # 预处理信号
                    processed_signal = self.hr_analyzer.preprocess_signal(signal)

                    # 检查信号质量
                    signal_std = np.std(processed_signal)
                    if signal_std < 0.1:  # 信号质量太低
                        continue

                    valid_signals.append(processed_signal)
                    result['valid_segments'] += 1

                    # 使用不同方法计算心率
                    if 'fft' in methods:
                        hr_fft = self.hr_analyzer.get_hr_fft_improved(processed_signal)
                        if np.isfinite(hr_fft):
                            segment_hrs['fft'].append(hr_fft)

                    if 'peak' in methods:
                        hr_peak = self.hr_analyzer.get_hr_peak_improved(processed_signal)
                        if np.isfinite(hr_peak):
                            segment_hrs['peak'].append(hr_peak)

                except Exception as e:
                    print(f"处理失败 (subject {subject_id}, segment {segment_idx}): {e}")
                    continue

        # 计算每种方法的平均心率
        for method in methods:
            if segment_hrs[method]:
                result[f'hr_{method}'] = segment_hrs[method]
                result[f'hr_{method}_mean'] = np.mean(segment_hrs[method])
            else:
                result[f'hr_{method}'] = []
                result[f'hr_{method}_mean'] = np.nan

        # 评估信号质量
        if result['valid_segments'] >= 2:
            result['signal_quality'] = 'good'
        elif result['valid_segments'] == 1:
            result['signal_quality'] = 'fair'
        else:
            result['signal_quality'] = 'poor'

        return result

    def analyze_dataset(self, test_split='test', methods=['fft', 'peak']):
        """
        分析整个数据集

        参数:
            test_split: 数据集分割 ('test', 'val', 'train')
            methods: 使用的心率计算方法列表

        返回:
            分析结果DataFrame和指标字典
        """
        # 加载数据集分割
        split_file = os.path.join(self.data_dir, f"Data File/{test_split}.csv")

        if not os.path.exists(split_file):
            print(f"分割文件 {split_file} 不存在，加载主数据集...")
            split_file = os.path.join(self.data_dir, "Data File/PPG-BP dataset.xlsx")
            df = pd.read_excel(split_file, header=1)

            # 根据分割类型筛选数据
            if test_split == 'test':
                test_ids = [14, 21, 25, 51, 52, 62, 67, 86, 90, 96, 103, 108, 110, 119, 123, 124,
                            130, 142, 144, 157, 172, 173, 174, 180, 182, 185, 192, 195, 200, 201,
                            211, 214, 219, 221, 228, 239, 250, 403, 405, 406, 410]
                df = df[df.subject_ID.isin(test_ids)]
            elif test_split == 'val':
                val_ids = [3, 11, 24, 27, 29, 30, 41, 43, 47, 64, 88, 91, 95, 115, 125, 127,
                           136, 145, 155, 156, 161, 163, 166, 178, 198, 203, 208, 213, 215, 222,
                           229, 232, 235, 237, 241, 245, 252, 254, 259, 411, 418]
                df = df[df.subject_ID.isin(val_ids)]
        else:
            df = pd.read_csv(split_file)

        # 重命名列
        if "Heart Rate(b/m)" in df.columns:
            df = df.rename(columns={"Heart Rate(b/m)": "hr"})

        print(f"分析 {test_split} 数据集: {len(df)} 个受试者")
        print(f"使用方法: {methods}")

        results = []

        # 处理每个受试者
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"分析受试者"):
            subject_id = row['subject_ID']
            true_hr = row['hr']

            if pd.isna(true_hr) or true_hr <= 0:
                continue

            # 计算受试者心率
            subject_result = self.calculate_subject_hr(subject_id, true_hr, methods)
            results.append(subject_result)

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        # 计算评估指标
        metrics = self.calculate_metrics(results_df, methods)

        return results_df, metrics

    def calculate_metrics(self, results_df, methods):
        """计算评估指标"""
        metrics = {}

        for method in methods:
            method_col = f'hr_{method}_mean'

            # 过滤有效数据
            valid_mask = (~results_df['true_hr'].isna()) & (~results_df[method_col].isna())
            valid_data = results_df[valid_mask]

            if len(valid_data) == 0:
                metrics[method] = {
                    'mae': np.nan,
                    'rmse': np.nan,
                    'correlation': np.nan,
                    'count': 0,
                    'mean_error': np.nan,
                    'std_error': np.nan
                }
                continue

            true_hrs = valid_data['true_hr'].values
            pred_hrs = valid_data[method_col].values

            # 计算指标
            mae = mean_absolute_error(true_hrs, pred_hrs)
            rmse = np.sqrt(mean_squared_error(true_hrs, pred_hrs))
            correlation = np.corrcoef(true_hrs, pred_hrs)[0, 1] if len(true_hrs) > 1 else np.nan

            errors = pred_hrs - true_hrs
            mean_error = np.mean(errors)
            std_error = np.std(errors)

            metrics[method] = {
                'mae': mae,
                'rmse': rmse,
                'correlation': correlation,
                'count': len(valid_data),
                'mean_error': mean_error,
                'std_error': std_error,
                'valid_rate': len(valid_data) / len(results_df)
            }

        return metrics

    def plot_comprehensive_results(self, results_df, metrics, methods, save_dir=None):
        """绘制综合分析结果"""
        n_methods = len(methods)
        fig, axes = plt.subplots(3, n_methods, figsize=(6 * n_methods, 15))

        if n_methods == 1:
            axes = axes.reshape(-1, 1)

        for i, method in enumerate(methods):
            method_col = f'hr_{method}_mean'

            # 过滤有效数据
            valid_mask = (~results_df['true_hr'].isna()) & (~results_df[method_col].isna())
            valid_data = results_df[valid_mask]

            if len(valid_data) == 0:
                continue

            true_hrs = valid_data['true_hr'].values
            pred_hrs = valid_data[method_col].values

            # 散点图
            axes[0, i].scatter(true_hrs, pred_hrs, alpha=0.6, s=30)
            min_hr, max_hr = min(true_hrs.min(), pred_hrs.min()), max(true_hrs.max(), pred_hrs.max())
            axes[0, i].plot([min_hr, max_hr], [min_hr, max_hr], 'r--', lw=2)
            axes[0, i].set_xlabel('真实心率 (bpm)')
            axes[0, i].set_ylabel('预测心率 (bpm)')
            axes[0, i].set_title(
                f'{method.upper()} 方法\nMAE: {metrics[method]["mae"]:.2f}, r: {metrics[method]["correlation"]:.3f}')
            axes[0, i].grid(True, alpha=0.3)

            # 误差分布
            errors = pred_hrs - true_hrs
            axes[1, i].hist(errors, bins=30, alpha=0.7, edgecolor='black', density=True)
            axes[1, i].axvline(0, color='red', linestyle='--', alpha=0.7)
            axes[1, i].axvline(np.mean(errors), color='blue', linestyle='-', alpha=0.7,
                               label=f'均值: {np.mean(errors):.2f}')
            axes[1, i].set_xlabel('预测误差 (bpm)')
            axes[1, i].set_ylabel('密度')
            axes[1, i].set_title(f'误差分布 (RMSE: {metrics[method]["rmse"]:.2f})')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)

            # Bland-Altman图
            mean_hr = (true_hrs + pred_hrs) / 2
            diff_hr = pred_hrs - true_hrs
            axes[2, i].scatter(mean_hr, diff_hr, alpha=0.6, s=30)
            axes[2, i].axhline(0, color='red', linestyle='-', alpha=0.7)
            axes[2, i].axhline(np.mean(diff_hr), color='blue', linestyle='--', alpha=0.7,
                               label=f'偏差: {np.mean(diff_hr):.2f}')
            axes[2, i].axhline(np.mean(diff_hr) + 1.96 * np.std(diff_hr), color='gray', linestyle='--', alpha=0.7)
            axes[2, i].axhline(np.mean(diff_hr) - 1.96 * np.std(diff_hr), color='gray', linestyle='--', alpha=0.7)
            axes[2, i].set_xlabel('平均心率 (bpm)')
            axes[2, i].set_ylabel('差值 (预测 - 真实)')
            axes[2, i].set_title('Bland-Altman图')
            axes[2, i].legend()
            axes[2, i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'comprehensive_hr_analysis.png'),
                        dpi=300, bbox_inches='tight')

        plt.show()

    def print_detailed_results(self, metrics, methods):
        """打印详细分析结果"""
        print("\n" + "=" * 60)
        print("PPG-BP 数据集心率分析结果")
        print("=" * 60)

        for method in methods:
            print(f"\n{method.upper()} 方法:")
            print(f"  MAE:        {metrics[method]['mae']:.3f} bpm")
            print(f"  RMSE:       {metrics[method]['rmse']:.3f} bpm")
            print(f"  相关系数:    {metrics[method]['correlation']:.3f}")
            print(f"  平均偏差:    {metrics[method]['mean_error']:.3f} bpm")
            print(f"  偏差标准差:  {metrics[method]['std_error']:.3f} bpm")
            print(f"  有效预测:    {metrics[method]['count']} 个受试者")
            print(f"  有效率:      {metrics[method]['valid_rate']:.1%}")

        print("\n" + "=" * 60)


def main():
    """主函数"""
    # 设置数据目录
    data_dir = r"E:\thsiu-ppg\5459299\PPG-BP Database"

    print("PPG-BP 数据集心率分析工具 (125Hz优化版本)")
    print("=" * 60)

    # 创建分析器
    analyzer = PPGDatasetAnalyzer(data_dir, fs=125)

    # 定义要测试的方法
    methods = ['fft', 'peak']

    # 分析测试集
    print("\n开始分析测试集...")
    results_df, metrics = analyzer.analyze_dataset(test_split='test', methods=methods)

    # 打印结果
    analyzer.print_detailed_results(metrics, methods)

    # 绘制结果
    print("\n生成分析图表...")
    analyzer.plot_comprehensive_results(results_df, metrics, methods,
                                        save_dir=os.path.join(data_dir, "hr_analysis_results"))

    # 保存结果
    output_dir = os.path.join(data_dir, "hr_analysis_results")
    os.makedirs(output_dir, exist_ok=True)

    # 保存详细结果
    results_df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)

    # 保存指标摘要
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"))

    print(f"\n结果已保存到: {output_dir}")
    print("分析完成!")


if __name__ == "__main__":
    main()