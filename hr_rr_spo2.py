import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.signal import butter, filtfilt, find_peaks, welch
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class BIDMCConfig:
    """BIDMC数据集配置"""
    # BIDMC数据集路径
    bidmc_dir = r"E:\thsiu-ppg\bidmc-ppg-and-respiration-dataset-1.0.0\bidmc-ppg-and-respiration-dataset-1.0.0"
    csv_dir = f"{bidmc_dir}/bidmc_csv"
    results_dir = f"{bidmc_dir}/vital_signs_results"

    # PPG信号参数 - 根据BIDMC数据集规格
    sampling_rate = 125  # BIDMC PPG采样频率 125Hz
    ppg_length = 1000  # 8秒 @ 125Hz (125 * 8 = 1000)
    window_overlap = 0.5  # 窗口重叠50%

    # HR计算参数 - 针对125Hz采样率优化
    hr_method = 'fft'  # 'fft' 或 'peak'
    hr_min = 30
    hr_max = 180
    hr_lowcut = 0.5
    hr_highcut = 3.0
    hr_filter_order = 3

    # RR计算参数 - 针对125Hz采样率优化
    rr_method = 'fft'  # 'fft' 或 'peak'
    rr_min = 6
    rr_max = 30
    rr_lowcut = 0.1
    rr_highcut = 0.5
    rr_filter_order = 3

    # SPO2计算参数
    spo2_ring_type = "ring1"  # "ring1" 或 "ring2"
    spo2_method = "ratio"


def bandpass_filter(data, lowcut=0.5, highcut=3, fs=125, order=3):
    """
    应用带通滤波器 - 针对125Hz采样率优化

    参数:
        data: 输入信号
        lowcut: 低频截止频率
        highcut: 高频截止频率
        fs: 采样频率 (默认125Hz)
        order: 滤波器阶数
    """
    if fs is None or fs <= 0:
        return np.zeros_like(data)

    nyq = 0.5 * fs  # 奈奎斯特频率 = 62.5Hz (125Hz/2)
    low = lowcut / nyq
    high = highcut / nyq

    # 检查频率是否超出奈奎斯特限制
    if low >= 1.0 or high >= 1.0:
        print(f"警告: 截止频率超出奈奎斯特频率。lowcut={lowcut}, highcut={highcut}, fs={fs}, nyq={nyq}")
        # 调整频率到安全范围
        if high >= 1.0:
            high = 0.95
            print(f"调整高频截止到: {high * nyq:.1f}Hz")
        if low >= 1.0:
            low = 0.05
            print(f"调整低频截止到: {low * nyq:.1f}Hz")

    try:
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    except Exception as e:
        print(f"滤波器应用失败: {e}")
        return data


def get_hr(y, fs=125, min_hr=30, max_hr=180, method='fft'):
    """
    使用FFT或峰值检测计算心率 - 针对125Hz采样率优化

    参数:
        y: 滤波后的PPG信号（1D数组）
        fs: 采样频率 (默认125Hz)
        min_hr: 最小心率
        max_hr: 最大心率
        method: 'fft' 或 'peak'

    返回:
        计算得到的心率值
    """
    # 确保是1D数组
    if len(y.shape) > 1:
        y = y.flatten()

    if method == 'fft':
        try:
            # 针对125Hz采样率调整FFT参数
            nfft_val = int(5e6 / fs)
            nperseg_val = np.min((len(y) - 1, 512))

            p, q = welch(y, fs, nfft=nfft_val, nperseg=nperseg_val)
            valid_freq = (p > min_hr / 60) & (p < max_hr / 60)

            if np.sum(valid_freq) == 0:
                return 80.0  # 默认心率

            fft_hr = p[valid_freq][np.argmax(q[valid_freq])] * 60
            return float(fft_hr)
        except Exception as e:
            return 80.0

    elif method == 'peak':
        try:
            # 针对125Hz采样率调整峰值检测参数
            min_distance = int(fs * 0.4)  # 最小峰值间距：0.4秒 (150bpm对应)

            ppg_peaks, _ = find_peaks(y, distance=min_distance)

            if len(ppg_peaks) <= 1:
                return 80.0

            peak_diffs = np.diff(ppg_peaks)
            mean_diff = np.mean(peak_diffs)

            if mean_diff <= 0:
                return 80.0

            peak_hr = 60 / (mean_diff / fs)

            if not np.isfinite(peak_hr):
                return 80.0

            return float(peak_hr)
        except Exception as e:
            return 80.0
    else:
        raise ValueError("Invalid method. Choose 'fft' or 'peak'.")


def get_rr(y, fs=125, min_rr=6, max_rr=30, method='fft'):
    """
    使用FFT或峰值检测计算呼吸频率 - 针对125Hz采样率优化

    参数:
        y: 0.1-0.5 Hz滤波后的PPG信号（1D数组）
        fs: 采样频率 (默认125Hz)
        min_rr: 最小呼吸频率
        max_rr: 最大呼吸频率
        method: 'fft' 或 'peak'

    返回:
        计算得到的呼吸频率值
    """
    # 确保是1D数组
    if len(y.shape) > 1:
        y = y.flatten()

    if method == 'fft':
        try:
            # 针对125Hz采样率和低频呼吸信号调整FFT参数
            nfft_val = int(5e6 / fs)
            nperseg_val = np.min((len(y) - 1, 512))

            p, q = welch(y, fs, nfft=nfft_val, nperseg=nperseg_val)
            valid_freq = (p > min_rr / 60) & (p < max_rr / 60)

            if np.sum(valid_freq) == 0:
                return 15.0  # 默认呼吸频率

            fft_rr = p[valid_freq][np.argmax(q[valid_freq])] * 60
            return float(fft_rr)
        except Exception as e:
            return 15.0

    elif method == 'peak':
        try:
            # 针对125Hz采样率调整峰值检测参数
            min_distance = int(fs * 2.0)  # 最小峰值间距：2秒 (30rpm对应)

            ppg_peaks, _ = find_peaks(y, distance=min_distance)

            if len(ppg_peaks) <= 1:
                return 15.0

            peak_intervals = np.diff(ppg_peaks) / fs

            if np.mean(peak_intervals) > 0:
                peak_rr = 60 / np.mean(peak_intervals)
            else:
                return 15.0

            return float(peak_rr)
        except Exception as e:
            return 15.0
    else:
        raise ValueError("Invalid method. Choose 'fft' or 'peak'.")


def get_spo2(ppg_ir, ppg_red, fs=125, ring_type="ring1", method="ratio"):
    """
    从原始PPG信号计算SpO2 - 针对125Hz采样率优化

    参数:
        ppg_red: 红光通道PPG信号
        ppg_ir: 红外通道PPG信号
        fs: 采样频率 (默认125Hz)
        ring_type: 环型传感器类型，"ring1" 或 "ring2"
        method: 计算方法，默认"ratio"

    返回:
        计算得到的SpO2值
    """
    try:
        ppg_red = np.array(ppg_red)
        ppg_ir = np.array(ppg_ir)

        # 计算AC分量 (峰-峰值)
        ac_red = np.max(ppg_red) - np.min(ppg_red)
        ac_ir = np.max(ppg_ir) - np.min(ppg_ir)

        # 计算DC分量 (平均值)
        dc_red = np.mean(ppg_red)
        dc_ir = np.mean(ppg_ir)

        # 避免除零
        epsilon = 1e-6
        dc_red = max(dc_red, epsilon)
        dc_ir = max(dc_ir, epsilon)

        # 计算每个信号的AC/DC
        acDivDcRed = ac_red / dc_red
        acDivDcIr = ac_ir / dc_ir

        # 计算比值，保护除零
        epsilon_ratio = 1e-6
        if abs(acDivDcIr) < epsilon_ratio:
            ratio = 1.0
        else:
            ratio = acDivDcRed / acDivDcIr

        # 根据环型传感器类型计算SpO2
        if ring_type == "ring1":
            SPO2 = 99 - 6 * ratio
        elif ring_type == "ring2":
            SPO2 = 87 + 6 * ratio
        else:
            SPO2 = 99 - 6 * ratio

        # 确保SpO2在有效范围内
        SPO2 = np.clip(SPO2, 80, 99)

        return float(np.mean(SPO2))
    except Exception as e:
        return np.nan


class BIDMCDataProcessor:
    def __init__(self, config):
        self.config = config
        self.data_records = []

    def load_bidmc_data(self):
        """加载BIDMC数据集"""
        print("开始加载BIDMC数据集...")

        success_subjects = 0
        total_segments = 0

        for i in range(1, 54):  # 01到53
            subject_id = f"{i:02d}"
            try:
                signals_file = f"{self.config.csv_dir}/bidmc_{subject_id}_Signals.csv"
                numerics_file = f"{self.config.csv_dir}/bidmc_{subject_id}_Numerics.csv"
                fix_file = f"{self.config.csv_dir}/bidmc_{subject_id}_Fix.txt"

                # 检查文件是否存在
                missing_files = []
                for file_path, file_name in [(signals_file, "Signals"), (numerics_file, "Numerics"), (fix_file, "Fix")]:
                    if not os.path.exists(file_path):
                        missing_files.append(file_name)

                if missing_files:
                    print(f"跳过受试者 {subject_id}: 缺失文件 {missing_files}")
                    continue

                # 读取数据文件
                signals_df = pd.read_csv(signals_file)
                numerics_df = pd.read_csv(numerics_file)

                # 设置时间索引
                signals_df.set_index('Time [s]', inplace=True)
                numerics_df.set_index('Time [s]', inplace=True)

                # 读取固定参数文件
                with open(fix_file, 'r') as f:
                    fix_content = f.read()

                age, gender = self.parse_fix_file(fix_content)

                # 处理信号和目标值
                subject_segments = self.process_signals_and_targets(
                    signals_df, numerics_df, subject_id, age, gender
                )

                if len(subject_segments) > 0:
                    self.data_records.extend(subject_segments)
                    success_subjects += 1
                    total_segments += len(subject_segments)
                    print(f"成功加载受试者 {subject_id}: {len(subject_segments)} 个信号段")
                else:
                    print(f"受试者 {subject_id}: 无有效信号段")

            except Exception as e:
                print(f"处理受试者 {subject_id} 时出错: {e}")
                continue

        print(f"数据加载完成: 成功处理 {success_subjects} 个受试者, 共 {total_segments} 个信号段")
        return self.data_records

    def parse_fix_file(self, content):
        """解析固定参数文件"""
        lines = content.strip().split('\n')
        age = 50  # 默认值
        gender = 'M'  # 默认值

        for line in lines:
            line_lower = line.lower()
            if 'age' in line_lower:
                try:
                    # 提取数字
                    age = int(''.join(filter(str.isdigit, line)))
                except:
                    pass
            elif 'gender' in line_lower or 'sex' in line_lower:
                if 'f' in line_lower or 'female' in line_lower:
                    gender = 'F'

        return age, gender

    def process_signals_and_targets(self, signals_df, numerics_df, subject_id, age, gender):
        """处理信号和目标值"""
        segments = []

        # 打印可用的列名进行调试
        print(f"受试者 {subject_id} - 信号列: {list(signals_df.columns)}")
        print(f"受试者 {subject_id} - 数值列: {list(numerics_df.columns)}")

        # 选择PPG信号列 - 更宽泛的搜索
        ppg_columns = []
        for col in signals_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['ppg', 'pleth', 'ii', 'v', 'pulse']):
                ppg_columns.append(col)

        if not ppg_columns:
            print(f"受试者 {subject_id}: 未找到PPG信号列。可用列: {list(signals_df.columns)}")
            return segments

        print(f"受试者 {subject_id}: 使用PPG信号列: {ppg_columns}")
        ppg_signal = signals_df[ppg_columns[0]].values

        # 如果有多个PPG通道，提取用于SpO2计算
        ppg_ir = ppg_signal  # 第一个通道作为红外
        ppg_red = signals_df[ppg_columns[1]].values if len(ppg_columns) > 1 else ppg_signal  # 第二个通道作为红光

        # 检查信号质量
        if len(ppg_signal) < self.config.ppg_length:
            print(f"受试者 {subject_id}: PPG信号太短 ({len(ppg_signal)} < {self.config.ppg_length})")
            return segments

        # 使用滑动窗口分割PPG信号
        window_size = self.config.ppg_length
        step_size = int(window_size * (1 - self.config.window_overlap))  # 50%重叠

        print(f"受试者 {subject_id}: PPG信号长度={len(ppg_signal)}, 窗口大小={window_size}, 步长={step_size}")

        for start_idx in range(0, len(ppg_signal) - window_size + 1, step_size):
            end_idx = start_idx + window_size

            # 提取信号段
            ppg_segment = ppg_signal[start_idx:end_idx]
            ppg_ir_segment = ppg_ir[start_idx:end_idx]
            ppg_red_segment = ppg_red[start_idx:end_idx]

            # 检查信号段质量
            if np.isnan(ppg_segment).any() or np.std(ppg_segment) < 1e-6:
                continue

            # 计算时间范围
            start_time = start_idx / self.config.sampling_rate
            end_time = end_idx / self.config.sampling_rate

            # 匹配时间段的数值数据 - 更灵活的列名匹配
            time_mask = (numerics_df.index >= start_time) & (numerics_df.index <= end_time)
            segment_numerics = numerics_df.loc[time_mask]

            # 获取真实的生理参数值 - 更灵活的列名匹配
            hr_true = np.nan
            rr_true = np.nan
            spo2_true = np.nan

            if not segment_numerics.empty:
                # 心率列名匹配
                hr_cols = [col for col in segment_numerics.columns if
                           any(keyword in col.lower() for keyword in ['hr', 'heart'])]
                if hr_cols:
                    hr_true = segment_numerics[hr_cols[0]].mean()

                # 呼吸频率列名匹配
                rr_cols = [col for col in segment_numerics.columns if
                           any(keyword in col.lower() for keyword in ['resp', 'rr', 'breath'])]
                if rr_cols:
                    rr_true = segment_numerics[rr_cols[0]].mean()

                # 血氧饱和度列名匹配
                spo2_cols = [col for col in segment_numerics.columns if
                             any(keyword in col.lower() for keyword in ['spo2', 'sp02', 'oxygen', 'sat'])]
                if spo2_cols:
                    spo2_true = segment_numerics[spo2_cols[0]].mean()

            # 计算生理参数
            vital_signs = self.calculate_vital_signs(
                ppg_segment, ppg_ir_segment, ppg_red_segment, self.config.sampling_rate
            )

            # 创建数据记录
            record = {
                'subject_id': subject_id,
                'start_time': start_time,
                'end_time': end_time,
                'age': age,
                'gender': gender,
                'ppg_signal': ppg_segment,
                'ppg_ir': ppg_ir_segment,
                'ppg_red': ppg_red_segment,

                # 真实值
                'hr_true': float(hr_true) if not np.isnan(hr_true) else np.nan,
                'rr_true': float(rr_true) if not np.isnan(rr_true) else np.nan,
                'spo2_true': float(spo2_true) if not np.isnan(spo2_true) else np.nan,

                # 计算值
                'hr_fft': vital_signs['hr_fft'],
                'hr_peak': vital_signs['hr_peak'],
                'rr_fft': vital_signs['rr_fft'],
                'rr_peak': vital_signs['rr_peak'],
                'spo2_calc': vital_signs['spo2'],

                # 滤波后的信号
                'hr_filtered': vital_signs['hr_filtered'],
                'rr_filtered': vital_signs['rr_filtered']
            }

            segments.append(record)

        print(f"受试者 {subject_id}: 生成 {len(segments)} 个有效信号段")
        return segments

    def calculate_vital_signs(self, ppg_signal, ppg_ir, ppg_red, fs):
        """计算生理信号"""
        # 预处理信号
        ppg_signal = np.nan_to_num(ppg_signal)
        ppg_ir = np.nan_to_num(ppg_ir)
        ppg_red = np.nan_to_num(ppg_red)

        # 检查信号质量
        if np.std(ppg_signal) < 1e-6:
            print("警告: PPG信号标准差太小，可能是无效信号")
            return {
                'hr_fft': 80.0,
                'hr_peak': 80.0,
                'rr_fft': 15.0,
                'rr_peak': 15.0,
                'spo2': np.nan,
                'hr_filtered': np.zeros_like(ppg_signal),
                'rr_filtered': np.zeros_like(ppg_signal)
            }

        # ===================
        # 心率计算
        # ===================
        try:
            # HR滤波 (0.5-3 Hz)
            hr_filtered = bandpass_filter(
                ppg_signal,
                lowcut=self.config.hr_lowcut,
                highcut=self.config.hr_highcut,
                fs=fs,
                order=self.config.hr_filter_order
            )

            # 使用FFT方法计算心率
            hr_fft = get_hr(
                hr_filtered,
                fs=fs,
                min_hr=self.config.hr_min,
                max_hr=self.config.hr_max,
                method='fft'
            )

            # 使用峰值检测方法计算心率
            hr_peak = get_hr(
                hr_filtered,
                fs=fs,
                min_hr=self.config.hr_min,
                max_hr=self.config.hr_max,
                method='peak'
            )
        except Exception as e:
            print(f"心率计算错误: {e}")
            hr_filtered = np.zeros_like(ppg_signal)
            hr_fft = 80.0
            hr_peak = 80.0

        # ===================
        # 呼吸频率计算
        # ===================
        try:
            # RR滤波 (0.1-0.5 Hz)
            rr_filtered = bandpass_filter(
                ppg_signal,
                lowcut=self.config.rr_lowcut,
                highcut=self.config.rr_highcut,
                fs=fs,
                order=self.config.rr_filter_order
            )

            # 使用FFT方法计算呼吸频率
            rr_fft = get_rr(
                rr_filtered,
                fs=fs,
                min_rr=self.config.rr_min,
                max_rr=self.config.rr_max,
                method='fft'
            )

            # 使用峰值检测方法计算呼吸频率
            rr_peak = get_rr(
                rr_filtered,
                fs=fs,
                min_rr=self.config.rr_min,
                max_rr=self.config.rr_max,
                method='peak'
            )
        except Exception as e:
            print(f"呼吸频率计算错误: {e}")
            rr_filtered = np.zeros_like(ppg_signal)
            rr_fft = 15.0
            rr_peak = 15.0

        # ===================
        # 血氧饱和度计算
        # ===================
        try:
            spo2 = get_spo2(
                ppg_ir,
                ppg_red,
                fs=fs,
                ring_type=self.config.spo2_ring_type,
                method=self.config.spo2_method
            )
        except Exception as e:
            print(f"血氧饱和度计算错误: {e}")
            spo2 = np.nan

        return {
            'hr_fft': hr_fft,
            'hr_peak': hr_peak,
            'rr_fft': rr_fft,
            'rr_peak': rr_peak,
            'spo2': spo2,
            'hr_filtered': hr_filtered,
            'rr_filtered': rr_filtered
        }


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
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


def analyze_vital_signs_performance(data_records, config):
    """分析生理信号计算性能"""
    print("\n=== 生理信号计算性能分析 ===")

    # 转换为DataFrame便于分析
    df = pd.DataFrame(data_records)

    print(f"总信号段数: {len(df)}")
    print(f"受试者数量: {df['subject_id'].nunique()}")

    # ===================
    # 数据诊断
    # ===================
    print(f"\n=== 数据诊断 ===")

    # 检查真实值的有效性
    hr_true_valid = df['hr_true'].notna().sum()
    rr_true_valid = df['rr_true'].notna().sum()
    spo2_true_valid = df['spo2_true'].notna().sum()

    print(f"有效真实心率数据: {hr_true_valid} / {len(df)} ({hr_true_valid / len(df) * 100:.1f}%)")
    print(f"有效真实呼吸频率数据: {rr_true_valid} / {len(df)} ({rr_true_valid / len(df) * 100:.1f}%)")
    print(f"有效真实血氧饱和度数据: {spo2_true_valid} / {len(df)} ({spo2_true_valid / len(df) * 100:.1f}%)")

    # 检查计算值的有效性
    hr_fft_valid = ((df['hr_fft'] != 80.0) & df['hr_fft'].notna()).sum()
    hr_peak_valid = ((df['hr_peak'] != 80.0) & df['hr_peak'].notna()).sum()
    rr_fft_valid = ((df['rr_fft'] != 15.0) & df['rr_fft'].notna()).sum()
    rr_peak_valid = ((df['rr_peak'] != 15.0) & df['rr_peak'].notna()).sum()
    spo2_calc_valid = df['spo2_calc'].notna().sum()

    print(f"有效FFT心率计算: {hr_fft_valid} / {len(df)} ({hr_fft_valid / len(df) * 100:.1f}%)")
    print(f"有效峰值心率计算: {hr_peak_valid} / {len(df)} ({hr_peak_valid / len(df) * 100:.1f}%)")
    print(f"有效FFT呼吸频率计算: {rr_fft_valid} / {len(df)} ({rr_fft_valid / len(df) * 100:.1f}%)")
    print(f"有效峰值呼吸频率计算: {rr_peak_valid} / {len(df)} ({rr_peak_valid / len(df) * 100:.1f}%)")
    print(f"有效血氧饱和度计算: {spo2_calc_valid} / {len(df)} ({spo2_calc_valid / len(df) * 100:.1f}%)")

    # 显示一些示例数据
    print(f"\n前5个样本的数据:")
    sample_cols = ['subject_id', 'hr_true', 'hr_fft', 'hr_peak', 'rr_true', 'rr_fft', 'rr_peak', 'spo2_true',
                   'spo2_calc']
    print(df[sample_cols].head().to_string())

    # 显示数据分布
    if hr_true_valid > 0:
        print(
            f"\n真实心率分布: 平均={df['hr_true'].mean():.1f}, 范围={df['hr_true'].min():.1f}-{df['hr_true'].max():.1f}")
    if rr_true_valid > 0:
        print(
            f"真实呼吸频率分布: 平均={df['rr_true'].mean():.1f}, 范围={df['rr_true'].min():.1f}-{df['rr_true'].max():.1f}")
    if spo2_true_valid > 0:
        print(
            f"真实血氧饱和度分布: 平均={df['spo2_true'].mean():.1f}, 范围={df['spo2_true'].min():.1f}-{df['spo2_true'].max():.1f}")

    # ===================
    # 心率分析
    # ===================
    print(f"\n=== 心率分析 ===")

    # FFT方法
    hr_fft_metrics = calculate_metrics(df['hr_true'].values, df['hr_fft'].values)
    print(f"心率 FFT方法:")
    if hr_fft_metrics['count'] > 0:
        print(f"  MAE: {hr_fft_metrics['mae']:.3f} bpm")
        print(f"  RMSE: {hr_fft_metrics['rmse']:.3f} bpm")
        print(f"  相关系数: {hr_fft_metrics['correlation']:.3f}")
    else:
        print(f"  无有效数据进行比较")
    print(f"  有效样本数: {hr_fft_metrics['count']}")

    # 峰值检测方法
    hr_peak_metrics = calculate_metrics(df['hr_true'].values, df['hr_peak'].values)
    print(f"\n心率 峰值检测方法:")
    if hr_peak_metrics['count'] > 0:
        print(f"  MAE: {hr_peak_metrics['mae']:.3f} bpm")
        print(f"  RMSE: {hr_peak_metrics['rmse']:.3f} bpm")
        print(f"  相关系数: {hr_peak_metrics['correlation']:.3f}")
    else:
        print(f"  无有效数据进行比较")
    print(f"  有效样本数: {hr_peak_metrics['count']}")

    # ===================
    # 呼吸频率分析
    # ===================
    print(f"\n=== 呼吸频率分析 ===")

    # FFT方法
    rr_fft_metrics = calculate_metrics(df['rr_true'].values, df['rr_fft'].values)
    print(f"呼吸频率 FFT方法:")
    if rr_fft_metrics['count'] > 0:
        print(f"  MAE: {rr_fft_metrics['mae']:.3f} rpm")
        print(f"  RMSE: {rr_fft_metrics['rmse']:.3f} rpm")
        print(f"  相关系数: {rr_fft_metrics['correlation']:.3f}")
    else:
        print(f"  无有效数据进行比较")
    print(f"  有效样本数: {rr_fft_metrics['count']}")

    # 峰值检测方法
    rr_peak_metrics = calculate_metrics(df['rr_true'].values, df['rr_peak'].values)
    print(f"\n呼吸频率 峰值检测方法:")
    if rr_peak_metrics['count'] > 0:
        print(f"  MAE: {rr_peak_metrics['mae']:.3f} rpm")
        print(f"  RMSE: {rr_peak_metrics['rmse']:.3f} rpm")
        print(f"  相关系数: {rr_peak_metrics['correlation']:.3f}")
    else:
        print(f"  无有效数据进行比较")
    print(f"  有效样本数: {rr_peak_metrics['count']}")

    # ===================
    # 血氧饱和度分析
    # ===================
    print(f"\n=== 血氧饱和度分析 ===")

    spo2_metrics = calculate_metrics(df['spo2_true'].values, df['spo2_calc'].values)
    print(f"血氧饱和度:")
    if spo2_metrics['count'] > 0:
        print(f"  MAE: {spo2_metrics['mae']:.3f} %")
        print(f"  RMSE: {spo2_metrics['rmse']:.3f} %")
        print(f"  相关系数: {spo2_metrics['correlation']:.3f}")
    else:
        print(f"  无有效数据进行比较")
    print(f"  有效样本数: {spo2_metrics['count']}")

    return {
        'hr_fft': hr_fft_metrics,
        'hr_peak': hr_peak_metrics,
        'rr_fft': rr_fft_metrics,
        'rr_peak': rr_peak_metrics,
        'spo2': spo2_metrics
    }


def plot_analysis_results(data_records, config):
    """绘制分析结果图表"""
    os.makedirs(config.results_dir, exist_ok=True)

    df = pd.DataFrame(data_records)

    # ===================
    # 1. 方法对比散点图
    # ===================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 心率 FFT vs 真实
    valid_hr_fft = ~(np.isnan(df['hr_true']) | np.isnan(df['hr_fft']))
    if valid_hr_fft.sum() > 0:
        axes[0, 0].scatter(df.loc[valid_hr_fft, 'hr_true'], df.loc[valid_hr_fft, 'hr_fft'], alpha=0.6, color='blue')
        min_hr = min(df.loc[valid_hr_fft, 'hr_true'].min(), df.loc[valid_hr_fft, 'hr_fft'].min())
        max_hr = max(df.loc[valid_hr_fft, 'hr_true'].max(), df.loc[valid_hr_fft, 'hr_fft'].max())
        axes[0, 0].plot([min_hr, max_hr], [min_hr, max_hr], 'r--', lw=2)
        axes[0, 0].set_xlabel('真实心率 (bpm)')
        axes[0, 0].set_ylabel('FFT计算心率 (bpm)')
        axes[0, 0].set_title('心率 FFT方法 vs 真实值')
        axes[0, 0].grid(True)

    # 心率 峰值 vs 真实
    valid_hr_peak = ~(np.isnan(df['hr_true']) | np.isnan(df['hr_peak']))
    if valid_hr_peak.sum() > 0:
        axes[0, 1].scatter(df.loc[valid_hr_peak, 'hr_true'], df.loc[valid_hr_peak, 'hr_peak'], alpha=0.6, color='red')
        min_hr = min(df.loc[valid_hr_peak, 'hr_true'].min(), df.loc[valid_hr_peak, 'hr_peak'].min())
        max_hr = max(df.loc[valid_hr_peak, 'hr_true'].max(), df.loc[valid_hr_peak, 'hr_peak'].max())
        axes[0, 1].plot([min_hr, max_hr], [min_hr, max_hr], 'r--', lw=2)
        axes[0, 1].set_xlabel('真实心率 (bpm)')
        axes[0, 1].set_ylabel('峰值检测心率 (bpm)')
        axes[0, 1].set_title('心率 峰值检测 vs 真实值')
        axes[0, 1].grid(True)

    # FFT vs 峰值心率对比
    valid_hr_both = ~(np.isnan(df['hr_fft']) | np.isnan(df['hr_peak']))
    if valid_hr_both.sum() > 0:
        axes[0, 2].scatter(df.loc[valid_hr_both, 'hr_fft'], df.loc[valid_hr_both, 'hr_peak'], alpha=0.6, color='purple')
        min_hr = min(df.loc[valid_hr_both, 'hr_fft'].min(), df.loc[valid_hr_both, 'hr_peak'].min())
        max_hr = max(df.loc[valid_hr_both, 'hr_fft'].max(), df.loc[valid_hr_both, 'hr_peak'].max())
        axes[0, 2].plot([min_hr, max_hr], [min_hr, max_hr], 'r--', lw=2)
        axes[0, 2].set_xlabel('FFT心率 (bpm)')
        axes[0, 2].set_ylabel('峰值检测心率 (bpm)')
        axes[0, 2].set_title('FFT vs 峰值检测心率对比')
        axes[0, 2].grid(True)

    # 呼吸频率 FFT vs 真实
    valid_rr_fft = ~(np.isnan(df['rr_true']) | np.isnan(df['rr_fft']))
    if valid_rr_fft.sum() > 0:
        axes[1, 0].scatter(df.loc[valid_rr_fft, 'rr_true'], df.loc[valid_rr_fft, 'rr_fft'], alpha=0.6, color='green')
        min_rr = min(df.loc[valid_rr_fft, 'rr_true'].min(), df.loc[valid_rr_fft, 'rr_fft'].min())
        max_rr = max(df.loc[valid_rr_fft, 'rr_true'].max(), df.loc[valid_rr_fft, 'rr_fft'].max())
        axes[1, 0].plot([min_rr, max_rr], [min_rr, max_rr], 'r--', lw=2)
        axes[1, 0].set_xlabel('真实呼吸频率 (rpm)')
        axes[1, 0].set_ylabel('FFT计算呼吸频率 (rpm)')
        axes[1, 0].set_title('呼吸频率 FFT方法 vs 真实值')
        axes[1, 0].grid(True)

    # 呼吸频率 峰值 vs 真实
    valid_rr_peak = ~(np.isnan(df['rr_true']) | np.isnan(df['rr_peak']))
    if valid_rr_peak.sum() > 0:
        axes[1, 1].scatter(df.loc[valid_rr_peak, 'rr_true'], df.loc[valid_rr_peak, 'rr_peak'], alpha=0.6,
                           color='orange')
        min_rr = min(df.loc[valid_rr_peak, 'rr_true'].min(), df.loc[valid_rr_peak, 'rr_peak'].min())
        max_rr = max(df.loc[valid_rr_peak, 'rr_true'].max(), df.loc[valid_rr_peak, 'rr_peak'].max())
        axes[1, 1].plot([min_rr, max_rr], [min_rr, max_rr], 'r--', lw=2)
        axes[1, 1].set_xlabel('真实呼吸频率 (rpm)')
        axes[1, 1].set_ylabel('峰值检测呼吸频率 (rpm)')
        axes[1, 1].set_title('呼吸频率 峰值检测 vs 真实值')
        axes[1, 1].grid(True)

    # 血氧饱和度 vs 真实
    valid_spo2 = ~(np.isnan(df['spo2_true']) | np.isnan(df['spo2_calc']))
    if valid_spo2.sum() > 0:
        axes[1, 2].scatter(df.loc[valid_spo2, 'spo2_true'], df.loc[valid_spo2, 'spo2_calc'], alpha=0.6, color='cyan')
        min_spo2 = min(df.loc[valid_spo2, 'spo2_true'].min(), df.loc[valid_spo2, 'spo2_calc'].min())
        max_spo2 = max(df.loc[valid_spo2, 'spo2_true'].max(), df.loc[valid_spo2, 'spo2_calc'].max())
        axes[1, 2].plot([min_spo2, max_spo2], [min_spo2, max_spo2], 'r--', lw=2)
        axes[1, 2].set_xlabel('真实血氧饱和度 (%)')
        axes[1, 2].set_ylabel('计算血氧饱和度 (%)')
        axes[1, 2].set_title('血氧饱和度计算 vs 真实值')
        axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'vital_signs_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # ===================
    # 2. 误差分布图
    # ===================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 心率误差分布
    if valid_hr_fft.sum() > 0:
        hr_fft_error = df.loc[valid_hr_fft, 'hr_fft'] - df.loc[valid_hr_fft, 'hr_true']
        axes[0, 0].hist(hr_fft_error, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('误差 (FFT - 真实) bpm')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].set_title('心率 FFT方法误差分布')
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].grid(True)

    if valid_hr_peak.sum() > 0:
        hr_peak_error = df.loc[valid_hr_peak, 'hr_peak'] - df.loc[valid_hr_peak, 'hr_true']
        axes[0, 1].hist(hr_peak_error, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_xlabel('误差 (峰值 - 真实) bpm')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('心率 峰值检测误差分布')
        axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].grid(True)

    # 心率方法对比
    if valid_hr_both.sum() > 0:
        hr_method_diff = df.loc[valid_hr_both, 'hr_fft'] - df.loc[valid_hr_both, 'hr_peak']
        axes[0, 2].hist(hr_method_diff, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 2].set_xlabel('差值 (FFT - 峰值) bpm')
        axes[0, 2].set_ylabel('频次')
        axes[0, 2].set_title('心率方法差异分布')
        axes[0, 2].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 2].grid(True)

    # 呼吸频率误差分布
    if valid_rr_fft.sum() > 0:
        rr_fft_error = df.loc[valid_rr_fft, 'rr_fft'] - df.loc[valid_rr_fft, 'rr_true']
        axes[1, 0].hist(rr_fft_error, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('误差 (FFT - 真实) rpm')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('呼吸频率 FFT方法误差分布')
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].grid(True)

    if valid_rr_peak.sum() > 0:
        rr_peak_error = df.loc[valid_rr_peak, 'rr_peak'] - df.loc[valid_rr_peak, 'rr_true']
        axes[1, 1].hist(rr_peak_error, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('误差 (峰值 - 真实) rpm')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('呼吸频率 峰值检测误差分布')
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].grid(True)

    # 血氧饱和度误差分布
    if valid_spo2.sum() > 0:
        spo2_error = df.loc[valid_spo2, 'spo2_calc'] - df.loc[valid_spo2, 'spo2_true']
        axes[1, 2].hist(spo2_error, bins=30, alpha=0.7, color='cyan', edgecolor='black')
        axes[1, 2].set_xlabel('误差 (计算 - 真实) %')
        axes[1, 2].set_ylabel('频次')
        axes[1, 2].set_title('血氧饱和度误差分布')
        axes[1, 2].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'error_distributions.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_signal_examples(data_records, config, num_examples=6):
    """绘制信号处理示例"""
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 3 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for i in range(min(num_examples, len(data_records))):
        record = data_records[i]

        time_axis = np.arange(len(record['ppg_signal'])) / config.sampling_rate

        # 原始PPG信号
        axes[i, 0].plot(time_axis, record['ppg_signal'], 'b-', linewidth=1)
        axes[i, 0].set_title(f'受试者 {record["subject_id"]} - 原始PPG信号')
        axes[i, 0].set_xlabel('时间 (秒)')
        axes[i, 0].set_ylabel('幅值')
        axes[i, 0].grid(True)

        # HR滤波信号
        axes[i, 1].plot(time_axis, record['hr_filtered'], 'r-', linewidth=1)
        axes[i, 1].set_title(
            f'HR滤波信号 (0.5-3Hz)\nFFT: {record["hr_fft"]:.1f}, 峰值: {record["hr_peak"]:.1f}, 真实: {record["hr_true"]:.1f}')
        axes[i, 1].set_xlabel('时间 (秒)')
        axes[i, 1].set_ylabel('幅值')
        axes[i, 1].grid(True)

        # RR滤波信号
        axes[i, 2].plot(time_axis, record['rr_filtered'], 'g-', linewidth=1)
        axes[i, 2].set_title(
            f'RR滤波信号 (0.1-0.5Hz)\nFFT: {record["rr_fft"]:.1f}, 峰值: {record["rr_peak"]:.1f}, 真实: {record["rr_true"]:.1f}')
        axes[i, 2].set_xlabel('时间 (秒)')
        axes[i, 2].set_ylabel('幅值')
        axes[i, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'signal_processing_examples.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def save_results(data_records, metrics, config):
    """保存分析结果"""
    os.makedirs(config.results_dir, exist_ok=True)

    # 保存详细结果
    df = pd.DataFrame(data_records)

    # 选择要保存的列
    save_columns = [
        'subject_id', 'start_time', 'end_time', 'age', 'gender',
        'hr_true', 'hr_fft', 'hr_peak',
        'rr_true', 'rr_fft', 'rr_peak',
        'spo2_true', 'spo2_calc'
    ]

    df_save = df[save_columns].copy()
    output_file = os.path.join(config.results_dir, 'bidmc_vital_signs_results.csv')
    df_save.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"详细结果已保存到: {output_file}")

    # 保存统计摘要
    summary_file = os.path.join(config.results_dir, 'bidmc_analysis_summary.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("BIDMC数据集生理信号计算分析结果摘要\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"总信号段数: {len(df)}\n")
        f.write(f"受试者数量: {df['subject_id'].nunique()}\n")
        f.write(f"采样频率: {config.sampling_rate} Hz\n")
        f.write(f"信号段长度: {config.ppg_length / config.sampling_rate:.1f} 秒\n\n")

        f.write("生理参数计算性能:\n")
        f.write("-" * 30 + "\n")

        for param, metric in metrics.items():
            param_name = {
                'hr_fft': '心率 (FFT方法)',
                'hr_peak': '心率 (峰值方法)',
                'rr_fft': '呼吸频率 (FFT方法)',
                'rr_peak': '呼吸频率 (峰值方法)',
                'spo2': '血氧饱和度'
            }.get(param, param)

            f.write(f"{param_name}:\n")
            f.write(f"  MAE: {metric['mae']:.3f}\n")
            f.write(f"  RMSE: {metric['rmse']:.3f}\n")
            f.write(f"  相关系数: {metric['correlation']:.3f}\n")
            f.write(f"  有效样本数: {metric['count']}\n\n")

    print(f"分析摘要已保存到: {summary_file}")


def main():
    """主函数"""
    config = BIDMCConfig()
    print("BIDMC数据集生理信号计算分析工具")
    print("=" * 50)
    print(f"数据目录: {config.bidmc_dir}")
    print(f"CSV目录: {config.csv_dir}")
    print(f"结果目录: {config.results_dir}")
    print(f"PPG采样频率: {config.sampling_rate} Hz")
    print(f"信号段长度: {config.ppg_length / config.sampling_rate:.1f} 秒 ({config.ppg_length} 个采样点)")
    print(f"窗口重叠: {config.window_overlap * 100:.0f}%")
    print(f"心率计算: {config.hr_lowcut}-{config.hr_highcut}Hz滤波, {config.hr_min}-{config.hr_max} bpm")
    print(f"呼吸频率计算: {config.rr_lowcut}-{config.rr_highcut}Hz滤波, {config.rr_min}-{config.rr_max} rpm")
    print(f"血氧饱和度计算: {config.spo2_ring_type}类型传感器")

    try:
        # 加载和处理数据
        processor = BIDMCDataProcessor(config)
        data_records = processor.load_bidmc_data()

        if len(data_records) == 0:
            print("未找到有效数据，请检查数据路径和文件格式")
            return

        # 分析生理信号计算性能
        metrics = analyze_vital_signs_performance(data_records, config)

        # 绘制分析结果
        print("\n正在生成分析图表...")
        plot_analysis_results(data_records, config)

        # 绘制信号处理示例
        print("正在生成信号处理示例...")
        plot_signal_examples(data_records, config, num_examples=6)

        # 保存结果
        print("正在保存结果...")
        save_results(data_records, metrics, config)

        print("\n=== 分析完成! ===")
        print(f"结果保存在: {config.results_dir}")

    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()