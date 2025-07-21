#!/usr/bin/env python3
"""
VitalDB最终处理器 - 修正版本
使用官方vitaldb包读取和处理PPG信号数据，兼容现有评估脚本
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import resample, butter, filtfilt
import warnings

warnings.filterwarnings('ignore')

# 首先需要安装vitaldb库
# pip install vitaldb

try:
    import vitaldb

    print("✅ vitaldb库已加载")
except ImportError:
    print("❌ 未安装vitaldb库，请运行: pip install vitaldb")
    print("如果安装失败，可以继续使用合成数据")
    vitaldb = None


class VitalDBFinalProcessor:
    def __init__(self, vitaldb_path, output_path="../data"):
        """
        初始化VitalDB处理器 - 最终版本
        """
        self.vitaldb_base_path = Path(vitaldb_path)
        self.vitaldb_path = self.vitaldb_base_path / "vital_files"
        self.output_path = Path(output_path)
        self.vital_dir = self.output_path / "vital"
        self.vitalppg_dir = self.output_path / "vitaldbppg"

        # 创建输出目录
        self.vital_dir.mkdir(parents=True, exist_ok=True)
        self.vitalppg_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔍 VitalDB最终处理器:")
        print(f"   VitalDB基础路径: {self.vitaldb_base_path}")
        print(f"   .vital文件路径: {self.vitaldb_path}")
        print(f"   输出路径: {self.output_path}")

    def check_and_use_existing_data(self):
        """检查是否已有处理好的数据"""
        print(f"\n🔍 检查现有数据...")

        # 检查是否已有VitalDB官方库生成的数据
        vitaldb_official_path = self.vital_dir / "train_clean_vitaldb_official.csv"
        if vitaldb_official_path.exists():
            try:
                df = pd.read_csv(vitaldb_official_path)
                print(f"✅ 发现VitalDB官方数据: {len(df)} 个样本")

                # 验证信号文件是否存在
                signal_files = list(self.vitalppg_dir.glob("vitaldb_*.npy"))
                print(f"✅ 发现信号文件: {len(signal_files)} 个")

                if len(signal_files) > 0:
                    print(f"📊 使用现有的VitalDB真实数据")
                    return self.prepare_final_dataset(df, "vitaldb_official")

            except Exception as e:
                print(f"⚠️ 读取VitalDB官方数据失败: {e}")

        # 检查是否有合成数据
        synthetic_path = self.vital_dir / "train_clean.csv"
        if synthetic_path.exists():
            try:
                df = pd.read_csv(synthetic_path)
                print(f"✅ 发现合成数据: {len(df)} 个样本")

                signal_files = list(self.vitalppg_dir.glob("case_*.npy"))
                print(f"✅ 发现信号文件: {len(signal_files)} 个")

                if len(signal_files) > 0:
                    print(f"📊 使用现有的合成数据")
                    return self.prepare_final_dataset(df, "synthetic")

            except Exception as e:
                print(f"⚠️ 读取合成数据失败: {e}")

        print(f"❌ 未找到可用的现有数据")
        return None

    def process_vitaldb_with_official_library(self, max_files=20, samples_per_case=3):
        """使用官方库处理VitalDB数据"""
        if vitaldb is None:
            print("❌ vitaldb库未安装，跳过真实数据处理")
            return None

        print(f"\n🚀 使用官方库处理VitalDB数据...")

        # 检查文件
        vital_files = list(self.vitaldb_path.glob("*.vital"))
        if not vital_files:
            print("❌ 未找到.vital文件")
            return None

        print(f"✅ 找到 {len(vital_files)} 个.vital文件")

        all_segments = []
        files_to_process = vital_files[:max_files]

        for i, vital_file in enumerate(files_to_process):
            print(f"\n处理文件 {i + 1}/{len(files_to_process)}: {vital_file.stem}")

            try:
                # 使用官方VitalFile读取
                vf = vitaldb.VitalFile(str(vital_file))
                track_names = vf.get_track_names()

                # 查找PPG轨道
                ppg_tracks = [t for t in track_names if
                              any(keyword in t.upper() for keyword in ['PLETH', 'PPG', 'SPO2'])]

                if not ppg_tracks:
                    print(f"   ❌ 未找到PPG轨道")
                    continue

                case_segments = 0

                # 处理每个PPG轨道
                for track_name in ppg_tracks[:2]:  # 限制每个文件最多处理2个轨道
                    try:
                        print(f"   💓 处理轨道: {track_name}")

                        # 读取数据，125Hz采样率
                        data = vf.to_numpy([track_name], interval=1 / 125)

                        if data is not None and len(data) > 0:
                            ppg_signal = data[:, 0]
                            ppg_signal = ppg_signal[~np.isnan(ppg_signal)]

                            if len(ppg_signal) > 1250:  # 至少10秒数据
                                segments = self.process_ppg_signal(
                                    ppg_signal,
                                    vital_file.stem,
                                    track_name,
                                    max_segments=samples_per_case
                                )
                                all_segments.extend(segments)
                                case_segments += len(segments)
                                print(f"      ✅ 提取 {len(segments)} 个信号段")
                            else:
                                print(f"      ❌ 数据太短: {len(ppg_signal)} 点")
                        else:
                            print(f"      ❌ 读取数据失败")

                    except Exception as e:
                        print(f"      ❌ 轨道处理失败: {e}")
                        continue

                print(f"   📊 文件总计: {case_segments} 个信号段")

            except Exception as e:
                print(f"   ❌ 文件处理失败: {e}")
                continue

        if all_segments:
            df_real = pd.DataFrame(all_segments)
            real_path = self.vital_dir / "train_clean_vitaldb_official.csv"
            df_real.to_csv(real_path, index=False)

            print(f"\n✅ VitalDB真实数据处理完成:")
            print(f"   - 处理文件数: {len(files_to_process)}")
            print(f"   - 总信号段数: {len(all_segments)}")
            print(f"   - 保存路径: {real_path}")

            return self.prepare_final_dataset(df_real, "vitaldb_official")
        else:
            print("\n❌ 未能提取任何真实数据")
            return None

    def create_enhanced_synthetic_data(self, num_cases=150):
        """创建增强的合成数据"""
        print(f"\n📊 创建增强的合成数据...")

        # 加载临床数据
        clinical_path = self.vitaldb_base_path / "clinical_data.csv"
        if not clinical_path.exists():
            print("⚠️ 未找到clinical_data.csv，使用默认参数生成数据")
            return self.create_default_synthetic_data(num_cases)

        try:
            clinical_df = pd.read_csv(clinical_path)
            print(f"✅ 加载临床数据: {len(clinical_df)} 个病例")

            # 选择前num_cases个病例
            sample_cases = clinical_df.head(num_cases)
            training_data = []

            for idx, row in sample_cases.iterrows():
                case_id = str(int(row['caseid'])).zfill(4)

                # 为每个病例创建3-5个信号段
                num_segments = np.random.randint(3, 6)

                for seg_idx in range(num_segments):
                    # 基于临床数据生成生理指标
                    bmi = row.get('bmi', 25.0)
                    if pd.isna(bmi):
                        bmi = 25.0

                    age = row.get('age', 50.0)
                    if pd.isna(age) or isinstance(age, str):
                        try:
                            age = float(age) if age != '' else 50.0
                        except (ValueError, TypeError):
                            age = 50.0

                    # 生成更真实的生理指标
                    svri = self.generate_realistic_svri(bmi, age)
                    skewness = self.generate_realistic_skewness(age)
                    ipa = self.generate_realistic_ipa(bmi, age)

                    # 生成合成PPG信号
                    segment_filename = f"case_{case_id}_seg_{seg_idx + 1:02d}.npy"
                    self.create_high_quality_ppg_signal(
                        self.vitalppg_dir / segment_filename,
                        svri, skewness, ipa, age, bmi
                    )

                    training_data.append({
                        'caseid': case_id,
                        'segments': segment_filename,
                        'svri': svri,
                        'skewness': skewness,
                        'ipa': ipa
                    })

            if training_data:
                df_synthetic = pd.DataFrame(training_data)
                synthetic_path = self.vital_dir / "train_clean.csv"
                df_synthetic.to_csv(synthetic_path, index=False)

                print(f"✅ 创建增强合成数据:")
                print(f"   - 病例数: {len(sample_cases)}")
                print(f"   - 信号段数: {len(training_data)}")
                print(f"   - 文件路径: {synthetic_path}")

                return self.prepare_final_dataset(df_synthetic, "synthetic")
            else:
                print("❌ 未能创建合成训练数据")
                return None

        except Exception as e:
            print(f"❌ 处理临床数据失败: {e}")
            return self.create_default_synthetic_data(num_cases)

    def create_default_synthetic_data(self, num_cases=100):
        """创建默认合成数据"""
        print(f"📊 创建默认合成数据 ({num_cases}个病例)...")

        training_data = []

        for case_idx in range(num_cases):
            case_id = f"{case_idx + 1:04d}"

            # 生成随机的患者特征
            age = np.random.normal(55, 15)
            age = np.clip(age, 18, 90)

            bmi = np.random.normal(25, 4)
            bmi = np.clip(bmi, 18, 40)

            num_segments = np.random.randint(3, 6)

            for seg_idx in range(num_segments):
                # 生成生理指标
                svri = self.generate_realistic_svri(bmi, age)
                skewness = self.generate_realistic_skewness(age)
                ipa = self.generate_realistic_ipa(bmi, age)

                # 生成信号
                segment_filename = f"case_{case_id}_seg_{seg_idx + 1:02d}.npy"
                self.create_high_quality_ppg_signal(
                    self.vitalppg_dir / segment_filename,
                    svri, skewness, ipa, age, bmi
                )

                training_data.append({
                    'caseid': case_id,
                    'segments': segment_filename,
                    'svri': svri,
                    'skewness': skewness,
                    'ipa': ipa
                })

        if training_data:
            df_synthetic = pd.DataFrame(training_data)
            synthetic_path = self.vital_dir / "train_clean.csv"
            df_synthetic.to_csv(synthetic_path, index=False)

            print(f"✅ 创建默认合成数据:")
            print(f"   - 病例数: {num_cases}")
            print(f"   - 信号段数: {len(training_data)}")

            return self.prepare_final_dataset(df_synthetic, "synthetic")

        return None

    def generate_realistic_svri(self, bmi, age):
        """生成真实的SVRI值"""
        # 基础SVRI受BMI和年龄影响
        base_svri = 0.8 + (bmi - 25) * 0.02 + (age - 50) * 0.005
        svri = base_svri + np.random.normal(0, 0.15)
        return np.clip(svri, 0.3, 1.8)

    def generate_realistic_skewness(self, age):
        """生成真实的偏度值"""
        # 年龄影响信号偏度
        base_skewness = (age - 50) * 0.015
        skewness = base_skewness + np.random.normal(0, 0.4)
        return np.clip(skewness, -2.5, 2.5)

    def generate_realistic_ipa(self, bmi, age):
        """生成真实的IPA值"""
        # IPA受多种因素影响
        base_ipa = (bmi - 25) * 0.1 + (age - 50) * 0.02
        ipa = base_ipa + np.random.normal(0, 2.5)
        return np.clip(ipa, -8, 8)

    def process_ppg_signal(self, ppg_signal, case_id, track_name, max_segments=3):
        """处理PPG信号数据"""
        segments = []

        target_fs = 125
        target_length = 1250  # 10秒

        # 信号预处理
        ppg_clean = self.preprocess_ppg_signal(ppg_signal)

        if len(ppg_clean) < target_length:
            return segments

        # 分割成多个段
        num_segments = min(max_segments, len(ppg_clean) // target_length)

        for i in range(num_segments):
            start_idx = i * target_length
            end_idx = start_idx + target_length

            segment = ppg_clean[start_idx:end_idx]

            # 检查信号质量
            if self.is_good_quality_segment(segment):
                # 归一化
                segment_norm = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)

                # 计算生理指标
                svri = self.calculate_svri_from_signal(segment_norm)
                skewness = self.calculate_skewness_from_signal(segment_norm)
                ipa = self.calculate_ipa_from_signal(segment_norm)

                # 保存信号段
                safe_track_name = track_name.replace('/', '_').replace('\\', '_')
                segment_filename = f"vitaldb_{case_id}_{safe_track_name}_seg_{i + 1:02d}.npy"
                segment_path = self.vitalppg_dir / segment_filename
                np.save(segment_path, segment_norm.astype(np.float32))

                segments.append({
                    'caseid': case_id,
                    'segments': segment_filename,
                    'svri': svri,
                    'skewness': skewness,
                    'ipa': ipa
                })

        return segments

    def preprocess_ppg_signal(self, signal):
        """PPG信号预处理"""
        # 移除异常值
        q1, q3 = np.percentile(signal, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        signal_clean = signal.copy()
        median_val = np.median(signal)
        signal_clean[(signal < lower_bound) | (signal > upper_bound)] = median_val

        # 带通滤波
        try:
            fs = 125
            low = 0.5 / (fs / 2)
            high = 8.0 / (fs / 2)
            b, a = butter(4, [low, high], btype='band')
            signal_filtered = filtfilt(b, a, signal_clean)
            return signal_filtered
        except:
            return signal_clean

    def is_good_quality_segment(self, segment):
        """检查信号段质量"""
        if len(segment) == 0:
            return False

        signal_range = np.max(segment) - np.min(segment)
        signal_std = np.std(segment)
        is_finite = np.all(np.isfinite(segment))
        is_variable = signal_range > 1e-6 and signal_std > 1e-6

        return is_finite and is_variable

    def calculate_svri_from_signal(self, signal):
        """从信号计算SVRI"""
        signal_std = np.std(signal)
        signal_mean_abs = np.mean(np.abs(signal))
        svri = signal_std / (signal_mean_abs + 1e-8)
        return np.clip(svri, 0.1, 2.0)

    def calculate_skewness_from_signal(self, signal):
        """从信号计算偏度"""
        if len(signal) == 0:
            return 0
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0
        skewness = np.mean(((signal - mean) / std) ** 3)
        return np.clip(skewness, -3, 3)

    def calculate_ipa_from_signal(self, signal):
        """从信号计算IPA"""
        peak_to_peak = np.max(signal) - np.min(signal)
        return np.clip(peak_to_peak, -8, 8)

    def create_high_quality_ppg_signal(self, output_path, svri, skewness, ipa, age, bmi, duration=10, fs=125):
        """创建高质量PPG信号"""
        length = fs * duration
        t = np.linspace(0, duration, length)

        # 基于年龄和BMI的心率
        base_hr = max(60, 100 - (age - 30) * 0.5 - (bmi - 25) * 0.3) + np.random.normal(0, 5)
        base_hr = np.clip(base_hr, 50, 120)
        freq = base_hr / 60

        # 生成复杂的PPG波形
        ppg = np.zeros(length)

        for i in range(length):
            phase = (t[i] * freq) % 1

            # 主脉冲波 - 基于SVRI调整形状
            if phase < 0.3:  # 收缩期
                amplitude = 1.0 + (svri - 1.0) * 0.3
                ppg[i] = amplitude * (np.sin(phase * np.pi / 0.3) ** 2)
            elif phase < 0.6:  # 舒张早期
                decay_rate = 0.1 + (svri - 1.0) * 0.05
                ppg[i] = 0.3 * np.exp(-(phase - 0.3) / decay_rate)
            else:  # 舒张晚期
                dicrotic_amplitude = 0.1 + abs(skewness) * 0.05
                ppg[i] = dicrotic_amplitude * np.sin((phase - 0.6) * np.pi / 0.4)

        # 基于IPA添加噪声和变异性
        noise_level = 0.02 + abs(ipa) * 0.005
        noise = np.random.normal(0, noise_level, length)

        # 基线漂移
        baseline_drift = 0.1 * np.sin(2 * np.pi * 0.05 * t)

        # 呼吸调制
        respiratory_freq = np.random.normal(15, 3) / 60  # 呼吸频率
        respiratory_modulation = 0.05 * np.sin(2 * np.pi * respiratory_freq * t)

        ppg_final = ppg + noise + baseline_drift + respiratory_modulation

        # 归一化
        ppg_final = (ppg_final - np.mean(ppg_final)) / (np.std(ppg_final) + 1e-8)

        # 保存
        np.save(output_path, ppg_final.astype(np.float32))

    def prepare_final_dataset(self, df, data_type):
        """准备最终数据集，确保与评估脚本兼容"""
        print(f"\n📋 准备最终数据集 ({data_type})...")

        # 确保数据格式正确
        df_final = df.copy()

        # 添加必要的列（如果不存在）
        if 'case_id' not in df_final.columns:
            df_final['case_id'] = 'vitaldbppg'
        if 'path' not in df_final.columns:
            df_final['path'] = '../data/'
        if 'fs' not in df_final.columns:
            df_final['fs'] = 125

        # 数据范围检查和清理
        original_count = len(df_final)

        # SVRI范围过滤
        df_final = df_final[(df_final['svri'] > 0) & (df_final['svri'] < 2)]

        # IPA范围过滤
        df_final = df_final[(df_final['ipa'] > -10) & (df_final['ipa'] < 10)]

        # 偏度范围过滤
        df_final = df_final[(df_final['skewness'] > -3) & (df_final['skewness'] < 3)]

        print(f"✅ 数据清理完成:")
        print(f"   - 原始样本数: {original_count}")
        print(f"   - 清理后样本数: {len(df_final)}")
        print(f"   - 保留率: {len(df_final) / original_count * 100:.1f}%")

        # 验证信号文件存在性
        signal_dir = self.vitalppg_dir
        actual_files = set(os.listdir(signal_dir))

        valid_rows = []
        for _, row in df_final.iterrows():
            if row['segments'] in actual_files:
                valid_rows.append(row)

        df_final = pd.DataFrame(valid_rows)

        print(f"✅ 文件验证完成:")
        print(f"   - 有效样本数: {len(df_final)}")
        print(f"   - 信号文件数: {len(actual_files)}")

        # 保存最终数据集
        final_path = self.vital_dir / "train_clean.csv"
        df_final.to_csv(final_path, index=False)

        # 显示数据统计
        print(f"\n📈 最终数据统计:")
        print(f"   - SVRI: {df_final['svri'].min():.3f} - {df_final['svri'].max():.3f}")
        print(f"   - 偏度: {df_final['skewness'].min():.3f} - {df_final['skewness'].max():.3f}")
        print(f"   - IPA: {df_final['ipa'].min():.3f} - {df_final['ipa'].max():.3f}")
        print(f"   - 最终文件: {final_path}")

        return df_final

    def run_complete_processing(self):
        """运行完整处理流程"""
        print("🚀 开始完整数据处理流程")
        print("=" * 60)

        # 1. 检查现有数据
        existing_data = self.check_and_use_existing_data()
        if existing_data is not None:
            print(f"\n✅ 使用现有数据，处理完成!")
            return existing_data

        # 2. 尝试处理真实VitalDB数据
        if vitaldb is not None and self.vitaldb_path.exists():
            print(f"\n🔍 尝试处理真实VitalDB数据...")
            real_data = self.process_vitaldb_with_official_library(max_files=15, samples_per_case=3)
            if real_data is not None:
                print(f"\n✅ 真实VitalDB数据处理完成!")
                return real_data

        # 3. 创建增强合成数据
        print(f"\n📊 创建增强合成数据作为备选...")
        synthetic_data = self.create_enhanced_synthetic_data(num_cases=200)
        if synthetic_data is not None:
            print(f"\n✅ 增强合成数据创建完成!")
            return synthetic_data

        print(f"\n❌ 所有数据处理方法都失败了")
        return None


def main():
    print("🚀 VitalDB最终处理器 - 修正版本")
    print("=" * 60)
    print("智能数据处理：真实数据优先，合成数据备选")
    print("=" * 60)

    # VitalDB数据集路径
    vitaldb_path = r"E:\thsiu-ppg\vitaldb-a-high-fidelity-multi-parameter-vital-signs-database-in-surgical-patients-1.0.0"

    # 创建处理器
    processor = VitalDBFinalProcessor(vitaldb_path)

    # 运行完整处理流程
    result = processor.run_complete_processing()

    if result is not None:
        print(f"\n🎉 数据处理成功完成!")
        print(f"📁 训练数据: ../data/vital/train_clean.csv")
        print(f"📁 信号数据: ../data/vitaldbppg/")
        print(f"\n🚀 下一步:")
        print(f"   1. 运行训练脚本进行模型训练")
        print(f"   2. 运行评估脚本查看模型性能")
        print(f"   3. 数据已兼容现有的评估代码")
    else:
        print(f"\n😞 数据处理失败")
        print(f"请检查：")
        print(f"   1. VitalDB数据路径是否正确")
        print(f"   2. 是否已安装vitaldb库")
        print(f"   3. 磁盘空间是否充足")


if __name__ == "__main__":
    main()