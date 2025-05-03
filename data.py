# data_generation.py
import numpy as np
import h5py
import pywt


# ---------------------------- 信号生成函数 ----------------------------
def generate_fhss_signal(num_samples, sample_rate, freq_set, hop_seq, snr_db):
    """生成跳频信号（含小波去噪）"""
    t = np.arange(num_samples) / sample_rate
    signal = np.zeros(num_samples, dtype=np.complex64)
    noise_power = 10 ** (-snr_db / 10)
    hop_duration = num_samples // len(hop_seq)

    # 生成跳频信号
    for i, freq_idx in enumerate(hop_seq):
        start = i * hop_duration
        end = (i + 1) * hop_duration
        freq = freq_set[freq_idx]
        signal[start:end] = np.exp(1j * 2 * np.pi * freq * t[start:end])

    # 添加噪声
    noise = np.random.normal(0, np.sqrt(noise_power), num_samples) + 1j * np.random.normal(0, np.sqrt(noise_power),
                                                                                           num_samples)
    noisy_signal = signal + noise

    # 小波去噪（复数信号分虚实处理）
    def denoise(s):
        real = pywt.wavedec(s.real, 'db4', level=3)
        imag = pywt.wavedec(s.imag, 'db4', level=3)
        real = [pywt.threshold(c, np.std(c) * 0.5, 'soft') for c in real]
        imag = [pywt.threshold(c, np.std(c) * 0.5, 'soft') for c in imag]
        return pywt.waverec(real, 'db4') + 1j * pywt.waverec(imag, 'db4')

    return denoise(noisy_signal)


def generate_fixed_frequency_signal(num_samples, sample_rate, freq, snr_db):
    """生成定频信号"""
    t = np.arange(num_samples) / sample_rate
    clean_signal = np.cos(2 * np.pi * freq * t)
    signal_power = np.mean(np.abs(clean_signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), num_samples)
    return clean_signal + noise


# ---------------------------- 数据标注函数 ----------------------------
def annotate_dataset(signals, sample_rate, freq_sets, hop_seqs):
    """标注数据集（强制统一信号长度）"""
    annotations = []
    for idx, signal in enumerate(signals):
        config_idx = idx % len(freq_sets)
        freq_set = freq_sets[config_idx]
        hop_seq = hop_seqs[config_idx]
        hop_duration = 100  # 单跳时长（示例值）
        total_length = hop_duration * len(hop_seq)

        # 截断信号至统一长度
        signal = signal[:total_length]
        for i, freq_idx in enumerate(hop_seq):
            start = i * hop_duration
            end = (i + 1) * hop_duration
            annotation = {
                'label': 0 if idx < 5 else 1,  # 预设标签：前5个为类0（跳频），后5个为类1（定频）
                'signal_segment': signal[start:end],
                'hop_seq': hop_seq,
                'freq_set': freq_set
            }
            annotations.append(annotation)
    return annotations


# ---------------------------- 主程序 ----------------------------
if __name__ == "__main__":
    np.random.seed(42)
    num_samples = 100000  # 单信号长度
    sample_rate = 10e6  # 10MHz采样率
    num_signals = 10  # 总信号数量
    snr_range = (5, 15)  # 信噪比范围

    # --- 频点配置生成 ---
    freq_configs = [
        {"base": 2.4e9, "step": 0.1e6, "num": 1000},  # 2.4-2.5GHz
        {"base": 3.5e9, "step": 0.2e6, "num": 750},  # 3.5-3.65GHz
        {"base": 5.8e9, "step": 0.5e6, "num": 400}  # 5.8-6.0GHz
    ]
    freq_sets = [np.arange(cfg['base'], cfg['base'] + cfg['step'] * cfg['num'], cfg['step'])
                 for cfg in freq_configs]
    hop_seqs = [np.random.permutation(len(fs))[:len(fs) // 2] for fs in freq_sets]

    # --- 信号生成 ---
    signals = []
    for i in range(num_signals):
        config_idx = i % len(freq_configs)
        freq_set = freq_sets[config_idx]
        hop_seq = hop_seqs[config_idx]
        snr = np.random.uniform(*snr_range)

        # 交替生成跳频/定频信号
        if i < 5:
            signal = generate_fhss_signal(num_samples, sample_rate, freq_set, hop_seq, snr)
        else:
            freq = np.random.choice(freq_set)
            signal = generate_fixed_frequency_signal(num_samples, sample_rate, freq, snr)
        signals.append(signal)

    # --- 数据标注与存储 ---
    annotations = annotate_dataset(signals, sample_rate, freq_sets, hop_seqs)

    with h5py.File('enhanced_communication_dataset.h5', 'w') as f:
        # 元数据
        f.attrs['num_samples'] = len(annotations)
        f.attrs['sample_rate'] = sample_rate

        # 信号数据（含跳频参数）
        signals_group = f.create_group('signals')
        for i, ann in enumerate(annotations):
            sig_group = signals_group.create_group(f'signal_{i}')
            sig_group.create_dataset('data', data=ann['signal_segment'])
            sig_group.attrs['hop_seq'] = ann['hop_seq']  # 存储跳频序列
            sig_group.attrs['freq_set'] = ann['freq_set']  # 存储频点集合

        # 标签数据（基于预设规则）
        f.create_dataset('labels', data=[a['label'] for a in annotations])

    print("数据集生成完成，保存至 enhanced_communication_dataset.h5")