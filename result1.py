# evaluation.py
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from h5py import File
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# ---------------------------- 特征提取模块 ----------------------------
class AdvancedFeatureExtractor:
    @staticmethod
    def extract_full_features(signal, sample_rate, hop_seq, freq_set):
        """提取信号多维特征"""
        # 示例特征集（需根据实际需求完善）
        features = []

        # 时域特征
        features.append(np.mean(np.abs(signal)))  # 幅度均值
        features.append(np.std(signal.real))  # 实部标准差
        features.append(np.max(signal.imag) - np.min(signal.imag))  # 虚部动态范围

        # 频域特征
        fft = np.fft.fft(signal)
        fft_mag = np.abs(fft)
        features.append(np.argmax(fft_mag))  # 主频位置
        features.append(np.sum(fft_mag[:10]))  # 低频能量

        # 跳频特征
        features.append(len(set(hop_seq)))  # 跳频点数
        features.append(np.mean(np.diff(freq_set)))  # 平均频率间隔

        return np.array(features)


# ---------------------------- 评估指标计算 ----------------------------
def cluster_entropy(labels):
    """计算聚类集熵值"""
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs + 1e-8))


def align_labels(y_true, y_pred):
    """标签对齐（匈牙利算法）"""
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    aligned = np.zeros_like(y_pred)
    for i, j in zip(row_ind, col_ind):
        aligned[y_pred == j] = i
    return aligned


def evaluate_performance(y_true, y_pred):
    """综合性能评估"""
    # 异常点识别（假设标签-1为噪声）
    noise_mask = (y_pred == -1)
    anomaly_acc = accuracy_score(y_true[noise_mask], y_pred[noise_mask]) if sum(noise_mask) > 0 else 0

    # 有效聚类评估（排除噪声点）
    valid_mask = ~noise_mask
    if sum(valid_mask) > 1:
        aligned = align_labels(y_true[valid_mask], y_pred[valid_mask])
        cluster_acc = accuracy_score(y_true[valid_mask], aligned)
        sil_score = silhouette_score(X[valid_mask], y_pred[valid_mask])
    else:
        cluster_acc = sil_score = 0

    return {
        "分类准确率": cluster_acc,
        "聚类熵": cluster_entropy(y_pred[valid_mask]),
        "异常识别率": anomaly_acc,
        "轮廓系数": sil_score
    }


# ---------------------------- 主流程 ----------------------------
# evaluation.py (修正部分)
if __name__ == "__main__":
    # 加载数据
    with File("enhanced_communication_dataset.h5", "r") as f:
        labels_true = f["labels"][:]
        sample_rate = f.attrs["sample_rate"]

        # 获取所有信号段的键名列表
        signal_keys = list(f["signals"].keys())
        signals = [f["signals"][k]["data"][:] for k in signal_keys]
        hop_seqs = [f["signals"][k].attrs["hop_seq"] for k in signal_keys]
        freq_sets = [f["signals"][k].attrs["freq_set"] for k in signal_keys]

    # 特征工程
    features = []
    for sig, hop_seq, freq_set in zip(signals, hop_seqs, freq_sets):
        feat = AdvancedFeatureExtractor.extract_full_features(
            sig, sample_rate, hop_seq, freq_set
        )
        features.append(feat)
    X = StandardScaler().fit_transform(np.array(features))


    # 聚类分析
    from sklearn.cluster import DBSCAN

    model = DBSCAN(eps=0.5, min_samples=3)
    labels_pred = model.fit_predict(X)

    # 性能评估
    metrics = evaluate_performance(labels_true, labels_pred)
    print(f"评估结果：{metrics}")

    # 可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=labels_true, cmap='viridis')
    plt.title("真实标签分布")

    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], c=labels_pred, cmap='viridis')
    plt.title("DBSCAN聚类结果")

    plt.savefig("cluster_comparison.png")
    plt.show()