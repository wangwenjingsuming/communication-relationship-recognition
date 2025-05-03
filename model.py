import torch
import torch.nn as nn
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
# ---------------------------- 混合模型架构 ----------------------------
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_channels=1, lstm_hidden=128, num_classes=2):
        super().__init__()

        # CNN特征提取器
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # BiLSTM时序建模
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden * 2, num_classes)  # 双向输出拼接
        )

    def forward(self, x):
        # 输入形状: (batch, channels, seq_len)

        # CNN特征提取
        cnn_out = self.cnn(x)  # 输出形状: (batch, 256, reduced_seq_len)

        # 维度转换 (batch, channels, seq) -> (batch, seq, channels)
        lstm_input = cnn_out.permute(0, 2, 1)

        # BiLSTM处理
        lstm_out, _ = self.lstm(lstm_input)  # 输出形状: (batch, seq, hidden*2)

        # 取最后一个时间步
        last_step = lstm_out[:, -1, :]

        # 分类
        return self.classifier(last_step)


# ---------------------------- 数据预处理 ----------------------------
def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        labels = f['labels'][:]
        signals = [f['signals'][k]['data'][:] for k in f['signals']]
        signals = np.array(signals)

    # 复数转幅度
    signals = np.abs(signals)

    # 标准化并调整维度
    signals = StandardScaler().fit_transform(signals.reshape(-1, signals.shape[-1])).reshape(signals.shape)
    return torch.FloatTensor(signals[:, np.newaxis, :]), torch.LongTensor(labels)  # [N, 1, L]


# ---------------------------- 训练流程 ----------------------------
def train_model():
    # 加载数据
    X, y = load_data('enhanced_communication_dataset.h5')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=64)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN_BiLSTM(input_channels=1, lstm_hidden=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    best_acc = 0.0
    for epoch in range(30):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        # 验证
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        avg_loss = total_loss / len(train_loader.dataset)
        acc = correct / len(test_loader.dataset)

        print(f"Epoch {epoch + 1}/30 | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"\nBest Test Accuracy: {best_acc:.4f}")

    # 加载最佳模型并进行可视化
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # 收集测试集预测结果
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])  # 取类别1的概率

    # 混淆矩阵可视化
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # ROC曲线可视化
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    print("可视化结果已保存至 confusion_matrix.png 和 roc_curve.png")


if __name__ == "__main__":
    train_model()