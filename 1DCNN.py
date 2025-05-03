import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# 定义1DCNN模型
class CNN1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):  # 修改为二分类
        super(CNN1D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 数据加载与预处理
with h5py.File('enhanced_communication_dataset.h5', 'r') as f:
    labels = f['labels'][:]
    signals = []
    for name in f['signals']:
        signals.append(f['signals'][name]['data'][:])
    signals = np.array(signals)



# 复数转幅度
signals = np.abs(signals)

# 标准化
scaler = StandardScaler()
signals = scaler.fit_transform(signals.reshape(-1, signals.shape[-1])).reshape(signals.shape)
signals = signals[:, np.newaxis, :]  # [样本数, 通道数, 序列长度]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    signals, labels, test_size=0.2, random_state=42
)

# 转换为PyTorch Dataset
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.LongTensor(y_train)
)  #

test_dataset = TensorDataset(
    torch.FloatTensor(X_test),
    torch.LongTensor(y_test)
)

# 创建DataLoader
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 初始化模型、损失函数和优化器
model = CNN1D(input_channels=1, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 完整训练循环（修复缩进问题）
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_data.size(0)

    # 验证阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for test_data, test_labels in test_loader:
            outputs = model(test_data)
            _, predicted = torch.max(outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()

    # 打印统计信息
    train_loss = train_loss / len(train_loader.dataset)
    test_acc = correct / total
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Test Acc: {test_acc:.4f}")

# 最终测试
model.eval()
with torch.no_grad():
    total_correct = 0
    for test_data, test_labels in test_loader:
        outputs = model(test_data)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == test_labels).sum().item()
final_acc = total_correct / len(test_loader.dataset)
print(f"\nFinal Test Accuracy: {final_acc:.4f}")



# ===== 收集测试集预测结果 =====
model.eval()
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for test_data, test_labels in test_loader:
        outputs = model(test_data)
        probabilities = torch.softmax(outputs, dim=1)  # 转换为概率
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(test_labels.numpy())
        all_preds.extend(predicted.numpy())
        all_probs.extend(probabilities[:, 1].numpy())  # 取正类概率

# ===== 混淆矩阵 =====
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ===== ROC曲线 =====
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()