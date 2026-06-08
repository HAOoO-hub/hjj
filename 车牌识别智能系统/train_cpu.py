#文件train_cpu
# 文件：train_on_cpu.py
#3层卷积 + 3层ReLU激活 + 3层池化 + 1层全连接 + 1层Dropout + 1层全连接
#卷积层（3 层）：提取图像特征，从基础边缘到抽象的车牌特征，是 CNN 的核心；
#池化层（3 层）：降维减参，提升计算效率，同时保留关键特征；
#ReLU 激活：引入非线性0/1，让模型能学习复杂的特征关系
#（没有 ReLU，CNN 退化为线性模型，无法处理复杂分类任务）；
#全连接层（2 层）：将卷积提取的特征整合，最终映射到分类结果；
#Dropout 层（1 层）：防止过拟合，提升模型在新数据上的识别准确率。
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import time
from pathlib import Path


# ==================== 超轻量级模型（专为CPU设计） ====================
class TinyPlateVerifier(nn.Module):
    """超轻量车牌验证模型（训练快，CPU友好）"""

    def __init__(self):
        super(TinyPlateVerifier, self).__init__()

        # 参数量 < 100K，训练极快
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 64x128
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 32x64
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 16x32
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x16

        self.fc1 = nn.Linear(64 * 8 * 16, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2)  # 2类：车牌/非车牌

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==================== 数据集加载 ====================
class PlateDataset(Dataset):
    """车牌验证数据集"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        # 加载正样本
        pos_dir = self.data_dir / "plates"
        for img_path in pos_dir.glob("*.jpg"):
            self.samples.append((img_path, 1))  # 标签：1=车牌

        # 加载负样本
        neg_dir = self.data_dir / "non_plates"
        for img_path in neg_dir.glob("*.jpg"):
            self.samples.append((img_path, 0))  # 标签：0=非车牌

        print(f"加载数据集: {len(self.samples)} 张图像")
        print(f"  - 正样本: {len(list(pos_dir.glob('*.jpg')))} 张")
        print(f"  - 负样本: {len(list(neg_dir.glob('*.jpg')))} 张")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ==================== 训练函数 ====================
def train_model():
    """在CPU上训练模型"""
    print("=" * 60)
    print("车牌验证模型训练")
    print("=" * 60)

    # 设置设备（CPU）
    device = torch.device('cpu')
    print(f"训练设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 加载数据集
    data_dir = Path(r"E:\CCPD 1.8\training_data")
    dataset = PlateDataset(data_dir, transform)

    # CPU建议batch_size=16或32，避免太慢
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 创建模型
    model = TinyPlateVerifier().to(device)

    # 损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#默认

    # 训练参数
    num_epochs = 12  # 12个epoch足够收敛
    best_accuracy = 0.0
    best_model_path = r"E:\CCPD 1.8\plate_verifier_best.pth"

    print(f"开始训练: {num_epochs}个epoch, batch_size={batch_size}")
    print("预计训练时间: 2-3小时（CPU）\n")

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        total_loss = 0
        correct = 0

        # 进度条
        for batch_idx, (images, labels) in enumerate(dataloader):
            # 移动数据到设备（CPU就原地操作）
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

            # 每20个batch显示一次进度
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        # 计算准确率
        accuracy = correct / len(dataset)

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"  💾 保存最佳模型 (准确率: {accuracy:.2%})")

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1}/{num_epochs} 完成:")
        print(f"  - 平均损失: {total_loss / len(dataloader):.4f}")
        print(f"  - 准确率: {accuracy:.2%}")
        print(f"  - 耗时: {epoch_time:.1f}秒\n")

    print("=" * 60)
    print("训练完成！")
    print(f"最佳模型已保存: {best_model_path}")
    print(f"最佳准确率: {best_accuracy:.2%}")
    print("=" * 60)

    return best_model_path


# ==================== 主入口 ====================
if __name__ == "__main__":
    # 开始训练
    model_path = train_model()

    print("\n下一步：")
    print(f"1. 在 test.py 中设置 verification_model_path = '{model_path}'")
    print("2. 重新运行你的车牌识别系统")