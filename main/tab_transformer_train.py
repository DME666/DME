import numpy as np
import torch
import math
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform
        self.label_map = dict(zip(self.data_frame['patient ID'], self.data_frame['continue injection']))

        self.img_files = [f for f in os.listdir(img_dir) if f in self.label_map]
        self.text_columns = ['gender', 'age', 'diagnosis', 'preVA', 'anti-VEGF', 'preCST', 'preIRF', 'preSRF', 'prePED', 'preHRF']
        self.mean = self.data_frame[self.text_columns].mean().values.astype(float)
        self.std = self.data_frame[self.text_columns].std().values.astype(float)
        self.text_data = self._preprocess_text_data()

    def _preprocess_text_data(self):
        """Prepare normalized text data."""
        text_data = {}
        for _, row in self.data_frame.iterrows():
            patient_id = row['patient ID']
            text = row[self.text_columns].values.astype(float)
            text_data[patient_id] = (text - self.mean) / self.std
        return text_data

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        label = self.label_map.get(img_name, 0)  # Default to 0 if label not found
        text = self.text_data.get(img_name, np.zeros(len(self.text_columns)))

        img = Image.open(img_path).convert('RGB') if os.path.exists(img_path) else Image.new('RGB', (224, 224))

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, text

# Simplified feature extractor for images
class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)

# 定义 Transformer 模型
class TransformerModel1(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, output_dim=2, dropout=0.1):
        super(TransformerModel1, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_dim, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Add positional encoding
        # print(x.shape)
        # print(self.input_projection(x).shape)
        # print(self.positional_encoding[:, :8, :].shape)
        x = self.input_projection(x) + self.positional_encoding[:, :x.size(1), :]

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Mean pooling over sequence dimension
        # x = x.mean(dim=1)

        # Output layer
        x = self.output_layer(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, output_dim=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        # 输入投影层，将输入维度转换为隐藏维度
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 定义一个线性层，将隐藏维度转换为 5 维
        self.dim_reduction = nn.Linear(hidden_dim, 5)

        # 定义一个线性层，将 5 维转换为 num_class 维
        self.dim_expansion = nn.Linear(5, output_dim)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # 输入投影，将输入维度转换为隐藏维度
        x = self.input_projection(x)  # (batch_size, input_dim) -> (batch_size, hidden_dim)

        # 通过 Transformer 编码器处理
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)  # (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)

        # 将隐藏维度转换为 5 维
        x = self.dim_reduction(x)  # (batch_size, hidden_dim) -> (batch_size, 5)

        # 将 5 维转换为 num_class 维
        x = self.dim_expansion(x)  # (batch_size, 5) -> (batch_size, num_class)

        return x


def compute_metrics(preds, labels):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    return precision, recall, f1

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 创建数据集和数据加载器
csv_file = '/home/wugang/data1/Projects/DME_APTOS2021_V4/pyfile_fusion_method/datasets/train.csv'
img_folder = '/home/wugang/data1/DME_dataset/resnet_DME_dataset/img_6'

dataset = MyDataset(csv_file=csv_file, img_dir=img_folder, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

csv_file_test = '/home/wugang/data1/Projects/DME_APTOS2021_V4/pyfile_fusion_method/datasets/test.csv'
img_folder_test = '/home/wugang/data1/DME_dataset/resnet_DME_dataset/img_6'

dataset_test = MyDataset(csv_file=csv_file_test, img_dir=img_folder_test, transform=transform)
test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=4)

# 初始化模型
model = TransformerModel(input_dim=10, output_dim=2)  # 根据实际特征维度调整
criterion = nn.CrossEntropyLoss()  # 如果是二分类任务
optimizer = torch.optim.Adam(model.parameters(), lr=3e-6, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

model.to(device)

# 训练循环
num_epochs = 100

train_losses = []
test_losses = []

train_accuracies = []
test_accuracies = []

best_acc = 0
best_epoch = 0
model_save_path = '/home/zhiqinkun/project/DME/tab-transformer/5_2_best_model.pth'

# 设置日志配置
logging.basicConfig(
    filename='/home/zhiqinkun/project/DME/tab-transformer/5_2.1_training.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 日志记录器示例
logging.info("Starting training process...")

for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    total_train = 0
    train_loss = []

    for images, labels, texts in tqdm(train_loader):
        images, labels, texts = images.to(device).float(), labels.to(device), texts.to(device).float()
        labels = labels.long()

        optimizer.zero_grad()
        outputs = model(texts)
        outputs = outputs.squeeze(0)
        _, preds = torch.max(outputs, 1)
        # print(labels.shape)
        # print(outputs.shape)

        loss = criterion(outputs, labels)  # 确保标签形状匹配

        train_loss.append(loss.cpu().item())

        loss.backward()
        optimizer.step()

        total_train += labels.size(0)
        train_correct += torch.sum(preds == labels).item()

    train_loss_mean = np.mean(np.delete(train_loss, [np.argmax(train_loss), np.argmin(train_loss)]))
    train_losses.append(train_loss_mean)
    train_accuracy = 100 * train_correct / total_train
    train_accuracies.append(train_accuracy)

    logging.info(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss_mean:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss_mean}, Train Accuracy: {train_accuracy:.4f}')

    # 计算准确率
    logging.info("Starting test phase...")
    correct = 0
    total = 0

    model.eval()  # Set the model to evaluation mode

    test_losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 不计算梯度
        for images, labels, texts in tqdm(test_loader):
            images, labels, texts = images.to(device).float(), labels.to(device).long(), texts.to(device).float()
            outputs = model(texts)
            outputs = outputs.squeeze(0)
            _, preds = torch.max(outputs, 1)
            # total += labels.size(0)  # 总样本数
            # correct += (preds == labels).sum().item()  # 计算正确预测的数量
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # accuracy = correct / total  # 计算准确率
    # print(f'Accuracy of the model on the test dataset: {accuracy:.2f}')
    # Calculate metrics outside the loop
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    test_accuracy_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # 记录测试阶段的日志
    logging.info(
        f'Final Test Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    logging.info(f'Confusion Matrix:\n{conf_matrix}')
    logging.info(f'Accuracy per class: {test_accuracy_per_class}')

    print(f'Accuracy_per_class: {test_accuracy_per_class}')
    print(f'Accuracy: {accuracy:.4f},Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    # 检查是否是最佳准确率
    if accuracy > best_acc:
        best_acc = accuracy
        best_epoch = epoch
        # 保存最佳模型
        torch.save(model.state_dict(), model_save_path)
        logging.info(f'New best model saved at epoch {epoch + 1} with accuracy {best_acc:.4f}')
        print(f'New best model saved at epoch {epoch + 1} with accuracy {best_acc:.4f}')

    # scheduler.step(np.mean(val_loss))
    # current_lr = scheduler.get_last_lr()
    # print(f'Current learning rate: {current_lr}')

print('Training complete!')
print(f'Best model saved at epoch {best_epoch + 1} with accuracy {best_acc:.4f}')