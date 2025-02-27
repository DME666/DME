import numpy
import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import AutoModelForImageClassification
import logging
import torch.nn.init as init
import warnings
warnings.filterwarnings("ignore")

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


class TabTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, output_dim=2, dropout=0.1):
        super(TabTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_projection(x)  # (batch_size, input_dim) -> (batch_size, hidden_dim)

        # 不需要加位置编码，直接 Transformer 处理
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)  # (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)

        x = self.output_layer(x)  # 输出分类
        return x

class MultiModel_EfficientNet_TabTransformer_CrossAttention(nn.Module):
    def __init__(self, image_pretrained_weights='IMAGENET1K_V1', num_non_image_features=10, num_classes=2):
        super(MultiModel_EfficientNet_TabTransformer_CrossAttention, self).__init__()
        self.tab_transformer_dim = 512
        self.table_out_dim = 1024

        self.mambavision = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-T-1K",
                                                                      trust_remote_code=True)  # 加载预训练的 EfficientNet-B2
        self.mambavision.model.head = nn.Linear(self.mambavision.model.head.in_features, self.table_out_dim)

        # Tabular modality: TabTransformer
        self.tab_transformer = TabTransformer(input_dim=10, output_dim=self.table_out_dim)

        # MultiheadAttention for cross-modality attention
        self.attention = nn.MultiheadAttention(embed_dim=self.table_out_dim, num_heads=8, dropout=0.1)

        # Fully connected layers for final classification
        self.fc = nn.Sequential(
            nn.Linear(self.table_out_dim + self.table_out_dim, 512),  # 拼接后送入全连接层，输出维度512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # 256层
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # 128层
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)  # 输出2类分类结果
        )

    def forward(self, image, tabular_data):
        # 处理图像数据
        image_features = self.mambavision(image)['logits']

        x_categ = torch.zeros(tabular_data.size(0), 0).to(tabular_data.device)  # 如果没有类别特征，x_categ 传递为零张量
        tabular_features = self.tab_transformer(tabular_data)

        # 为了使用 MultiheadAttention，需要对特征进行维度转换
        image_features = image_features.unsqueeze(0)  # shape: [1, batch_size, feature_dim] 需要加上一个维度
        tabular_features = tabular_features.unsqueeze(0)  # shape: [1, batch_size, feature_dim]

        # 交叉注意力：图像作为 query，表格作为 key 和 value
        cross_attention_output, _ = self.attention(image_features, tabular_features, tabular_features)

        # 融合后的特征（通过最大池化来获得最终的特征向量）
        combined_features = torch.cat((cross_attention_output.squeeze(0), tabular_features.squeeze(0)), dim=1)

        # 通过全连接层做最终分类
        output = self.fc(combined_features)
        return output

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
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

csv_file_test = '/home/wugang/data1/Projects/DME_APTOS2021_V4/pyfile_fusion_method/datasets/test.csv'
img_folder_test = '/home/wugang/data1/DME_dataset/resnet_DME_dataset/img_6'

dataset_test = MyDataset(csv_file=csv_file_test, img_dir=img_folder_test, transform=transform)
test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=4)


# 初始化模型
model = MultiModel_EfficientNet_TabTransformer_CrossAttention()  # 根据实际特征维度调整
criterion = nn.CrossEntropyLoss()  # 如果是二分类任务
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
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

# 设置日志配置
logging.basicConfig(
    filename='/home/zhiqinkun/project/DME/mamba_tab/training.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
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
        outputs = model(images, texts)
        # outputs = outputs.squeeze(0)
        _, preds = torch.max(outputs, 1)
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

    logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(train_loss):.4f}, accuracy: {train_accuracy:.4f}')
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss_mean}, Train Accuracy: {train_accuracy:.4f}')

    # 计算准确率
    logging.info("Starting test phase...")
    correct = 0
    total = 0

    model.eval()
    test_losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, texts in tqdm(test_loader):
            images, labels, texts = images.to(device).float(), labels.to(device), texts.to(device).float()
            labels = labels.long()
            outputs = model(images, texts)
            # outputs = outputs.squeeze(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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
            logging.info(f'New best model saved at epoch {epoch + 1} with accuracy {best_acc:.4f}')
            print(f'New best model saved at epoch {epoch + 1} with accuracy {best_acc:.4f}')



print('Training complete!')
