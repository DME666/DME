import numpy as np
import torch
import math
import os
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
import pandas as pd
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F
from transformers import AutoModelForImageClassification
from models.MambaVision_downsample import MambaVision
from models.TabTransformer import TabTransformer
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '2'

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')



# 定义数据集
class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform

        self.label_map = dict(zip(self.data_frame['patient ID'], self.data_frame['continue injection']))
        self.img_files = [f for f in os.listdir(img_dir) if f in self.label_map]

        # 显式拆分 类别/连续 文本特征列
        self.cat_text_cols = ['gender', 'diagnosis', 'anti-VEGF', 'preIRF', 'preSRF', 'prePED', 'preHRF']
        self.cont_text_cols = ['age', 'preVA', 'preCST']
        self.all_text_cols = self.cat_text_cols + self.cont_text_cols

        # 显式转换连续特征列为数值类型
        for col in self.cont_text_cols:
            self.data_frame[col] = pd.to_numeric(self.data_frame[col], errors='coerce')
        # 预处理：统计类别特征取值数、计算连续特征 mean/std
        self.cat_dims = self._get_cat_dims()  # 关键：统计每个类别特征的取值数量
        self.cont_mean = self.data_frame[self.cont_text_cols].mean().values
        self.cont_std = self.data_frame[self.cont_text_cols].std().values
        self.text_data = self._preprocess_text_data()

    def _get_cat_dims(self):
        """统计每个类别特征的不同取值数量，给 TabTransformer 用"""
        cat_dims = []
        for col in self.cat_text_cols:
            # 类别特征需是整数编码（如 0,1,2...），若原始是字符串需先 map
            cat_dims.append(self.data_frame[col].nunique())
        return cat_dims

    def _preprocess_text_data(self):
        text_data = {}
        for _, row in self.data_frame.iterrows():
            patient_id = row['patient ID']

            # 提取连续特征并处理缺失值
            cont_vals = row[self.cont_text_cols].values
            cont_vals = np.nan_to_num(cont_vals, nan=0.0)  # 用0填充缺失值

            # 标准化
            cont_vals = (cont_vals - self.cont_mean) / self.cont_std

            text_data[patient_id] = {
                'cat': row[self.cat_text_cols].values.astype(np.int64),
                'cont': cont_vals
            }
        return text_data

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        label = self.label_map.get(img_name, 0)
        text_dict = self.text_data.get(img_name, {
            'cat': np.zeros(len(self.cat_text_cols), dtype=np.int64),
            'cont': np.zeros(len(self.cont_text_cols), dtype=np.float32)
        })

        # 读取图像
        img = Image.open(img_path).convert('RGB') if os.path.exists(img_path) else Image.new('RGB', (224, 224))
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, text_dict['cat'], text_dict['cont'], img_name


# 自定义 BatchSampler
class CustomBatchSampler(BatchSampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.grouped_indices = self._group_indices()
        self.batches = list(self.grouped_indices.values())

    def _group_indices(self):
        """ 按照 `patient ID` 第一部分分组，把人作为基本单位动态划分batch """
        grouped_indices = {}
        for idx, img_name in enumerate(self.dataset.img_files):
            group_key = img_name.split('_')[0]  # 取 patient ID 第一部分
            if group_key not in grouped_indices:
                grouped_indices[group_key] = []
            grouped_indices[group_key].append(idx)
        return grouped_indices

    def __iter__(self):
        """ 生成 batch，每个 batch 包含相同 `patient ID` 的样本 """
        for batch in self.batches:
            yield batch

    def __len__(self):
        """ batch 数量 = patient ID 的不同数量 """
        return len(self.batches)



class MultiModel_MambaVision_TabTransformer_CrossAttention(nn.Module):
    def __init__(self, cat_cols, num_cat_features=7, num_cont_features=3, num_classes=2):
        super(MultiModel_MambaVision_TabTransformer_CrossAttention, self).__init__()
        self.table_out_dim = 1024

        self.mambavision = MambaVision(num_classes = 2)
        self.mambavision.head = nn.Sequential(
            nn.Linear(self.mambavision.head[0].in_features, self.table_out_dim),
            nn.ReLU()
        )

        # cat_cols = dataset.cat_text_cols
        self.tab_transformer = TabTransformer(
            cat_cols = cat_cols,  # 从 dataset 拿统计好的类别维度
            num_cont_features=num_cont_features,
            cat_embed_dim=64,
            num_heads=8,
            num_layers=4,
            ff_hidden_dim=256,
            max_seq_len=100,
            num_classes=num_classes,
            extract_features = True
        )

        # MultiheadAttention for cross-modality attention
        self.attention = nn.MultiheadAttention(embed_dim=self.table_out_dim, num_heads=8, dropout=0.1, batch_first=True)

        # Fully connected layers for final classification
        self.fc = nn.Sequential(
            nn.Linear(self.table_out_dim * 2, 512),  # 拼接后送入全连接层，输出维度512
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
        self.loaded_init_weights = False

    def load_gcondnet_weights(self, weight_path='/home/zhiqinkun/DME/MambaVision/mambavision-main/gcondnet_init_weights.pt'):
        """从文件加载GCondNet生成的权重"""
        if os.path.exists(weight_path):
            try:
                device = next(self.parameters()).device
                weight_matrix = torch.load(weight_path, map_location=device)
                # 确保权重维度匹配
                if self.fc[0].weight.shape == weight_matrix.shape:
                    self.fc[0].weight.data = weight_matrix
                    # 初始化偏置
                    if self.fc[0].bias is not None:
                        self.fc[0].bias.data.zero_()
                    self.loaded_init_weights = True
                    print(f"成功加载GCondNet权重到self.fc第一层，形状: {weight_matrix.shape}")
                else:
                    print(f"权重维度不匹配: 期望 {self.fc[0].weight.shape}, 得到 {weight_matrix.shape}")
            except Exception as e:
                print(f"加载权重时出错: {e}")
        else:
            print(f"权重文件不存在: {weight_path}")

    def forward(self, image, cat_text, cont_text):
        # 处理图像数据
        image_features = self.mambavision(image)
        # print(f"Image features shape: {image_features.shape}")

        tabular_features = self.tab_transformer(cat_text, cont_text)
        # print(f"Tabular features shape: {tabular_features.shape}")

        # 为了使用 MultiheadAttention，需要对特征进行维度转换
        image_features = image_features.unsqueeze(1)  # shape: [1, batch_size, feature_dim] 需要加上一个维度
        tabular_features = tabular_features.unsqueeze(1)  # shape: [1, batch_size, feature_dim]
        # print(f"After unsqueeze: image={image_features.shape}, tabular={tabular_features.shape}")

        cross_attention_output_image, _ = self.attention(query=image_features, key=tabular_features, value=tabular_features)
        cross_attention_output_table, _ = self.attention(query=tabular_features, key=image_features, value=tabular_features)
        # print(f"Attention outputs: image={cross_attention_output_image.shape}, table={cross_attention_output_table.shape}")

        # 融合后的特征（通过最大池化来获得最终的特征向量）
        combined_features = torch.cat((cross_attention_output_image.squeeze(1), cross_attention_output_table.squeeze(1)), dim=1)
        # print(f"Combined features shape: {combined_features.shape}")

        assert combined_features.device == next(self.parameters()).device, \
            f"combined_features device: {combined_features.device}, model device: {next(self.parameters()).device}"

        # 通过全连接层做最终分类
        output = self.fc(combined_features)
        # print(f"Output shape: {output.shape}")
        return output


# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def custom_collate(batch):
    imgs = []
    labels = []
    cat_texts = []
    cont_texts = []
    img_names = []

    for img, label, cat, cont, name in batch:
        imgs.append(img)
        labels.append(label)
        cat_texts.append(cat)

        # 确保连续特征是数值类型
        cont = np.array(cont, dtype=np.float32)
        cont_texts.append(cont)

        img_names.append(name)

    # 转 tensor
    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    cat_texts = torch.tensor(np.array(cat_texts), dtype=torch.long)
    cont_texts = torch.tensor(np.array(cont_texts), dtype=torch.float)

    return imgs, labels, cat_texts, cont_texts, img_names

# 创建数据集和数据加载器
csv_file = '/home/wugang/data1/Projects/DME_APTOS2021_V4/pyfile_fusion_method/datasets/train.csv'
img_folder = '/home/wugang/data1/DME_dataset/resnet_DME_dataset/img_6'

dataset = MyDataset(csv_file=csv_file, img_dir=img_folder, transform=transform)
batch_sampler = CustomBatchSampler(dataset)
train_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4, collate_fn=custom_collate)

csv_file_test = '/home/wugang/data1/Projects/DME_APTOS2021_V4/pyfile_fusion_method/datasets/test.csv'
img_folder_test = '/home/wugang/data1/DME_dataset/resnet_DME_dataset/img_6'

cat_text_cols = ['gender', 'diagnosis', 'anti-VEGF', 'preIRF', 'preSRF', 'prePED', 'preHRF']
cont_text_cols = ['age', 'preVA', 'preCST']

dataset_test = MyDataset(csv_file=csv_file_test, img_dir=img_folder_test, transform=transform)
batch_sampler = CustomBatchSampler(dataset_test)
test_loader = DataLoader(dataset_test, batch_sampler=batch_sampler, num_workers=4, collate_fn=custom_collate)

# 初始化模型
model = MultiModel_MambaVision_TabTransformer_CrossAttention(cat_cols=dataset.cat_text_cols)  # 根据实际特征维度调整
criterion = nn.CrossEntropyLoss()  # 如果是二分类任务
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

model.to(device)

model.load_gcondnet_weights('/home/zhiqinkun/DME/MambaVision/mambavision-main/gcondnet_init_weights.pt')

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
    filename='/home/zhiqinkun/project/DME/mamba_tab_graph/training_2.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.info("Starting training process...")

for epoch in range(num_epochs):
    logging.info("")
    logging.info("-" * 30)
    model.train()
    train_correct = 0
    total_train = 0
    train_loss = []

    for images, labels, cat_texts, cont_texts ,img_names in tqdm(train_loader):
        images, labels, cat_texts, cont_texts = images.to(device).float(), labels.to(device), cat_texts.to(device).long(), cont_texts.to(device).float()
        labels = labels.long()

        optimizer.zero_grad()
        outputs = model(images, cat_texts, cont_texts)
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
        for images, labels, cat_texts, cont_texts, img_names in tqdm(test_loader):
            images, labels, cat_texts, cont_texts = images.to(device).float(), labels.to(device), cat_texts.to(device).long(), cont_texts.to(device).float()
            labels = labels.long()
            outputs = model(images, cat_texts, cont_texts)
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