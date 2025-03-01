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
from models.MambaVision import MambaVision


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')



# 定义数据集
class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform

        # 创建映射关系
        self.label_map = dict(zip(self.data_frame['patient ID'], self.data_frame['continue injection']))

        # 只保留数据集中有图像的部分
        self.img_files = [f for f in os.listdir(img_dir) if f in self.label_map]

        # 需要的文本特征
        self.text_columns = ['gender', 'age', 'diagnosis', 'preVA', 'anti-VEGF', 'preCST', 'preIRF', 'preSRF', 'prePED',
                             'preHRF']
        self.mean = self.data_frame[self.text_columns].mean().values.astype(float)
        self.std = self.data_frame[self.text_columns].std().values.astype(float)
        self.text_data = self._preprocess_text_data()

    def _preprocess_text_data(self):
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

        # 获取对应标签和文本特征
        label = self.label_map.get(img_name, 0)
        text = self.text_data.get(img_name, np.zeros(len(self.text_columns)))

        # 读取图像
        img = Image.open(img_path).convert('RGB') if os.path.exists(img_path) else Image.new('RGB', (224, 224))
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, text, img_name  # 需要返回 img_name 用于 batch 分组


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

# mambavision模型
mambavision = MambaVision(num_classes = 2)
checkpoint = torch.load('/home/zhiqinkun/project/DME/ManbaVision/log/纯图片/100ep5_2/mambavision-T-1K minlr=3e-7/best_model_epoch_3.pth', map_location='cpu')
mambavision.load_state_dict(checkpoint)


mambavision.head = nn.Sequential(
    mambavision.model.head[0],  # nn.Linear(mambavision.model.head.in_features, 5)
    mambavision.model.head[1]   # nn.ReLU()
)
mambavision.to(device)
# mambavision模型


# 冻结 mambavision 模型除头部外的所有参数
for name, param in mambavision.named_parameters():
    if 'head' not in name:
        param.requires_grad = False

#tab-transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, output_dim=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        # 输入投影层，将输入维度转换为隐藏维度
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.dim_reduction = nn.Linear(hidden_dim, 5)

        # self.dim_expansion = nn.Linear(5, output_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_projection(x)  # (batch_size, input_dim) -> (batch_size, hidden_dim)

        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)  # (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)

        x = self.dim_reduction(x)  # (batch_size, hidden_dim) -> (batch_size, 5)

        # x = self.dim_expansion(x)  # (batch_size, 5) -> (batch_size, num_class)
        return x

TabTransformer = TransformerModel(input_dim=10, output_dim=2)

checkpoint = torch.load('/home/zhiqinkun/project/DME/tab-transformer/5_2_best_model.pth', map_location='cpu')
TabTransformer.load_state_dict(checkpoint, strict=False)

TabTransformer.to(device)
#tab-transformer模型

# 冻结 TabTransformer 模型除输出层外的所有参数
for name, param in TabTransformer.named_parameters():
    if 'output_layer' not in name:
        param.requires_grad = False

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 创建数据集和数据加载器
# csv_file = '/home/wugang/data1/Projects/DME_APTOS2021_V4/pyfile_fusion_method/datasets/train.csv'
# img_folder = '/home/wugang/data1/DME_dataset/resnet_DME_dataset/img_6'
#
# dataset = MyDataset(csv_file=csv_file, img_dir=img_folder, transform=transform)
# batch_sampler = CustomBatchSampler(dataset)
# train_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

csv_file_test = '/home/wugang/data1/Projects/DME_APTOS2021_V4/pyfile_fusion_method/datasets/test.csv'
img_folder_test = '/home/wugang/data1/DME_dataset/resnet_DME_dataset/img_6'

dataset_test = MyDataset(csv_file=csv_file_test, img_dir=img_folder_test, transform=transform)
batch_sampler = CustomBatchSampler(dataset_test)
test_loader = DataLoader(dataset_test, batch_sampler=batch_sampler, num_workers=4)

all_combined_outputs = None
gap_layer = nn.AdaptiveAvgPool1d(1)

for images, labels, texts, img_names in tqdm(test_loader):
    images, labels, texts = images.to(device).float(), labels.to(device), texts.to(device).float()
    labels = labels.long()

    images_outputs = mambavision(images)['logits']
    texts_outputs = TabTransformer(texts)

    combined_outputs = torch.cat((images_outputs, texts_outputs), dim=1)

    # 全局平均池化
    output = gap_layer(combined_outputs.unsqueeze(2)).squeeze(2)

    # 对 batch_size 维度求平均，得到 1×10 的张量
    combined_outputs = torch.mean(output, dim=0, keepdim=True)  # (1, 10)

    # 计算 labels 的众数,非必须，减少出错的可能。
    mode_label = labels.mode()[0].view(1, 1)  # (1, 1)

    # 拼接众数到 `combined_outputs`
    combined_outputs = torch.cat((combined_outputs, mode_label), dim=1)  # (1, 11)

    # 累积 `all_combined_outputs`
    if all_combined_outputs is None:
        all_combined_outputs = combined_outputs
    else:
        all_combined_outputs = torch.cat((all_combined_outputs, combined_outputs), dim=0)  # 按 batch 维度拼接

print(all_combined_outputs.shape)  # 形状应为 (num_batches, 11)
df = pd.DataFrame(all_combined_outputs.detach().cpu().numpy())
df.to_csv("/home/zhiqinkun/project/gnn_test.csv", index=False, header=False)