import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.decoder = nn.Linear(out_channels, in_channels)  # 解码器
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x_recon = self.decoder(x)  # 重构特征
        return x_recon, x  # 返回重构结果和嵌入


def create_knn_graph(x, k=5):
    """处理全零特征和单样本情况"""
    if np.all(x == 0) or len(x) == 1:
        edge_index = torch.tensor([[], []], dtype=torch.long)
    else:
        # 处理k过大情况
        actual_k = min(k, len(x) - 1)
        adj_matrix = kneighbors_graph(
            x.reshape(-1, 1),
            n_neighbors=actual_k,
            mode='connectivity',
            include_self=False
        ).toarray()

        edge_index = []
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i, j] == 1:
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    return Data(x=x_tensor, edge_index=edge_index)


def generate_graph_embeddings(data, gcn_model, device, k=5):
    n_samples, n_features = data.shape
    embeddings = []

    print(f"Generating embeddings for {n_features} features...")
    for feature_idx in range(n_features):
        graph = create_knn_graph(data[:, feature_idx], k=k)
        graph = graph.to(device)

        with torch.no_grad():
            _, node_embeddings = gcn_model(graph.x, graph.edge_index)
            graph_embedding = torch.mean(node_embeddings, dim=0)
            embeddings.append(graph_embedding.cpu().numpy())

    return np.array(embeddings)


def train_gcn_model(graphs, in_channels, out_channels, device, epochs=100):
    gcn_model = GCN(in_channels, 64, out_channels).to(device)
    optimizer = optim.Adam(gcn_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        gcn_model.train()
        total_loss = 0

        for graph in graphs:
            graph = graph.to(device)
            optimizer.zero_grad()

            node_recon, _ = gcn_model(graph.x, graph.edge_index)
            loss = criterion(node_recon, graph.x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(graphs)

        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return gcn_model


def generate_weight_matrix(embeddings, input_dim, output_dim, alpha=1.0):
    n_features = embeddings.shape[0]

    # 确保不超出目标维度
    if n_features > input_dim:
        n_features = input_dim
        embeddings = embeddings[:input_dim]

    # 直接使用嵌入矩阵作为权重基础
    w_gnn = embeddings.T  # 转置为 (output_dim, n_features)

    # 创建匹配目标维度的随机权重矩阵
    w_random = np.random.randn(output_dim, input_dim) * 0.01

    # 将学习到的嵌入融合到随机矩阵的前n_features列
    w_random[:, :n_features] = alpha * w_gnn + (1 - alpha) * w_random[:, :n_features]

    return torch.FloatTensor(w_random)


def generate_and_save_weights(csv_path, output_path, input_dim=2048, output_dim=512, k=5, alpha=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # 读取并全局标准化数据
    data = pd.read_csv(csv_path, header=None).values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    print(f"Data shape: {data_scaled.shape}")

    # 创建所有特征的图
    graphs = []
    for i in range(data_scaled.shape[1]):
        graphs.append(create_knn_graph(data_scaled[:, i], k=k))

    # 训练GCN模型
    gcn_model = train_gcn_model(
        graphs,
        in_channels=1,
        out_channels=output_dim,
        device=device,
        epochs=100
    )
    gcn_model.eval()

    # 生成嵌入
    embeddings = generate_graph_embeddings(
        data_scaled,
        gcn_model,
        device,
        k=k
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # 生成并保存权重
    weight_matrix = generate_weight_matrix(
        embeddings,
        input_dim,
        output_dim,
        alpha
    )
    torch.save(weight_matrix, output_path)
    print(f"Weights saved to {output_path}, shape: {weight_matrix.shape}")
    return True


if __name__ == "__main__":
    generate_and_save_weights(
        csv_path='/home/zhiqinkun/DME/MambaVision/mambavision-main/nolabel_gnn_train.csv',
        output_path='/home/zhiqinkun/DME/MambaVision/mambavision-main/gcondnet_init_weights.pt',
        input_dim=2048,
        output_dim=512,
        k=5,
        alpha=1.0
    )