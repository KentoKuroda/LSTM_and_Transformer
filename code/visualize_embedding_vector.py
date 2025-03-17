import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from Transformer.Transformer_kuroda import TransformerClassification
from get_dataset import googledrive_download, init_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # the number of players
    num_player = 22
    # ハイパーパラメータ
    input_dim = (num_player + 1) * 2  # 入力の次元数
    hidden_dim = 20  # 隠れ層の次元数
    target_size = 18  # クラス数（分類タスク）
    num_heads = 4  # マルチヘッドアテンションのヘッド数
    num_layers = 2  # Transformerの層の数
    batch_size = 2048

    sequence_np, label_np = googledrive_download(bepro=True) 
    print("Dataset shape:", sequence_np.shape, label_np.shape)
    train_loader, val_loader, test_loader = init_dataset(sequence_np, label_np, batch_size)

    # モデルの初期化
    model = TransformerClassification(input_dim, hidden_dim, target_size, num_heads, num_layers)
    model = model.to(device)

    # 各モデルの特徴ベクトル取得
    transformer_features, train_labels = get_feature_vectors(model, train_loader)

    print("Feature shape:", transformer_features.shape)
    print("Label shape:", train_labels.shape)

    # t-SNEによる次元削減
    tsne_transformer = reduce_dim(transformer_features)

    # プロット (train_loader のラベルを使用)
    tactical_action_name = ['Others', 'Build up', 'Progression', 'Final third', 'Counter-attack', 'High press', 'Mid block', 'Low block', 'Counter-press', 'Recovery']
    plot_features(tsne_transformer, train_labels, "Transformer Feature Space", tactical_action_name)


# モデルの特徴ベクトルを取得する関数
def get_feature_vectors(model, loader, edge_index=None):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if edge_index is not None:  # GAT + Transformer の場合
                features = model.gnn(inputs, edge_index)
                features = features.view(inputs.shape[0], -1)  # (B, N*H) に変換
            else:  # Transformer の場合
                features = model.embedding(inputs)
                features = model.pos_encoder(features)
                features = model.transformer_encoder(features)
                features = features[:, -1, :]  # (B, H)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.vstack(all_features), np.concatenate(all_labels)


# 特徴ベクトルをt-SNEで次元削減
def reduce_dim(features):
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    return tsne.fit_transform(features)


# ラベルの変換関数
def convert_labels(labels, tactical_action_name):
    # (N, 9) の中で 0.75 以上の値があるかどうかのマスク
    mask = labels[:, :9] >= 0.75  

    # 最も左の 0.75 以上の戦術のインデックスを取得（1つのみ）
    max_indices = np.argmax(mask, axis=1)  # これで最も左の 1 つを選択
    
    # 0.75 以上の戦術が1つもない場合は9（others）にする
    max_indices[~mask.any(axis=1)] = 9  
    
    # tactical_action_name を NumPy 配列に変換
    tactical_action_name = np.array(tactical_action_name)  

    # tactical_action_name に変換
    new_labels = tactical_action_name[max_indices]

    return new_labels


# 特徴ベクトルをプロット
def plot_features(features, labels, title, tactical_action_name):
    labels = convert_labels(labels, tactical_action_name)  # ラベルを変換
    # tactical_action_name の順番通りに並べ替える
    unique_labels = [name for name in tactical_action_name if name in labels]

    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        idx = labels == label
        plt.scatter(features[idx, 0], features[idx, 1], label=f"{label}")
    
    plt.legend()
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    main()
