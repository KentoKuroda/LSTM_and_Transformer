import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
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
    model.load_state_dict(torch.load("model/transformer_model/fine_tuning_best_params.pth", map_location=device), strict=False)
    model = model.to(device)

    # 各モデルの特徴ベクトル取得
    features, labels = get_feature_vectors(model, train_loader)

    print("Feature shape:", features.shape)
    print("Label shape:", labels.shape)

    # t-SNEによる次元削減
    tsne_transformer = reduce_dim(features)

    # プロット (train_loader のラベルを使用)
    tactical_action_name = ['Build up', 'Progression', 'Final third', 'Counter-attack', 'High press', 'Mid block', 'Low block', 'Counter-press', 'Recovery']
    output_dir = "output\\transformer"
    labels = convert_labels(labels, tactical_action_name)
    # plot_features(tsne_transformer, labels, "Transformer Feature Space", tactical_action_name, output_dir)
    evaluate_similarity(features, labels, output_dir)


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
    # max_indices[~mask.any(axis=1)] = 9  
    
    # tactical_action_name を NumPy 配列に変換
    tactical_action_name = np.array(tactical_action_name)  

    # tactical_action_name に変換
    new_labels = tactical_action_name[max_indices]

    return new_labels


# 特徴ベクトルをプロットして保存
def plot_features(features, labels, title, tactical_action_name, output_dir):
    # 戦術ごとの色を定義
    TACTICAL_COLORS = {
        'Build up': 'blue',
        'Progression': 'skyblue',
        'Final third': 'green',
        'Counter-attack': 'yellowgreen',
        'High press': 'red',
        'Mid block': 'pink',
        'Low block': 'orange',
        'Counter-press': 'yellow',
        'Recovery': 'brown'
    }
    unique_labels = [name for name in tactical_action_name if name in labels]

    # 全戦術のプロット
    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        idx = labels == label
        color = TACTICAL_COLORS[label]
        plt.scatter(features[idx, 0], features[idx, 1], color=color, label=f"{label}")

    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(output_dir, "all_tactics.png"))  # すべての戦術を保存
    plt.close()

    # 各戦術ごとのプロットを個別に保存（他の戦術も表示）
    for label in unique_labels:
        plt.figure(figsize=(8, 6))

        # 他の戦術を彩度5%でプロット
        legend_handles = []
        for other_label in unique_labels:
            idx = labels == other_label
            color = TACTICAL_COLORS[other_label]
            if other_label == label:
                # 注目戦術（目立たせる）
                plt.scatter(features[idx, 0], features[idx, 1], color=color, s=50, alpha=1.0, label=f"{other_label}")
            else:
                # 他の戦術（目立たなくする）
                adaptive_alpha = min(0.05, 1 / (1 + np.sqrt(len(idx))))  # データ数が多いほど薄く
                plt.scatter(features[idx, 0], features[idx, 1], color=color, s=10, alpha=adaptive_alpha, label=f"{other_label}")

            # 凡例の色変更
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                            markersize=8, alpha=(1.0 if other_label == label else 0.3), label=other_label))

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        legend = plt.legend(handles=legend_handles, loc="upper right")

        # 凡例の色変更（注目戦術は黒、それ以外は灰色）
        for text, handle in zip(legend.get_texts(), legend_handles):
            text.set_color("black" if text.get_text() == label else "gray")

        plt.title(f"{title} - {label}")
        plt.savefig(os.path.join(output_dir, f"{label}.png"))  # 各戦術ごとに保存
        plt.close()


# 定量評価：同じ戦術 vs. 異なる戦術の類似度
def evaluate_similarity(features, labels, output_dir, batch_size=1000):
    unique_labels = np.unique(labels)

    # 各戦術ごとの統計データを保存するリスト
    tactic_results = []

    # 平均類似度を保存するための変数
    total_same_tactic_similarity = 0
    total_same_tactic_count = 0
    total_different_tactic_similarity = 0
    total_different_tactic_count = 0

    for label in unique_labels:
        idx = labels == label
        tactic_vectors = features[idx]

        # 戦術ごとの類似度を計算
        same_tactic_similarity_sum = 0
        same_tactic_count = 0
        different_tactic_similarity_sum = 0
        different_tactic_count = 0

        # 同じ戦術内の類似度
        if len(tactic_vectors) > 1:
            sim_matrix = cosine_similarity(tactic_vectors)
            upper_triangle_indices = np.triu_indices(len(tactic_vectors), k=1)
            upper_triangle_values = sim_matrix[upper_triangle_indices]

            if upper_triangle_values.size > 0:  # スカラー値でないことを確認
                same_tactic_similarity_sum = np.sum(upper_triangle_values)
                same_tactic_count = upper_triangle_values.size

        # 異なる戦術間の類似度
        for other_label in unique_labels:
            if label == other_label:
                continue
            other_idx = labels == other_label
            other_vectors = features[other_idx]

            if len(tactic_vectors) > 0 and len(other_vectors) > 0:
                for i in range(0, len(tactic_vectors), batch_size):
                    batch_tactic = tactic_vectors[i:i+batch_size]
                    for j in range(0, len(other_vectors), batch_size):
                        batch_other = other_vectors[j:j+batch_size]
                        sim_matrix = cosine_similarity(batch_tactic, batch_other)
                        if sim_matrix.size > 0:  # スカラー値でないことを確認
                            different_tactic_similarity_sum += np.sum(sim_matrix)
                            different_tactic_count += sim_matrix.size

        # 平均類似度を計算
        same_mean = (same_tactic_similarity_sum / same_tactic_count) if same_tactic_count > 0 else 0
        different_mean = (different_tactic_similarity_sum / different_tactic_count) if different_tactic_count > 0 else 0

        # 各戦術の統計データを保存
        tactic_results.append([label, same_mean, different_mean])

        # 全体の統計にも加算
        total_same_tactic_similarity += same_tactic_similarity_sum
        total_same_tactic_count += same_tactic_count
        total_different_tactic_similarity += different_tactic_similarity_sum
        total_different_tactic_count += different_tactic_count

    # 全体の平均類似度を計算
    total_same_mean = (total_same_tactic_similarity / total_same_tactic_count) if total_same_tactic_count > 0 else 0
    total_different_mean = (total_different_tactic_similarity / total_different_tactic_count) if total_different_tactic_count > 0 else 0

    # 結果をCSVに保存
    csv_path = os.path.join(output_dir, "similarity_results.csv")
    df = pd.DataFrame(tactic_results, columns=["Tactic", "Same_Tactic_Mean", "Different_Tactic_Mean"])
    df.loc[len(df)] = ["Overall", total_same_mean, total_different_mean]  # 全体の平均を追加
    df.to_csv(csv_path, index=False)
    print(f"Similarity results saved to {csv_path}")

    # 結果をプリント
    print(df)

    return total_same_mean, total_different_mean


if __name__ == "__main__":
    main()
