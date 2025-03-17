import torch
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
from Transformer.Transformer_kuroda import TransformerClassification
from get_dataset import googledrive_download, init_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve

# GPUチェック
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to the fine-tuned model')
    parser.add_argument('--output_file', required=False, help='Path to the fine-tuned model')
    parser.add_argument('--make_graph', action='store_true', help='Whether to create a graph-based dataset')
    return parser.parse_args()


def main():
    args = parse_arguments()
    model_name = args.model
    output_file = args.output_file
    make_graph = args.make_graph

    # the number of players
    num_player = 22
    # ハイパーパラメータ
    input_dim = (num_player + 1) * 2  # 入力の次元数
    hidden_dim = 20  # 隠れ層の次元数
    target_size = 18  # クラス数（分類タスク）
    num_heads = 4  # マルチヘッドアテンションのヘッド数
    num_layers = 2  # Transformerの層の数
    batch_size = 2048

    # 各戦術的行動の名前 
    tactical_action_name_list = ['Build up 1', 'Progression 1', 'Final third 1', 'Counter-attack 1', 'High press 1', 'Mid block 1', 'Low block 1', 'Counter-press 1', 'Recovery 1', 'Build up 2', 'Progression 2', 'Final third 2', 'Counter-attack 2', 'High press 2', 'Mid block 2', 'Low block 2', 'Counter-press 2', 'Recovery 2']

    # 学習済みモデルのロード
    model = TransformerClassification(input_dim, hidden_dim, target_size, num_heads, num_layers)
    model.load_state_dict(torch.load(f"model/{model_name}_model/fine_tuning_best_params.pth", map_location=device), strict=False)

    if make_graph:
        sequence_np, label_np = googledrive_download(make_graph=make_graph, bepro=True)
        print(sequence_np.shape, label_np.shape)
        graph_testloader = init_dataset(sequence_np, label_np, batch_size, make_graph=True)
        outputs_list, labels_list = evaluate(model, graph_testloader)
        outputs_np = np.array(outputs_list)
        labels_np = np.array(labels_list)
        print(labels_np.shape, outputs_np.shape)
        df = generate_sequence_result(outputs_np, labels_np, tactical_action_name_list)
        # CSVファイルに保存
        df.to_csv(output_file, index=False)
        print(f"Output file saved to {output_file}")

    else:
        # numpy load
        sequence_np, label_np = googledrive_download(bepro=True) 
        print(sequence_np.shape, label_np.shape)
        train_loader, val_loader, test_loader = init_dataset(sequence_np, label_np, batch_size)
        outputs_list, labels_list = evaluate(model, test_loader)
        outputs_np = np.array(outputs_list)
        labels_np = np.array(labels_list)
        print(labels_np.shape, outputs_np.shape)
        get_error(outputs_np, labels_np, tactical_action_name_list, output_dir=f"output/{model_name}/{model_name}")


def evaluate(model, loader):
    model = model.to(device)
    model.eval()

    outputs_list = []
    labels_list = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels_i = data
            inputs,labels_i = inputs.to(device), labels_i.to(device)
            labels_i_list = labels_i.tolist()

            outputs_i = model(inputs)
            # outputs_i = F.softmax(outputs_i, dim = 1)
            outputs_i_list = outputs_i.tolist()

            for j in range(len(outputs_i)):
                outputs_list.append(outputs_i_list[j])

            for j in range(len(labels_i)):
                labels_list.append(labels_i_list[j])

    return outputs_list, labels_list


# modelの評価
def get_error(outputs_np, labels_np, tactical_action_name_list, output_dir="."):
    num_actions = labels_np.shape[1]
    num_tactics = num_actions // 2  # 9戦術

    # 戦術名の統一（チーム名を削除）
    unique_tactical_action_names = [name.rsplit(" ", 1)[0] for name in tactical_action_name_list[:num_tactics]]


    # === 回帰タスクの評価 ===
    abs_errors = np.abs(outputs_np - labels_np)
    squared_errors = (outputs_np - labels_np) ** 2
    
    mae = np.mean(abs_errors, axis=0)
    mse = np.mean(squared_errors, axis=0)
    rmse = np.sqrt(mse)
    
    overall_mae = np.mean(abs_errors)
    overall_mse = np.mean(squared_errors)
    overall_rmse = np.sqrt(overall_mse)

    # 9戦術ごとに平均を計算
    mae_avg = (mae[:num_tactics] + mae[num_tactics:]) / 2
    mse_avg = (mse[:num_tactics] + mse[num_tactics:]) / 2
    rmse_avg = (rmse[:num_tactics] + rmse[num_tactics:]) / 2
    
    regression_df = pd.DataFrame({
        "Tactical Action": unique_tactical_action_names,
        "MAE": mae_avg,
        "MSE": mse_avg,
        "RMSE": rmse_avg
    })
    
    overall_regression_df = pd.DataFrame({
        "Tactical Action": ["Overall"],
        "MAE": [overall_mae],
        "MSE": [overall_mse],
        "RMSE": [overall_rmse]
    })
    
    regression_df = pd.concat([regression_df, overall_regression_df], ignore_index=True)
    regression_df.to_csv(f"{output_dir}_regression_metrics.csv", index=False)


    # === チーム内順位の分析 ===
    num_teams = 2  # チーム数（1チーム9戦術）
    num_actions_per_team = num_actions // num_teams  # 各チームの戦術数（9）

    total_count = np.zeros(num_actions)  # 各戦術でラベルが1.0の回数
    top1_count = np.zeros(num_actions)  # ラベルが1.0のとき、チーム内1位を取った回数
    top2_count = np.zeros(num_actions)  # ラベルが1.0のとき、チーム内2位を取った回数
    top3_count = np.zeros(num_actions)  # ラベルが1.0のとき、チーム内3位を取った回数

    for i in range(labels_np.shape[0]):
        label_indices = np.where(labels_np[i] >= 0.75)[0]  # ラベルが1.0の戦術のインデックス

        if len(label_indices) == 0:
            continue

        # チームごとの予測値ランキングを計算
        for team_id in range(num_teams):
            start_idx = team_id * num_actions_per_team
            end_idx = (team_id + 1) * num_actions_per_team
            team_pred_ranking = np.argsort(outputs_np[i, start_idx:end_idx])[::-1] + start_idx  # 降順ソート

            for label_idx in label_indices:
                if start_idx <= label_idx < end_idx:  # その戦術が該当チームに属する場合
                    total_count[label_idx] += 1
                    if label_idx == team_pred_ranking[0]:
                        top1_count[label_idx] += 1
                    if label_idx in team_pred_ranking[:2]:  # 2位以内
                        top2_count[label_idx] += 1
                    if label_idx in team_pred_ranking[:3]:  # 3位以内
                        top3_count[label_idx] += 1

    aggregated_data = defaultdict(lambda: {"Total Count (1.0)": 0, "Top-1 Count": 0, "Top-2 Count": 0, "Top-3 Count": 0})

    for idx, action in enumerate(tactical_action_name_list):
        base_action = action.rsplit(" ", 1)[0]  # "Build up 1" → "Build up"
        
        aggregated_data[base_action]["Total Count (1.0)"] += total_count[idx]
        aggregated_data[base_action]["Top-1 Count"] += top1_count[idx]
        aggregated_data[base_action]["Top-2 Count"] += top2_count[idx]
        aggregated_data[base_action]["Top-3 Count"] += top3_count[idx]

    # 統合データをDataFrameに変換
    final_data = []
    for action, values in aggregated_data.items():
        total = values["Total Count (1.0)"]
        final_data.append({
            "Tactical Action": action,
            "Total Count (1.0)": total,
            "Top-1 Count": values["Top-1 Count"],
            "Top-1 Ratio": values["Top-1 Count"] / max(total, 1),
            "Top-2 Count": values["Top-2 Count"],
            "Top-2 Ratio": values["Top-2 Count"] / max(total, 1),
            "Top-3 Count": values["Top-3 Count"],
            "Top-3 Ratio": values["Top-3 Count"] / max(total, 1),
        })

    top_k_df = pd.DataFrame(final_data)

    # 全体の平均を追加
    overall_top_k_df = pd.DataFrame({
        "Tactical Action": ["Overall"],
        "Total Count (1.0)": [np.sum(top_k_df["Total Count (1.0)"])],
        "Top-1 Count": [np.sum(top_k_df["Top-1 Count"])],
        "Top-1 Ratio": [np.mean(top_k_df["Top-1 Ratio"])],
        "Top-2 Count": [np.sum(top_k_df["Top-2 Count"])],
        "Top-2 Ratio": [np.mean(top_k_df["Top-2 Ratio"])],
        "Top-3 Count": [np.sum(top_k_df["Top-3 Count"])],
        "Top-3 Ratio": [np.mean(top_k_df["Top-3 Ratio"])],
    })

    # CSV出力
    top_k_df = pd.concat([top_k_df, overall_top_k_df], ignore_index=True)
    top_k_df.to_csv(f"{output_dir}_top_k_analysis.csv", index=False)


    # === 二値分類タスクの評価 ===
    binarized_labels = (labels_np >= 0.75).astype(int)
    best_thresholds, best_f1_scores = optimize_threshold(outputs_np, binarized_labels)
    binarized_outputs = (outputs_np >= best_thresholds).astype(int)
    
    accuracy = np.mean(binarized_labels == binarized_outputs, axis=0)
    recall = recall_score(binarized_labels, binarized_outputs, average=None, zero_division=0)
    precision = precision_score(binarized_labels, binarized_outputs, average=None, zero_division=0)
    f1 = f1_score(binarized_labels, binarized_outputs, average=None, zero_division=0)

    best_thresholds_avg = (best_thresholds[:num_tactics] + best_thresholds[num_tactics:]) / 2
    accuracy_avg = (accuracy[:num_tactics] + accuracy[num_tactics:]) / 2
    recall_avg = (recall[:num_tactics] + recall[num_tactics:]) / 2
    precision_avg = (precision[:num_tactics] + precision[num_tactics:]) / 2
    f1_avg = (f1[:num_tactics] + f1[num_tactics:]) / 2
    
    classification_df = pd.DataFrame({
        "Tactical Action": unique_tactical_action_names,
        "Threshold": best_thresholds_avg,
        "Accuracy": accuracy_avg,
        "Recall": recall_avg,
        "Precision": precision_avg,
        "F1-score": f1_avg
    })
    
    overall_classification_df = pd.DataFrame({
        "Tactical Action": ["Overall"],
        "Threshold": [np.mean(best_thresholds)],
        "Accuracy": [accuracy_score(binarized_labels.flatten(), binarized_outputs.flatten())],
        "Recall": [recall_score(binarized_labels, binarized_outputs, average="macro", zero_division=0)],
        "Precision": [precision_score(binarized_labels, binarized_outputs, average="macro", zero_division=0)],
        "F1-score": [f1_score(binarized_labels, binarized_outputs, average="macro", zero_division=0)]
    })
    
    classification_df = pd.concat([classification_df, overall_classification_df], ignore_index=True)
    classification_df.to_csv(f"{output_dir}_classification_metrics.csv", index=False)
    
    print("Evaluation completed. Metrics saved in CSV files.")


def optimize_threshold(outputs_np, labels_np):
    thresholds = np.linspace(0, 1, 100)
    best_thresholds = []
    best_f1_scores = []
    
    for i in range(labels_np.shape[1]):
        precision, recall, thresh = precision_recall_curve(labels_np[:, i], outputs_np[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.nanargmax(f1_scores)  # NaN回避
        best_thresholds.append(thresh[best_idx] if best_idx < len(thresh) else 0.5)
        best_f1_scores.append(f1_scores[best_idx])
    
    return np.array(best_thresholds), np.array(best_f1_scores)


def generate_sequence_result(outputs_np, labels_np, tactical_action_name_list, half=1):

    # 新しいdf
    sequence_outcome_df = pd.DataFrame()

    slice_len = len(labels_np) % len(outputs_np)
    labels_np = np.delete(labels_np, slice(len(labels_np) - slice_len, len(labels_np)), 0)

    for j in range(len(tactical_action_name_list)):
        sequence_outcome_df[tactical_action_name_list[j]] = labels_np[:, j]
        sequence_outcome_df['output_' + tactical_action_name_list[j]] = outputs_np[:, j]

    return sequence_outcome_df


if __name__ == '__main__':
    main()