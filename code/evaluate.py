import torch
import pandas as pd
import numpy as np
import argparse
from LSTM.LSTM_kuroda import LSTMClassification
from get_dataset import googledrive_download, init_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# GPUチェック
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--output_file')
    parser.add_argument('--make_graph', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()
    model_name = args.model
    output_file = args.output_file
    make_graph = args.make_graph

    # the number of players
    num_player = 22

    batch_size = 2048
    input_dim = (num_player + 1) * 2
    hidden_dim = 20
    target_size = 18

    # 各戦術的行動の名前 
    tactical_action_name_list = ['Build up 1', 'Progression 1', 'Final third 1', 'Counter-attack 1', 'High press 1', 'Mid block 1', 'Low block 1', 'Counter-press 1', 'Recovery 1', 'Build up 2', 'Progression 2', 'Final third 2', 'Counter-attack 2', 'High press 2', 'Mid block 2', 'Low block 2', 'Counter-press 2', 'Recovery 2']

    # 学習済みモデルのロード
    model = LSTMClassification(input_dim=input_dim, hidden_dim=hidden_dim, target_size=target_size)
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
        get_error(outputs_np, labels_np, tactical_action_name_list)


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
    
    # === 回帰タスクの評価 ===
    abs_errors = np.abs(outputs_np - labels_np)
    squared_errors = (outputs_np - labels_np) ** 2
    
    mae = np.mean(abs_errors, axis=0)
    mse = np.mean(squared_errors, axis=0)
    rmse = np.sqrt(mse)
    
    overall_mae = np.mean(abs_errors)
    overall_mse = np.mean(squared_errors)
    overall_rmse = np.sqrt(overall_mse)
    
    regression_df = pd.DataFrame({
        "Tactical Action": tactical_action_name_list,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })
    
    overall_regression_df = pd.DataFrame({
        "Tactical Action": ["Overall"],
        "MAE": [overall_mae],
        "MSE": [overall_mse],
        "RMSE": [overall_rmse]
    })
    
    regression_df = pd.concat([regression_df, overall_regression_df], ignore_index=True)
    regression_df.to_csv(f"{output_dir}_regression_metrics.csv", index=False)
    
    # === Top-1, Top-2, Top-3 Accuracy ===
    top1_correct_per_action = np.argmax(outputs_np, axis=1) == np.argmax(labels_np, axis=1)
    top1_accuracy_per_action = np.mean(top1_correct_per_action, axis=0)
    
    top_k_correct = np.zeros((labels_np.shape[0], 3, num_actions))
    for i in range(labels_np.shape[0]):
        top_k_preds = np.argsort(outputs_np[i])[-3:][::-1]  # 上位3つの予測
        for j in range(num_actions):
            top_k_correct[i, 0, j] = np.argmax(labels_np[i]) in top_k_preds[:1]  # Top-1
            top_k_correct[i, 1, j] = np.argmax(labels_np[i]) in top_k_preds[:2]  # Top-2
            top_k_correct[i, 2, j] = np.argmax(labels_np[i]) in top_k_preds[:3]  # Top-3
    
    top_accuracies_per_action = np.mean(top_k_correct, axis=0)
    
    top_k_df = pd.DataFrame({
        "Tactical Action": tactical_action_name_list,
        "Top-1 Accuracy": top_accuracies_per_action[0],
        "Top-2 Accuracy": top_accuracies_per_action[1],
        "Top-3 Accuracy": top_accuracies_per_action[2]
    })
    
    overall_top_k_df = pd.DataFrame({
        "Tactical Action": ["Overall"],
        "Top-1 Accuracy": [np.mean(top_accuracies_per_action[0])],
        "Top-2 Accuracy": [np.mean(top_accuracies_per_action[1])],
        "Top-3 Accuracy": [np.mean(top_accuracies_per_action[2])]
    })
    
    top_k_df = pd.concat([top_k_df, overall_top_k_df], ignore_index=True)
    top_k_df.to_csv(f"{output_dir}_top_k_accuracy.csv", index=False)
    
    # === 分類タスクの評価 ===
    binarized_labels = (labels_np > 0.50).astype(int)
    binarized_outputs = (outputs_np > 0.50).astype(int)
    
    accuracy = np.mean(binarized_labels == binarized_outputs, axis=0)
    recall = recall_score(binarized_labels, binarized_outputs, average=None, zero_division=0)
    precision = precision_score(binarized_labels, binarized_outputs, average=None, zero_division=0)
    f1 = f1_score(binarized_labels, binarized_outputs, average=None, zero_division=0)
    
    classification_df = pd.DataFrame({
        "Tactical Action": tactical_action_name_list,
        "Accuracy": accuracy,
        "Recall": recall,
        "Precision": precision,
        "F1-score": f1
    })
    
    overall_classification_df = pd.DataFrame({
        "Tactical Action": ["Overall"],
        "Accuracy": [accuracy_score(binarized_labels.flatten(), binarized_outputs.flatten())],
        "Recall": [recall_score(binarized_labels, binarized_outputs, average="macro", zero_division=0)],
        "Precision": [precision_score(binarized_labels, binarized_outputs, average="macro", zero_division=0)],
        "F1-score": [f1_score(binarized_labels, binarized_outputs, average="macro", zero_division=0)]
    })
    
    classification_df = pd.concat([classification_df, overall_classification_df], ignore_index=True)
    classification_df.to_csv(f"{output_dir}_classification_metrics.csv", index=False)
    
    print("Evaluation completed. Metrics saved in CSV files.")


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