import torch
import torch.nn as nn
from torch import optim
import time
import numpy as np

from transformer_kuroda import TransformerClassification

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from get_dataset import googledrive_download, init_dataset

# GPUチェック
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


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
    num_epochs = 500
    lr = 0.0001

    # 各戦術的行動の名前 
    tactical_action_name_list = ['Build up 1', 'Progression 1', 'Final third 1', 'Counter-attack 1', 'High press 1', 'Mid block 1', 'Low block 1', 'Counter-press 1', 'Recovery 1', 'Build up 2', 'Progression 2', 'Final third 2', 'Counter-attack 2', 'High press 2', 'Mid block 2', 'Low block 2', 'Counter-press 2', 'Recovery 2']

    # numpy load
    sequence_np, label_np = googledrive_download(bepro=True) # _0_or_1, 
    print(sequence_np.shape, label_np.shape)

    train_loader, val_loader, test_loader = init_dataset(sequence_np, label_np, batch_size)

    # モデルを定義
    model = TransformerClassification(input_dim, hidden_dim, target_size, num_heads, num_layers)

    history, model = train(train_loader, val_loader, test_loader, model, num_epochs, lr)


def train(train_loader, valid_loader, test_loader, model, num_epochs, lr, patience=int(20), mode='only_fine_tuning'): # pretrain, fine_tuning, only_fine_tuning
    
    model = model.to(device)
    loss_function = nn.HuberLoss() # SmoothL1, CrossEntropyLoss, HuberLoss
    optimizer = optim.SGD(model.parameters(), lr=lr)

    history = {
        'loss': []
    }
    best_train_loss = float("inf")
    best_valid_loss = float("inf")
    test_loss_best_valid = float("inf")
    total_train_loss = None
    train_losses = []
    val_losses = []
    no_improvement = 0
    for epoch in range(num_epochs):
        print(f"\nepoch: {epoch}", flush=True)

        model.eval()
        total_valid_loss = 0.0
        n_valid = 0
        with torch.no_grad():
            for valid_tensors in valid_loader:
                inputs, labels = valid_tensors
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.long()
                model.zero_grad()
                tag_scores = model(inputs)
                val_loss = loss_function(tag_scores, labels.float())
                val_losses.append(float(val_loss))
                total_valid_loss += val_loss.item()
                n_valid += 1

            print(tag_scores.view(labels.shape), flush=True)
            print(labels.view(labels.shape), flush=True)

        total_valid_loss /= n_valid

        if total_valid_loss < best_valid_loss:
            best_valid_loss = total_valid_loss
            no_improvement = 0
            torch.save(optimizer.state_dict(), f"model/transformer_model/{mode}_best_optimizer.pth")
            torch.save(model.state_dict(), f"model/transformer_model/{mode}_best_params.pth")

        elif no_improvement < int(patience):
            no_improvement = int(no_improvement) + 1
            if no_improvement == patience:
                print("Now training full model.")
                train_params = [params for params in model.parameters()]
                optimizer = optim.Adam(train_params, lr=lr)

        print(f"total_train_loss: {total_train_loss}")
        print(f"best_train_loss: {best_train_loss}")
        print(f"total_valid_loss: {total_valid_loss}")
        print(f"best_valid_loss: {best_valid_loss}")
        print(f"test_loss_best_valid: {test_loss_best_valid}")

        model.train()
        total_train_loss = 0.0
        n_train = 0
        start_time = time.time()
        for (train_idx, train_tensors) in enumerate(train_loader):
            inputs, labels = train_tensors
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()
            model.zero_grad()
            tag_scores = model(inputs)
            train_loss = loss_function(tag_scores, labels.float())
            train_loss.backward()
            optimizer.step()
            train_losses.append(float(train_loss))
            total_train_loss += train_loss.item()
            n_train += 1

        epoch_time = time.time() - start_time

        total_train_loss /= n_train
        if total_train_loss < best_train_loss:
            best_train_loss = total_train_loss

        print(f"epoch_time: {epoch_time:.2f}", flush=True)

    avg_train_loss = np.mean(train_losses)
    history['loss'].append(avg_train_loss)
    print("Epoch {} / {}: train_Loss = {:.3f}".format(epoch+1, num_epochs, avg_train_loss))

    avg_val_loss = np.mean(val_losses)
    history['loss'].append(avg_val_loss)
    print("Epoch {} / {}: val_Loss = {:.3f}".format(epoch+1, num_epochs, avg_val_loss))

    return history, model


if __name__ == "__main__":
    main()