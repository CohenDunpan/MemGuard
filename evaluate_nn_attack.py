import argparse
import importlib.machinery
import importlib.util
import os

import configparser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import input_data_class


def _load_network_module(path: str, module_name: str):
    full_path = path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    loader = importlib.machinery.SourceFileLoader(module_name, full_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _build_dataloader(x_data: np.ndarray, y_data: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, criterion) -> tuple:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = torch.sigmoid(logits) > 0.5
            total_correct += (preds.float() == batch_y).sum().item()
            total_samples += batch_x.size(0)
    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='location')
    parser.add_argument('-scenario', default='full')
    parser.add_argument('-adv', default='adv1')
    parser.add_argument('-version', default='v0')
    args = parser.parse_args()

    torch.manual_seed(10000)
    np.random.seed(10000)

    dataset = args.dataset
    input_data = input_data_class.InputData(dataset=dataset)
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
    config.read(config_path)

    user_label_dim = int(config[dataset]["num_classes"])
    num_classes = 1
    epochs = int(config[dataset]["attack_epochs"])
    user_epochs = int(config[dataset]["user_epochs"])
    attack_epochs = int(config[dataset]["attack_shallow_model_epochs"])
    batch_size = int(config[dataset]["defense_batch_size"])
    result_folder = config[dataset]["result_folder"]
    network_architecture = str(config[dataset]["network_architecture"])
    network_name = str(config[dataset]["network_name"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fccnet = _load_network_module(network_architecture, network_name)

    # Load defense data
    x_evaluate, y_evaluate, l_evaluate = input_data.input_data_attacker_evaluate()
    evaluation_noise_filepath = os.path.join(result_folder, "attack", "noise_data_evaluation.npz")
    print(evaluation_noise_filepath)
    if not os.path.isfile(evaluation_noise_filepath):
        raise FileNotFoundError
    npz_defense = np.load(evaluation_noise_filepath)
    f_evaluate_noise = npz_defense['defense_output']
    f_evaluate_origin = npz_defense['tc_output']

    f_evaluate_defense = np.zeros(f_evaluate_noise.shape, dtype=np.float32)
    np.random.seed(100)
    for i in np.arange(f_evaluate_defense.shape[0]):
        f_evaluate_defense[i, :] = f_evaluate_noise[i, :]

    # Load attacker's shadow model
    x_train, y_train, l_train = input_data.input_data_attacker_adv1()
    shadow_weights_path = os.path.join(result_folder, "models", "epoch_{}_weights_attack_shallow_model_{}.pt".format(user_epochs, args.adv))
    if not os.path.isfile(shadow_weights_path):
        raise FileNotFoundError("Shadow model weights not found at {}".format(shadow_weights_path))
    shadow_model = fccnet.model_user(input_shape=x_train.shape[1:], labels_dim=user_label_dim).to(device)
    shadow_state = torch.load(shadow_weights_path, map_location=device)
    shadow_model.load_state_dict(shadow_state['state_dict'])
    shadow_model.eval()

    with torch.no_grad():
        logits_train = []
        loader_shadow = DataLoader(torch.tensor(x_train, dtype=torch.float32), batch_size=batch_size, shuffle=False)
        for batch in loader_shadow:
            batch = batch.to(device)
            logits = shadow_model(batch)
            logits_train.append(torch.softmax(logits, dim=1).cpu().numpy())
        f_train = np.concatenate(logits_train, axis=0)
    del shadow_model

    f_train = np.sort(f_train, axis=1)
    f_evaluate_defense = np.sort(f_evaluate_defense, axis=1)
    f_evaluate_origin = np.sort(f_evaluate_origin, axis=1)

    if args.scenario == 'full':
        b_train = f_train[:, :]
        b_test = f_evaluate_defense[:, :]
        b_test_origin = f_evaluate_origin[:, :]
    else:
        raise NotImplementedError

    label_train = l_train.astype(np.float32)
    label_test = l_evaluate.astype(np.float32)

    input_shape = b_train.shape[1:]
    attack_model = fccnet.model_attack_nn(input_shape=input_shape, labels_dim=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(attack_model.parameters(), lr=0.01)

    train_loader = _build_dataloader(b_train, label_train, batch_size, shuffle=True)
    test_loader_defense = _build_dataloader(b_test, label_test, batch_size, shuffle=False)
    test_loader_origin = _build_dataloader(b_test_origin, label_test, batch_size, shuffle=False)

    for epoch in range(epochs):
        attack_model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = attack_model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 300 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print("Learning rate: {:.6f}".format(optimizer.param_groups[0]['lr']))

        if (epoch + 1) % 100 == 0:
            print("Epochs: {}".format(epoch))
            loss_defense, acc_defense = _evaluate(attack_model, test_loader_defense, device, criterion)
            loss_nodefense, acc_nodefense = _evaluate(attack_model, test_loader_origin, device, criterion)
            loss_train, acc_train = _evaluate(attack_model, train_loader, device, criterion)
            print('Test loss defense: {:.4f}'.format(loss_defense))
            print('Test accuracy defense: {:.4f}'.format(acc_defense))
            print('Test loss no defense: {:.4f}'.format(loss_nodefense))
            print('Test accuracy no defense: {:.4f}'.format(acc_nodefense))
            print('Train loss: {:.4f}'.format(loss_train))
            print('Train accuracy: {:.4f}'.format(acc_train))

    result_filepath = os.path.join(result_folder, config[dataset]["result_file_publish"])
    print(result_folder)
    os.makedirs(result_folder, exist_ok=True)
    if not os.path.isfile(result_filepath):
        fp = open(result_filepath, 'w+')
        fp.close()

    evaluation_noise_filepath = os.path.join(result_folder, "attack", "noise_data_evaluation.npz")
    if not os.path.isfile(evaluation_noise_filepath):
        raise FileNotFoundError
    npz_defense = np.load(evaluation_noise_filepath)
    f_evaluate_noise = npz_defense['defense_output']
    f_evaluate_origin = npz_defense['tc_output']
    f_evaluate_origin_score = npz_defense['predict_origin']
    f_evaluate_defense_score = npz_defense['predict_modified']

    with torch.no_grad():
        predict_result_origin = (torch.sigmoid(
            attack_model(torch.tensor(np.sort(f_evaluate_origin, axis=1), dtype=torch.float32, device=device))).cpu().numpy() > 0.5).astype(int)
        predict_result_defense = (torch.sigmoid(
            attack_model(torch.tensor(np.sort(f_evaluate_noise, axis=1), dtype=torch.float32, device=device))).cpu().numpy() > 0.5).astype(int)

    predict_result_origin = predict_result_origin.reshape(-1)
    predict_result_defense = predict_result_defense.reshape(-1)
    label_test = label_test.reshape(-1)

    epsilon_value_list = [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]
    inference_accuracy_list = []

    for epsilon_value in epsilon_value_list:
        inference_accuracy = 0.0

        np.random.seed(100)
        for i in np.arange(f_evaluate_origin.shape[0]):
            distortion_noise = np.sum(np.abs(f_evaluate_origin[i, :] - f_evaluate_noise[i, :]))
            p_value = 0.0
            if np.abs(f_evaluate_origin_score[i] - 0.5) <= np.abs(f_evaluate_defense_score[i] - 0.5):
                p_value = 0.0
            else:
                p_value = min(epsilon_value / max(distortion_noise, 1e-12), 1.0)

            if predict_result_origin[i] == label_test[i]:
                inference_accuracy += 1.0 - p_value
            if predict_result_defense[i] == label_test[i]:
                inference_accuracy += p_value
        inference_accuracy_list.append(inference_accuracy / float(f_evaluate_origin.shape[0]))

    print("Budget list: {}".format(epsilon_value_list))
    print("inference accuracy list: {}".format(inference_accuracy_list))


if __name__ == "__main__":
    main()
