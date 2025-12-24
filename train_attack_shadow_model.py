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
    loader = importlib.machinery.SourceFileLoader(module_name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _build_dataloader(x_data: np.ndarray, y_data: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.long)
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
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)
    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='location')
    parser.add_argument('-adv', default='adv1')
    args = parser.parse_args()

    torch.manual_seed(1000)
    np.random.seed(1000)

    dataset = args.dataset
    input_data = input_data_class.InputData(dataset=dataset)
    config = configparser.ConfigParser()
    config.read('config.ini')

    num_classes = int(config[dataset]["num_classes"])
    save_model = True
    user_epochs = int(config[dataset]["user_epochs"])
    batch_size = int(config[dataset]["attack_shallow_model_batch_size"])
    result_folder = config[dataset]["result_folder"]
    network_architecture = str(config[dataset]["network_architecture"])
    network_name = str(config[dataset]["network_name"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fccnet = _load_network_module(network_architecture, network_name)

    (x_train, y_train), (x_test, y_test) = input_data.input_data_attacker_shallow_model_adv1()

    input_shape = x_train.shape[1:]
    model = fccnet.model_user(input_shape=input_shape, labels_dim=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_loader = _build_dataloader(x_train, y_train, batch_size, shuffle=True)
    test_loader = _build_dataloader(x_test, y_test, batch_size, shuffle=False)

    print(x_train.shape)
    print(x_test.shape)

    for epoch in range(user_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 150 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print("Learning rate: {:.6f}".format(optimizer.param_groups[0]['lr']))

        if (epoch + 1) % 100 == 0:
            train_loss, train_acc = _evaluate(model, train_loader, device, criterion)
            test_loss, test_acc = _evaluate(model, test_loader, device, criterion)
            print("Epochs: {}".format(epoch))
            print('Test loss: {:.4f}'.format(test_loss))
            print('Test accuracy: {:.4f}'.format(test_acc))
            print('Train loss: {:.4f}'.format(train_loss))
            print('Train accuracy: {:.4f}'.format(train_acc))

    if save_model:
        weights_dir = os.path.join(result_folder, "models")
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, "epoch_{}_weights_attack_shallow_model_{}.pt".format(user_epochs, args.adv))
        torch.save({'state_dict': model.state_dict()}, weights_path)
        print("Saved shadow model weights to {}".format(weights_path))


if __name__ == "__main__":
    main()