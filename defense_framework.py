import argparse
import importlib.machinery
import importlib.util
import os

import configparser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import input_data_class


def _load_network_module(path: str, module_name: str):
    full_path = path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    loader = importlib.machinery.SourceFileLoader(module_name, full_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _evaluate_binary(model: torch.nn.Module, features: np.ndarray, labels: np.ndarray, device: torch.device) -> tuple:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(torch.tensor(features, dtype=torch.float32), batch_size=128, shuffle=False)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    start = 0
    with torch.no_grad():
        for batch in loader:
            end = start + batch.size(0)
            target = labels_tensor[start:end].to(device)
            batch = batch.to(device)
            logits = model(batch).squeeze(-1)
            loss = criterion(logits, target)
            total_loss += loss.item() * batch.size(0)
            preds = torch.sigmoid(logits) > 0.5
            total_correct += (preds.float() == target).sum().item()
            total_samples += batch.size(0)
            start = end
    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def main():
    config = configparser.ConfigParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('-qt', type=str, default='evaluation')
    parser.add_argument('-dataset', default='location')
    args = parser.parse_args()

    torch.manual_seed(1000)
    np.random.seed(1000)

    dataset = args.dataset
    input_data = input_data_class.InputData(dataset=dataset)
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
    config.read(config_path)

    user_label_dim = int(config[dataset]["num_classes"])
    num_classes = 1

    user_epochs = int(config[dataset]["user_epochs"])
    defense_epochs = int(config[dataset]["defense_epochs"])
    result_folder = config[dataset]["result_folder"]
    network_architecture = str(config[dataset]["network_architecture"])
    network_name = str(config[dataset]["network_name"])

    print("Config: ")
    print("dataset: {}".format(dataset))
    print("result folder: {}".format(result_folder))
    print("network architecture: {}".format(network_architecture))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fccnet = _load_network_module(network_architecture, network_name)

    print("Loading Evaluation dataset...")
    x_evaluate, y_evaluate, l_evaluate = input_data.input_data_attacker_evaluate()

    print("Loading target model...")
    user_weights_path = os.path.join(result_folder, "models", "epoch_{}_weights_user.pt".format(user_epochs))
    if not os.path.isfile(user_weights_path):
        raise FileNotFoundError("User model weights not found at {}".format(user_weights_path))
    user_model = fccnet.model_user(input_shape=x_evaluate.shape[1:], labels_dim=user_label_dim).to(device)
    user_state = torch.load(user_weights_path, map_location=device)
    user_model.load_state_dict(user_state['state_dict'])
    user_model.eval()

    batch_predict = 100
    loader_eval = DataLoader(torch.tensor(x_evaluate, dtype=torch.float32), batch_size=batch_predict, shuffle=False)
    f_evaluate_list = []
    f_evaluate_logits_list = []
    with torch.no_grad():
        for batch in loader_eval:
            batch = batch.to(device)
            logits = user_model(batch)
            probs = torch.softmax(logits, dim=1)
            f_evaluate_logits_list.append(logits.cpu().numpy())
            f_evaluate_list.append(probs.cpu().numpy())
    f_evaluate = np.concatenate(f_evaluate_list, axis=0)
    f_evaluate_logits = np.concatenate(f_evaluate_logits_list, axis=0)
    del user_model

    f_evaluate_origin = np.copy(f_evaluate)
    f_evaluate_logits_origin = np.copy(f_evaluate_logits)

    sort_index = np.argsort(f_evaluate, axis=1)
    back_index = np.copy(sort_index)
    for i in np.arange(back_index.shape[0]):
        back_index[i, sort_index[i, :]] = np.arange(back_index.shape[1])
    f_evaluate = np.sort(f_evaluate, axis=1)
    f_evaluate_logits = np.sort(f_evaluate_logits, axis=1)

    print("f evaluate shape: {}".format(f_evaluate.shape))
    print("f evaluate logits shape: {}".format(f_evaluate_logits.shape))

    input_shape = f_evaluate.shape[1:]
    print("Loading defense model...")
    defense_weights_path = os.path.join(result_folder, "models", "epoch_{}_weights_defense.pt".format(defense_epochs))
    if not os.path.isfile(defense_weights_path):
        raise FileNotFoundError("Defense model weights not found at {}".format(defense_weights_path))
    model_opt = fccnet.model_defense_optimize(input_shape=input_shape, labels_dim=num_classes).to(device)
    defense_state = torch.load(defense_weights_path, map_location=device)
    model_opt.load_state_dict(defense_state['state_dict'])
    model_opt.eval()

    eval_loss, eval_acc = _evaluate_binary(model_opt, f_evaluate_logits, l_evaluate, device)
    print('evaluate loss on model: {:.4f}'.format(eval_loss))
    print('evaluate accuracy on model: {:.4f}'.format(eval_acc))

    c1 = 1.0
    c2 = 10.0
    c3_initial = 0.1
    max_iteration = 300

    result_array = np.zeros(f_evaluate.shape, dtype=np.float32)
    result_array_logits = np.zeros(f_evaluate.shape, dtype=np.float32)
    success_fraction = 0.0

    for test_sample_id in np.arange(0, f_evaluate.shape[0]):
        if test_sample_id % 100 == 0:
            print("test sample id: {}".format(test_sample_id))

        max_label = int(np.argmax(f_evaluate[test_sample_id, :]))
        origin_value = torch.tensor(f_evaluate[test_sample_id, :], dtype=torch.float32, device=device).view(1, -1)
        origin_value_logits = torch.tensor(f_evaluate_logits[test_sample_id, :], dtype=torch.float32, device=device).view(1, -1)

        label_mask_array = torch.zeros((1, user_label_dim), dtype=torch.float32, device=device)
        label_mask_array[0, max_label] = 1.0

        def _compute_loss(sample_f: torch.Tensor, c3_value: float):
            sample_f.retain_grad()
            output_logit = model_opt(sample_f).squeeze(-1)
            correct_label = torch.sum(label_mask_array * sample_f, dim=1)
            wrong_label = torch.max((1 - label_mask_array) * sample_f - 1e8 * label_mask_array, dim=1)[0]
            loss1 = torch.abs(output_logit)
            loss2 = F.relu(wrong_label - correct_label)
            loss3 = torch.sum(torch.abs(F.softmax(sample_f, dim=1) - origin_value))
            return c1 * loss1 + c2 * loss2 + c3_value * loss3, output_logit

        sample_f = origin_value_logits.clone().detach().requires_grad_(True)
        with torch.no_grad():
            result_predict_scores_initial = torch.sigmoid(model_opt(sample_f)).item()

        if np.abs(result_predict_scores_initial - 0.5) <= 1e-5:
            success_fraction += 1.0
            result_array[test_sample_id, :] = origin_value.cpu().numpy()[0, back_index[test_sample_id, :]]
            result_array_logits[test_sample_id, :] = origin_value_logits.cpu().numpy()[0, back_index[test_sample_id, :]]
            continue

        last_iteration_result = origin_value.cpu().numpy()[0, back_index[test_sample_id, :]].copy()
        last_iteration_result_logits = origin_value_logits.cpu().numpy()[0, back_index[test_sample_id, :]].copy()

        success = True
        c3_value = c3_initial
        iterate_time = 1

        while success:
            sample_f = origin_value_logits.clone().detach().requires_grad_(True)
            result_max_label = -1
            result_predict_scores = result_predict_scores_initial
            j = 1
            while j < max_iteration and (result_max_label != max_label or (result_predict_scores - 0.5) * (result_predict_scores_initial - 0.5) > 0):
                loss, _ = _compute_loss(sample_f, c3_value)
                loss.backward()
                gradient_values = sample_f.grad.detach()
                grad_norm = torch.norm(gradient_values)
                if grad_norm > 0:
                    gradient_values = gradient_values / grad_norm
                sample_f = (sample_f - 0.1 * gradient_values).detach().requires_grad_(True)
                with torch.no_grad():
                    result_predict_scores = torch.sigmoid(model_opt(sample_f)).item()
                    result_max_label = int(torch.argmax(sample_f).item())
                j += 1

            if max_label != result_max_label:
                if iterate_time == 1:
                    print("failed sample for label not same for id: {}, c3:{} not add noise".format(test_sample_id, c3_value))
                    success_fraction -= 1.0
                break

            if (result_predict_scores - 0.5) * (result_predict_scores_initial - 0.5) > 0:
                if iterate_time == 1:
                    with torch.no_grad():
                        max_score = torch.max(F.softmax(sample_f, dim=1)).item()
                    print("max iteration reached with id: {}, max score: {:.6f}, prediction_score: {:.6f}, c3: {}, not add noise".format(test_sample_id, max_score, result_predict_scores, c3_value))
                break

            with torch.no_grad():
                softmax_sample = F.softmax(sample_f, dim=1).cpu().numpy()[0, back_index[test_sample_id, :]]
                sample_logits_cpu = sample_f.cpu().numpy()[0, back_index[test_sample_id, :]]
                last_iteration_result[:] = softmax_sample
                last_iteration_result_logits[:] = sample_logits_cpu

            iterate_time += 1
            c3_value *= 10
            if c3_value > 100000:
                break

        success_fraction += 1.0
        result_array[test_sample_id, :] = last_iteration_result[:]
        result_array_logits[test_sample_id, :] = last_iteration_result_logits[:]

    print("Success fraction: {}".format(success_fraction / float(f_evaluate.shape[0])))

    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(os.path.join(result_folder, "attack"), exist_ok=True)

    model_defense = fccnet.model_defense(input_shape=input_shape, labels_dim=num_classes).to(device)
    model_defense.load_state_dict(defense_state['state_dict'])
    model_defense.eval()

    with torch.no_grad():
        predict_origin = torch.sigmoid(model_defense(torch.tensor(np.sort(f_evaluate_origin, axis=1), dtype=torch.float32, device=device))).cpu().numpy()
        predict_modified = torch.sigmoid(model_defense(torch.tensor(np.sort(result_array, axis=1), dtype=torch.float32, device=device))).cpu().numpy()

    np.savez(os.path.join(result_folder, "attack", "noise_data_{}.npz".format(args.qt)),
             defense_output=result_array,
             defense_output_logits=result_array_logits,
             tc_output=f_evaluate_origin,
             tc_output_logits=f_evaluate_logits_origin,
             predict_origin=predict_origin,
             predict_modified=predict_modified)


if __name__ == "__main__":
    main()
