import os
import torch
import torch.nn as nn
from configs.default import *
import torch.nn.functional as F
from tqdm import tqdm
import random
from utils.classification_metric import classification_update, classification_results

def Shuffle_Batch_Data(data_in):
    len_total = len(data_in)
    idx_list = list(range(len_total))
    random.shuffle(idx_list)
    return data_in[idx_list]

def site_evaluation(model, dataloader):
    model.eval()
    total_correct_count = 0
    total_count = 0
    total_loss = 0
    with torch.no_grad():
        for imgs, labels, domain_labels, in dataloader:
            imgs = imgs.cuda()
            output = model(imgs)
            correct_count, count, loss = classification_update(output, labels)
            total_correct_count += correct_count
            total_count += count
            total_loss += loss
    results_dict = classification_results(total_correct_count, total_count, total_loss)
    return results_dict


def test_func(comm_round, test_domain, model_dict, log_file, args, dataloader_dict, train_domain_list, note):
    test_dataloader = dataloader_dict[test_domain]['test']
    client_domain = train_domain_list.copy()
    client_domain.remove(test_domain)

    results = {}  # Dictionary to store results for each train domain

    for k in client_domain:
        model = model_dict[k]
        model = model.cuda()
        model.eval()
        total_correct_count = 0
        total_count = 0
        total_loss = 0
        with torch.no_grad():
            for imgs, labels, domain_labels, in test_dataloader:
                imgs = imgs.cuda()
                output = model(imgs)
                correct_count, count, loss = classification_update(output, labels)
                total_correct_count += correct_count
                total_count += count
                total_loss += loss
        results_dict = classification_results(total_correct_count, total_count, total_loss)
        log_file.info(f'{note} Round: {comm_round:3d} | Train Domain: {k} | Test Domain: {test_domain} | Acc: {results_dict["acc"] * 100:.2f}%')
        results[k] = results_dict["acc"]

    return results

def SaveCheckPoint(args, model, epochs, path, optimizer=None, schedule=None, note='best_val'):
    check_dict = {'args': args, 'epochs': epochs, 'model': model.state_dict(), 'note': note}
    if optimizer is not None:
        check_dict['optimizer'] = optimizer.state_dict()
    if schedule is not None:
        check_dict['scheduler'] = schedule.state_dict()
    if not os.path.isdir(path):
        os.makedirs(path)

    torch.save(check_dict, os.path.join(path, note + '.pt'))

def remove_module_prefix(model_state_dict):
    state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
            state_dict[new_key] = value
        else:
            state_dict[key] = value
    return state_dict


def adjust_model_prefix(state_dict, model):
    new_state_dict = {}
    model_state = model.state_dict()
    model_has_module = any(k.startswith('module.') for k in model_state.keys())
    state_has_module = any(k.startswith('module.') for k in state_dict.keys())

    if model_has_module and not state_has_module:
        for k, v in state_dict.items():
            new_state_dict[f'module.{k}'] = v
        return new_state_dict
    elif not model_has_module and state_has_module:
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    else:
        return state_dict



def load_from_checkpoint(checkpoint_path, client_dual_model, client_single_model, dual_model_dict, single_model_dict,
                         client_dual_optimizer, client_single_optimizer, dual_optimizer_dict, single_optimizer_dict,
                         dual_scheduler_dict, single_scheduler_dict, dual_ci_dict, single_ci_dict, dual_c, single_c,
                         weight_dict, train_domain_list, log_file):
    """
    Load models, optimizers, schedulers, and control variables from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        log_file.info(f"Checkpoint not found at {checkpoint_path}. Starting from beginning.")
        return 0, weight_dict, dual_c, single_c, dual_ci_dict, single_ci_dict

    log_file.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Load start round
    start_round = checkpoint['round'] + 1

    # Load weight dictionary
    if 'weight_dict' in checkpoint:
        weight_dict = checkpoint['weight_dict']

    # Load global control variables
    if 'dual_c' in checkpoint:
        dual_c = checkpoint['dual_c']
    if 'single_c' in checkpoint:
        single_c = checkpoint['single_c']

    # Load dual client model
    if 'global_dual_model' in checkpoint:
        state_dict = adjust_model_prefix(checkpoint['global_dual_model'], client_dual_model)
        client_dual_model.load_state_dict(state_dict)

    # Load single client model
    if 'global_single_model' in checkpoint:
        state_dict = adjust_model_prefix(checkpoint['global_single_model'], client_single_model)
        client_single_model.load_state_dict(state_dict)

    # Load domain-specific models, optimizers, schedulers, and control variables
    for domain_name in train_domain_list:
        if f'dual_model_{domain_name}' in checkpoint:
            state_dict = adjust_model_prefix(checkpoint[f'dual_model_{domain_name}'], dual_model_dict[domain_name])
            dual_model_dict[domain_name].load_state_dict(state_dict)

        if f'single_model_{domain_name}' in checkpoint:
            state_dict = adjust_model_prefix(checkpoint[f'single_model_{domain_name}'], single_model_dict[domain_name])
            single_model_dict[domain_name].load_state_dict(state_dict)

        # Load optimizers and schedulers directly (no prefix adjustment needed)
        if f'dual_optimizer_{domain_name}' in checkpoint:
            dual_optimizer_dict[domain_name].load_state_dict(checkpoint[f'dual_optimizer_{domain_name}'])

        if f'single_optimizer_{domain_name}' in checkpoint:
            single_optimizer_dict[domain_name].load_state_dict(checkpoint[f'single_optimizer_{domain_name}'])

        if f'dual_scheduler_{domain_name}' in checkpoint:
            dual_scheduler_dict[domain_name].load_state_dict(checkpoint[f'dual_scheduler_{domain_name}'])

        if f'single_scheduler_{domain_name}' in checkpoint:
            single_scheduler_dict[domain_name].load_state_dict(checkpoint[f'single_scheduler_{domain_name}'])

        if f'dual_ci_{domain_name}' in checkpoint:
            dual_ci_dict[domain_name] = checkpoint[f'dual_ci_{domain_name}']

        if f'single_ci_{domain_name}' in checkpoint:
            single_ci_dict[domain_name] = checkpoint[f'single_ci_{domain_name}']

    # Load client optimizers if they exist in checkpoint
    if 'client_dual_optimizer' in checkpoint:
        client_dual_optimizer.load_state_dict(checkpoint['client_dual_optimizer'])
    if 'client_single_optimizer' in checkpoint:
        client_single_optimizer.load_state_dict(checkpoint['client_single_optimizer'])

    log_file.info(f"Resuming training from round {start_round}")
    return start_round, weight_dict, dual_c, single_c, dual_ci_dict, single_ci_dict



def save_checkpoint_for_resume(args, round_idx, client_dual_model, client_single_model, dual_model_dict, single_model_dict,
                              client_dual_optimizer, client_single_optimizer, dual_optimizer_dict, single_optimizer_dict,
                              dual_scheduler_dict, single_scheduler_dict, dual_ci_dict, single_ci_dict, dual_c, single_c,
                              weight_dict, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'round': round_idx,
        'weight_dict': weight_dict,
        'dual_c': dual_c,
        'single_c': single_c,
        'global_dual_model': client_dual_model.state_dict(),
        'global_single_model': client_single_model.state_dict(),
        'client_dual_optimizer': client_dual_optimizer.state_dict(),
        'client_single_optimizer': client_single_optimizer.state_dict()
    }

    for domain_name in dual_model_dict.keys():
        checkpoint[f'dual_model_{domain_name}'] = dual_model_dict[domain_name].state_dict()
        checkpoint[f'single_model_{domain_name}'] = single_model_dict[domain_name].state_dict()
        checkpoint[f'dual_optimizer_{domain_name}'] = dual_optimizer_dict[domain_name].state_dict()
        checkpoint[f'single_optimizer_{domain_name}'] = single_optimizer_dict[domain_name].state_dict()
        checkpoint[f'dual_scheduler_{domain_name}'] = dual_scheduler_dict[domain_name].state_dict()
        checkpoint[f'single_scheduler_{domain_name}'] = single_scheduler_dict[domain_name].state_dict()
        checkpoint[f'dual_ci_{domain_name}'] = dual_ci_dict[domain_name]
        checkpoint[f'single_ci_{domain_name}'] = single_ci_dict[domain_name]

    torch.save(checkpoint, save_path)
