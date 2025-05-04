import torch

def Dict_weight(dict_in, weight_in):
    for k,v in dict_in.items():
        dict_in[k] = weight_in*v
    return dict_in

def Dict_Add(dict1, dict2):
    for k,v in dict1.items():
        dict1[k] = v + dict2[k]
    return dict1

def Dict_Minus(dict1, dict2):
    for k,v in dict1.items():
        dict1[k] = v - dict2[k]
    return dict1

def Cal_Weight_Dict(dataset_dict, site_list=None):
    if site_list is None:
        site_list = list(dataset_dict.keys())
    weight_dict = {}
    total_len = 0
    for site_name in site_list:
        total_len += len(dataset_dict[site_name]['test'])
    for site_name in site_list:
        site_len = len(dataset_dict[site_name]['test'])
        weight_dict[site_name] = site_len/total_len
    return weight_dict


def get_invariant_adapter(model):
    invariant_adapter_param = {}
    for name, param in model.named_parameters():
        if 'adapter_invariant_down' in name or 'adapter_invariant_up' in name or 'adapter_norm' in name:
            invariant_adapter_param[name] = param
    return invariant_adapter_param

def get_aware_adapter(model):
    aware_adapter_param = {}
    for name, param in model.named_parameters():
        if 'adapter_aware_down' in name or 'adapter_aware_up' in name or 'adapter_norm' in name:
            aware_adapter_param[name] = param
    return aware_adapter_param


def update_adapter_state_dict(model, adapter_state):
    model_state = model.state_dict()
    for name, param in adapter_state.items():
        model_state[name].copy_(param)


def MomentumUpdate(model, teacher, alpha=0.99):
    teacher_dict = teacher.state_dict()
    model_dict = model.state_dict()
    for k,v in teacher_dict.items():
        teacher_dict[k] = alpha * v + (1-alpha)*model_dict[k]
    teacher.load_state_dict(teacher_dict)


def aggregate(grads_list, weight_dict):
    aggregated = []
    client_weights = [weight_dict[domain_name] for domain_name in weight_dict.keys()]

    for grad_layer in zip(*grads_list):
        weighted_sum = torch.zeros_like(grad_layer[0])
        for grad, weight in zip(grad_layer, client_weights):
            weighted_sum += grad * weight
        aggregated.append(weighted_sum)
    return aggregated


def apply_grads(model_parameters, aggregated_grads, optimizer, c_ci):
    optimizer.zero_grad()
    for param, grad in zip(model_parameters, aggregated_grads):
        param.grad = grad
    optimizer.step(c_ci)

