import numpy as np

def refine_weight_dict_by_GA(weight_dict, site_before_results_dict, site_after_results_dict, step_size=0.1, fair_metric='loss'):

    # Check for NaN values in weight_dict and reset if needed
    for site_name in weight_dict.keys():
        if np.isnan(weight_dict[site_name]):
            for k in weight_dict:
                weight_dict[k] = 1.0 / len(weight_dict)
            return weight_dict

    if fair_metric == 'acc':
        signal = -1.0
    elif fair_metric == 'loss':
        signal = 1.0
    else:
        raise ValueError('fair_metric must be acc or loss')

    value_list = []
    for site_name in site_before_results_dict.keys():
        # Check for NaN values in results
        before_val = site_before_results_dict[site_name][fair_metric]
        after_val = site_after_results_dict[site_name][fair_metric]
        if np.isnan(before_val) or np.isnan(after_val):
            # Return current weights without changes if NaN detected
            return weight_dict
        value_list.append(after_val - before_val)
    value_list = np.array(value_list)
    step_size = 1./4. * step_size

    # Safe normalization with epsilon to prevent division by zero
    max_abs_value = np.max(np.abs(value_list))
    if max_abs_value < 1e-10:  # If all values are very close to zero
        return weight_dict  # Keep weights unchanged

    norm_gap_list = value_list / max_abs_value

    for i, site_name in enumerate(weight_dict.keys()):
        weight_dict[site_name] += signal * norm_gap_list[i] * step_size

        # Immediate check to prevent NaN propagation
        if np.isnan(weight_dict[site_name]):
            weight_dict[site_name] = 1.0 / len(weight_dict)
    weight_dict = weight_clip(weight_dict)

    return weight_dict

def weight_clip(weight_dict):
    new_total_weight = 0.0
    for key_name in weight_dict.keys():
        weight_dict[key_name] = np.clip(weight_dict[key_name], 0.0, 1.0)
        new_total_weight += weight_dict[key_name]

    for key_name in weight_dict.keys():
        weight_dict[key_name] /= new_total_weight

    return weight_dict

