import torch.nn.functional as F

def classification_update(pred, label, easy_model=False):
    pred_list = []
    label_list = []
    correct_count = 0
    count = 0
    loss = 0

    pred = pred.cpu()
    label = label.cpu()

    if easy_model:
        pass
    else:
        loss = F.cross_entropy(pred, label).item() * len(label)
        pred = pred.data.max(1)[1]
    pred_list.extend(pred.numpy())
    label_list.extend(label.numpy())
    correct_count += pred.eq(label.data.view_as(pred)).sum()
    count += len(label)

    return correct_count, count, loss

def classification_results(correct_count, total_count, loss):

    result_dict = {}
    result_dict['acc'] = float(correct_count) / float(total_count)
    result_dict['loss'] = float(loss) / float(total_count)

    return result_dict
