import sys
import torch
import torch.nn as nn
from network.get_network import GetNetwork
from utils.log_utils import *
from torch.utils.tensorboard.writer import SummaryWriter
from utils.classification_metric import classification_update, classification_results
from utils.fed_merge import get_shared_adapter, get_local_adapter, aggregate, apply_grads
from utils.trainval_func import site_evaluation, SaveCheckPoint, site_evaluation_for_all_domain, test_func, load_from_checkpoint, save_checkpoint_for_resume
from utils.weight_adjust import refine_weight_dict_by_GA
from network.FedOptimizer.Scaffold import *
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='vlcs',
                        choices=['pacs', 'officehome', 'terrainc', 'office_caltech_10', 'domainnet', 'vlcs'],
                        help='Name of dataset')
    parser.add_argument("--model", type=str, default='vit_b16',
                        choices=['vit_b16', 'resnet18', 'resnet50'], help='model name')
    parser.add_argument("--test_domain", type=str, default='c',
                        choices=['p', 'a', 'c', 's', 'r'], help='the domain name for testing')
    parser.add_argument('--batch_size', help='batch_size', type=int, default=64)
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=30)
    parser.add_argument('--comm', help='epochs number', type=int, default=400)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--step_size', help='rate weight step', type=float, default=0.2)
    parser.add_argument("--lr_policy", type=str, default='step', choices=['step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--fair", type=str, default='acc', choices=['acc', 'loss'],
                        help="the fairness metric for FedAvg")
    parser.add_argument('--note', help='note of experimental settings', type=str, default='generalization_adjustment')
    parser.add_argument('--display', help='display in controller', default=True, action='store_true')
    parser.add_argument('--resume', help='path to checkpoint to resume from', type=str,
                        default=None)
    parser.add_argument("--ckpt_freq", type=int, default=5, help="frequency of saving resume checkpoints")
    return parser.parse_args()


args = get_argparse()
if args.dataset == 'pacs':
    from data_loader.pacs_dataset import PACS_FedDG
    dataobj = PACS_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    train_domain_list = ['p', 'a', 'c', 's']
    num_classes = 7
elif args.dataset == 'officehome':
    from data_loader.officehome_dataset import OfficeHome_FedDG
    dataobj = OfficeHome_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    train_domain_list = ['a', 'c', 'p', 'r']
    num_classes = 65
elif args.dataset == 'vlcs':
    from data_loader.vlcs_dataset import VLCS_FedDG
    dataobj = VLCS_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    train_domain_list = ['v', 'l', 'c', 's']
    num_classes = 5
elif args.dataset == 'domainnet':
    # Import and use the DomainNet dataset class
    from data_loader.domainnet_dataset import DomainNet_FedDG
    dataobj = DomainNet_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    train_domain_list = ['c', 'i', 'p', 'q', 'r', 's']
    num_classes = 345
else:
    raise ValueError(f"Dataset '{args.dataset}' not supported")


def epoch_site_train(epochs, site_name, model_dual, model_single, optimizer_dual, optimizer_single, scheduler_dual,
                     scheduler_single, c_ci_dual, c_ci_single, dataloader, log_ten):

    model_dual.train()
    model_single.train()

    dual_correct_count, dual_total_count, dual_loss = 0, 0, 0
    single_correct_count, single_total_count, single_loss = 0, 0, 0

    scaler_dual = GradScaler()
    scaler_single = GradScaler()

    for i, data_list in enumerate(dataloader):
        imgs, labels, domain_labels = data_list
        imgs = imgs.cuda()
        labels = labels.cuda()
        domain_labels = domain_labels.cuda()

        if labels == None:
            print("label is None!")

        for param in model_dual.parameters():
            param.requires_grad = False
        for name, param in get_shared_adapter(model_single).items():
            param.requires_grad = True

        with torch.no_grad():
            with autocast():
                output_dual = model_dual(imgs)
        with autocast():
            output_single = model_single(imgs)

            loss_KL_single = F.kl_div(F.log_softmax(output_single, dim=1), F.softmax(output_dual, dim=1),
                                      reduction='batchmean')
            loss_CE_single = F.cross_entropy(output_single, labels)
            loss_single = loss_KL_single + loss_CE_single

        optimizer_single.zero_grad()
        scaler_single.scale(loss_single).backward()
        scaler_single.step(optimizer_single, c_ci_single)
        scaler_single.update()

        for param in model_single.parameters():
            param.requires_grad = False
        for name, param in get_local_adapter(model_dual).items():
            param.requires_grad = True

        with torch.no_grad():
            with autocast():
                output_single = model_single(imgs)
        with autocast():
            output_dual = model_dual(imgs)
            loss_KL_dual = F.kl_div(F.log_softmax(output_dual, dim=1), F.softmax(output_single, dim=1),
                                    reduction='batchmean')
            loss_CE_dual = F.cross_entropy(output_dual, labels)
            loss_dual = loss_KL_dual + loss_CE_dual

        optimizer_dual.zero_grad()
        scaler_dual.scale(loss_dual).backward()
        scaler_dual.step(optimizer_dual, c_ci_dual)
        scaler_dual.update()

        log_ten.add_scalar(f'{site_name}_train_loss_dual', loss_dual.item(), epochs * len(dataloader) + i)
        log_ten.add_scalar(f'{site_name}_train_loss_single', loss_single.item(), epochs * len(dataloader) + i)

        epoch_dual_correct, epoch_dual_total, epoch_dual_loss = classification_update(output_dual, labels)
        epoch_single_correct, epoch_single_total, epoch_single_loss = classification_update(output_single, labels)

        dual_correct_count += epoch_dual_correct
        dual_total_count += epoch_dual_total
        dual_loss += epoch_dual_loss
        single_correct_count += epoch_single_correct
        single_total_count += epoch_single_total
        single_loss += epoch_single_loss

    log_ten.add_scalar(f'{site_name}_train_acc_dual',
                       classification_results(dual_correct_count, dual_total_count, dual_loss)['acc'], epochs)
    log_ten.add_scalar(f'{site_name}_train_acc_single',
                       classification_results(single_correct_count, single_total_count, single_loss)['acc'], epochs)

    scheduler_dual.step()
    scheduler_single.step()

    return model_dual, model_single


def site_train(comm_rounds, site_name, args, model_dual, model_single, optimizer_dual, optimizer_single,
               scheduler_dual, scheduler_single, c_ci_dual, c_ci_single, dataloader, log_ten):

    tbar = tqdm(range(args.local_epochs))

    site_single_gradients = []

    for local_epoch in tbar:
        tbar.set_description(f'{site_name}_train')
        model_dual, model_single = epoch_site_train(comm_rounds * args.local_epochs + local_epoch, site_name,
                                                    model_dual, model_single, optimizer_dual,
                                                    optimizer_single, scheduler_dual, scheduler_single, c_ci_dual,
                                                    c_ci_single, dataloader, log_ten)

        orig_single_adapter = list(get_shared_adapter(deepcopy(model_single)).values())
        updated_single_adapter = list(get_shared_adapter(model_single).values())
        for orig_param, updated_param in zip(orig_single_adapter, updated_single_adapter):
            site_single_gradients.append(orig_param.detach() - updated_param.detach())

    return site_single_gradients


def GetFedModel(args, num_classes):
    dim = 768
    adapter_dim = dim // 4
    shared_adapter_down = nn.Linear(dim, adapter_dim)
    shared_adapter_up = nn.Linear(adapter_dim, dim)
    nn.init.normal_(shared_adapter_down.weight, mean=0.0, std=0.02)
    nn.init.normal_(shared_adapter_up.weight, mean=0.0, std=0.02)

    client_dual_model, client_single_model, _ = GetNetwork(args, num_classes, shared_adapter_down, shared_adapter_up)
    client_dual_model = client_dual_model.cuda()
    client_single_model = client_single_model.cuda()

    client_dual_model = nn.DataParallel(client_dual_model)
    client_single_model = nn.DataParallel(client_single_model)

    dual_model_dict = {}
    single_model_dict = {}

    dual_optimizer_dict = {}
    single_optimizer_dict = {}

    client_dual_optimizer = Scaffold(
        [param for name, param in get_local_adapter(client_dual_model).items()],
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    client_single_optimizer = Scaffold(
        [param for name, param in get_shared_adapter(client_single_model).items()],
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    dual_scheduler_dict = {}
    single_scheduler_dict = {}
    dual_ci_dict = {}
    single_ci_dict = {}

    dual_c = GenZeroParamList([param for name, param in get_local_adapter(client_dual_model).items()])
    single_c = GenZeroParamList([param for name, param in get_shared_adapter(client_single_model).items()])

    for domain_name in train_domain_list:
        dual_model_dict[domain_name], single_model_dict[domain_name], _ = (
            GetNetwork(args, num_classes, shared_adapter_down, shared_adapter_up))
        dual_model_dict[domain_name] = dual_model_dict[domain_name].cuda()
        single_model_dict[domain_name] = single_model_dict[domain_name].cuda()

        dual_model_dict[domain_name] = nn.DataParallel(dual_model_dict[domain_name])
        single_model_dict[domain_name] = nn.DataParallel(single_model_dict[domain_name])

        dual_ci_dict[domain_name] = GenZeroParamList(
            [param for name, param in get_local_adapter(dual_model_dict[domain_name]).items()])
        single_ci_dict[domain_name] = GenZeroParamList(
            [param for name, param in get_shared_adapter(single_model_dict[domain_name]).items()])

        dual_optimizer_dict[domain_name] = Scaffold(
            [param for name, param in get_local_adapter(dual_model_dict[domain_name]).items()],
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        single_optimizer_dict[domain_name] = Scaffold(
            [param for name, param in get_shared_adapter(single_model_dict[domain_name]).items()],
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4
        )

        if args.lr_policy == 'step':
            dual_scheduler_dict[domain_name] = torch.optim.lr_scheduler.StepLR(dual_optimizer_dict[domain_name],
                                                                               step_size=args.local_epochs * args.comm,
                                                                               gamma=0.1)
            single_scheduler_dict[domain_name] = torch.optim.lr_scheduler.StepLR(single_optimizer_dict[domain_name],
                                                                                 step_size=args.local_epochs * args.comm,
                                                                                 gamma=0.1)

    return client_dual_model, client_single_model, dual_model_dict, single_model_dict, client_dual_optimizer, client_single_optimizer, dual_optimizer_dict, single_optimizer_dict, dual_scheduler_dict, single_scheduler_dict, dual_ci_dict, single_ci_dict, dual_c, single_c


def main():
    file_name = 'FedSDAF_' + os.path.split(__file__)[1].replace('.py', '')
    args = get_argparse()

    log_dir, tensorboard_dir = Gen_Log_Dir(args, file_name=file_name)
    log_ten = SummaryWriter(log_dir=tensorboard_dir)
    log_file = Get_Logger(file_name=log_dir + 'train.log', display=args.display)
    Save_Hyperparameter(log_dir, args)

    dataloader_dict, dataset_dict = dataobj.GetData()

    (client_dual_model, client_single_model, dual_model_dict, single_model_dict,
     client_dual_optimizer, client_single_optimizer, dual_optimizer_dict, single_optimizer_dict,
     dual_scheduler_dict, single_scheduler_dict, dual_ci_dict, single_ci_dict, dual_c, single_c) = GetFedModel(args, num_classes)

    weight_dict = {}
    site_results_before_avg = {}
    site_results_after_avg = {}
    for site_name in train_domain_list:
        weight_dict[site_name] = 1.0/len(train_domain_list)
        site_results_before_avg[site_name] = None
        site_results_after_avg[site_name] = None

    step_size_decay = args.step_size / args.comm

    best_domain_acc = {}
    best_domain_info = {}
    for test_domain in train_domain_list:
        best_domain_acc[test_domain] = 0.0
        best_domain_info[test_domain] = {'round': 0, 'train_domain': '', 'acc': 0.0, 'phase': ''}

    start_round = 0
    if args.resume and args.resume != '':
        start_round, weight_dict, dual_c, single_c, dual_ci_dict, single_ci_dict = load_from_checkpoint(
            args.resume, client_dual_model, client_single_model, dual_model_dict, single_model_dict,
            client_dual_optimizer, client_single_optimizer, dual_optimizer_dict, single_optimizer_dict,
            dual_scheduler_dict, single_scheduler_dict, dual_ci_dict, single_ci_dict, dual_c, single_c,
            weight_dict, train_domain_list, log_file
        )


    for i in range(args.comm + 1):

        single_gradients = []

        for domain_name in train_domain_list:

            dual_c_ci = ListMinus(dual_c, dual_ci_dict[domain_name])
            single_c_ci = ListMinus(single_c, single_ci_dict[domain_name])
            print(len(dataloader_dict[domain_name]['train']) * args.local_epochs)

            K = len(dataloader_dict[domain_name]['train']) * args.local_epochs

            site_single_gradients = site_train(i, domain_name, args, dual_model_dict[domain_name],
                                               single_model_dict[domain_name],
                                               dual_optimizer_dict[domain_name], single_optimizer_dict[domain_name],
                                               dual_scheduler_dict[domain_name],
                                               single_scheduler_dict[domain_name], dual_c_ci, single_c_ci,
                                               dataloader_dict[domain_name]['train'], log_ten)
            single_gradients.append(site_single_gradients)

            site_results_before_avg[domain_name] = site_evaluation(i, domain_name, args, dual_model_dict[domain_name],
                                                                   dataloader_dict[domain_name]['val'],
                                                                   log_file, log_ten, note='before_avg')

            dual_ci_dict[domain_name] = UpdateLocalControl(dual_c, dual_ci_dict[domain_name],
                                                           [param for name, param in
                                                            get_local_adapter(client_dual_model).items()],
                                                           [param for name, param in
                                                            get_local_adapter(dual_model_dict[domain_name]).items()], K)
            single_ci_dict[domain_name] = UpdateLocalControl(single_c, single_ci_dict[domain_name],
                                                             [param for name, param in
                                                              get_shared_adapter(client_single_model).items()],
                                                             [param for name, param in get_shared_adapter(
                                                                 single_model_dict[domain_name]).items()], K)

        log_file.info("\n===== Before Avg Domain Valid =====")
        for val_domain in train_domain_list:
            site_evaluation_for_all_domain(i, val_domain, dual_model_dict, log_file, args, dataloader_dict,
                                           train_domain_list, note='before_avg')
        log_file.info("===== Before Avg Valid Complete =====")

        log_file.info("\n===== Before Avg Domain Test =====")
        before_avg_results = {}
        for test_domain in train_domain_list:
            before_avg_results[test_domain] = test_func(i, test_domain, dual_model_dict, log_file, args,
                                                        dataloader_dict, train_domain_list, note='before_avg')

            for train_domain, acc in before_avg_results[test_domain].items():
                if acc > best_domain_acc[test_domain]:
                    best_domain_acc[test_domain] = acc
                    best_domain_info[test_domain] = {
                        'round': i,
                        'train_domain': train_domain,
                        'acc': acc,
                        'phase': 'before_avg'
                    }

                    SaveCheckPoint(args, dual_model_dict[train_domain], args.comm,
                                   os.path.join(log_dir, 'checkpoints'),
                                   dual_optimizer_dict[train_domain],
                                   dual_scheduler_dict[train_domain],
                                   note=f'best_domain_{test_domain}_dual')
                    SaveCheckPoint(args, single_model_dict[train_domain], args.comm,
                                   os.path.join(log_dir, 'checkpoints'),
                                   single_optimizer_dict[train_domain],
                                   single_scheduler_dict[train_domain],
                                   note=f'best_domain_{test_domain}_single')
                    log_file.info(
                        f'New Best Model for Test Domain {test_domain}! Round: {i}, Train Domain: {train_domain}, Phase: before_avg, Acc: {acc * 100:.2f}%')
                    info = best_domain_info[test_domain]
                    log_file.info(
                        f"Test Domain: {test_domain} | Best Round: {info['round']} | Train Domain: {info['train_domain']} | Phase: {info['phase']} | Acc: {info['acc'] * 100:.2f}%")

        log_file.info("===== Before Avg Test Complete =====")

        dual_c = UpdateServerControl(dual_c, dual_ci_dict, weight_dict)
        single_c = UpdateServerControl(single_c, single_ci_dict, weight_dict)

        single_grads = aggregate(single_gradients, weight_dict)

        apply_grads([param for name, param in get_shared_adapter(client_single_model).items()], single_grads,
                    client_single_optimizer, single_c)

        for domain_name in train_domain_list:
            single_model_dict[domain_name].load_state_dict(
                client_single_model.state_dict(),
                strict=True
            )

        shared_adapter_params = get_shared_adapter(client_single_model)
        for domain_name in train_domain_list:
            for name, param in shared_adapter_params.items():
                dual_model_dict[domain_name].state_dict()[name].copy_(param)

        for domain_name in train_domain_list:
            site_results_after_avg[domain_name] = site_evaluation(i, domain_name, args, dual_model_dict[domain_name],
                                                                  dataloader_dict[domain_name]['val'],
                                                                  log_file, log_ten, note='after_avg')


        weight_dict = refine_weight_dict_by_GA(weight_dict, site_results_before_avg, site_results_after_avg,
                                               args.step_size - (i - 1) * step_size_decay, fair_metric=args.fair)
        log_str = f'Round {i} FedAvg weight: {weight_dict}'
        log_file.info(log_str)


        log_file.info("\n===== After Avg Domain Valid =====")
        for val_domain in train_domain_list:
            site_evaluation_for_all_domain(i, val_domain, dual_model_dict, log_file, args, dataloader_dict,
                                           train_domain_list, note='before_avg')
        log_file.info("===== After Avg Valid Complete =====")


        log_file.info("\n===== After Avg Domain Test =====")
        after_avg_results = {}
        for test_domain in train_domain_list:
            after_avg_results[test_domain] = test_func(i, test_domain, dual_model_dict, log_file, args, dataloader_dict,
                                                       train_domain_list, note='after_avg')

            for train_domain, acc in after_avg_results[test_domain].items():
                if acc > best_domain_acc[test_domain]:
                    best_domain_acc[test_domain] = acc
                    best_domain_info[test_domain] = {
                        'round': i,
                        'train_domain': train_domain,
                        'acc': acc,
                        'phase': 'after_avg'
                    }

                    SaveCheckPoint(args, dual_model_dict[train_domain], args.comm,
                                   os.path.join(log_dir, 'checkpoints'),
                                   dual_optimizer_dict[train_domain],
                                   dual_scheduler_dict[train_domain],
                                   note=f'best_domain_{test_domain}_dual')
                    SaveCheckPoint(args, single_model_dict[train_domain], args.comm,
                                   os.path.join(log_dir, 'checkpoints'),
                                   single_optimizer_dict[train_domain],
                                   single_scheduler_dict[train_domain],
                                   note=f'best_domain_{test_domain}_single')

                    log_file.info(
                        f'New Best Model for Test Domain {test_domain}! Round: {i}, Train Domain: {train_domain}, Phase: after_avg, Acc: {acc * 100:.2f}%')
                    info = best_domain_info[test_domain]
                    log_file.info(
                        f"Test Domain: {test_domain} | Best Round: {info['round']} | Train Domain: {info['train_domain']} | Phase: {info['phase']} | Acc: {info['acc'] * 100:.2f}%")
        log_file.info("===== After Avg Test Complete =====")

        if i % args.ckpt_freq == 0 or i == args.comm:
            save_path = os.path.join(log_dir, 'checkpoints', 'latest_checkpoint.pt')
            save_checkpoint_for_resume(
                args, i, client_dual_model, client_single_model, dual_model_dict, single_model_dict,
                client_dual_optimizer, client_single_optimizer, dual_optimizer_dict, single_optimizer_dict,
                dual_scheduler_dict, single_scheduler_dict, dual_ci_dict, single_ci_dict, dual_c, single_c,
                weight_dict, save_path
            )
            log_file.info(f"Saved checkpoint at round {i} to {save_path}")

        log_file.info("\n===== Current Best Performance Summary =====")
        for test_domain in train_domain_list:
            info = best_domain_info[test_domain]
            log_file.info(
                f"Test Domain: {test_domain} | Best Round: {info['round']} | Train Domain: {info['train_domain']} | Phase: {info['phase']} | Acc: {info['acc'] * 100:.2f}%")


if __name__ == '__main__':
    main()