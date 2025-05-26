import sys
import torch
import torch.nn as nn
from utils.log_utils import *
from torch.utils.tensorboard.writer import SummaryWriter
from network.modified_vit import DualAdapterViT, SingleAdapterViT
from utils.classification_metric import classification_update, classification_results
from utils.fed_merge import get_invariant_adapter, get_aware_adapter, aggregate, apply_grads
from utils.trainval_func import site_evaluation, SaveCheckPoint, test_func, load_from_checkpoint, save_checkpoint_for_resume
from utils.weight_adjust import refine_weight_dict_by_GA
from network.FedOptimizer.Scaffold import *
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='pacs', choices=['pacs', 'officehome', 'domainnet', 'vlcs'],
                        help='Name of dataset')
    parser.add_argument("--model", type=str, default='vit_b16', help='model name')
    parser.add_argument("--test_domain", type=str, default='c',
                        choices=['p', 'a', 'c', 's', 'r'], help='the domain name for testing')
    parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=30)
    parser.add_argument('--comm', help='communication rounds', type=int, default=200)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--step_size', help='rate weight step', type=float, default=0.2)
    parser.add_argument("--lr_policy", type=str, default='step', choices=['step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--fair", type=str, default='acc', choices=['acc', 'loss'],
                        help="the fairness metric for FedAvg")
    parser.add_argument('--note', help='note of experimental settings', type=str, default='domain_generalization')
    parser.add_argument('--display', help='display in controller', default=True, action='store_true')
    parser.add_argument('--resume', help='path to checkpoint to resume from', type=str,
                        default=None)
    parser.add_argument("--ckpt_freq", type=int, default=5, help="frequency of saving resume checkpoints")
    return parser.parse_args()


args = get_argparse()
if args.dataset == 'pacs':
    from data_loader.pacs_dataset import PACS_FedDG
    dataobj = PACS_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    all_domain_list = ['p', 'a', 'c', 's']
    num_classes = 7
elif args.dataset == 'officehome':
    from data_loader.officehome_dataset import OfficeHome_FedDG
    dataobj = OfficeHome_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    all_domain_list = ['a', 'c', 'p', 'r']
    num_classes = 65
elif args.dataset == 'vlcs':
    from data_loader.vlcs_dataset import VLCS_FedDG
    dataobj = VLCS_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    all_domain_list = ['v', 'l', 'c', 's']
    num_classes = 5
elif args.dataset == 'domainnet':
    from data_loader.domainnet_dataset import DomainNet_FedDG
    dataobj = DomainNet_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    all_domain_list = ['c', 'i', 'p', 'q', 'r', 's']
    num_classes = 345
else:
    raise ValueError(f"Dataset '{args.dataset}' not supported")

# Source domains exclude the test domain
source_domain_list = all_domain_list.copy()
source_domain_list.remove(args.test_domain)


def epoch_site_train(epochs, site_name, model_dual, model_single, optimizer_dual, optimizer_single, scheduler_dual,
                     scheduler_single, c_ci_dual, c_ci_single, dataloader, log_ten, args):
    # Set models to training mode
    model_dual.train()
    model_single.train()

    dual_correct_count, dual_total_count, dual_loss = 0, 0, 0
    single_correct_count, single_total_count, single_loss = 0, 0, 0

    # Initialize GradScalers for mixed precision
    scaler_dual = GradScaler()
    scaler_single = GradScaler()

    # Iterate over the dataloader
    for i, data_list in enumerate(dataloader):
        imgs, labels, domain_labels = data_list
        imgs = imgs.cuda()
        labels = labels.cuda()
        domain_labels = domain_labels.cuda()

        if labels == None:
            print("label is None!")

        # Freeze model_dual, update model_single
        for param in model_dual.parameters():
            param.requires_grad = False
        for name, param in get_invariant_adapter(model_single).items():
            param.requires_grad = True

        with torch.no_grad():
            with autocast():
                output_dual = model_dual(imgs)  # Forward pass for model_dual
        with autocast():
            output_single = model_single(imgs)  # Forward pass for model_single
            # Compute KL divergence and cross-entropy losses
            loss_KL_single = F.kl_div(F.log_softmax(output_single, dim=1), F.softmax(output_dual, dim=1),
                                      reduction='batchmean')
            loss_CE_single = F.cross_entropy(output_single, labels)
            loss_single = loss_CE_single + loss_KL_single

        # Update model_single with mixed precision
        optimizer_single.zero_grad()
        scaler_single.scale(loss_single).backward()
        scaler_single.step(optimizer_single, c_ci_single)
        scaler_single.update()

        # Freeze model_single, update model_dual
        for param in model_single.parameters():
            param.requires_grad = False
        for name, param in get_aware_adapter(model_dual).items():
            param.requires_grad = True

        with torch.no_grad():
            with autocast():
                output_single = model_single(imgs)  # Forward pass for model_single
        with autocast():
            output_dual = model_dual(imgs)  # Forward pass for model_dual
            # Compute KL divergence and cross-entropy losses
            loss_KL_dual = F.kl_div(F.log_softmax(output_dual, dim=1), F.softmax(output_single, dim=1),
                                    reduction='batchmean')
            loss_CE_dual = F.cross_entropy(output_dual, labels)
            loss_dual = loss_CE_dual + loss_KL_dual

        # Update model_dual with mixed precision
        optimizer_dual.zero_grad()
        scaler_dual.scale(loss_dual).backward()
        scaler_dual.step(optimizer_dual, c_ci_dual)
        scaler_dual.update()

        # Log training losses
        log_ten.add_scalar(f'{site_name}_train_loss_dual', loss_dual.item(), epochs * len(dataloader) + i)
        log_ten.add_scalar(f'{site_name}_train_loss_single', loss_single.item(), epochs * len(dataloader) + i)

        # Update classification metrics
        epoch_dual_correct, epoch_dual_total, epoch_dual_loss = classification_update(output_dual, labels)
        epoch_single_correct, epoch_single_total, epoch_single_loss = classification_update(output_single, labels)

        # Accumulate metrics
        dual_correct_count += epoch_dual_correct
        dual_total_count += epoch_dual_total
        dual_loss += epoch_dual_loss
        single_correct_count += epoch_single_correct
        single_total_count += epoch_single_total
        single_loss += epoch_single_loss

    # Log training accuracy
    log_ten.add_scalar(f'{site_name}_train_acc_dual',
                       classification_results(dual_correct_count, dual_total_count, dual_loss)['acc'], epochs)
    log_ten.add_scalar(f'{site_name}_train_acc_single',
                       classification_results(single_correct_count, single_total_count, single_loss)['acc'], epochs)

    # Update learning rate schedulers
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
                                                    c_ci_single, dataloader, log_ten, args)

        orig_single_adapter = list(get_invariant_adapter(deepcopy(model_single)).values())
        updated_single_adapter = list(get_invariant_adapter(model_single).values())
        for orig_param, updated_param in zip(orig_single_adapter, updated_single_adapter):
            site_single_gradients.append(orig_param.detach() - updated_param.detach())

    return site_single_gradients


def GetFedModel(args, num_classes, source_domains):
    # Global models (trained on aggregated source domains)
    client_dual_model = DualAdapterViT(num_classes)
    client_single_model = SingleAdapterViT(num_classes)

    client_dual_model = client_dual_model.cuda()
    client_single_model = client_single_model.cuda()

    client_dual_model = nn.DataParallel(client_dual_model)
    client_single_model = nn.DataParallel(client_single_model)

    # Domain-specific models (only for source domains)
    dual_model_dict = {}
    single_model_dict = {}

    dual_optimizer_dict = {}
    single_optimizer_dict = {}
    
    client_dual_optimizer = Scaffold(
        [param for name, param in get_aware_adapter(client_dual_model).items()],
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    client_single_optimizer = Scaffold(
        [param for name, param in get_invariant_adapter(client_single_model).items()],
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    dual_scheduler_dict = {}
    single_scheduler_dict = {}
    dual_ci_dict = {}
    single_ci_dict = {}

    dual_c = GenZeroParamList([param for name, param in get_aware_adapter(client_dual_model).items()])
    single_c = GenZeroParamList([param for name, param in get_invariant_adapter(client_single_model).items()])

    # Only create models for source domains (exclude test domain)
    for domain_name in source_domains:
        dual_model_dict[domain_name] = DualAdapterViT(num_classes)
        single_model_dict[domain_name] = SingleAdapterViT(num_classes)

        dual_model_dict[domain_name] = dual_model_dict[domain_name].cuda()
        single_model_dict[domain_name] = single_model_dict[domain_name].cuda()

        dual_model_dict[domain_name] = nn.DataParallel(dual_model_dict[domain_name])
        single_model_dict[domain_name] = nn.DataParallel(single_model_dict[domain_name])

        dual_ci_dict[domain_name] = GenZeroParamList(
            [param for name, param in get_aware_adapter(dual_model_dict[domain_name]).items()])
        single_ci_dict[domain_name] = GenZeroParamList(
            [param for name, param in get_invariant_adapter(single_model_dict[domain_name]).items()])

        dual_optimizer_dict[domain_name] = Scaffold(
            [param for name, param in get_aware_adapter(dual_model_dict[domain_name]).items()],
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        single_optimizer_dict[domain_name] = Scaffold(
            [param for name, param in get_invariant_adapter(single_model_dict[domain_name]).items()],
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

    return (client_dual_model, client_single_model, dual_model_dict, single_model_dict,
            client_dual_optimizer, client_single_optimizer, dual_optimizer_dict, single_optimizer_dict,
            dual_scheduler_dict, single_scheduler_dict, dual_ci_dict, single_ci_dict, dual_c, single_c)


def test_on_target_domain(model_dict, dataloader_dict, target_domain, source_domains, log_file, comm_round, note=""):
    """Test all source domain models on the target domain"""
    test_dataloader = dataloader_dict[target_domain]['test']
    results = {}
    
    log_file.info(f"\n===== {note} Testing on Target Domain: {target_domain} =====")
    
    best_acc = 0.0
    best_model_domain = None
    
    for source_domain in source_domains:
        model = model_dict[source_domain]
        model = model.cuda()
        model.eval()
        
        total_correct_count = 0
        total_count = 0
        total_loss = 0
        
        with torch.no_grad():
            for imgs, labels, domain_labels in test_dataloader:
                imgs = imgs.cuda()
                labels = labels.cuda()
                output = model(imgs)
                correct_count, count, loss = classification_update(output, labels)
                total_correct_count += correct_count
                total_count += count
                total_loss += loss
        
        results_dict = classification_results(total_correct_count, total_count, total_loss)
        acc = results_dict["acc"]
        results[source_domain] = acc
        
        if acc > best_acc:
            best_acc = acc
            best_model_domain = source_domain
        
        log_file.info(f'{note} Round: {comm_round:3d} | Source Domain: {source_domain} | Target Domain: {target_domain} | Acc: {acc * 100:.2f}%')
    
    log_file.info(f"Best performing source domain: {best_model_domain} with accuracy: {best_acc * 100:.2f}%")
    log_file.info("=" * 60)
    
    return results, best_acc, best_model_domain


def main():
    file_name = 'DomainGeneralization_' + os.path.split(__file__)[1].replace('.py', '')

    args = get_argparse()

    log_dir, tensorboard_dir = Gen_Log_Dir(args, file_name=file_name)
    log_ten = SummaryWriter(log_dir=tensorboard_dir)
    log_file = Get_Logger(file_name=log_dir + 'train.log', display=args.display)
    Save_Hyperparameter(log_dir, args)

    log_file.info(f"Dataset: {args.dataset}")
    log_file.info(f"Target Domain: {args.test_domain}")
    log_file.info(f"Source Domains: {source_domain_list}")

    dataloader_dict, dataset_dict = dataobj.GetData()

    # Initialize models, optimizers, etc. (only for source domains)
    (client_dual_model, client_single_model, dual_model_dict, single_model_dict,
     client_dual_optimizer, client_single_optimizer, dual_optimizer_dict, single_optimizer_dict,
     dual_scheduler_dict, single_scheduler_dict, dual_ci_dict, single_ci_dict, dual_c, single_c) = GetFedModel(args, num_classes, source_domain_list)

    # Initialize weights only for source domains
    weight_dict = {}
    site_results_before_avg = {}
    site_results_after_avg = {}
    for site_name in source_domain_list:
        weight_dict[site_name] = 1.0/len(source_domain_list)
        site_results_before_avg[site_name] = None
        site_results_after_avg[site_name] = None

    step_size_decay = args.step_size / args.comm

    # Save best accuracy for target domain
    best_target_acc = 0.0
    best_target_info = {'round': 0, 'source_domain': '', 'acc': 0.0, 'phase': ''}

    # Load from checkpoint if resume path is provided
    start_round = 0
    if args.resume and args.resume != '':
        start_round, weight_dict, dual_c, single_c, dual_ci_dict, single_ci_dict = load_from_checkpoint(
            args.resume, client_dual_model, client_single_model, dual_model_dict, single_model_dict,
            client_dual_optimizer, client_single_optimizer, dual_optimizer_dict, single_optimizer_dict,
            dual_scheduler_dict, single_scheduler_dict, dual_ci_dict, single_ci_dict, dual_c, single_c,
            weight_dict, source_domain_list, log_file
        )

    for i in range(start_round, args.comm + 1):
        log_file.info(f"\n{'='*50}")
        log_file.info(f"Communication Round: {i}")
        log_file.info(f"{'='*50}")

        single_gradients = []

        # Train only on source domains
        for domain_name in source_domain_list:
            dual_c_ci = ListMinus(dual_c, dual_ci_dict[domain_name])
            single_c_ci = ListMinus(single_c, single_ci_dict[domain_name])

            K = len(dataloader_dict[domain_name]['train']) * args.local_epochs

            site_single_gradients = site_train(i, domain_name, args, dual_model_dict[domain_name],
                                               single_model_dict[domain_name],
                                               dual_optimizer_dict[domain_name], single_optimizer_dict[domain_name],
                                               dual_scheduler_dict[domain_name],
                                               single_scheduler_dict[domain_name], dual_c_ci, single_c_ci,
                                               dataloader_dict[domain_name]['train'], log_ten)
            single_gradients.append(site_single_gradients)

            # Evaluate on validation set of source domain
            site_results_before_avg[domain_name] = site_evaluation(dual_model_dict[domain_name], dataloader_dict[domain_name]['val'])

            # Update control variables
            dual_ci_dict[domain_name] = UpdateLocalControl(dual_c, dual_ci_dict[domain_name],
                                                           [param for name, param in
                                                            get_aware_adapter(client_dual_model).items()],
                                                           [param for name, param in
                                                            get_aware_adapter(dual_model_dict[domain_name]).items()], K)
            single_ci_dict[domain_name] = UpdateLocalControl(single_c, single_ci_dict[domain_name],
                                                             [param for name, param in
                                                              get_invariant_adapter(client_single_model).items()],
                                                             [param for name, param in get_invariant_adapter(
                                                                 single_model_dict[domain_name]).items()], K)

        # Test on target domain before aggregation
        before_avg_results, before_best_acc, before_best_domain = test_on_target_domain(
            dual_model_dict, dataloader_dict, args.test_domain, source_domain_list, log_file, i, "Before Avg"
        )

        # Check if new best accuracy
        if before_best_acc > best_target_acc:
            best_target_acc = before_best_acc
            best_target_info = {
                'round': i,
                'source_domain': before_best_domain,
                'acc': before_best_acc,
                'phase': 'before_avg'
            }
            # Save the best model
            SaveCheckPoint(args, dual_model_dict[before_best_domain], args.comm,
                           os.path.join(log_dir, 'checkpoints'),
                           dual_optimizer_dict[before_best_domain],
                           dual_scheduler_dict[before_best_domain],
                           note=f'best_target_{args.test_domain}_dual')
            SaveCheckPoint(args, single_model_dict[before_best_domain], args.comm,
                           os.path.join(log_dir, 'checkpoints'),
                           single_optimizer_dict[before_best_domain],
                           single_scheduler_dict[before_best_domain],
                           note=f'best_target_{args.test_domain}_single')
            log_file.info(f'🏆 New Best Model for Target Domain {args.test_domain}! Round: {i}, Source Domain: {before_best_domain}, Phase: before_avg, Acc: {before_best_acc * 100:.2f}%')

        # Federated aggregation
        dual_c = UpdateServerControl(dual_c, dual_ci_dict, weight_dict)
        single_c = UpdateServerControl(single_c, single_ci_dict, weight_dict)

        single_grads = aggregate(single_gradients, weight_dict)
        apply_grads([param for name, param in get_invariant_adapter(client_single_model).items()], single_grads,
                    client_single_optimizer, single_c)

        # Update all source domain models with aggregated parameters
        for domain_name in source_domain_list:
            single_model_dict[domain_name].load_state_dict(
                client_single_model.state_dict(),
                strict=True
            )

        # Share invariant adapter parameters
        shared_adapter_params = get_invariant_adapter(client_single_model)
        for domain_name in source_domain_list:
            for name, param in shared_adapter_params.items():
                dual_model_dict[domain_name].state_dict()[name].copy_(param)

        # Evaluate after aggregation
        for domain_name in source_domain_list:
            site_results_after_avg[domain_name] = site_evaluation(dual_model_dict[domain_name], dataloader_dict[domain_name]['val'])

        # Update weights
        weight_dict = refine_weight_dict_by_GA(weight_dict, site_results_before_avg, site_results_after_avg,
                                               args.step_size - (i - 1) * step_size_decay, fair_metric=args.fair)

        log_str = f'Round {i} FedAvg weight: {weight_dict}'
        log_file.info(log_str)

        # Test on target domain after aggregation
        after_avg_results, after_best_acc, after_best_domain = test_on_target_domain(
            dual_model_dict, dataloader_dict, args.test_domain, source_domain_list, log_file, i, "After Avg"
        )

        # Check if new best accuracy after aggregation
        if after_best_acc > best_target_acc:
            best_target_acc = after_best_acc
            best_target_info = {
                'round': i,
                'source_domain': after_best_domain,
                'acc': after_best_acc,
                'phase': 'after_avg'
            }
            # Save the best model
            SaveCheckPoint(args, dual_model_dict[after_best_domain], args.comm,
                           os.path.join(log_dir, 'checkpoints'),
                           dual_optimizer_dict[after_best_domain],
                           dual_scheduler_dict[after_best_domain],
                           note=f'best_target_{args.test_domain}_dual')
            SaveCheckPoint(args, single_model_dict[after_best_domain], args.comm,
                           os.path.join(log_dir, 'checkpoints'),
                           single_optimizer_dict[after_best_domain],
                           single_scheduler_dict[after_best_domain],
                           note=f'best_target_{args.test_domain}_single')
            log_file.info(f'🏆 New Best Model for Target Domain {args.test_domain}! Round: {i}, Source Domain: {after_best_domain}, Phase: after_avg, Acc: {after_best_acc * 100:.2f}%')

        # Save checkpoint for resumption every ckpt_freq rounds
        if i % args.ckpt_freq == 0 or i == args.comm:
            save_path = os.path.join(log_dir, 'checkpoints', 'latest_checkpoint.pt')
            save_checkpoint_for_resume(
                args, i, client_dual_model, client_single_model, dual_model_dict, single_model_dict,
                client_dual_optimizer, client_single_optimizer, dual_optimizer_dict, single_optimizer_dict,
                dual_scheduler_dict, single_scheduler_dict, dual_ci_dict, single_ci_dict, dual_c, single_c,
                weight_dict, save_path
            )
            log_file.info(f"Saved checkpoint at round {i} to {save_path}")

        # Print summary of the best performance 
        log_file.info(f"\n📊 Current Best Performance Summary:")
        info = best_target_info
        log_file.info(f"Target Domain: {args.test_domain} | Best Round: {info['round']} | Best Source Domain: {info['source_domain']} | Phase: {info['phase']} | Acc: {info['acc'] * 100:.2f}%")

    # Final summary
    log_file.info(f"\n🎯 FINAL RESULTS:")
    log_file.info(f"Dataset: {args.dataset}")
    log_file.info(f"Target Domain: {args.test_domain}")
    log_file.info(f"Source Domains: {source_domain_list}")
    info = best_target_info
    log_file.info(f"Best Target Domain Accuracy: {info['acc'] * 100:.2f}% (Round {info['round']}, Source Domain: {info['source_domain']}, Phase: {info['phase']})")


if __name__ == '__main__':
    main()
