import argparse
from torch._C import NoopLogger
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from models.resnet_multi_bn_cluster import resnet18 as resnet18_cluster
from models.resnet_multi_bn_cluster import proj_head as proj_head_cluster
from models.resnet_multi_bn import resnet18, proj_head
from utils import *
from pgd_contrastive import *
import torchvision.transforms as transforms

import numpy as np

from data.cifar10_cluster import CustomCIFAR10, CustomCIFAR100
from torchvision.transforms import InterpolationMode
from optimizer.lars import LARS

import pickle

import datetime


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('experiment', type=str, help='location for saving trained models')
parser.add_argument('--data', type=str, default='./data', help='location of the data')
parser.add_argument('--dataset', type=str, default='cifar10', help='which dataset to be used, (cifar10 or cifar100)')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--resume', action='store_true', help='if resume training')
parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer type')
parser.add_argument('--lr', default=0.5, type=float, help='optimizer lr')
parser.add_argument('--temp', default=0.5, type=float, help='temperature for contrastive loss')
parser.add_argument('--scheduler', default='cosine', type=str, help='lr scheduler type')
parser.add_argument('--twoLayerProj', action='store_true', help='if specified, use two layers linear head for simclr proj head')
parser.add_argument('--pgd_iter', default=5, type=int, help='how many iterations employed to attack the model')
parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--num_workers', type=int, default=4, help='num_workers')

parser.add_argument('--contrastive_at', action='store_true', help='if specified, use the contrastive adversarial training mode')

parser.add_argument('--contrastive_trades', action='store_true', help='if specified, use the contrastive TRADES mode')
parser.add_argument('--trades_lamb', type=float, help='hyperparameter for contrastive TRADES')

parser.add_argument('--contrastive_mart', action='store_true', help='if specified, use the contrastive MART mode')
parser.add_argument('--mart_lamb', type=float, help='hyperparameter for contrastive MART')

parser.add_argument('--contrastive_hat', action='store_true', help='if specified, use the contrastive HAT mode')
parser.add_argument('--hat_beta', type=float, help='hyperparameter beta for contrastive HAT')
parser.add_argument('--hat_gamma', type=float, help='hyperparameter gamma for contrastive HAT')
parser.add_argument('--hat_std_model_path', type=str, default='../../data', help='location of the data')

parser.add_argument('--save_freq', default=1, type=int, help='model save frequency')
parser.add_argument('--test_freq', default=1000, type=int, help='model save frequency')

parser.add_argument('--no_pre_normalize', action='store_true', help="if specified, don't do pre-process normalization")


parser.add_argument('--cluster', action='store_true', help="if specified, use ClusterFit")
parser.add_argument('--cluster_lamb', type=float, help='hyperparameter for ClusterFit')
parser.add_argument('--cluster_ori_att', action='store_true', help="if specified, use ClusterFit attack without eval()")
parser.add_argument('--cluster_iter', default=5, type=int, help='how many iterations employed to attack the model')



def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr


def main():
    global args
    args = parser.parse_args()

    assert args.dataset in ['cifar100', 'cifar10']

    save_dir = os.path.join('checkpoints', args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    log = logger(path=save_dir)
    log.info(str(args))
    setup_seed(args.seed)

    # different attack corresponding to different bn settings
    if args.contrastive_trades or args.contrastive_at or args.contrastive_mart or args.contrastive_hat:
        bn_names = ['normal', 'pgd']

        if args.contrastive_hat:
            bn_names.append('pgd_helper')

        if args.cluster:
            bn_names.append('pgd_cluster')
    else:
        bn_names = ['normal', ]

    pre_normalize = not args.no_pre_normalize

    # define model
    if args.cluster:
        model = resnet18_cluster(pretrained=False, bn_names=bn_names, use_normalize=pre_normalize)
        ch = model.fc.in_features
        model.fc = proj_head_cluster(ch, bn_names=bn_names, twoLayerProj=args.twoLayerProj)
    else:
        model = resnet18(pretrained=False, bn_names=bn_names, use_normalize=pre_normalize)
        ch = model.fc.in_features
        model.fc = proj_head(ch, bn_names=bn_names, twoLayerProj=args.twoLayerProj)

    model.cuda()
    cudnn.benchmark = True

    strength = 1.0
    rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * strength, 0.4 * strength, 0.4 * strength, 0.1 * strength)], p=0.8 * strength)
    rnd_gray = transforms.RandomGrayscale(p=0.2 * strength)

    tfs_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(1.0 - 0.9 * strength, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        rnd_color_jitter,
        rnd_gray,
        transforms.ToTensor(),
    ])

    tfs_test = transforms.Compose([
        transforms.ToTensor(),
    ])


    tfs_anch = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    if args.cluster:
        label_pseudo_train_list = []
        num_classes_list = [2, 10, 50, 100, 500]

        if args.dataset == 'cifar10':
            dict_name = 'data/imagenet_clPretrain_pseudo_labels.pkl'
        elif args.dataset == 'cifar100':
            dict_name = 'data/cifar100_pseudo_labels.pkl'
        
        print('read pseudo-labels from: ', dict_name)

        f = open(dict_name, 'rb')  # Pickle file is newly created where foo1.py is
        feat_label_dict = pickle.load(f)  # dump data to f
        f.close()
        for i in range(5):
            class_num = num_classes_list[i]
            key_train = 'pseudo_train_{}'.format(class_num)
            label_pseudo_train = feat_label_dict[key_train]
            label_pseudo_train_list.append(label_pseudo_train)

    # dataset process
    if args.dataset == 'cifar10':
        if args.cluster:
            train_datasets = CustomCIFAR10(root=args.data, train=True, transform=tfs_train, download=True, anchTrans=tfs_anch, 
                                                pseudoLabel_002=label_pseudo_train_list[0],
                                                pseudoLabel_010=label_pseudo_train_list[1],
                                                pseudoLabel_050=label_pseudo_train_list[2],
                                                pseudoLabel_100=label_pseudo_train_list[3],
                                                pseudoLabel_500=label_pseudo_train_list[4])
        else:
            train_datasets = CustomCIFAR10(root=args.data, train=True, transform=tfs_train, download=True, anchTrans=tfs_anch)

        val_train_datasets = datasets.CIFAR10(root=args.data, train=True, transform=tfs_test, download=True)
        test_datasets = datasets.CIFAR10(root=args.data, train=False, transform=tfs_test, download=True)
        num_classes = 10

    elif args.dataset == 'cifar100':
        if args.cluster:
            train_datasets = CustomCIFAR100(root=args.data, train=True, transform=tfs_train, download=True, anchTrans=tfs_anch, 
                                                pseudoLabel_002=label_pseudo_train_list[0],
                                                pseudoLabel_010=label_pseudo_train_list[1],
                                                pseudoLabel_050=label_pseudo_train_list[2],
                                                pseudoLabel_100=label_pseudo_train_list[3],
                                                pseudoLabel_500=label_pseudo_train_list[4])
        else:
            train_datasets = CustomCIFAR100(root=args.data, train=True, transform=tfs_train, download=True, anchTrans=tfs_anch)

        val_train_datasets = datasets.CIFAR100(root=args.data, train=True, transform=tfs_test, download=True)
        test_datasets = datasets.CIFAR100(root=args.data, train=False, transform=tfs_test, download=True)
        num_classes = 100
    else:
        print("unknow dataset")
        assert False

    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True)

    val_train_loader = torch.utils.data.DataLoader(
        val_train_datasets,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        )

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=1e-6)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        print("no defined optimizer")
        assert False

    if args.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs * len(train_loader) * 10, ], gamma=1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    args.epochs * len(train_loader),
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps=10 * len(train_loader))
        )
    else:
        print("unknown schduler: {}".format(args.scheduler))
        assert False

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(os.path.join(save_dir, 'model.pt'))
        model.load_state_dict(checkpoint['state_dict'])

        if 'epoch' in checkpoint and 'optim' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optim'])
            for i in range((start_epoch - 1) * len(train_loader)):
                scheduler.step()
            log.info("resume the checkpoint from epoch {}".format(checkpoint['epoch']))
        else:
            log.info("cannot resume due to lack of files")
            assert False



    if args.contrastive_trades:
        contrastive_criterion = ContrastiveTrades(args.trades_lamb, t=args.temp)
    elif args.contrastive_hat:

        hat_std_model = resnet18(pretrained=False, bn_names=['normal'])
        ch = model.fc.in_features
        hat_std_model.fc = proj_head(ch, bn_names=['normal'], twoLayerProj=False)

        checkpoint = torch.load(args.hat_std_model_path)
        hat_std_model.load_state_dict(checkpoint['state_dict'])
        hat_std_model.cuda()
        
        contrastive_criterion = ContrastiveHat(args.hat_beta, args.hat_gamma, hat_std_model, t=args.temp)

    elif args.contrastive_mart:
        contrastive_criterion = ContrastiveMart(args.mart_lamb, t=args.temp)
    else:
        contrastive_criterion = ContrastiveAt(t=args.temp)


    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch - start_epoch != 0:
            eta = ((time.time() - start_time) / (epoch - start_epoch)) * ((args.epochs + 1) - epoch) 
            eta = str(datetime.timedelta(seconds=eta))
        else:
            eta = None

        log.info("current lr is {}. ETA {}".format(optimizer.state_dict()['param_groups'][0]['lr'], eta))
        train_start_time = time.time()

        train(contrastive_criterion, train_loader, model, optimizer, scheduler, epoch, log, num_classes=num_classes)

        log.info("epoch time " + str(datetime.timedelta(seconds=time.time() - train_start_time)))


        if epoch % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, filename=os.path.join(save_dir, 'model.pt'))

        if epoch % 100 == 0 and epoch > 0:
            log.info("elapsed time {}".format(time.time() - start_time))

            if epoch == args.epochs:
                log.info("total training time is {}".format(time.time() - start_time))

            if epoch % args.test_freq == 0:
            
                acc, tacc = validate(val_train_loader, test_loader, model, log, num_classes=num_classes)
                
                log.info('train_accuracy {acc:.3f}'
                        .format(acc=acc))
                log.info('test_accuracy {tacc:.3f}'
                        .format(tacc=tacc))

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, filename=os.path.join(save_dir, 'model_{}.pt'.format(epoch)))



def train(contrastive_criterion, train_loader, model, optimizer, scheduler, epoch, log, num_classes):

    
    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()
    
    if args.cluster:
        ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    end = time.time()
    for i, data_tuple in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)

        if args.cluster:
            class_inputs, anchor_inputs, cluster_targets = data_tuple

            tt = []
            for tt_ in cluster_targets:
                tt.append(tt_.cuda(non_blocking=True).long())
            cluster_targets = tt
        else:
            class_inputs, anchor_inputs = data_tuple

        class_inputs = class_inputs.cuda(non_blocking=True)
        anchor_inputs = anchor_inputs.cuda(non_blocking=True)

        if args.contrastive_at:
            adv_anchor = contrastive_criterion.pgd(model, class_inputs, anchor_inputs, iters=args.pgd_iter)
            
            cls_feat = model.train()(class_inputs, 'normal')
            adv_anchor_feat = model.train()(adv_anchor, 'pgd')
            
            loss = contrastive_criterion(cls_feat, adv_anchor_feat)

        elif args.contrastive_trades:
            adv_anchor = contrastive_criterion.pgd(model, class_inputs, anchor_inputs, iters=args.pgd_iter)
            
            cls_feat = model.train()(class_inputs, 'normal')
            anchor_feat = model.train()(anchor_inputs, 'normal')

            adv_anchor_feat = model.train()(adv_anchor, 'pgd')

            loss = contrastive_criterion(cls_feat, anchor_feat, adv_anchor_feat)

        elif args.contrastive_mart:
            adv_anchor = contrastive_criterion.pgd(model, class_inputs, anchor_inputs, iters=args.pgd_iter)

            cls_feat = model.train()(class_inputs, 'normal')
            anchor_feat = model.train()(anchor_inputs, 'normal')
            adv_anchor_feat = model.train()(adv_anchor, 'pgd')

            loss = contrastive_criterion(cls_feat, anchor_feat, adv_anchor_feat)
        elif args.contrastive_hat:
            adv_anchor, helper_anchor, helper_y = contrastive_criterion.pgd(model, class_inputs, anchor_inputs, iters=args.pgd_iter)

            cls_feat = model.train()(class_inputs, 'normal')
            anchor_feat = model.train()(anchor_inputs, 'normal')
            adv_anchor_feat = model.train()(adv_anchor, 'pgd')
            helper_anchor_feat = model.train()(helper_anchor, 'pgd_helper')

            loss = contrastive_criterion(cls_feat, anchor_feat, adv_anchor_feat, helper_anchor_feat, helper_y)
        else:
            cls_feat = model.train()(class_inputs, 'normal')
            anch_feat = model.train()(anchor_inputs, 'normal')

            loss = contrastive_criterion(cls_feat, anch_feat)
        

        if args.cluster:
            cluster_input = cluster_attack(model, anchor_inputs, cluster_targets, iters=args.cluster_iter, cluster_lamb=args.cluster_lamb, ori_att=args.cluster_ori_att)
            cluster_feats = model.train()(cluster_input, 'pgd_cluster', cluster=True)

            ce_loss = 0
            for label_idx in range(5):
                tgt = cluster_targets[label_idx].long()
                lgt = cluster_feats[label_idx]
                ce_loss += ce_criterion(lgt, tgt) / 5.

            loss = loss + ce_loss * args.cluster_lamb


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(float(loss.detach().cpu()), class_inputs.shape[0])

        train_time = time.time() - end
        end = time.time()
        train_time_meter.update(train_time)

        # torch.cuda.empty_cache()
        if i % args.print_freq == 0:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'data_time: {data_time.val:.2f}\t'
                     'iter_train_time: {train_time.avg:.2f}\t'.format(
                          epoch, i, len(train_loader), loss=losses,
                          data_time=data_time_meter, train_time=train_time_meter))

    return losses.avg


def validate(train_loader, val_loader, model, log, num_classes=10, bn_name='normal'):
    """
    Run evaluation
    """
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    train_time_meter = AverageMeter()
    losses = AverageMeter()
    losses.reset()
    end = time.time()

    # train a fc on the representation
    for param in model.parameters():
        param.requires_grad = False

    previous_fc = model.fc
    ch = model.fc.in_features
    model.fc = nn.Linear(ch, num_classes)
    model.cuda()

    epochs_max = 100
    lr = 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(step,
                                                epochs_max * len(train_loader),
                                                1,  # since lr_lambda computes multiplicative factor
                                                1e-6 / lr,
                                                warmup_steps=0)
    )

    for epoch in range(epochs_max):
        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        

        for i, (sample) in enumerate(train_loader):

            x, y = sample[0].cuda(non_blocking=True), sample[1].cuda(non_blocking=True)
            p = model.eval()(x, bn_name)
            loss = criterion(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(float(loss.detach().cpu()))

            train_time = time.time() - end
            end = time.time()
            train_time_meter.update(train_time)

        log.info('Test epoch: ({0})\t'
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'train_time: {train_time.avg:.2f}\t'.format(
                    epoch, loss=losses, train_time=train_time_meter))

    acc = []
    for loader in [train_loader, val_loader]:
        losses = AverageMeter()
        losses.reset()
        top1 = AverageMeter()

        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # compute output
            with torch.no_grad():
                outputs = model.eval()(inputs, bn_name)
                loss = criterion(outputs, targets)

            outputs = outputs.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, targets)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            if i % args.print_freq == 0:
                log.info('Test: [{0}/{1}]\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                             i, len(loader), loss=losses, top1=top1))

        acc.append(top1.avg)

    # recover every thing
    model.fc = previous_fc
    model.cuda()
    for param in model.parameters():
        param.requires_grad = True

    return acc


def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)


if __name__ == '__main__':
    main()


