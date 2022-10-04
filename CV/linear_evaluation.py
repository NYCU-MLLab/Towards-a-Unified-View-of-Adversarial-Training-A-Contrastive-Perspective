from __future__ import print_function
from ast import arg
import os
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from models.resnet import resnet18

import time
from utils import AverageMeter, logger, pgd_attack_le, accuracy, setup_seed
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import copy

from collections import OrderedDict

from autoattack import AutoAttack

import datetime

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser(description='CIFAR linear evaluation')

parser.add_argument('experiment', type=str, help='exp name')
parser.add_argument('--data', type=str, default='./data', help='location of the data')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to be used (cifar10 or cifar100)')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--epsilon', type=float, default=8. / 255.,
                    help='perturbation')
parser.add_argument('--num-steps-train', type=int, default=10,
                    help='perturb number of steps')
parser.add_argument('--num-steps-test', type=int, default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=2. / 255.,
                    help='perturb step size')

parser.add_argument('--beta', type=float, default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

parser.add_argument('--trainmode', default='normal', type=str,
                    help='at or trades or normal')

parser.add_argument('--checkpoint', default='', type=str,
                    help='path to pretrained model')

parser.add_argument('--decreasing_lr', default='15,20', help='decreasing strategy')

parser.add_argument('--bnNameCnt', default=-1, type=int)


parser.add_argument('--save_path', type=str, default='checkpoints_eval',
                    help='save path of model checkpoints')

parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--resume', action='store_true', help="if specified, resume training")


args = parser.parse_args()

# settings
model_dir = os.path.join(args.save_path, args.experiment)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

log = logger(os.path.join(model_dir))
use_cuda = not args.no_cuda and torch.cuda.is_available()
setup_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

log.info(args.__str__())

# setup data loader
transform_train = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
])
transform_test = transforms.Compose([
  transforms.ToTensor(),
])

if args.dataset == 'cifar10':
    train_datasets = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
    num_classes = 10
elif args.dataset == 'cifar100':
    train_datasets = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform_test)
    num_classes = 100
else:
    print("dataset {} is not supported".format(args.dataset))
    assert False


train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, pin_memory=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)



def train(args, encoder, linear, device, train_loader, optimizer, epoch, log):
    encoder.eval()
    linear.train()

    dataTimeAve = AverageMeter()
    totalTimeAve = AverageMeter()
    end = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        dataTime = time.time() - end
        dataTimeAve.update(dataTime)

        optimizer.zero_grad()

        if args.trainmode == 'trades':

            batch_size = len(data)
            criterion_kl = nn.KLDivLoss(reduction='sum')

            # generate adversarial example
            x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()

            for _ in range(args.num_steps_train):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    encoder.eval()
                    linear.eval()

                    loss_kl = criterion_kl(F.log_softmax(linear(encoder(x_adv)), dim=1),
                                            F.softmax(linear(encoder(data)), dim=1))

                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, data - args.epsilon), data + args.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

            x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

            # zero gradient
            optimizer.zero_grad()
            # calculate robust loss
            encoder.eval()
            linear.train()

            with torch.no_grad():
                clean_feat = encoder(data)
            logits = linear(clean_feat.detach())

            loss = F.cross_entropy(logits, target)

            with torch.no_grad():
                adv_feat = encoder(x_adv)
            logits_adv = linear(adv_feat.detach())

            loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                            F.softmax(logits, dim=1))
            loss += args.beta * loss_robust

        elif args.trainmode == 'at':
            # generate adversarial example
            x_adv = pgd_attack_le(encoder, linear, data, target, device, eps=args.epsilon, iters=args.num_steps_train, alpha=args.step_size).data

            optimizer.zero_grad()
            encoder.eval()
            linear.train()

            with torch.no_grad():
                feat = encoder(x_adv)
            logits = linear(feat.detach())
            loss = F.cross_entropy(logits, target)

        elif args.trainmode == 'normal':
            optimizer.zero_grad()
            encoder.eval()
            linear.train()

            with torch.no_grad():
                feat = encoder(data)
            logits = linear(feat.detach())
            loss = F.cross_entropy(logits, target)


        loss.backward()
        optimizer.step()

        totalTime = time.time() - end
        totalTimeAve.update(totalTime)
        end = time.time()
        # print progress
        if batch_idx % args.log_interval == 0:
            log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tData time: {:.3f}\tTotal time: {:.3f}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
                     100. * batch_idx / len(train_loader), loss.item(), dataTimeAve.avg, totalTimeAve.avg))



def eval(encoder, linear, device, loader, log):
    encoder.eval()
    linear.eval()

    test_loss = 0
    correct = 0
    whole = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = linear(encoder(data))
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            whole += len(target)
    test_loss /= len(loader.dataset)
    log.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({})'.format(
      test_loss, correct, whole,
      100. * correct / whole))
    test_accuracy = correct / whole
    return test_loss, test_accuracy * 100



def eval_adv_test(encoder, linear, device, test_loader, epsilon, alpha, criterion, log, attack_iter=40):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # fix random seed for testing
    torch.manual_seed(1)

    encoder.eval()
    linear.eval()

    end = time.time()

    for i, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)

        input_adv = pgd_attack_le(encoder, linear, input, target, device, eps=epsilon, iters=attack_iter, alpha=alpha).data

        # compute output
        with torch.no_grad():
            output = linear(encoder(input_adv))
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1, input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % 10 == 0) or (i == len(test_loader) - 1):
            log.info(
                'Adv Test: [{}/{}]\t'
                'Time: {batch_time.avg:.4f}\t'
                'Loss: {loss.avg:.3f}\t'
                'Accuracy: {top1.avg}\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1
                )
            )
            
    # log.info(str(target.detach().cpu().numpy()[:25]))
    # log.info(str(output.detach().cpu().numpy()[:25]))

    log.info(' * Robust Accuracy {top1.avg}'.format(top1=top1))

    return float(top1.avg)


def get_state_dict():

    encoder_state_dict, linear_state_dict, optimizor_state_dict = None, None, None

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    state_dict = checkpoint['state_dict']
    
    temp_state_dict = OrderedDict()
    for k, v in state_dict.items():
        temp_state_dict[k] = v

    # deal with adv bn
    state_dict_new = copy.deepcopy(temp_state_dict)

    if args.bnNameCnt >= 0:
        for name, item in temp_state_dict.items():
            if 'bn' in name:
                assert 'bn_list' in name
                state_dict_new[name.replace('.bn_list.{}'.format(args.bnNameCnt), '')] = item

    name_to_del = []
    for name, item in state_dict_new.items():
        # print(name)
        if 'bn' in name and 'adv' in name:
            name_to_del.append(name)
        if 'bn_list' in name:
            name_to_del.append(name)
        if 'fc' in name:
            name_to_del.append(name)
        if 'head' in name:
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    # deal with down sample layer
    keys = list(state_dict_new.keys())[:]
    name_to_del = []
    for name in keys:
        if 'downsample.conv' in name:
            state_dict_new[name.replace('downsample.conv', 'downsample.0')] = state_dict_new[name]
            name_to_del.append(name)
        if 'downsample.bn' in name:
            state_dict_new[name.replace('downsample.bn', 'downsample.1')] = state_dict_new[name]
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]
    
    encoder_state_dict = state_dict_new
    
    if args.resume:
        checkpoint = torch.load(os.path.join(model_dir, 'model.pt'))
        linear_state_dict = checkpoint['linear']
        optimizor_state_dict = checkpoint['optim']
    

    return encoder_state_dict, linear_state_dict, optimizor_state_dict


def get_train_stats():
    last_model = torch.load(os.path.join(model_dir, 'model.pt'))
    latest_epoch = int(last_model['epoch'])
    latest_epoch_acc = float(last_model['acc'])
    latest_epoch_acc_rob = float(last_model['rob_acc'])
    
    best_model = torch.load(os.path.join(model_dir, 'best_model.pt'))
    best_model_epoch = int(best_model['epoch'])
    best_acc = float(best_model['acc'])
    best_acc_rob = float(best_model['rob_acc'])
    
    rob_best_model = torch.load(os.path.join(model_dir, 'rob_best_model.pt'))
    rob_best_model_epoch = int(rob_best_model['epoch'])
    best_rob_acc_cln = float(rob_best_model['acc'])
    best_rob_acc = float(rob_best_model['rob_acc'])

    return latest_epoch, latest_epoch_acc, latest_epoch_acc_rob, best_model_epoch, best_acc, best_acc_rob, rob_best_model_epoch, best_rob_acc_cln, best_rob_acc
    


def main():
    # init model, ResNet18() can be also used here for training

    encoder = resnet18(with_fc=False).to(device)
    
    linear = nn.Linear(512, num_classes).to(device)

    optimizer = optim.SGD(linear.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    if args.checkpoint != '':

        encoder_state_dict, linear_state_dict, optimizor_state_dict = get_state_dict()
            
        encoder.load_state_dict(encoder_state_dict)

        if args.resume:
            linear.load_state_dict(linear_state_dict)
            

        log.info('read checkpoint {}'.format(args.checkpoint))
    else:
        print('empty checkpoint.')
        exit()
    

    best_acc = 0
    best_rob_acc = 0

    starting_epoch = 0

    if args.resume:
        starting_epoch, _, _, _, best_acc, _, _, _, best_rob_acc = get_train_stats()
        for i in range(starting_epoch):
            scheduler.step()

        optimizer.load_state_dict(optimizor_state_dict)
        
        print('resumed from checkpoint, starting epoch: ', starting_epoch + 1, ' best_acc: ', best_acc, ' best_rob_acc: ', best_rob_acc)


    for epoch in range(starting_epoch + 1, args.epochs + 1):

        log.info("Epoch {}, current lr is {}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))

        # adversarial training
        train_start_time = time.time()
        train(args, encoder, linear, device, train_loader, optimizer, epoch, log)
        log.info("epoch time " + str(datetime.timedelta(seconds=time.time() - train_start_time)))

        # evaluation on natural examples
        print('================================================================')
        eval(encoder, linear, device, train_loader, log)
        
        _, vali_acc = eval(encoder, linear, device, test_loader, log)

        print('================================================================')

        # adv testing
        vali_rob_acc = eval_adv_test(encoder, linear, device, test_loader, epsilon=args.epsilon, alpha=args.step_size, criterion=F.cross_entropy, log=log, attack_iter=args.num_steps_test)

        #adjust learning rate for SGD
        scheduler.step()

        # save checkpoint
        if epoch % args.save_freq == 0:
            info_dict = {
                'epoch': epoch,
                'linear': linear.state_dict(),
                'optim': optimizer.state_dict(),
                'acc': vali_acc,
                'rob_acc': vali_rob_acc
            }
            torch.save(info_dict, os.path.join(model_dir, 'model_{}.pt'.format(epoch)))
            torch.save(info_dict, os.path.join(model_dir, 'model.pt'))

        is_best = vali_acc > best_acc
        best_acc = max(vali_acc, best_acc)

        rob_acc_is_best = vali_rob_acc > best_rob_acc
        best_rob_acc = max(vali_rob_acc, best_rob_acc)

        if is_best:
            torch.save({
                'epoch': epoch,
                'linear': linear.state_dict(),
                'optim': optimizer.state_dict(),
                'acc': vali_acc,
                'rob_acc': vali_rob_acc
            }, os.path.join(model_dir, 'best_model.pt'))

        if rob_acc_is_best:
            torch.save({
                'epoch': epoch,
                'linear': linear.state_dict(),
                'optim': optimizer.state_dict(),
                'acc': vali_acc,
                'rob_acc': vali_rob_acc
            }, os.path.join(model_dir, 'rob_best_model.pt'))
        
        

    _, latest_epoch_acc, latest_epoch_acc_rob, best_model_epoch, best_acc, best_acc_rob, rob_best_model_epoch, best_rob_acc_cln, best_rob_acc = get_train_stats()

    log.info("On the rob_best_model (epoch {}), test acc is {}, test robust acc is {}".format(rob_best_model_epoch, best_rob_acc_cln, best_rob_acc))

    log.info("On the best_model (epoch {}), test acc is {}, test robust acc is {}".format(best_model_epoch, best_acc, best_acc_rob))

    log.info("On the final model, test acc is {}, test robust acc is {}".format(latest_epoch_acc, latest_epoch_acc_rob))


if __name__ == '__main__':
    main()
