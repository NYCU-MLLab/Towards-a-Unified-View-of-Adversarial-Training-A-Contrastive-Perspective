from builtins import print
import torch
import torch.nn as nn
import os
import time
import numpy as np
import random
import copy
from pdb import set_trace
from collections import OrderedDict

import torch.nn.functional as F


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def pgd_attack(model, images, labels, device, eps=8. / 255., alpha=2. / 255., iters=20, advFlag=None, forceEval=True, randomInit=True):
    # images = images.to(device)
    # labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    # init
    if randomInit:
        delta = torch.rand_like(images) * eps * 2 - eps
    else:
        delta = torch.zeros_like(images)
    delta = torch.nn.Parameter(delta, requires_grad=True)

    for i in range(iters):
        model.eval()
        outputs = model(images + delta)

        model.zero_grad()
        cost = loss(outputs, labels)
        # cost.backward()
        delta_grad = torch.autograd.grad(cost, [delta])[0]

        delta.data = delta.data + alpha * delta_grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

    model.zero_grad()

    return (images + delta).detach()


def pgd_attack_le(encoder, linear, images, labels, device, eps=8. / 255., alpha=2. / 255., iters=20):
    ce_loss = nn.CrossEntropyLoss()

    delta = torch.rand_like(images) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)
    # delta = torch.nn.Parameter(delta, requires_grad=True)

    for i in range(iters):
        encoder.eval()
        linear.eval()

        outputs = linear(encoder(images + delta))

        encoder.zero_grad()
        linear.zero_grad()

        loss = ce_loss(outputs, labels)

        loss.backward()
        delta.data = delta.data + alpha * delta.grad.sign()

        # delta_grad = torch.autograd.grad(loss, [delta])[0]
        # delta.data = delta.data + alpha * delta_grad.sign()

        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

    encoder.zero_grad()
    linear.zero_grad()

    return (images + delta).detach()


def eval_adv_test(model, device, test_loader, epsilon, alpha, criterion, log, attack_iter=40):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # fix random seed for testing
    torch.manual_seed(1)

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        input_adv = pgd_attack(model, input, target, device, eps=epsilon, iters=attack_iter, alpha=alpha).data

        # compute output
        output = model.eval()(input_adv)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1, input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % 10 == 0) or (i == len(test_loader) - 1):
            log.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1
                )
            )

    log.info(' * Adv Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def eval_adv_test_dist(model, device, test_loader, epsilon, alpha, criterion, log, world_size, attack_iter=40, randomInit=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # fix random seed for testing
    torch.manual_seed(1)

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        input_adv = pgd_attack(model, input, target, device, eps=epsilon, iters=attack_iter, alpha=alpha, randomInit=randomInit).data

        # compute output
        output = model(input_adv)
        output_list = [torch.zeros_like(output) for _ in range(world_size)]
        torch.distributed.all_gather(output_list, output)
        output = torch.cat(output_list)

        target_list = [torch.zeros_like(target) for _ in range(world_size)]
        torch.distributed.all_gather(target_list, target)
        target = torch.cat(target_list)

        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1, input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % 10 == 0) or (i == len(test_loader) - 1):
            log.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1
                )
            )

    log.info(' * Adv Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class logger(object):
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, "log.txt"), 'a') as f:
            f.write(msg + "\n")


def fix_bn(model, fixmode):
    if fixmode == 'f1':
        # fix none
        pass
    elif fixmode == 'f2':
        # fix previous three layers
        for name, m in model.named_modules():
            if not ("layer4" in name or "fc" in name):
                m.eval()
    elif fixmode == 'f3':
        # fix every layer except fc
        # fix previous four layers
        for name, m in model.named_modules():
            if not ("fc" in name):
                m.eval()
    else:
        assert False


# loss
def pair_cosine_similarity(x, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps)


def nt_xent(x, t=0.5):
    # print("device of x is {}".format(x.device))
    sim_mat = pair_cosine_similarity(x)
    sim_mat = torch.exp(sim_mat / t)
    idx = torch.arange(sim_mat.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    sim_mat = sim_mat[idx]
    # subtract the similarity of 1 from the numerator
    y = sim_mat.diag() / (sim_mat.sum(0) - torch.exp(torch.tensor(1 / t)))

    return -torch.log(y).mean()


def nt_xent_dual(x1, x2, t=0.5):
    x1_n = nn.functional.normalize(x1)
    x2_n = nn.functional.normalize(x2)
    sim_mat = torch.matmul(x1_n, x2_n.t())

    sim_mat = torch.exp(sim_mat / t)
    self_sim = sim_mat.diag()

    idx = torch.arange(sim_mat.size()[0])
    idx[::2] += 1
    idx[1::2] -= 1
    sim_mat = sim_mat[idx]

    y = sim_mat.diag() / (sim_mat.sum(0) - self_sim)

    return -torch.log(y).mean()



class ContrastiveLossSPMultiAug:

    def __init__(self, batch_size, aug_size, t=0.5):

        self.batch_size = batch_size
        self.aug_size = aug_size
        self.t = t

        self.neg_gather_idx, self.pos_gather_idx = self.generate_gather_index(batch_size)

        
    def generate_gather_index(self, batch_size):
        mask = torch.triu(torch.full((batch_size - 1, batch_size), 1), 1).repeat_interleave(self.aug_size, dim=0)
        neg_gather_idx = torch.arange(0, (batch_size - 1) * self.aug_size).unsqueeze(1).repeat(1, batch_size) + self.aug_size
        neg_gather_idx[mask == 1] = torch.arange(0, (batch_size - 1) * self.aug_size).unsqueeze(1).repeat(1, batch_size)[mask == 1]

        neg_gather_idx = torch.repeat_interleave(neg_gather_idx, self.aug_size, dim=1).cuda(non_blocking=True)
        pos_gather_idx = torch.arange(0, batch_size * self.aug_size).unsqueeze(0).cuda(non_blocking=True)

        # if no_anch_aug:
            # neg_gather_idx = neg_gather_idx.cuda(non_blocking=True)
            # pos_gather_idx = torch.arange(0, batch_size).repeat_interleave(self.aug_size).unsqueeze(1).cuda(non_blocking=True)
        
        return neg_gather_idx, pos_gather_idx

    
    def __call__(self, cls_feat, anch_feat):
        batch_size = int(cls_feat.size()[0] / self.aug_size)

        if batch_size != self.batch_size:
            neg_gather_idx, pos_gather_idx = self.generate_gather_index(batch_size)
        else:
            neg_gather_idx, pos_gather_idx = self.neg_gather_idx, self.pos_gather_idx

        cls_feat_n = nn.functional.normalize(cls_feat)
        anch_feat_n = nn.functional.normalize(anch_feat)
        
        sim_mat = torch.matmul(cls_feat_n, anch_feat_n.t()) / self.t

        neg_sim = sim_mat.gather(0, neg_gather_idx)
        pos_sim = sim_mat.gather(0, pos_gather_idx)

        logits = torch.vstack([neg_sim, pos_sim]).t()

        # if no_anch_aug:
        #     neg_sim = sim_mat.gather(0, neg_gather_idx).repeat_interleave(self.aug_size, dim=1)
        #     pos_sim = sim_mat.gather(1, pos_gather_idx)

        #     logits = torch.hstack([neg_sim.t(), pos_sim])


        y = torch.full((batch_size * self.aug_size, ), logits.size()[1] - 1).cuda(non_blocking=True)

        loss = F.cross_entropy(logits, y)

        return loss






def clae_cl(x, t=0.5):
    x1_feat, x2_feat = x[::2], x[1::2]
    x1_feat = nn.functional.normalize(x1_feat)
    x2_feat = nn.functional.normalize(x2_feat)
    x1_x2_mat = torch.exp(torch.matmul(x1_feat, x2_feat.t()) / t)

    prob = torch.diag(x1_x2_mat) / torch.sum(x1_x2_mat, dim=0)
    loss = -torch.mean(torch.log(prob))
    
    return loss


def clae_cl_at(x1_feat, x2_feat, t=0.5):
    x1_feat = nn.functional.normalize(x1_feat)
    x2_feat = nn.functional.normalize(x2_feat)
    x1_x2_mat = torch.exp(torch.matmul(x1_feat, x2_feat.t()) / t)

    prob = torch.diag(x1_x2_mat) / torch.sum(x1_x2_mat, dim=0)
    loss = -torch.mean(torch.log(prob))
    
    return loss


def contrastive_trades_kl(x, x_adv, t=0.5):
    criterion_kl = nn.KLDivLoss(reduction='sum')

    mask = torch.arange(x.shape[0] - 1).unsqueeze(1).repeat(1, x.shape[0]).cuda(non_blocking=True)
    mask += torch.full(mask.shape, 1).tril().cuda(non_blocking=True)
    # print(mask)

    x_n = nn.functional.normalize(x)
    x_adv_n = nn.functional.normalize(x_adv)

    sim_mat_nat = torch.matmul(x_n, x_n.t()) / t
    sim_mat_adv = torch.matmul(x_n, x_adv_n.t()) / t
    # print(sim_mat_nat)
    # print(sim_mat_adv)

    logits_nat = torch.gather(sim_mat_nat, dim=0, index=mask).t()
    logits_adv = torch.gather(sim_mat_adv, dim=0, index=mask).t()
    # print(sim_mat_nat)
    # print(sim_mat_adv)

    loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1))

    return loss_kl


def contrastive_trades_loss(x, x_adv, lamb, t=0.5):
    criterion_kl = nn.KLDivLoss(reduction='sum')
    batch_size_double = x.shape[0]

    mask = torch.arange(x.shape[0] - 1).unsqueeze(1).repeat(1, x.shape[0]).cuda(non_blocking=True)
    mask += torch.full(mask.shape, 1).tril().cuda(non_blocking=True)
    # print(mask)

    x_n = nn.functional.normalize(x)
    x_adv_n = nn.functional.normalize(x_adv)

    sim_mat_nat = torch.matmul(x_n, x_n.t()) / t
    sim_mat_adv = torch.matmul(x_n, x_adv_n.t()) / t
    # print(sim_mat_nat)
    # print(sim_mat_adv)

    logits_nat = torch.gather(sim_mat_nat, dim=0, index=mask).t()
    logits_adv = torch.gather(sim_mat_adv, dim=0, index=mask).t()
    # print(sim_mat_nat)
    # print(sim_mat_adv)

    y = torch.arange(0, batch_size_double - 1, 2).repeat_interleave(2).cuda(non_blocking=True)
    loss_natural = F.cross_entropy(logits_nat, y)

    loss_robust = (1.0 / batch_size_double) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1))

    return loss_natural + lamb * loss_robust



class ContrastiveTradesMultiAug:

    def __init__(self, batch_size, aug_size, lamb, t=0.5):

        self.batch_size = batch_size
        self.aug_size = aug_size
        self.t = t

        self.criterion_kl = nn.KLDivLoss(reduction='sum')

        self.lamb = lamb

        self.neg_gather_idx, self.pos_gather_idx = self.generate_gather_index(batch_size)

        
    def generate_gather_index(self, batch_size):
        mask = torch.triu(torch.full((batch_size - 1, batch_size), 1), 1).repeat_interleave(self.aug_size, dim=0)
        neg_gather_idx = torch.arange(0, (batch_size - 1) * self.aug_size).unsqueeze(1).repeat(1, batch_size) + self.aug_size
        neg_gather_idx[mask == 1] = torch.arange(0, (batch_size - 1) * self.aug_size).unsqueeze(1).repeat(1, batch_size)[mask == 1]

        neg_gather_idx = torch.repeat_interleave(neg_gather_idx, self.aug_size, dim=1).cuda(non_blocking=True)
        pos_gather_idx = torch.arange(0, batch_size * self.aug_size).unsqueeze(0).cuda(non_blocking=True)

        # if no_anch_aug:
        #     neg_gather_idx = neg_gather_idx.cuda(non_blocking=True)
        #     pos_gather_idx = torch.arange(0, batch_size).repeat_interleave(self.aug_size).unsqueeze(1).cuda(non_blocking=True)
        
        return neg_gather_idx, pos_gather_idx
    
    def get_logits(self, sim_mat, neg_gather_idx, pos_gather_idx):

        neg_sim = sim_mat.gather(0, neg_gather_idx)
        pos_sim = sim_mat.gather(0, pos_gather_idx)

        logits = torch.vstack([neg_sim, pos_sim]).t()

        # if no_anch_aug:
        #     neg_sim = sim_mat.gather(0, neg_gather_idx).repeat_interleave(self.aug_size, dim=1)
        #     pos_sim = sim_mat.gather(1, pos_gather_idx)
        #     logits = torch.hstack([neg_sim.t(), pos_sim])
        
        return logits


    def pgd(self, model, cls_inputs, anch_inputs, eps=8. / 255., alpha=2. / 255., iters=5):
        delta = torch.rand_like(anch_inputs) * eps * 2 - eps
        delta = torch.nn.Parameter(delta)
        
        for i in range(iters):
            cls_feat = model.eval()(cls_inputs, 'normal').detach()
            cln_anch_feat = model.eval()(anch_inputs, 'normal').detach()

            adv_anch_feat = model.eval()(anch_inputs + delta, 'pgd')

            model.zero_grad()
            loss = self.kl(cls_feat, cln_anch_feat, adv_anch_feat)
            loss.backward()
            # print("loss is {}".format(loss))

            delta.data = delta.data + alpha * delta.grad.sign()
            delta.grad = None
            delta.data = torch.clamp(delta.data, min=-eps, max=eps)
            delta.data = torch.clamp(anch_inputs + delta.data, min=0, max=1) - anch_inputs 

        return (anch_inputs + delta).detach()


    def kl(self, cls_feat, anch_feat, adv_anch_feat):
        batch_size = int(cls_feat.size()[0] / self.aug_size)

        if batch_size != self.batch_size:
            neg_gather_idx, pos_gather_idx = self.generate_gather_index(batch_size)
        else:
            neg_gather_idx, pos_gather_idx = self.neg_gather_idx, self.pos_gather_idx

        cls_feat_n = nn.functional.normalize(cls_feat)
        anch_feat_n = nn.functional.normalize(anch_feat)
        adv_anch_feat_n = nn.functional.normalize(adv_anch_feat)

        sim_mat_nat = torch.matmul(cls_feat_n, anch_feat_n.t()) / self.t
        sim_mat_adv = torch.matmul(cls_feat_n, adv_anch_feat_n.t()) / self.t

        logits_nat = self.get_logits(sim_mat_nat, neg_gather_idx, pos_gather_idx)
        logits_adv = self.get_logits(sim_mat_adv, neg_gather_idx, pos_gather_idx)

        return self.criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1))

    
    def __call__(self, cls_feat, anch_feat, adv_anch_feat):

        batch_size = int(cls_feat.size()[0] / self.aug_size)

        if batch_size != self.batch_size:
            neg_gather_idx, pos_gather_idx = self.generate_gather_index(batch_size)
        else:
            neg_gather_idx, pos_gather_idx = self.neg_gather_idx, self.pos_gather_idx

        cls_feat_n = nn.functional.normalize(cls_feat)
        anch_feat_n = nn.functional.normalize(anch_feat)
        adv_anch_feat_n = nn.functional.normalize(adv_anch_feat)

        sim_mat_nat = torch.matmul(cls_feat_n, anch_feat_n.t()) / self.t
        sim_mat_adv = torch.matmul(cls_feat_n, adv_anch_feat_n.t()) / self.t

        logits_nat = self.get_logits(sim_mat_nat, neg_gather_idx, pos_gather_idx)
        logits_adv = self.get_logits(sim_mat_adv, neg_gather_idx, pos_gather_idx)

        y = torch.full((batch_size * self.aug_size, ), logits_nat.size()[1] - 1).cuda(non_blocking=True)

        loss_natural = F.cross_entropy(logits_nat, y)

        loss_robust = (1.0 / logits_nat.size()[0]) * self.criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1))

        return loss_natural + self.lamb * loss_robust


class ContrastiveAt:

    def __init__(self, t=0.5):
        self.t = t
        
    def pgd(self, model, cls_inputs, anch_inputs, eps=8. / 255., alpha=2. / 255., iters=5):

        delta = torch.rand_like(anch_inputs) * eps * 2 - eps
        delta = torch.nn.Parameter(delta)
        
        for i in range(iters):

            feat1 = model.eval()(cls_inputs, 'normal').detach()
            feat2 = model.eval()(anch_inputs + delta, 'pgd')

            model.zero_grad()
            
            loss = self.__call__(feat1, feat2)

            loss.backward()

            delta.data = delta.data + alpha * delta.grad.sign()
            delta.grad = None
            delta.data = torch.clamp(delta.data, min=-eps, max=eps)
            delta.data = torch.clamp(anch_inputs + delta.data, min=0, max=1) - anch_inputs 

        return (anch_inputs + delta).detach()
    

    def free_pgd(self, model, cls_inputs, anch_inputs, eps=8. / 255., alpha=2. / 255., iters=10):

        delta = torch.rand_like(anch_inputs) * eps * 2 - eps
        delta = torch.nn.Parameter(delta)
        
        for i in range(iters):

            feat1 = model.train()(cls_inputs, 'normal')
            feat2 = model.train()(anch_inputs + delta, 'pgd')

            loss = self.__call__(feat1, feat2) / (iters + 1)

            loss.backward()

            delta.data = delta.data + alpha * delta.grad.sign()
            delta.grad = None
            delta.data = torch.clamp(delta.data, min=-eps, max=eps)
            delta.data = torch.clamp(anch_inputs + delta.data, min=0, max=1) - anch_inputs 

        return (anch_inputs + delta).detach()
    
    def __call__(self, cls_feat, anch_feat):
        cls_feat_n = nn.functional.normalize(cls_feat)
        anch_feat_n = nn.functional.normalize(anch_feat)
        
        sim_mat = torch.matmul(cls_feat_n, anch_feat_n.t()) / self.t

        logits = sim_mat.t()

        y = torch.arange(logits.size(0)).cuda(non_blocking=True)

        loss = F.cross_entropy(logits, y)

        return loss


class ContrastiveTrades:

    def __init__(self, lamb, t=0.5):

        self.t = t
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.lamb = lamb


    def pgd(self, model, cls_inputs, anch_inputs, eps=8. / 255., alpha=2. / 255., iters=5):
        delta = torch.rand_like(anch_inputs) * eps * 2 - eps
        delta = torch.nn.Parameter(delta)
        
        for i in range(iters):
            cls_feat = model.eval()(cls_inputs, 'normal').detach()
            cln_anch_feat = model.eval()(anch_inputs, 'normal').detach()
            adv_anch_feat = model.eval()(anch_inputs + delta, 'pgd')

            model.zero_grad()

            cls_feat_n = nn.functional.normalize(cls_feat)
            anch_feat_n = nn.functional.normalize(cln_anch_feat)
            adv_anch_feat_n = nn.functional.normalize(adv_anch_feat)

            sim_mat_nat = torch.matmul(cls_feat_n, anch_feat_n.t()) / self.t
            sim_mat_adv = torch.matmul(cls_feat_n, adv_anch_feat_n.t()) / self.t

            logits_nat = sim_mat_nat.t()
            logits_adv = sim_mat_adv.t()

            loss = self.criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1))

            loss.backward()
            # print("loss is {}".format(loss))

            delta.data = delta.data + alpha * delta.grad.sign()
            delta.grad = None
            delta.data = torch.clamp(delta.data, min=-eps, max=eps)
            delta.data = torch.clamp(anch_inputs + delta.data, min=0, max=1) - anch_inputs 

        return (anch_inputs + delta).detach()
    
    def __call__(self, cls_feat, anch_feat, adv_anch_feat):
        cls_feat_n = nn.functional.normalize(cls_feat)
        anch_feat_n = nn.functional.normalize(anch_feat)
        adv_anch_feat_n = nn.functional.normalize(adv_anch_feat)

        sim_mat_nat = torch.matmul(cls_feat_n, anch_feat_n.t()) / self.t
        sim_mat_adv = torch.matmul(cls_feat_n, adv_anch_feat_n.t()) / self.t

        logits_nat = sim_mat_nat.t()
        logits_adv = sim_mat_adv.t()

        y = torch.arange(logits_nat.size(0)).cuda(non_blocking=True)

        loss_natural = F.cross_entropy(logits_nat, y)

        loss_robust = (1.0 / logits_nat.size()[0]) * self.criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1))

        return loss_natural + self.lamb * loss_robust




class ContrastiveMart:

    def __init__(self, lamb, t=0.5):

        self.t = t
        self.criterion_kl = nn.KLDivLoss(reduction='none')
        self.lamb = lamb
    
    def pgd(self, model, cls_inputs, anch_inputs, eps=8. / 255., alpha=2. / 255., iters=10):

        delta = torch.rand_like(anch_inputs) * eps * 2 - eps
        delta = torch.nn.Parameter(delta)
        
        for i in range(iters):
            feat1 = model.eval()(cls_inputs, 'normal').detach()

            feat2 = model.eval()(anch_inputs + delta, 'pgd')

            model.zero_grad()
            
            cls_feat_n = nn.functional.normalize(feat1)
            anch_feat_n = nn.functional.normalize(feat2)
            
            sim_mat = torch.matmul(cls_feat_n, anch_feat_n.t()) / self.t

            logits = sim_mat.t()

            y = torch.arange(logits.size(0)).cuda(non_blocking=True)

            loss = F.cross_entropy(logits, y)

            loss.backward()

            delta.data = delta.data + alpha * delta.grad.sign()
            delta.grad = None
            delta.data = torch.clamp(delta.data, min=-eps, max=eps)
            delta.data = torch.clamp(anch_inputs + delta.data, min=0, max=1) - anch_inputs 

        return (anch_inputs + delta).detach()

    
    def __call__(self, cls_feat, anch_feat, adv_anch_feat):
        cls_feat_n = nn.functional.normalize(cls_feat)
        anch_feat_n = nn.functional.normalize(anch_feat)
        adv_anch_feat_n = nn.functional.normalize(adv_anch_feat)

        sim_mat_nat = torch.matmul(cls_feat_n, anch_feat_n.t()) / self.t
        sim_mat_adv = torch.matmul(cls_feat_n, adv_anch_feat_n.t()) / self.t

        logits_nat = sim_mat_nat.t()
        logits_adv = sim_mat_adv.t()

        y = torch.arange(logits_nat.size(0)).cuda(non_blocking=True)


        adv_probs = F.softmax(logits_adv, dim=1)

        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

        loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

        nat_probs = F.softmax(logits_nat, dim=1)

        true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

        loss_robust = (1.0 / logits_nat.size(0)) * torch.sum(torch.sum(self.criterion_kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

        # print(loss_adv.item())
        # print(loss_robust.item())

        return loss_adv + float(self.lamb) * loss_robust


class ContrastiveHat:

    def __init__(self, beta, gamma, std_model, h=2, t=0.5):

        self.t = t
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.beta = beta
        self.gamma = gamma

        self.std_model = std_model
        self.h = h



    def pgd(self, model, cls_inputs, anch_inputs, eps=8. / 255., alpha=2. / 255., iters=5):
        delta = torch.rand_like(anch_inputs) * eps * 2 - eps
        delta = torch.nn.Parameter(delta)
        
        for i in range(iters):
            cls_feat = model.eval()(cls_inputs, 'normal').detach()
            cln_anch_feat = model.eval()(anch_inputs, 'normal').detach()

            adv_anch_feat = model.eval()(anch_inputs + delta, 'pgd')

            model.zero_grad()
            loss = self.kl(cls_feat, cln_anch_feat, adv_anch_feat)

            loss.backward()
            # print("loss is {}".format(loss))

            delta.data = delta.data + alpha * delta.grad.sign()
            delta.grad = None
            delta.data = torch.clamp(delta.data, min=-eps, max=eps)
            delta.data = torch.clamp(anch_inputs + delta.data, min=0, max=1) - anch_inputs 
        
        adv_inputs = (anch_inputs + delta).detach()

        helper_inputs = (anch_inputs + self.h * delta).detach()

        with torch.no_grad():
            cls_feat = (self.std_model.eval()(cls_inputs, 'normal')).detach()
            adv_feat = (self.std_model.eval()(adv_inputs, 'normal')).detach()

            cls_feat_n = nn.functional.normalize(cls_feat)
            adv_feat_n = nn.functional.normalize(adv_feat)

            helper_y = ((torch.matmul(cls_feat_n, adv_feat_n.t())).argmax(dim=0)).detach()

        return adv_inputs, helper_inputs, helper_y


    def kl(self, cls_feat, anch_feat, adv_anch_feat):
        cls_feat_n = nn.functional.normalize(cls_feat)
        anch_feat_n = nn.functional.normalize(anch_feat)
        adv_anch_feat_n = nn.functional.normalize(adv_anch_feat)

        sim_mat_nat = torch.matmul(cls_feat_n, anch_feat_n.t()) / self.t
        sim_mat_adv = torch.matmul(cls_feat_n, adv_anch_feat_n.t()) / self.t

        logits_nat = sim_mat_nat.t()
        logits_adv = sim_mat_adv.t()

        return self.criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1))

    
    def __call__(self, cls_feat, anch_feat, adv_anch_feat, helper_anch_feat, helper_y):
        cls_feat_n = nn.functional.normalize(cls_feat)
        anch_feat_n = nn.functional.normalize(anch_feat)
        adv_anch_feat_n = nn.functional.normalize(adv_anch_feat)
        helper_anch_feat_n = nn.functional.normalize(helper_anch_feat)

        sim_mat_nat = torch.matmul(cls_feat_n, anch_feat_n.t()) / self.t
        sim_mat_adv = torch.matmul(cls_feat_n, adv_anch_feat_n.t()) / self.t
        sim_mat_helper = torch.matmul(cls_feat_n, helper_anch_feat_n.t()) / self.t

        logits_nat = sim_mat_nat.t()
        logits_adv = sim_mat_adv.t()
        logits_helper = sim_mat_helper.t()

        y = torch.arange(logits_nat.size(0)).cuda(non_blocking=True)

        loss_natural = F.cross_entropy(logits_nat, y)

        loss_robust = (1.0 / logits_nat.size()[0]) * self.criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1))

        loss_help = F.cross_entropy(logits_helper, helper_y)

        return loss_natural + self.beta * loss_robust + self.gamma * loss_help


class ContrastiveHatAt:

    def __init__(self, lamb, std_model, h=2, t=0.5):

        self.t = t
        self.lamb = lamb

        self.std_model = std_model
        self.h = h


    def pgd(self, model, cls_inputs, anch_inputs, eps=8. / 255., alpha=2. / 255., iters=5):
        delta = torch.rand_like(anch_inputs) * eps * 2 - eps
        delta = torch.nn.Parameter(delta)
        
        for i in range(iters):
            feat1 = model.eval()(cls_inputs, 'normal').detach()
            feat2 = model.eval()(anch_inputs + delta, 'pgd')

            model.zero_grad()
            
            cls_feat_n = nn.functional.normalize(feat1)
            anch_feat_n = nn.functional.normalize(feat2)
            
            sim_mat = torch.matmul(cls_feat_n, anch_feat_n.t()) / self.t

            logits = sim_mat.t()

            y = torch.arange(logits.size(0)).cuda(non_blocking=True)

            loss = F.cross_entropy(logits, y)

            loss.backward()

            delta.data = delta.data + alpha * delta.grad.sign()
            delta.grad = None
            delta.data = torch.clamp(delta.data, min=-eps, max=eps)
            delta.data = torch.clamp(anch_inputs + delta.data, min=0, max=1) - anch_inputs 

        adv_inputs = (anch_inputs + delta).detach()

        helper_inputs = (anch_inputs + self.h * delta).detach()

        with torch.no_grad():
            cls_feat = (self.std_model.eval()(cls_inputs, 'normal')).detach()
            adv_feat = (self.std_model.eval()(adv_inputs, 'normal')).detach()

            cls_feat_n = nn.functional.normalize(cls_feat)
            adv_feat_n = nn.functional.normalize(adv_feat)

            helper_y = ((torch.matmul(cls_feat_n, adv_feat_n.t())).argmax(dim=0)).detach()

        return adv_inputs, helper_inputs, helper_y

    
    def __call__(self, cls_feat, adv_anch_feat, helper_anch_feat, helper_y):
        cls_feat_n = nn.functional.normalize(cls_feat)
        adv_anch_feat_n = nn.functional.normalize(adv_anch_feat)
        helper_anch_feat_n = nn.functional.normalize(helper_anch_feat)

        sim_mat_adv = torch.matmul(cls_feat_n, adv_anch_feat_n.t()) / self.t
        sim_mat_helper = torch.matmul(cls_feat_n, helper_anch_feat_n.t()) / self.t

        logits_adv = sim_mat_adv.t()
        logits_helper = sim_mat_helper.t()

        y = torch.arange(logits_adv.size(0)).cuda(non_blocking=True)

        loss_adv = F.cross_entropy(logits_adv, y)

        loss_help = F.cross_entropy(logits_helper, helper_y)

        return loss_adv + self.lamb * loss_help



def cvtPrevious2bnToCurrent2bn(state_dict):
    """
    :param state_dict: old state dict with bn and bn adv
    :return:
    """
    new_state_dict = OrderedDict()
    for name, value in state_dict.items():
        if ('bn1' in name) and ('adv' not in name):
            newName = name.replace('bn1.', 'bn1.bn_list.0.')
        elif ('bn1' in name) and ('adv' in name):
            newName = name.replace('bn1_adv.', 'bn1.bn_list.1.')
        elif ('bn2' in name) and ('adv' not in name):
            newName = name.replace('bn2.', 'bn2.bn_list.0.')
        elif ('bn2' in name) and ('adv' in name):
            newName = name.replace('bn2_adv.', 'bn2.bn_list.1.')
        elif ('bn.' in name):
            newName = name.replace('bn.', 'bn.bn_list.0.')
        elif ('bn_adv.' in name):
            newName = name.replace('bn_adv.', 'bn.bn_list.1.')
        elif 'bn3' in name:
            assert False
        else:
            newName = name

        print("convert name: {} to {}".format(name, newName))
        new_state_dict[newName] = value
    return new_state_dict


class augStrengthScheduler(object):
    """Computes and stores the average and current value"""
    def __init__(self, aug_dif_scheduler_strength_range, aug_dif_scheduler_epoch_range, transGeneFun):
        if ',' in aug_dif_scheduler_strength_range:
            self.aug_dif_scheduler_strength_range = list(map(float, aug_dif_scheduler_strength_range.split(',')))
        else:
            self.aug_dif_scheduler_strength_range = []

        if ',' in aug_dif_scheduler_epoch_range:
            self.aug_dif_scheduler_epoch_range = list(map(int, aug_dif_scheduler_epoch_range.split(',')))
        else:
            self.aug_dif_scheduler_epoch_range = []
        self.transGeneFun = transGeneFun
        self.epoch = 0

        assert (len(self.aug_dif_scheduler_strength_range) == 2 and len(self.aug_dif_scheduler_epoch_range) == 2) or \
               (len(self.aug_dif_scheduler_strength_range) == 0 and len(self.aug_dif_scheduler_epoch_range) == 0)

    def step(self):
        self.epoch += 1

        if len(self.aug_dif_scheduler_strength_range) == 0 and len(self.aug_dif_scheduler_epoch_range) == 0:
            return self.transGeneFun(1.0)
        else:
            startStrength, endStrength = self.aug_dif_scheduler_strength_range
            startEpoch, endEpoch = self.aug_dif_scheduler_epoch_range
            strength = min(max(0, self.epoch - startEpoch), endEpoch - startEpoch) / (endEpoch - startEpoch) * (endStrength - startStrength) + startStrength
            return self.transGeneFun(strength)

# new_state_dict = cvtPrevious2bnToCurrent2bn(checkpoint['state_dict'])
# model.load_state_dict(new_state_dict)
