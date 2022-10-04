import torch
from utils import *

def PGD_contrastive(model, inputs, eps=8. / 255., alpha=2. / 255., iters=10, singleImg=False, feature_gene=None, sameBN=False):
    # init
    delta = torch.rand_like(inputs) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)

    if singleImg:
        # project half of the delta to be zero
        idx = [i for i in range(1, delta.data.shape[0], 2)]
        delta.data[idx] = torch.clamp(delta.data[idx], min=0, max=0)

    for i in range(iters):
        if feature_gene is None:
            if sameBN:
                features = model.eval()(inputs + delta, 'normal')
            else:
                features = model.eval()(inputs + delta, 'pgd')
        else:
            features = feature_gene(model, inputs + delta, 'eval')

        model.zero_grad()
        loss = nt_xent(features)
        loss.backward()
        # print("loss is {}".format(loss))

        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(inputs + delta.data, min=0, max=1) - inputs

        if singleImg:
            # project half of the delta to be zero
            idx = [i for i in range(1, delta.data.shape[0], 2)]
            delta.data[idx] = torch.clamp(delta.data[idx], min=0, max=0)

    return (inputs + delta).detach()



def pgd_contrastive_at(model, cls_inputs, anch_inputs, criterion, eps=8. / 255., alpha=2. / 255., iters=10):

    delta = torch.rand_like(anch_inputs) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)
    
    for i in range(iters):
        feat1 = model.eval()(cls_inputs, 'normal').detach()

        feat2 = model.eval()(anch_inputs + delta, 'pgd')

        model.zero_grad()
        
        loss = criterion(feat1, feat2)

        loss.backward()

        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(anch_inputs + delta.data, min=0, max=1) - anch_inputs 

    return (anch_inputs + delta).detach()


def pgd_contrastive_at_cluster(model, cls_inputs, anch_inputs, criterion, targets, cluster_lamb, eps=8. / 255., alpha=2. / 255., iters=10):
    

    delta = torch.rand_like(anch_inputs) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)

    cluster_delta = torch.rand_like(anch_inputs) * eps * 2 - eps
    cluster_delta = torch.nn.Parameter(cluster_delta)
    
    for i in range(iters):
        feat1 = model.eval()(cls_inputs, 'normal').detach()

        feat2 = model.eval()(anch_inputs + delta, 'pgd')

        cluster_feats = model.eval()(anch_inputs + cluster_delta, 'pgd_cluster', cluster=True)

        loss_cl = criterion(feat1, feat2)

        loss_ce = 0
        for label_idx in range(5):
            tgt = targets[label_idx].long()
            lgt = cluster_feats[label_idx]
            loss_ce += F.cross_entropy(lgt, tgt, size_average=False, ignore_index=-1) / 5.
        
        loss = loss_cl + loss_ce * cluster_lamb

        model.zero_grad()
        loss.backward()

        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(anch_inputs + delta.data, min=0, max=1) - anch_inputs 

        cluster_delta.data = cluster_delta.data + alpha * cluster_delta.grad.sign()
        cluster_delta.grad = None
        cluster_delta.data = torch.clamp(cluster_delta.data, min=-eps, max=eps)
        cluster_delta.data = torch.clamp(anch_inputs + cluster_delta.data, min=0, max=1) - anch_inputs 

    return (anch_inputs + delta).detach(), (anch_inputs + cluster_delta).detach()


def cluster_attack(model, anch_inputs, targets, cluster_lamb=1, eps=8. / 255., alpha=2. / 255., iters=10, ori_att=False):

    cluster_delta = torch.rand_like(anch_inputs) * eps * 2 - eps
    cluster_delta = torch.nn.Parameter(cluster_delta)

    for i in range(iters):
        if ori_att:
            cluster_feats = model.train()(anch_inputs + cluster_delta, bn_name='pgd_cluster', cluster=True)
        else:
            cluster_feats = model.eval()(anch_inputs + cluster_delta, bn_name='pgd_cluster', cluster=True)
        
        loss_ce = 0
        for label_idx in range(5):
            tgt = targets[label_idx].long()
            lgt = cluster_feats[label_idx]
            loss_ce += F.cross_entropy(lgt, tgt, size_average=False, ignore_index=-1) / 5.
        
        loss = loss_ce * cluster_lamb

        model.zero_grad()
        loss.backward()

        cluster_delta.data = cluster_delta.data + alpha * cluster_delta.grad.sign()

        cluster_delta.grad = None
        cluster_delta.data = torch.clamp(cluster_delta.data, min=-eps, max=eps)
        cluster_delta.data = torch.clamp(anch_inputs + cluster_delta.data, min=0, max=1) - anch_inputs 
    
    return (anch_inputs + cluster_delta).detach()


def pgd_contrastive_trades(model, inputs, eps=8. / 255., alpha=2. / 255., iters=10, sameBN=False):
    delta = torch.rand_like(inputs) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)
    
    for i in range(iters):
        feat1 = model.eval()(inputs, 'normal').detach()
        if sameBN:
            feat2 = model.eval()(inputs + delta, 'normal')
        else:
            feat2 = model.eval()(inputs + delta, 'pgd')

        model.zero_grad()
        loss = contrastive_trades_kl(feat1, feat2)
        loss.backward()
        # print("loss is {}".format(loss))

        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(inputs + delta.data, min=0, max=1) - inputs 

    return (inputs + delta).detach()


def cluster_attack_ori(model, images_org, targets, num_steps, epsilon=8. / 255, step_size=2. / 255):

    x_ce = images_org.clone().detach()

    noise = torch.zeros_like(x_ce).uniform_(-epsilon, epsilon)

    x_ce = x_ce + noise

    for i in range(num_steps):
        x_ce.requires_grad_()
        with torch.enable_grad():
            logits_ce = model.eval()(x_ce, bn_name='pgd_cluster', cluster=True)

            loss_ce = 0
            for label_idx in range(5):
                tgt = targets[label_idx].long()
                lgt = logits_ce[label_idx]
                loss_ce += F.cross_entropy(lgt, tgt, size_average=False, ignore_index=-1) / 5.

            loss = loss_ce 

        grad_x_ce = torch.autograd.grad(loss, [x_ce])[0]

        x_ce = x_ce.detach() + step_size * torch.sign(grad_x_ce.detach())

        x_ce = torch.min(torch.max(x_ce, images_org - epsilon), images_org + epsilon)
        x_ce = torch.clamp(x_ce, 0, 1)
    
    return x_ce.detach()



