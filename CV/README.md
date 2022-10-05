# Contrastive Unified Adversarial Training

## Setup
```
pip install -r requirements.txt
```

## Training
To train with unsupervised AT, run the following command:
```
python train.py <experiment name> \
    --contrastive_at \
    --dataset <cifar10 or cifar100>
```

To train with unsupervised TRADES, run the following command:
```
python train.py <experiment name> \
    --contrastive_trades \
    --trades_lamb <lambda hyperparameter> \
    --dataset <cifar10 or cifar100>
```

To train with unsupervised HAT, you need to download the standard model checkpoint from [here](https://drive.google.com/file/d/107LBST4EbfzyAzUaFxGHEOt_xM2JVyEA/view?usp=sharing), then run the following command:
```
python train.py <experiment name> \
    --contrastive_hat \
    --hat_beta <beta hyperparameter> \
    --hat_gamma <gamma hyperparameter> \
    --hat_std_model_path <path to standard model checkpoint> \
    --dataset <cifar10 or cifar100>
```

### Using pseudo-labels
To incorporate the pseudo-labels from AdvCL, add the following arguments to the above commands:
```
--cluster --cluster_lamb <lambda hyperparameter>
```


## Evaluation
To run linear evaluation, use the following command:
```
python linear_evaluation.py <evaluation experiment name> \
    --checkpoint <path to encoder checkpoint> \ 
    --save_path <save path> \
    --bnNameCnt 1 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight-decay 0.0002 \
    --decreasing_lr 15,20 \
    --epochs 25 \
    --batch-size 512 \
    --num-steps-test 20 \
    --epsilon 0.031 \
    --step-size 0.007 \
    --trainmode normal \
    --test-batch-size <batch size for inference>

```

## Checkpoints



|                          | CIFAR10                                                                                    | CIFAR100                                                                                   |
| ------------------------ |:------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Unsupervised AT          | [link](https://drive.google.com/file/d/1DvkGMuxd3T2MPRMee1FS63ALGWaKugAg/view?usp=sharing) | [link](https://drive.google.com/file/d/13y7-U2omadbNQ9YxlQ6QDA8O2tCB4Xsg/view?usp=sharing) |
| Unsupervised AT ($K=3$)  | [link](https://drive.google.com/file/d/1ExMOO-bSQez8_QGgcxKyYZR13w2e7l3O/view?usp=sharing) | [link](https://drive.google.com/file/d/1GyGo9F8d9YTxA7bZPlm3psZR_mpos1Di/view?usp=sharing) |
| Unsupervised AT + PL     | [link](https://drive.google.com/file/d/1GhkZbMQyWacbGCXRuAtB37X7bM4xxbcx/view?usp=sharing) | [link](https://drive.google.com/file/d/1z4OMHQpwwIjAtCNxSFZzpLuOPeVD-Q6k/view?usp=sharing) |
| Unsupervised TRADES      | [link](https://drive.google.com/file/d/1yfwvolvCMZzT6eTsLHUT9t3CZUmUnEu2/view?usp=sharing) | [link](https://drive.google.com/file/d/1jBaRPkc-W5MYysVWZZAsJBqe-L-iEWI5/view?usp=sharing) |
| Unsupervised TRADES + PL | [link](https://drive.google.com/file/d/1wR2dqnRwMUB5OWQZ_AjRWRYtw8LDdr_Q/view?usp=sharing) | [link](https://drive.google.com/file/d/1I2lDPnvqQ1pVjQEtgw1dylFapf4ajTyB/view?usp=sharing) |
| Unsupervised HAT         | [link](https://drive.google.com/file/d/1lg1ci02FDbw89GljptWRlGiaJbN8pqVr/view?usp=sharing) |                                                                                            |

