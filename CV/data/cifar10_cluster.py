import torch
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image

class CustomCIFAR10(CIFAR10):
    def __init__(self, anchTrans=None,
                pseudoLabel_002=None,
                pseudoLabel_010=None,
                pseudoLabel_050=None,
                pseudoLabel_100=None,
                pseudoLabel_500=None, 
                **kwds):

        super().__init__(**kwds)

        self.anchTrans = anchTrans

        self.pseudo_label_002 = pseudoLabel_002
        self.pseudo_label_010 = pseudoLabel_010
        self.pseudo_label_050 = pseudoLabel_050
        self.pseudo_label_100 = pseudoLabel_100
        self.pseudo_label_500 = pseudoLabel_500

    def __getitem__(self, idx):

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        
        cls_imgs = self.transform(img)
        anch_imgs = self.anchTrans(img) 

        if self.pseudo_label_002 is not None:
            label_p_002 = self.pseudo_label_002[idx]
            label_p_010 = self.pseudo_label_010[idx]
            label_p_050 = self.pseudo_label_050[idx]
            label_p_100 = self.pseudo_label_100[idx]
            label_p_500 = self.pseudo_label_500[idx]

            label_p = (label_p_002,
                    label_p_010,
                    label_p_050,
                    label_p_100,
                    label_p_500)
            
            return cls_imgs, anch_imgs, label_p
        
        return cls_imgs, anch_imgs


class CustomCIFAR100(CIFAR100):
    def __init__(self, anchTrans=None, 
                pseudoLabel_002=None,
                pseudoLabel_010=None,
                pseudoLabel_050=None,
                pseudoLabel_100=None,
                pseudoLabel_500=None, 
                **kwds):
        super().__init__(**kwds)

        self.anchTrans = anchTrans

        self.pseudo_label_002 = pseudoLabel_002
        self.pseudo_label_010 = pseudoLabel_010
        self.pseudo_label_050 = pseudoLabel_050
        self.pseudo_label_100 = pseudoLabel_100
        self.pseudo_label_500 = pseudoLabel_500

    def __getitem__(self, idx):

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        
        cls_imgs = self.transform(img) 
        anch_imgs = self.anchTrans(img)


        if self.pseudo_label_002 is not None:
            label_p_002 = self.pseudo_label_002[idx]
            label_p_010 = self.pseudo_label_010[idx]
            label_p_050 = self.pseudo_label_050[idx]
            label_p_100 = self.pseudo_label_100[idx]
            label_p_500 = self.pseudo_label_500[idx]

            label_p = (label_p_002,
                    label_p_010,
                    label_p_050,
                    label_p_100,
                    label_p_500)
                    
            return cls_imgs, anch_imgs, label_p

        return cls_imgs, anch_imgs

