import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import datetime
import os



def get_segmentation(seg_model, estimated_image_for_nnunet, mini_batch=False):
    """
    Get the segmentation logits from reconstructed CT.
    The CT needs recover to HU.
    """
    
    # mini-batch to reduce memory used.
    if mini_batch:
        pred_logits = []
        for i in range(estimated_image_for_nnunet.size(0)):
            mini_slice = estimated_image_for_nnunet[i:i+1].float()
            pred = seg_model(mini_slice)
            pred_logits.append(pred)
        pred_logits = torch.cat(pred_logits, dim=0)
    else:
        pred_logits = seg_model(estimated_image_for_nnunet.float())
    
    return pred_logits


def compute_adaptive_gan_weight(recon_loss, adv_loss, last_layer_weight):
    """
    Compute adaptive weight from Taming Transformers / LDM paper.
    Balances GAN and recon gradients at the decoder's last layer.
    """
    recon_grads = torch.autograd.grad(recon_loss, last_layer_weight, retain_graph=True)[0]
    adv_grads   = torch.autograd.grad(adv_loss,   last_layer_weight, retain_graph=True)[0]

    d_weight = torch.norm(recon_grads) / (torch.norm(adv_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight