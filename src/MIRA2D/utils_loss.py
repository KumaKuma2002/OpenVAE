import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from utils import center_crop
import datetime
import os
import json
import lpips

"""
Load the record module
"""
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"../train_log/loss/loss_error_{timestamp}.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
def loss_log(msg):
        with open(log_path, "a") as f:
            f.write(msg + "\n")

"""
Load the dataset-non-aware module
"""
def load_label_map(dataset_json_path):
    """
    Returns:
        label_name_to_id: dict[str, int]
        label_id_to_name: dict[int, str]
    """
    with open(dataset_json_path, "r") as f:
        data = json.load(f)

    label_name_to_id = data["labels"]
    label_id_to_name = {v: k for k, v in label_name_to_id.items()}

    return label_name_to_id, label_id_to_name
label_name_to_id, label_id_to_name = load_label_map(
    "/projects/bodymaps/jliu452/nnUNet/Dataset_results/Dataset911/nnUNetTrainer__nnUNetPlans__2d/dataset.json"
    )

 
# clinical importance, NOT dataset-dependent
DEFAULT_ORGAN_PENALTIES = {
    "aorta": 100.0,
    "kidney_left": 100.0,
    "kidney_right": 100.0,
    "liver": 100.0,
    "pancreas": 100.0,
    "stomach": 10.0,
    "spleen": 10.0,
    "pancreatic_lesion": 500.0, # high-value target 
}


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
    

def get_classification(phase_classifier, estimated_image, repeat_channel=False):
    b, c, h, w = estimated_image.shape
    
    # NOTE: classification model, In-channel=3
    if repeat_channel:# timm requires processed
        classify_logits = phase_classifier(estimated_image.float().reshape(b*c, 1, h, w).repeat(1, 3, 1, 1)) 
    else:
        estimated_image_01 = (estimated_image + 1000.0) /2000.0
        classify_logits = phase_classifier(estimated_image_01.float()) #MONAI, requires (0, 1)
    
    
    return classify_logits


def unchanged_region_loss(batch, estimated_image, HU_threshold=800):
    """
    Pixel-wise loss on the unchanged region (air and bone) between cond image and translated image.
    estimated_image shall be in HU domain. [-1000, 1000]
    """
    unchanged_mask = batch["unchanged_mask"].to(estimated_image.device)  # 1 where unchanged, 0 where changed
    # estimated_image_unchanged_mask = ((estimated_image < -HU_threshold) | (estimated_image > HU_threshold))
    estimated_image_unchanged_mask = ((estimated_image < -HU_threshold))
    mask_loss = F.mse_loss(unchanged_mask.float(), estimated_image_unchanged_mask.float())
    
    return mask_loss


def organ_mean_hu(x, mask, organ_id):
    """
    Organ-volume averaged HU value.
    x is preferablly in [0, 1] domain
    """

    organ_mask = (mask == organ_id)
    organ_area = organ_mask.sum().clamp(min=1.0)
    
    organ_mask = organ_mask.reshape(x.shape)
    return (x * organ_mask.float()).sum() / organ_area


def HU_avg_loss(
    vae,
    batch,
    estimated_image,
    pred_logits=None,
    *,
    label_id_to_name,
    organ_penalties=DEFAULT_ORGAN_PENALTIES,
    use_cond_gt=False,
):
    """
    Organ-wise average HU MSE loss with semantic penalties.
    Pixel-level alignment NOT required.
    """

    device = vae.device
    b, c, h, w = pred_logits.shape

    # ground-truth image & mask
    pixel_values = batch["pixel_values"].to(device)          # b, 3, 512, 512
    gt_mask = batch["mask_values"].to(device).long().reshape(-1, 3, h, w)

    # predicted / translated image → [0,1]
    translated_values = ((estimated_image + 1000.0) / 2000.0).squeeze(1)

    # predicted mask
    """
    Strong prior supervision
        avoid the model learns to generate nothing, so the mask is "void" and therefore loss to zero
    """
    pred_mask = gt_mask.clone()

    batch_losses = []

    for i in range(pixel_values.size(0)):
        gt_mask_i = gt_mask[i]
        pred_mask_i = pred_mask[i]
        gt_img_i = pixel_values[i]
        pred_img_i = translated_values[i]

        # valid overlapping organs
        gt_ids = torch.unique(gt_mask_i)
        gt_ids = gt_ids[gt_ids != 0]

        pred_ids = torch.unique(pred_mask_i)
        overlap_ids = [oid.item() for oid in gt_ids if oid in pred_ids]

        if len(overlap_ids) < 2:
            continue

        organ_losses = []

        for organ_id in overlap_ids:
            organ_name = label_id_to_name.get(organ_id, None)
            if organ_name is None:
                continue

            gt_hu = organ_mean_hu(gt_img_i, gt_mask_i, organ_id)
            pred_hu = organ_mean_hu(pred_img_i, pred_mask_i, organ_id)

            if gt_hu is None or pred_hu is None:
                continue

            mse = (pred_hu - gt_hu) ** 2
            weight = organ_penalties.get(organ_name, 1.0)

            organ_losses.append(weight * mse)

        if organ_losses:
            batch_losses.append(torch.mean(torch.stack(organ_losses)))

    if batch_losses:
        return torch.mean(torch.stack(batch_losses))
    else:
        return torch.tensor(0.0, device=device)



def segmentation_loss(batch, pred_logits, estimated_image, use_gt_mask=False):  
    """
    Via the MONAI-DiceCELoss.

    estimated_image: (B, C, H, W), shall be in HU domain.
    """
    b, c, h, w = estimated_image.shape
    N = b * c

    if use_gt_mask:
        cond_mask = batch["cond_mask_values"].reshape(N, h, w).long()
    else:
        cond_mask = batch["mask_values"] 
        # pred by segmenter from high resolution images
        if cond_mask.ndim == 4 and cond_mask.shape[1] == 1:
            cond_mask = cond_mask.squeeze(1)
        cond_mask = cond_mask.long()
        
    seg_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="mean") # still CE
    eff_seg_loss = seg_loss_fn(pred_logits, cond_mask) 
    # add safecheck
    if torch.isnan(eff_seg_loss) or torch.isinf(eff_seg_loss):
        eff_seg_loss = torch.tensor(0.0, device=estimated_image.device)
    return eff_seg_loss



def classification_loss(classify_logits, gt_phase, repeat_channel=False):
    """phase-classfication CE loss"""
    
    if repeat_channel:# for old timm model version
        gt_phase = gt_phase.repeat(3)
    cls_loss_per = F.cross_entropy(classify_logits, gt_phase)
    eff_cls_loss = (cls_loss_per).mean()
    
    return eff_cls_loss


def cycle_mse_loss(batch, estimated_image, cycle_estimated_image):
    """
        Cycle Image MSE loss, pixel-wise
        
        estimated_image shall be in HU domain.
    """

    cycle_estimated_image_01 = ((cycle_estimated_image + 1000.0)/2000.0).to(estimated_image.device)# to [0, 1] range
    cycle_origin_pixel_values_01 = batch["cond_pixel_values_original"].to(estimated_image.device)# translation source
    cycle_loss = F.mse_loss(cycle_estimated_image_01, cycle_origin_pixel_values_01)
    return cycle_loss


def strong_supervision_loss(batch, estimated_image, loss_choice="l1", lpips_weight=1e-1, mask_area=True):
    """
    strong per-pixel loss, for PCCT training
    
    ⚠️ If enable the `mask_area` option, the loss will be only calculated on the segmentation mask from ground truth
    """
    
    estimated_image_01 = ((estimated_image.clamp(-1000, 1000) + 1000.0)/2000.0).to(estimated_image.device)# same as load_slice_h5 module
    target_image_01 = batch["input_pixel_values_original"].to(estimated_image.device)

    if mask_area:
        mask = batch["mask_values"].to(estimated_image.device)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = (mask != 0).float().reshape(estimated_image_01.shape)
        
        estimated_image_01 = estimated_image_01 * mask
        target_image_01 = target_image_01 * mask

    if loss_choice == "l1":
        pixel_loss =  F.l1_loss(estimated_image_01, target_image_01)
    elif loss_choice == "l2":
        pixel_loss = F.mse_loss(estimated_image_01, target_image_01)
    elif loss_choice == "huber":
        pixel_loss = F.smooth_l1_loss(estimated_image_01, target_image_01)

    else:
        raise ValueError(f"Unknown loss_choice: {loss_choice}")
    
    loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False).to(estimated_image.device)
    lpips_loss = loss_fn_vgg(estimated_image_01, target_image_01).mean()
    
    return pixel_loss + lpips_weight*lpips_loss




#========================== Uncertainty Weighted Loss Module ==========================# 
class UncertaintyWeightedLoss(torch.nn.Module):
    """
    Uncertainty Weighted Multi-Task Loss, Kendall et al., 2018.
    https://arxiv.org/abs/1705.07115
    """
    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = torch.nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):  # list of scalar losses
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss





#========================== General Step Loss Functions ==========================#

def step3_loss(batch, estimated_image, cycle_estimated_image, classify_logits, diffusion_loss, uc_area_lambda=1, cls_lambda=1e-2, cycle_lambda=1):
    """
    Loss function for step3, where the phase loss is added.
    Cycle process is also deployed to ensure structure consistancy.
    """

    b, c, h, w = estimated_image.shape
    unchanged_loss = unchanged_region_loss(batch, estimated_image)
    cls_loss = classification_loss(classify_logits, batch["gt_phase_id"].to(classify_logits.device))
    cycle_loss = cycle_mse_loss(batch, estimated_image, cycle_estimated_image)
    # print(f"UC area loss: {unchanged_loss.item()}")
    return diffusion_loss + uc_area_lambda*unchanged_loss + cls_lambda*cls_loss + cycle_lambda*cycle_loss 


def step4_loss(vae, batch, estimated_image, classify_logits, pred_logits, diffusion_loss, uc_area_lambda=1, cls_lambda=1e-2, seg_dsc_lambda=1e-2, hu_mse_lambda=1, if_use_gt=False):
    """
    Controling the segmentation structure correctness and translated HU correlation.
    Formula:
        Loss = diffusion_loss + segmentation_loss + classification_loss + hu_average_loss
    """
    b, c, h, w = estimated_image.shape
    unchanged_loss = unchanged_region_loss(batch, estimated_image)
    cls_loss = classification_loss(classify_logits, batch["gt_phase_id"].to(classify_logits.device))
    seg_loss = segmentation_loss(batch, pred_logits, estimated_image, use_gt_mask=if_use_gt)
    hu_avg_loss = HU_avg_loss(vae, batch, estimated_image, pred_logits, label_id_to_name=label_id_to_name, use_cond_gt=if_use_gt)
    loss_log(f"Step4 Losses: Diffusion {diffusion_loss.item():.4f}, Cls {cls_loss.item():.4f}, Seg {seg_loss.item():.4f}, UC area: {unchanged_loss.item():.4f}, HU {hu_avg_loss.item():.4f}")
    
    return diffusion_loss + uc_area_lambda*unchanged_loss + cls_lambda*cls_loss + seg_dsc_lambda*seg_loss + hu_mse_lambda*hu_avg_loss


def step5_loss(vae, batch, estimated_image, cycle_estimated_image, classify_logits, pred_logits, diffusion_loss, auto_adjust=False, uc_area_lambda=1, cls_lambda=1e-2, seg_dsc_lambda=1, hu_mse_lambda=1, cycle_lambda=100, if_use_gt=False):
    """
    Add Cycle Diffusion MSE Loss
    """
    fore_loss = step4_loss(vae, batch, estimated_image, classify_logits, pred_logits, diffusion_loss, uc_area_lambda, cls_lambda, seg_dsc_lambda, hu_mse_lambda)
    cycle_loss = cycle_mse_loss(batch, estimated_image, cycle_estimated_image)
    return fore_loss + cycle_lambda * cycle_loss


def step6_loss(
    vae, 
    batch, 
    estimated_image, 
    cycle_estimated_image, 
    classify_logits, pred_logits, 
    diffusion_loss, 
    uncertainty_loss_module=None, 
    uc_area_lambda=1, cls_lambda=1e-2, seg_dsc_lambda=1, hu_mse_lambda=100, cycle_lambda=100,
    if_use_gt=False):
    """
    Add learnable parameters, to punish more on more obvious loss.
    """
    b, c, h, w = estimated_image.shape
    unchanged_loss = unchanged_region_loss(batch, estimated_image)
    cls_loss = classification_loss(classify_logits, batch["gt_phase_id"].to(classify_logits.device))
    seg_loss = segmentation_loss(batch, pred_logits, estimated_image, use_gt_mask=if_use_gt)
    hu_avg_loss = HU_avg_loss(vae, batch, estimated_image, pred_logits, label_id_to_name=label_id_to_name, use_cond_gt=if_use_gt)
    cycle_loss = cycle_mse_loss(batch, estimated_image, cycle_estimated_image)
    
    if uncertainty_loss_module is not None:
        # auto-adjust the classification, segmentation and HU losses
        composite_loss = uncertainty_loss_module([cls_loss, seg_loss, hu_avg_loss])
        total_loss = diffusion_loss + uc_area_lambda*unchanged_loss + cls_lambda * composite_loss + cycle_lambda * cycle_loss

        return total_loss
    
    else:
        total_loss = (
            diffusion_lambda * diffusion_loss +
            uc_area_lambda * unchanged_loss +
            cls_lambda * cls_loss +
            seg_dsc_lambda * seg_loss +
            hu_mse_lambda * hu_avg_loss +
            cycle_lambda * cycle_loss
        )

    return total_loss





