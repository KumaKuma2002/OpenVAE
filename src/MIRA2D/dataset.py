import torch
import os
import numpy as np
import nibabel as nib
import albumentations as A
import cv2
from torch.utils.data import Dataset, DataLoader
import random
import time
from tqdm import tqdm
import pandas as pd
import h5py
from collections import defaultdict
import datetime

from utils import find_GT_BDMAP
from utils_degrade import degrade_sparse_view


import matplotlib.pyplot as plt


# NOTE IMPORTANT!! 


"""
Dataset script for super-resolution project


Low-resolution:
    degrade methods:
        -- low-resolution (randomly from 256 to 400)
        -- noise
            -- gaussian and poisson noise
            -- directional noise (radiational noise from CT scan)
        -- metal artifacts
"""
def degrade_ct(
    img,
    target_size_range=(200, 500),
    gaussian_std_range=(0.0, 0.2),
    poisson_scale=1e5,
    n_views_range=(80, 200),
    directional_noise_prob=0.1,
    sparse_view_prob=0.5,
    debug_visualization=False,
    ):
    """
    img: np.ndarray (H, W, C), range [0, 1]
    returns: np.ndarray (H, W, C)
    """
    img = img.astype(np.float32, copy=False)
    H, W, C = img.shape
    out = np.empty_like(img)

    # low-resolution
    target_size = np.random.randint(*target_size_range)
    scale = target_size / max(H, W)
    th, tw = int(H * scale), int(W * scale)

    for c in range(C):
        low = cv2.resize(
            img[..., c], (tw, th), interpolation=cv2.INTER_AREA
        )
        out[..., c] = cv2.resize(
            low, (W, H), interpolation=cv2.INTER_LINEAR
        )

    # noise (Poisson + Gaussian)
    lam = np.clip(out, 1e-6, 1.0) * poisson_scale
    out = np.random.poisson(lam).astype(np.float32) / poisson_scale

    sigma = np.random.uniform(*gaussian_std_range)
    out += np.random.normal(0, sigma, out.shape).astype(np.float32)

    # directional noise
    if np.random.rand() < directional_noise_prob:
        mask = np.zeros((H, W), np.float32)
        angle = np.random.uniform(0, 180)
        length = np.random.randint(H // 4, H // 2)
        thickness = np.random.randint(1, 3)
        cx, cy = np.random.randint(0, W), np.random.randint(0, H)

        dx = int(length * np.cos(np.deg2rad(angle)))
        dy = int(length * np.sin(np.deg2rad(angle)))

        cv2.line(mask, (cx - dx, cy - dy), (cx + dx, cy + dy), 1.0, thickness)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=6, sigmaY=6)
        mask /= (mask.max() + 1e-6)

        out += mask[..., None] * np.random.uniform(0.05, 0.15)
    
    # sparse 2D downgrade
    if np.random.rand() < sparse_view_prob:
        n_views = np.random.randint(n_views_range[0], n_views_range[1])
        for i in range(out.shape[-1]):
            out[:,:,i] = degrade_sparse_view(out[:,:,i], n_views)


    """
    WARNING: only enable in debug mode
    """
    if debug_visualization:
        mid = C // 2
        orig = img[..., mid]
        deg = out[..., mid]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(np.rot90(orig), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(np.rot90(deg), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Degraded")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig("downgraded_debug.png", dpi=150)
        plt.close()

    return np.clip(out, 0.0, 1.0)




def collate_fn(examples):

        valid_examples = [
                ex for ex in examples
                if ex is not None and isinstance(ex, dict) and ex.get("pixel_values") is not None
            ]
        if len(valid_examples) == 0:
            return None 
        examples = valid_examples
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
        cond_mask_values = torch.stack([example["cond_mask_values"] for example in examples])
        cond_mask_values = cond_mask_values.to(memory_format=torch.contiguous_format).float()
        gt_mask_values = torch.stack([example["mask_values"] for example in examples])
        gt_mask_values = gt_mask_values.to(memory_format=torch.contiguous_format).float()
        gt_phase_id = torch.stack([example["gt_phase_id"] for example in examples])
        gt_phase_id = gt_phase_id.to(memory_format=torch.contiguous_format).long()  
        cond_phase_id = torch.stack([example["cond_phase_id"] for example in examples])
        cond_phase_id = cond_phase_id.to(memory_format=torch.contiguous_format).long()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        cond_ids = torch.stack([example["cond_ids"] for example in examples])
        input_pixel_values_orginal = torch.stack([example["input_pixel_values_original"] for example in examples])
        input_pixel_values_orginal = input_pixel_values_orginal.to(memory_format=torch.contiguous_format).float()
        cond_pixel_values_orginal = torch.stack([example["cond_pixel_values_original"] for example in examples])
        cond_pixel_values_orginal = cond_pixel_values_orginal.to(memory_format=torch.contiguous_format).float()
        unchanged_mask = torch.stack([example["unchanged_mask"] for example in examples])
        unchanged_mask = unchanged_mask.to(memory_format=torch.contiguous_format)

        return {
            "pixel_values": pixel_values, 
            "mask_values": gt_mask_values,
            "input_ids": input_ids, 
            "cond_ids": cond_ids,   
            "cond_pixel_values": cond_pixel_values, 
            "cond_mask_values": cond_mask_values,
            "gt_phase_id": gt_phase_id,
            "cond_phase_id": cond_phase_id,
            "input_pixel_values_original": input_pixel_values_orginal,
            "cond_pixel_values_original": cond_pixel_values_orginal,
            "unchanged_mask": unchanged_mask
        }


def collate_fn_inference(examples):
        # pixel_values = torch.stack([example["pixel_values"] for example in examples])
        # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
        input_prompt = [example["input_prompt"] for example in examples]
        slice_idx = [example["slice_idx"] for example in examples]
        return {
            # "pixel_values": pixel_values, 
            "input_prompt": input_prompt,   # NOTE: different from training
            "cond_pixel_values": cond_pixel_values,
            # "gt_pixel_values": gt_pixel_values,
            "slice_idx": slice_idx
        }



def varifyh5(filename): # read the h5 file to see if the conversion is finished or not
    try:
        with h5py.File(filename, "r") as hf:   # can read successfully
            pass
        return True
    except OSError:     # transform not complete
        return False




def load_CT_sliceh5(ct_path, slice_idx=None, is_mask=False):
    """
    For our training data: ranging from [-1000, 1000], shape of (H W D) 

    Loading three adjacent slices as three channels, for diffusion model structure consistency.
    """
    with h5py.File(ct_path, 'r') as hf:
        nii = hf['image']
        z_shape = nii.shape[2]

        # NOTE: take adjacent 3 slices into the 3 RGB channel
        if slice_idx is None:
            slice_idx = random.randint(0, z_shape - 3)   # `random.randint` includes end point

        ct_slice = nii[:, :, slice_idx:slice_idx + 3]   

    if not is_mask:
        # target range: [-1000, 1000] -> [-1, 1]
        ct_slice[ct_slice > 1000.] = 1000.          # clipping range and normalize
        ct_slice[ct_slice < -1000.] = -1000.
        ct_slice = (ct_slice + 1000.) / 2000.       # [-1000, 1000] --> [0, 1]
    
    if ct_slice.shape[2]!= 3:
        raise ValueError

    return ct_slice  # (H W 3)[0, 1]

def load_CT_sliceniigz(ct_data, slice_idx=None):
    """
        For any nii data during inference: ranging from [-1000, 1000], shape of (H W D) 
    """
    ct_slice = ct_data[:, :, slice_idx:slice_idx + 3].copy() 
            
    # target range: [-1000, 1000] -> [-1, 1]
    ct_slice[ct_slice > 1000.] = 1000.   
    ct_slice[ct_slice < -1000.] = -1000.
    ct_slice = (ct_slice + 1000.) / 2000.     

    if ct_slice.shape[-1] != 3:
        ct_slice = ct_slice[:,:,3]
    return ct_slice 



def load_CT_slice_from_h5py(nii, slice_idx=None):
    """
    For inference:

        AbdomenAtlasPro data: ranging from [-1000, 1000], shape of (H W D) 
    """
    
    z_shape = nii.shape[2]

    # NOTE: take adjacent 3 slices into the 3 RGB channel
    if slice_idx is None:
        slice_idx = random.randint(0, z_shape - 3)   # `random.randint` includes end point

    end_idx = slice_idx + 3
    if end_idx > z_shape:
        # use the last one for padding
        slices = [nii[:, :, i] for i in range(slice_idx, z_shape)]
        while len(slices) < 3:
            slices.append(slices[-1])  
        ct_slice = np.stack(slices, axis=-1)
    else:
        ct_slice = nii[:, :, slice_idx:end_idx]

    # target range: [-1000, 1000] -> [-1, 1]
    ct_slice[ct_slice > 1000.] = 1000.    # clipping range and normalize
    ct_slice[ct_slice < -1000.] = -1000.
    ct_slice = (ct_slice + 1000.) / 2000.       # [-1000, 1000] --> [0, 1]
    return ct_slice  # (H W 3)[0, 1]


class HWCarrayToCHWtensor(A.ImageOnlyTransform):
    """Converts (H, W, C) NumPy array to (C, H, W) PyTorch tensor."""
    def apply(self, img, **kwargs):
        return torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) → (C, H, W)




class CTSuperResolutionDataset(Dataset):
    """
    CT Super-Resolution dataset
    
    """

    def __init__(self, file_paths, data_root, image_transforms=None, tokenizer=None, resolution=512, ignore_no_label_area=True,
    ):
        self.data_root = data_root
        self.file_paths = file_paths
        self.image_transforms = image_transforms
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.ignore_no_label_area = ignore_no_label_area

        self.norm = A.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            max_pixel_value=1.0,
            p=1.0
        )

        self.phases_id_mapping = {
            "non-contrast": 0,
            "arterial": 1,
            "venous": 2,
            "delayed": 3,
        }

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = f"../train_log/sr_dataset_error_{timestamp}.log"
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True) if os.path.dirname(self.log_path) else None

    def log(self, msg):
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        ct_dir = self.file_paths[idx]
        ct_path = os.path.join(ct_dir, "ct.h5")
        gt_path = os.path.join(ct_dir, "gt.h5")

        try:
            with h5py.File(ct_path, "r") as hf:
                z_dim = hf["image"].shape[2]

            rel_pos = random.uniform(0.05, 0.95)
            slice_idx = int(rel_pos * (z_dim - 1))

            ct_hr = load_CT_sliceh5(ct_path, slice_idx=slice_idx)
            gt_slice = load_CT_sliceh5(gt_path, slice_idx=slice_idx, is_mask=True)

            if self.ignore_no_label_area and len(np.unique(gt_slice)) < 2:
                self.log(f"[No Label] {ct_path}, slice={slice_idx}")
                return None

        except Exception as e:
            self.log(f"[Bad Slice] {ct_path}, slice={slice_idx}, err={repr(e)}")
            return None

        # resolution check
        if ct_hr.shape[:2] != (self.resolution, self.resolution):
            ct_hr = cv2.resize(
                ct_hr, 
                (self.resolution, self.resolution), 
                interpolation=cv2.INTER_LINEAR
            )
            gt_slice = cv2.resize(
                gt_slice.astype("uint8"),
                (self.resolution, self.resolution),
                interpolation=cv2.INTER_NEAREST,
            )

        ct_lr = degrade_ct(ct_hr)

        # unchanged mask (optional, but kept for compatibility)
        ct_hr_hu = ct_hr * 2000.0 - 1000.0
        unchanged_mask = (ct_hr_hu < -800).astype(np.uint8)

        if self.image_transforms:
            try:
                transformed = self.image_transforms(
                    image=ct_hr,
                    cond=ct_lr,
                    mask=gt_slice,
                    cond_mask=gt_slice,
                )
                ct_hr = transformed["image"]
                ct_lr = transformed["cond"]
                gt_slice = transformed["mask"]
            except Exception as e:
                self.log(f"[Bad Transform] {ct_path}, slice={slice_idx}, err={repr(e)}")
                return None

        ct_hr = HWCarrayToCHWtensor(p=1.)(
            image=self.norm(image=ct_hr)["image"]
        )["image"]

        ct_lr = HWCarrayToCHWtensor(p=1.)(
            image=self.norm(image=ct_lr)["image"]
        )["image"]

        gt_slice = HWCarrayToCHWtensor(p=1.)(image=gt_slice)["image"]
        unchanged_mask = HWCarrayToCHWtensor(p=1.)(image=unchanged_mask)["image"]

        text_prompt = "A high resolution CT slice."
        source_prompt = "A low resolution CT slice."

        example = dict()
        example["mask_values"] = gt_slice
        example["pixel_values"] = ct_hr                  # target (HR)
        example["cond_pixel_values"] = ct_lr              # condition (LR)
        example["cond_mask_values"] = example["mask_values"]
        example["input_ids"] = self.tokenize_caption(text_prompt)
        example["cond_ids"] = self.tokenize_caption(source_prompt)
        example["gt_phase_id"] = torch.tensor(0).long()
        example["cond_phase_id"] = torch.tensor(0).long()
        example["input_pixel_values_original"] = ct_hr
        example["cond_pixel_values_original"] = ct_lr
        example["unchanged_mask"] = unchanged_mask

        return example

    def tokenize_caption(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return inputs.input_ids
    
"""
Inference Dataloaders
"""
class CTDatasetInference(Dataset):    # for a single CT volume
    """
    General inference dataloader, for .nii.gz form data
    
    """
    def __init__(self, file_path, image_transforms=None, cond_transforms=None):
        """ (inference on CT volume only)
        Args:
            file_path (string): The CT volume to inference (.nii.gz).
            transform (albumentations.Compose): Transformations to apply to 2D slices. 
        """

        # read CT volume data
        self.file_path = file_path
        self.bdmap_id = self.file_path.split("/")[-2]
        self.ct_volume_nii = nib.load(self.file_path)

        vol = self.ct_volume_nii.get_fdata()
        """
        CT intensity value standardlization

        for preprocessed lung CT, which values falls between [0, 1]
        """
        if vol.min() >= 0.0 and vol.max() <= 1.0:
            vol = vol * 2000.0 - 1000.0
        print(vol.min(), vol.max())
        self.ct_volume = vol
        self.ct_xyz_shape = self.ct_volume_nii.shape   # (H W D)
        self.ct_z_shape = self.ct_xyz_shape[2]
        
        # normalization
        self.norm_to_zero_centered = A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=1.0,
                p=1.0
            )

        ### Deprecated
        self.image_transforms = image_transforms


    def __len__(self):
        return self.ct_z_shape-3 + 1   # 3 adjacent clices as input unit

    def __getitem__(self, slice_idx): # slice_idx will always in order by setting `shuffle=False`

        cond_ct_slice_raw = load_CT_sliceniigz(self.ct_volume, slice_idx)     # [0, 1]
        cond_ct_slice = self.image_transforms(image=cond_ct_slice_raw)["image"]

        if cond_ct_slice.shape[2] !=3:# border protection
            return None

        cond_ct_slice = HWCarrayToCHWtensor(p=1.)(
            image=self.norm_to_zero_centered(
                image=cond_ct_slice)["image"]
                )["image"] # array to tensor    [0, 1] -> ~[-1, 1]
        
        text_prompt = f""   # default
        example = dict()
        example["cond_pixel_values"] = cond_ct_slice
        example["input_prompt"] = text_prompt
        example["slice_idx"] = slice_idx    

        return example  # Shape: (C, H, W)





# class CTDatasetInferenceH5(Dataset):    # for a single CT volume
#     """
#     The inference container for .H5 dataset form
    
#     """
#     def __init__(self, file_path, image_transforms=None, cond_transforms=None):
#         """ (inference on CT volume only)
#         Args:
#             file_path (string): The CT volume to inference (.nii.gz).
#             transform (albumentations.Compose): Transformations to apply to 2D slices. 
#         """
#         # read CT volume data
#         self.file_path = file_path
#         self.bdmap_id = self.file_path.split("/")[-2]
#         self.ct_volume_nii = h5py.File(self.file_path, 'r')['image']  # directly read the 'image' dataset
#         # self.ct_volume_data = self.ct_volume_nii.get_fdata()
#         self.ct_xyz_shape = self.ct_volume_nii.shape   # (H W D)
#         self.ct_z_shape = self.ct_xyz_shape[2]
        
#         # normalization
#         self.norm_to_zero_centered = A.Normalize(
#                 mean=(0.5, 0.5, 0.5),
#                 std=(0.5, 0.5, 0.5),
#                 max_pixel_value=1.0,
#                 p=1.0
#             )

#         ### Deprecated
#         self.image_transforms = image_transforms
#         # self.phases = __all_phases__

#     def __len__(self):
#         return self.ct_z_shape   # 3 adjacent clices as input unit

#     def __getitem__(self, slice_idx, phase='arterial'): # slice_idx will always in order by setting `shuffle=False`
#         cond_ct_slice_raw = load_CT_slice_from_h5py(self.ct_volume_nii, slice_idx)     # [0, 1]
#         cond_ct_slice = self.image_transforms(image=cond_ct_slice_raw)["image"]

#         cond_ct_slice = HWCarrayToCHWtensor(p=1.)(
#             image=self.norm_to_zero_centered(
#                 image=cond_ct_slice)["image"]
#                 )["image"] # array to tensor    [0, 1] -> ~[-1, 1]
        
#         text_prompt = f""   # default

#         example = dict()
#         example["cond_pixel_values"] = cond_ct_slice
#         example["input_prompt"] = text_prompt
#         example["slice_idx"] = slice_idx    # haha.

#         return example  # Shape: (C, H, W)





if __name__ == "__main__":


    train_data_dir = "/projects/bodymaps/jliu452/Data/Dataset803_SMILE_PCCT/h5"
    paths = sorted([entry.path for entry in os.scandir(train_data_dir)])
    paths = [entry.path.replace("ct.h5", "") for path in  paths
                                            for entry in os.scandir(path) if entry.name == "ct.h5"]
    print(len(paths), "CT scans found!")


    train_transforms = A.Compose([
        A.Resize(512, 512, interpolation=cv2.INTER_LINEAR),
        A.RandomResizedCrop((512, 512), scale=(0.75, 1.0), ratio=(1., 1.), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])

    from transformers import CLIPTextModel, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", 
        subfolder="tokenizer", 
    )


    train_dataset = CTSuperResolutionDataset(paths, 
        data_root=train_data_dir,
        image_transforms=train_transforms, 
        tokenizer=tokenizer)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=1,
        pin_memory=True
    )

    for batch in tqdm(train_dataloader):
        batch = batch["pixel_values"]
        exit(0)