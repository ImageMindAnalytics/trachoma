import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.ops import nms

import argparse
import SimpleITK as sitk
import numpy as np 

import plotly.graph_objects as go
from plotly.subplots import make_subplots


from lightning import LightningDataModule
import monai
import random 
import albumentations as A

import pickle

class bcolors:
    HEADER = '\033[95m'
    OK = '\033[94m'
    INFO = '\033[96m'
    SUCCESS = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TTDatasetBX(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", class_column = 'class',sev_column='sev'):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column
        self.severity_column = sev_column
        self.transform = transform

        self.df_subject = self.df[img_column].drop_duplicates().reset_index()
        self.target_size = (768, 1536)

    def __len__(self):
        return len(self.df_subject.index)

    def __getitem__(self, idx):
        
        subject = self.df_subject.iloc[idx][self.img_column]
        img_path = os.path.join(self.mount_point, subject)
        # self.seg_path = img_path.replace('img', 'segmentation_cleaned').replace('.jpg', '.nrrd')
        # seg_path = img_path.replace('img', 'seg').replace('.jpg', '.nrrd')
        seg_path = os.path.join(self.mount_point, 'mtss_pret_combined_seg', subject).replace('.jpg', '.nrrd')

        df_patches = self.df.loc[ self.df[self.img_column] == subject]        

        seg = torch.tensor(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).copy())).to(torch.float32)
        img = torch.tensor(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(img_path)).copy())).to(torch.float32)
        img = img.permute((2, 0, 1))
        img = img/255.0

        ## crop img within segmentation
        bbx_eye = self.compute_eye_bbx(seg, pad=0.05)
        img_cropped = img[:,bbx_eye[1]:bbx_eye[3],bbx_eye[0]:bbx_eye[2] ]
        seg_cropped = seg[bbx_eye[1]:bbx_eye[3],bbx_eye[0]:bbx_eye[2] ]
        # seg_cropped[ seg_cropped!=3 ] =0
        h,w = seg_cropped.shape
        
        self.pad = int(img_cropped.shape[1]/10)

        df_filtered = df_patches[(df_patches['x_patch'] >= bbx_eye[0].numpy()) & (df_patches['x_patch'] <= bbx_eye[2].numpy())]
        df_filtered = df_filtered[(df_filtered['y_patch'] >= bbx_eye[1].numpy()) & (df_filtered['y_patch'] <= bbx_eye[3].numpy())]
        # df_filtered = df_patches

        bbx, classes = [], []
        if not df_filtered.empty:
            for k, row in df_filtered.iterrows():
                class_idx =  torch.tensor(row[self.class_column]).to(torch.long)
                x, y = row['x_patch'], row['y_patch']

                cropped_x, cropped_y = x - bbx_eye[0], y -bbx_eye[1]
                box = torch.tensor([max((cropped_x-2*self.pad/3), 0),
                                    max((cropped_y-5*self.pad/3), 0),
                                    min((cropped_x+2*self.pad/3), img_cropped.shape[2]),
                                    min((cropped_y+self.pad/3), img_cropped.shape[1])])

                classes.append(class_idx.unsqueeze(0))
                bbx.append(box.unsqueeze(0))

        else:
            classes.append(torch.tensor(1).to(torch.long).unsqueeze(0))
            bbx.append(torch.tensor([5,5,img_cropped.shape[2]-5, img_cropped.shape[1]-5]).unsqueeze(0))
        bbx, classes = torch.cat(bbx), torch.cat(classes)

        augmented = self.transform(img_cropped.permute(1,2,0).numpy(), bbx.numpy(), classes.numpy(), seg_cropped.numpy())

        aug_coords = torch.tensor(augmented['bboxes'])
        aug_image = augmented['image']
        aug_seg = augmented['mask']
        aug_image = torch.tensor(aug_image).permute(2,0,1)

        indices = nms(aug_coords, 0.5*torch.ones_like(aug_coords[:,0]), iou_threshold=1.0) ## iou as args
        return {"img": aug_image, 
                "labels": classes[indices], 
                "boxes": aug_coords[indices] ,
                'mask':torch.tensor(aug_seg), 
                'fn': subject
                }


    def compute_eye_bbx(self, seg, label=1, pad=0):

        shape = seg.shape
        
        ij = torch.argwhere(seg.squeeze() != 0)

        bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

        bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
        bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
        bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
        bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        
        return bb


    def get_xy_coordinates_from_patch_name(self,patch_name):
        for elt in patch_name.split('_'):
            if 'x' == elt[-1]:
                x = elt[:-1]
            elif elt == 'Wavy':
                pass
            elif 'y' == elt[-1]:
                y = elt[:-1]
        return int(x), int(y)

class TTDataModuleBX(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column='class', severity_column='sev', balanced=False, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.img_column = img_column
        self.class_column = class_column   
        self.severity_column = severity_column   
        
        self.balanced = balanced
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = monai.data.Dataset(data=TTDatasetBX(self.df_train, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, sev_column=self.severity_column, transform=self.train_transform))
        self.val_ds = monai.data.Dataset(TTDatasetBX(self.df_val, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, sev_column=self.severity_column, transform=self.valid_transform))
        self.test_ds = monai.data.Dataset(TTDatasetBX(self.df_test, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, sev_column=self.severity_column, transform=self.test_transform))

    def train_dataloader(self):

        # if self.balanced: 
        #     g = self.df_train.groupby(self.class_column)
        #     df_train = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)
        #     self.train_ds = monai.data.Dataset(data=TTDatasetBX(df_train, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.train_transform))

        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=self.drop_last, collate_fn=self.custom_collate_fn, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        # remove balancing for evaluation step for acc or p,r,f metrics
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.custom_collate_fn)

    def custom_collate_fn(self,batch):
        targets = []
        imgs = []
        for targets_dic in batch:
            img = targets_dic.pop('img', None)
            imgs.append(img.unsqueeze(0))
            targets.append(targets_dic)
        return torch.cat(imgs), targets
    
    def balance_batch_collate_fn(self, batch):
        """Custom collate function that balances classes in a batch"""
        images = torch.stack([item['img'] for item in batch])
       
        masks = torch.stack([item['mask'] for item in batch])

        original_boxes = [item['boxes'] for item in batch]
        original_labels = [item['labels'] for item in batch]
        
        # Count boxes per class across the entire batch
        all_labels = []
        for item in batch:
            all_labels.append(item['labels'])
        
        # Find minimum count across classes
        all_labels = torch.cat(all_labels)
        classes, counts = torch.unique(all_labels, return_counts=True)
        min_count = counts.min().item()
        
        targets = []
        for i, (boxes, labels, mask) in enumerate(zip(original_boxes, original_labels, masks)):
            
            img_classes = torch.unique(labels)
            img_balanced_boxes, img_balanced_labels = [], []

            for cls in img_classes:
                m_label = (labels == cls)
                cls_boxes = boxes[m_label]
                cls_labels = labels[m_label]
                
                # Calculate how many to keep of this class
                proportion = len(cls_boxes) / counts[classes == cls].item()
                keep_count = max(1, int(min_count * proportion))
                keep_count = min(keep_count, len(cls_boxes))
                
                # Randomly sample
                if len(cls_boxes) > keep_count:
                    indices = torch.randperm(len(cls_boxes))[:keep_count]
                    cls_boxes = cls_boxes[indices]
                    cls_labels = cls_labels[indices]
                
                img_balanced_boxes.append(cls_boxes)
                img_balanced_labels.append(cls_labels)
            
            if img_balanced_boxes:
                img_boxes = torch.cat(img_balanced_boxes)
                img_labels = torch.cat(img_balanced_labels)
            else:
                img_boxes = boxes
                img_labels = labels
            
            dic_i = {'labels': img_labels, 
                     'boxes': img_boxes,
                     'mask': mask}
            targets.append(dic_i)

        labels = [t['labels'] for t in targets]
        classes, counts = torch.unique(torch.cat(labels), return_counts=True)
        # print(counts)

        return images, targets

class BBXImageTestTransform():
    def __init__(self):

        self.transform = A.Compose(
            [
                # A.NoOp(),
                # A.Resize(height=1024, width=1024),
                A.LongestMaxSize(max_size=1024),
            ], 
            bbox_params=A.BboxParams(format='pascal_voc', min_area=32, min_visibility=0.1, label_fields=['category_ids']),
            additional_targets={'mask': 'mask'},
        )

    def __call__(self, image, bboxes, category_ids, mask=None):
        return self.transform(image=image, bboxes=bboxes, category_ids=category_ids, mask=mask)

def _clamp_xyxy(boxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W - 1)  # x1,x2
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H - 1)  # y1,y2
    return boxes

@torch.no_grad()
def labelmap_from_boxes_intersect_label3(
    imgs: torch.Tensor,
    targets: list[dict],
    mask_value: int = 3,          # intersect with mask == 3
    background: int = 0,
    overlap_rule: str = "last", # "last" | "largest" | "score",
    pad_y = 100,
    pad_x = 5,
) -> torch.Tensor:
    """
    Returns:
      out: LongTensor [B,H,W]
    """
    assert imgs.dim() == 4, f"imgs should be [B,C,H,W], got {imgs.shape}"
    B, _, H, W = imgs.shape
    device = imgs.device

    out = torch.full((B, H, W), background, dtype=torch.long, device=device)

    for b in range(B):
        t = targets[b]
        boxes  = t["boxes"].to(device)
        labels = t["labels"].to(device).long() + 1
        mask_lm = t["mask"].to(device)

        if mask_lm.dim() != 2:
            raise ValueError(f"Expected mask labelmap [H,W], got {mask_lm.shape}")
        if mask_lm.shape != (H, W):
            raise ValueError(f"mask shape {mask_lm.shape} must match image {(H,W)}")

        if boxes.numel() == 0:
            continue

        # boolean region of interest: only where labelmap == mask_value (i.e., == 3)
        roi = (mask_lm == mask_value)        

        # clamp + integerize boxes, use exclusive slicing
        boxes = _clamp_xyxy(boxes, H, W)
        x1 = boxes[:, 0].floor().long().clamp(0, W - 1)
        y1 = boxes[:, 1].floor().long().clamp(0, H - 1)
        x2 = boxes[:, 2].ceil().long()
        y2 = boxes[:, 3].ceil().long()

        # make x2/y2 exclusive and at least 1 pixel wide/tall
        x2 = torch.maximum(x2, x1 + 1).clamp(1, W)
        y2 = torch.maximum(y2, y1 + 1).clamp(1, H)

        N = boxes.shape[0]

        # choose painting order (later overwrites earlier)
        if overlap_rule == "last":
            order = torch.sort(labels)[1]
            # order = torch.arange(N, device=device)
        elif overlap_rule == "largest":
            areas = (x2 - x1) * (y2 - y1)
            order = torch.argsort(areas)  # small -> large; large painted last = wins
        elif overlap_rule == "score":
            if "scores" not in t:
                raise ValueError("overlap_rule='score' requires targets[i]['scores']")
            scores = t["scores"].to(device)
            order = torch.argsort(scores)  # low -> high; high painted last = wins
        else:
            raise ValueError(f"Unknown overlap_rule: {overlap_rule}")

        lm = roi.clone().long() 

        for j in order.tolist():
            xa, xb = int(x1[j]) - pad_x, int(x2[j]) + pad_x
            ya, yb = int(y1[j]) - pad_y, int(y2[j]) + pad_y

            xa = max(xa, 0)
            xb = min(xb, W)
            ya = max(ya, 0)
            yb = min(yb, H)
            if xa >= xb or ya >= yb:
                continue

            allowed = roi[ya:yb, xa:xb]
            if allowed.any():
                patch = lm[ya:yb, xa:xb]
                patch[allowed] = labels[j]
                lm[ya:yb, xa:xb] = patch

        out[b] = lm

    return out

def _resize_with_grid_sample(
    x: torch.Tensor,
    out_hw: tuple[int, int],
    mode: str,
    align_corners: bool = False,
) -> torch.Tensor:
    """
    x: (N,C,H,W) tensor
    out_hw: (H_out, W_out)
    mode: 'bilinear' for images/logits, 'nearest' for label maps
    """
    assert x.ndim == 4, f"Expected (N,C,H,W), got {x.shape}"

    mesh_grid_params = [torch.arange(start=-1.0, end=1.0, step=(2.0/s), device=x.device) for s in out_hw]
    mesh_grid = torch.stack(torch.meshgrid(mesh_grid_params, indexing='xy'), dim=-1).to(torch.float32).unsqueeze(0)

    y = F.grid_sample(
        x, mesh_grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )
    return y

label_colors = {    
    0: "black",
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
    4: "tab:red",
    5: "tab:purple",
    6: "tab:brown",
}

def map_seg(seg: torch.Tensor) -> torch.Tensor:
    """
    seg: (N,1,H,W) float tensor in [0,1]
    Returns:
      seg_rgb: (N,3,H,W) float tensor in [0,1]
    """
    assert seg.ndim == 4 and seg.shape[1] == 1, f"Expected (N,1,H,W), got {seg.shape}"
    color_map = torch.tensor(
        [
            [0, 0, 0],        # class 0: black
            [31, 119, 180],   # 1: tab:blue
            [255, 127, 14],   # 2: tab:orange
            [44, 160, 44],    # 3: tab:green
            [214, 39, 40],    # 4: tab:red
            [148, 103, 189],  # 5: tab:purple
            [140, 86, 75],    # 6: tab:brown
            [0, 256, 256],    # 7: cyan
        ],
        device=seg.device,
        dtype=torch.float32
    ) / 255.0  # Normalize to [0,1]

    N, _, H, W = seg.shape
    seg_long = seg.long().squeeze(1)  # (N,H,W)
    seg_rgb = color_map[seg_long]     # (N,H,W,3)
    seg_rgb = seg_rgb.permute(0, 3, 1, 2)  # (N,3,H,W)
    return seg_rgb

def blend(img: torch.Tensor, seg: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    img: (N,3,H,W) float tensor in [0,1]
    seg: (N,1,H,W) float tensor in [0,1]
    """
    assert img.ndim == 4 and img.shape[1] == 3, f"Expected (N,3,H,W), got {img.shape}"
    assert seg.ndim == 4 and seg.shape[1] == 1, f"Expected (N,1,H,W), got {seg.shape}"
    seg_rgb = map_seg(seg)
    blended = (1.0 - alpha) * img + alpha * seg_rgb
    blended = torch.clamp(blended, 0.0, 1.0)
    return blended


def merge_bb(batch, out_dir, out_dir_comp):
    imgs, targets = batch
    fn = targets[0].get('fn')
    if fn is None:
        raise ValueError("Each target dict must include 'fn' for filename")

    labelmaps = labelmap_from_boxes_intersect_label3(
        imgs=imgs,
        targets=targets,
        background=0,
        overlap_rule="last",
    )

    out_d = {
        'img': (imgs * 255.0).to(torch.uint8),
        'seg': labelmaps.to(torch.uint8),
    }

    out_fn = os.path.join(out_dir, fn.replace('.jpg', '.pkl'))
    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
    print(bcolors.INFO, "Writing:", out_fn, bcolors.ENDC)
    with open(out_fn, 'wb') as f:
        pickle.dump(out_d, f)

    if out_dir_comp is not None:
        composite_t = blend(imgs, labelmaps.unsqueeze(1), alpha=0.6)
        composite_64 = _resize_with_grid_sample(
            composite_t, (64, 64), mode="nearest", align_corners=False
        ).squeeze()
        composite_64 = (composite_64.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        composite = sitk.GetImageFromArray(composite_64, isVector=True)

        ouf_comp_fn = os.path.join(out_dir_comp, fn)
        os.makedirs(os.path.dirname(ouf_comp_fn), exist_ok=True)

        print(bcolors.INFO, "Writing:", ouf_comp_fn, bcolors.ENDC)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(ouf_comp_fn)
        writer.UseCompressionOn()
        writer.Execute(composite)


def process_loader(loader, out_dir, out_dir_comp):
    for batch in loader:
        merge_bb(batch, out_dir, out_dir_comp)


def main(args):
    mount_point = args.mount_point
    df_train = pd.read_csv(args.csv_train) if args.csv_train else pd.DataFrame({args.img_column: pd.Series(dtype=str)})
    df_val = pd.read_csv(args.csv_val) if args.csv_val else pd.DataFrame({args.img_column: pd.Series(dtype=str)})
    df_test = pd.read_csv(args.csv_test) if args.csv_test else pd.DataFrame({args.img_column: pd.Series(dtype=str)})

    dm = TTDataModuleBX(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        mount_point=mount_point,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_column=args.img_column,
        class_column=args.class_column,
        balanced=False,
        train_transform=BBXImageTestTransform(),
        valid_transform=BBXImageTestTransform(),
        test_transform=BBXImageTestTransform(),
        drop_last=False,
    )
    dm.setup()

    out_dir = args.output_dir
    out_dir_comp = args.output_dir_comp

    if args.csv_train:
        print(bcolors.INFO, "Processing train split", bcolors.ENDC)
        process_loader(dm.train_dataloader(), out_dir, out_dir_comp)
    if args.csv_val:
        print(bcolors.INFO, "Processing val split", bcolors.ENDC)
        process_loader(dm.val_dataloader(), out_dir, out_dir_comp)
    if args.csv_test:
        print(bcolors.INFO, "Processing test split", bcolors.ENDC)
        process_loader(dm.test_dataloader(), out_dir, out_dir_comp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write labelmaps from bounding-box targets and mask intersections', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mount-point', type=str, default='/MEDUSA_STOR/jprieto/EGower/', help='Root path used to resolve image filenames')
    parser.add_argument('--csv-train', type=str, default=os.path.join('/MEDUSA_STOR/jprieto/EGower/', 'CSV_files/mtss_pret_combined_train_train.csv'), help='Train CSV with patch targets')
    parser.add_argument('--csv-val', type=str, default=os.path.join('/MEDUSA_STOR/jprieto/EGower/', 'CSV_files/mtss_pret_combined_train_test.csv'), help='Validation CSV with patch targets')
    parser.add_argument('--csv-test', type=str, default=os.path.join('/MEDUSA_STOR/jprieto/EGower/', 'CSV_files/mtss_pret_combined_test.csv'), help='Test CSV with patch targets')
    parser.add_argument('--img-column', type=str, default='filename', help='CSV image filename column name')
    parser.add_argument('--class-column', type=str, default='class', help='CSV class label column name')
    parser.add_argument('--batch-size', type=int, default=1, help='DataLoader batch size')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of DataLoader workers')
    parser.add_argument('--output-dir', type=str, default='/MEDUSA_STOR/jprieto/EGower/mtss_pred_combined_merged_segpredv53', help='Directory to write labelmap pickle outputs')
    parser.add_argument('--output-dir-comp', type=str, default='/MEDUSA_STOR/jprieto/EGower/mtss_pred_combined_merged_segpredv53_composite', help='Directory to write composite preview images')
    args = parser.parse_args()
    main(args)
