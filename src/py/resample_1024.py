from torch.utils.data import Dataset, DataLoader

import os
import pandas as pd
import numpy as np
import SimpleITK as sitk

import torch
import pickle 

import monai
from monai.transforms import (    
    
    Compose,    
    EnsureChannelFirstd,
    Padd, 
    Resized,
    SpatialPad,
)


class SquarePad:
    def __init__(self, keys):
        self.keys = keys
    def __call__(self, X):

        max_shape = []
        for k in self.keys:
            max_shape.append(torch.max(torch.tensor(X[k].shape)))
        max_shape = torch.max(torch.tensor(max_shape)).item()
        
        return Padd(self.keys, padder=SpatialPad(spatial_size=(max_shape, max_shape)))(X)

class TransformSeg:
    def __init__(self):
        self.train_transform = Compose(
            [
                EnsureChannelFirstd(strict_check=False, keys=["img"], channel_dim=2),
                EnsureChannelFirstd(strict_check=False, keys=["seg"], channel_dim='no_channel'),
                SquarePad(keys=["img", "seg"]),
                Resized(keys=["img", "seg"], spatial_size=[1024, 1024], mode=['area', 'nearest']),                
            ]
        )
    def __call__(self, inp):
        return self.train_transform(inp)
    


    
class DS(Dataset):
    def __init__(self, df, mount_point="./", img_column="img_path", seg_column="seg_path", class_column='class'):
        self.df = df        
        self.mount_point = mount_point
        self.img_column = img_column
        self.seg_column = seg_column
        self.class_column = class_column
    def __len__(self):
        return len(self.df.index)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        
        img = os.path.join(self.mount_point, row[self.img_column])
        seg = os.path.join(self.mount_point, row[self.seg_column])

        img_t = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(img)))
        seg_t = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(seg)))

        return {"img": img_t, "seg": seg_t, "fn": row[self.img_column]}
    

# df = pd.read_csv("/MEDUSA_STOR/jprieto/EGower/CSV_files/full_seg_train_fixed.csv")

# ds = DS(df, mount_point="/MEDUSA_STOR/jprieto/EGower/", img_column="img", seg_column="seg")

# transform = TransformSeg()

# for i in range(len(ds)):
#     sample = ds[i]
#     out = transform(sample)
#     out["img"] = out["img"].to(torch.uint8)
#     out["seg"] = out["seg"].to(torch.uint8)
    
#     out_fn = os.path.join("/MEDUSA_STOR/jprieto/EGower/UNET_Resampled_1024/", sample["fn"].replace(".jpg", ".pkl"))
#     out_dir = os.path.dirname(out_fn)
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)

#     print(out_fn)

#     pickle.dump(out, open(out_fn, "wb"))



df_t = pd.read_csv("/MEDUSA_STOR/jprieto/EGower/CSV_files/full_seg_test.csv")

ds_t = DS(df_t, mount_point="/MEDUSA_STOR/jprieto/EGower/", img_column="img", seg_column="seg")

transform = TransformSeg()

for i in range(len(ds_t)):
    sample = ds_t[i]
    out = transform(sample)
    out["img"] = out["img"].to(torch.uint8)
    out["seg"] = out["seg"].to(torch.uint8)
    
    out_fn = os.path.join("/MEDUSA_STOR/jprieto/EGower/UNET_Resampled_1024/", sample["fn"].replace(".jpg", ".pkl"))
    out_dir = os.path.dirname(out_fn)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(out_fn)

    pickle.dump(out, open(out_fn, "wb"))


    
