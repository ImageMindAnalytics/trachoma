import SimpleITK as sitk
import numpy as np
import argparse
from collections import namedtuple

import torch
import torch.nn.functional as F

import os
import sys
import pandas as pd

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
            [255, 0, 0],      # class 1: red
            [0, 255, 0],      # class 2: green
            [0, 0, 255],      # class 3: blue
            [255, 255, 0],    # class 4: yellow
            [255, 0, 255],    # class 5: magenta
            [0, 255, 255],    # class 6: cyan
            [255, 165, 0],    # class 7: orange
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

def main(args): 

    img_out = []

    if args.csv:
        
        df = pd.read_csv(args.csv)

        for idx, row in df.iterrows():
            
            out_comp = os.path.normpath(os.path.join(args.out_composite, row[args.img_column]))                    

            if args.ow or (not os.path.exists(out_comp)):

                out_comp_dir = os.path.dirname(out_comp)

                if not os.path.exists(out_comp_dir):
                    os.makedirs(out_comp_dir)

                out_d = {'img': row[args.img_column],
                        'seg': row[args.seg_column],
                        'out_composite': out_comp}            
                
                img_out.append(out_d)

        if len(img_out) == 0:
            print(bcolors.INFO, "All images have been processed!", bcolors.ENDC)
            quit()

    else:
        img_out.append({'img': args.img})

    for obj in img_out:

        try:
            print(bcolors.INFO, "Reading:", obj["img"], bcolors.ENDC)
            img = sitk.ReadImage(obj["img"])            
            img_np = sitk.GetArrayFromImage(img)  
            img_t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float()/255.0  


            seg_np = sitk.ReadImage(obj["seg"])
            seg_np = sitk.GetArrayFromImage(seg_np)
            seg_t = torch.from_numpy(seg_np).unsqueeze(0).unsqueeze(0)

            
            print(bcolors.SUCCESS, "Writing:", obj["out_composite"], bcolors.ENDC)
            
            composite_t = blend(img_t, seg_t, alpha=args.alpha)

            composite_64 = _resize_with_grid_sample(
                composite_t, (64,64), mode="nearest", align_corners=False
            ).squeeze()
            composite_64 = (composite_64.permute(1,2,0).cpu().numpy()*255.0).astype(np.uint8)
            
            composite = sitk.GetImageFromArray(composite_64, isVector=True)

            writer = sitk.ImageFileWriter()
            writer.SetFileName(obj["out_composite"])
            writer.UseCompressionOn()
            writer.Execute(composite)
            
        except Exception as e:
            print(bcolors.FAIL, e, bcolors.ENDC, file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Blend img and seg for qc purposes', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_group = parser.add_argument_group('Input parameters')

    in_group = input_group.add_mutually_exclusive_group(required=True)
    in_group.add_argument('--img', type=str, help='Input image for prediction')
    in_group.add_argument('--csv', type=str, help='CSV file with images. Uses column name "image"')
    parser.add_argument('--csv_root', type=str, help='Root path to replace for output', default=None)
    parser.add_argument('--img_column', type=str, help='Name of column in csv file', default="image")
    parser.add_argument('--seg_column', type=str, help='Name of seg column in csv file', default="seg")

    output_group = parser.add_argument_group('Output parameters')    
    output_group.add_argument('--out_composite', type=str, help='Output for composite images', default='./composite')
    output_group.add_argument('--alpha', type=float, help='Alpha blending factor for composite images', default=0.7)
    output_group.add_argument('--ow', type=bool, help='Overwrite outputs', default=False)    

    args = parser.parse_args()
    main(args)