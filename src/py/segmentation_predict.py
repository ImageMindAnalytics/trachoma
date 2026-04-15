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

@torch.no_grad()
def run_segmentation_resize_roundtrip(
    model,
    img: torch.Tensor,
    *,
    in_hw: tuple[int, int] = (512, 512),
    align_corners: bool = False
):
    """
    img: (N,C,H,W)
    Returns:
      - logits_back: (N,K,H,W) if return_logits=True
      - or labels_back: (N,1,H,W) long tensor if return_logits=False
    """
    assert img.ndim == 4, f"Expected (N,C,H,W), got {img.shape}"
    n, c, h0, w0 = img.shape

    # 1) resize to model input
    img_512 = _resize_with_grid_sample(
        img, in_hw, mode="nearest", align_corners=align_corners
    )

    # 2) run model (expects 512x512)
    logits_512 = model(img_512)  # (N,K,512,512) typically

    # 3) resize back to original size
    logits = _resize_with_grid_sample(
        logits_512.float(), (w0, h0), mode="nearest", align_corners=align_corners
    )

    # If you want discrete labels, do argmax at original resolution
    labels = torch.argmax(logits, dim=1, keepdim=True).to(torch.long)  # (N,1,H,W)
    return labels, img_512, logits_512

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

    device = torch.device("cuda:0")

    model_seg = torch.jit.load(args.model, map_location=device)
    model_seg.eval()

    img_out = []

    if args.csv:
        
        df = pd.read_csv(args.csv)

        for idx, row in df.iterrows():

            out_d = {'img': row[args.img_column]}

            img = row[args.img_column]

            if args.csv_root:
                img = img.replace(args.csv_root, '')

            out_seg = None

            if args.out:

                ext = os.path.splitext(img)[1]
                out_seg = os.path.normpath(os.path.join(args.out, img)).replace(ext, ".nrrd")                    

                out_seg_dir = os.path.dirname(out_seg)

                if not os.path.exists(out_seg_dir):
                    os.makedirs(out_seg_dir)

                out_d['seg'] = out_seg
            else:
                out_d['seg'] = None

            if args.out_composite:

                out_comp = os.path.normpath(os.path.join(args.out_composite, img))                    

                out_comp_dir = os.path.dirname(out_comp)

                if not os.path.exists(out_comp_dir):
                    os.makedirs(out_comp_dir)
                
                out_d['out_composite'] = out_comp
            
            if args.ow or (not os.path.exists(out_seg)):
                img_out.append(out_d)

        

        if len(img_out) == 0:
            print(bcolors.INFO, "All images have been processed!", bcolors.ENDC)
            quit()

        df_out = pd.DataFrame(img_out)
        df['seg'] = df_out['seg']        
        csv_split_ext = os.path.splitext(args.csv)
        out_csv = csv_split_ext[0] + "_seg" + csv_split_ext[1]
        df.to_csv(out_csv, index=False)

    else:
        img_out.append({'img': args.img})

    for obj in img_out:

        try:
            print(bcolors.INFO, "Reading:", obj["img"], bcolors.ENDC)
            img = sitk.ReadImage(obj["img"])            
            img_np = sitk.GetArrayFromImage(img)  

            seg_t, img_512, logits_512 = run_segmentation_resize_roundtrip(model_seg, torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float().cuda()/255.0)

            if obj["seg"] is not None:
                print(bcolors.SUCCESS, "Writing:", obj["seg"], bcolors.ENDC)
                
                seg = sitk.GetImageFromArray(seg_t.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8))                
                seg.CopyInformation(img)

                writer = sitk.ImageFileWriter()
                writer.SetFileName(obj["seg"])
                writer.UseCompressionOn()
                writer.Execute(seg)

            if obj["out_composite"] is not None:
                print(bcolors.SUCCESS, "Writing:", obj["out_composite"], bcolors.ENDC)

                seg_512 = torch.argmax(logits_512, dim=1, keepdim=True).float()
                composite_t = blend(img_512, seg_512, alpha=args.alpha)
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
    parser = argparse.ArgumentParser(description='Prediction the segmentation only', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_group = parser.add_argument_group('Input parameters')

    in_group = input_group.add_mutually_exclusive_group(required=True)
    in_group.add_argument('--img', type=str, help='Input image for prediction')
    in_group.add_argument('--csv', type=str, help='CSV file with images. Uses column name "image"')
    parser.add_argument('--csv_root', type=str, help='Root path to replace for output', default=None)
    parser.add_argument('--img_column', type=str, help='Name of column in csv file', default="img")
    
    parser.add_argument('--model', type=str, help='Segmentation pytorch script model', default='/work/jprieto/data/remote/EGower/trained_models/segmentation/v5.3/epoch=189-val_loss=0.17.pt')
    

    output_group = parser.add_argument_group('Output parameters')
    output_group.add_argument('--out', type=str, help='Output dir', default=None)
    output_group.add_argument('--out_composite', type=str, help='Output for composite images', default=None)
    output_group.add_argument('--alpha', type=float, help='Alpha blending factor for composite images', default=0.7)
    output_group.add_argument('--ow', type=bool, help='Overwrite outputs', default=False)    

    args = parser.parse_args()
    main(args)