import os
import argparse

import torch

from nets.segmentation import TTUNet


def main(args):

    model = TTUNet.load_from_checkpoint(
        args.ckpt,
        out_channels=args.out_channels,
    )

    model.eval()
    model.cpu()

    example_input = torch.randn(1, 3, 512, 512)

    traced = torch.jit.trace(model.model, example_input)

    out_ts = os.path.splitext(args.ckpt)[0] + ".ts"

    torch.jit.save(
        traced,
        out_ts
    )

    print("saved:", out_ts)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to checkpoint"
    )

    parser.add_argument(
        "--out_channels",
        type=int,
        default=7,
    )

    args = parser.parse_args()

    main(args)