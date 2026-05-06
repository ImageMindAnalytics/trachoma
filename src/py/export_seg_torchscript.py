import torch
from nets.segmentation import TTUNet

CKPT = "/home/yin816/tt_training/output/seg_run_gpu_trans_dicece_sqrt/epoch=16-val_loss=1.53.ckpt"

model = TTUNet.load_from_checkpoint(
    CKPT,
    out_channels=7,
)

model.eval()
model.cpu()

example_input = torch.randn(1, 3, 512, 512)

traced = torch.jit.trace(model.model, example_input)

torch.jit.save(
    traced,
    "/home/yin816/tt_training/output/seg_run_gpu_trans_dicece_sqrt/model_nocj_epoch=16-val_loss=1.53.ts"
)

print("saved")