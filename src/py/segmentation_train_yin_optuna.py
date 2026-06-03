import os

# os.environ['CUDA_VISIBLE_DEVICES']="0"

import argparse



import math

import pandas as pd

import numpy as np 



import torch

torch.set_float32_matmul_precision('medium')



from nets.segmentation import TTUNet,TTRCNN

from loaders import tt_dataset

from callbacks.logger import SegImageLoggerNeptune, MaskRCNNImageLoggerNeptune



from lightning import Trainer

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch.strategies.ddp import DDPStrategy

from lightning.pytorch.loggers import NeptuneLogger



from sklearn.utils import class_weight



def main(args):
    
    DM = getattr(tt_dataset, args.data_module)
    ttdata = DM(**vars(args))
        





    checkpoint_callback = ModelCheckpoint(dirpath=args.out, filename='{epoch}-{val_loss:.2f}', save_top_k=2, monitor='val_loss' )

    checkpoint_callback = ModelCheckpoint(dirpath=args.out, filename='{epoch}-{val_rare_dice:.4f}', save_top_k=2, monitor='val_rare_dice', mode='max')

    

    # image_logger = MaskRCNNImageLoggerNeptune(log_steps = args.log_every_n_steps)

    

    if args.model:

        model = TTUNet.load_from_checkpoint(args.model, out_channels=2, **vars(args), strict=False)

        # model = TTRCNN.load_from_checkpoint(args.model, out_channels=4, **vars(args), strict=False)

    else:

        model = TTUNet(**vars(args))

        # model = TTRCNN(num_classes=4,  **vars(args))

    



    # ckpt = '/CMF/data/lumargot/trachoma/output/segmentation/first_model/epoch.192-val_loss.0.52.ckpt'

    # checkpoint = torch.load(ckpt, map_location="cpu")



    # state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # model_dict = model.state_dict()



    # pretrained_dict = {

    #     k: v for k, v in state_dict.items()

    #     if k in model_dict and v.shape == model_dict[k].shape

    # }



    # print(f"Loaded {len(pretrained_dict)} / {len(model_dict)} layers")



    # # Update and load

    # model_dict.update(pretrained_dict)

    # model.load_state_dict(model_dict)

    



    # early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    early_stop_callback = EarlyStopping(monitor="val_rare_dice", min_delta=0.00, patience=args.patience, verbose=True, mode="max")

    

    logger = None

    

    callbacks = [early_stop_callback, checkpoint_callback]

    if args.neptune_tags:

        logger = NeptuneLogger(project='ImageMindAnalytics/trachoma',

                               tags=args.neptune_tags,

                               api_key=os.environ['NEPTUNE_API_TOKEN'],

                               log_model_checkpoints=False)



        image_logger = SegImageLoggerNeptune(num_images=4, nrow=2, log_steps=args.log_every_n_steps)

        callbacks.append(image_logger)



    trainer = Trainer(

        logger=logger,

        max_epochs=args.epochs,

        callbacks=callbacks,

        devices=torch.cuda.device_count(), 

        accelerator="gpu", 

        strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else 'auto',

        log_every_n_steps=args.log_every_n_steps, 

        accumulate_grad_batches=args.accumulate_grad_batches

    )

    trainer.fit(model, datamodule=ttdata, ckpt_path=args.model)

    trainer.test(model, datamodule=ttdata)





if __name__ == '__main__':





    parser = argparse.ArgumentParser(description='TT segmentation Training')



    input_group = parser.add_argument_group('Input')

    input_group.add_argument('--data_module', help='Data module to use', type=str, default="TTDataModuleSegPklGPUResize")

    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)

    input_group.add_argument('--img_column', type=str, default="img_path", help='Name of image column in csv')

    input_group.add_argument('--seg_column', type=str, default="seg_path", help='Name of segmentation column in csv')



    hparams_group = parser.add_argument_group('Hyperparameters')

    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')

    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)

    hparams_group.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')

    hparams_group.add_argument('--patience', help='Max number of patience steps for EarlyStopping', type=int, default=30)

    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    

    # hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=256)

    hparams_group.add_argument('--out_channels', help='Number of output channels', type=int, default=4)

    hparams_group.add_argument('--ce_weight', help='Cross entropy weight', type=float, default=None, nargs='+')

    hparams_group.add_argument('--accumulate_grad_batches', help='Number of gradient accumulation steps', type=int, default=1) 

    

    logger_group = parser.add_argument_group('Logger')

    logger_group.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)

    logger_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)

    logger_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="segmentation_unet")

    

    logger_group.add_argument('--neptune_tags', help='neptune tag', type=str, nargs='+', default=None)



    output_group = parser.add_argument_group('Output')

    output_group.add_argument('--out', help='Output', type=str, default="./")



    args, _ = parser.parse_known_args()

    DM = getattr(tt_dataset, args.data_module)
    parser = DM.add_data_specific_args(parser)
    
    args = parser.parse_args()



    main(args)