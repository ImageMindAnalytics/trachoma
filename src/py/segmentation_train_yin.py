import argparse
import os

import torch

from loaders import tt_dataset
from nets import segmentation

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.loggers import MLFlowLogger

torch.set_float32_matmul_precision('medium')


def add_train_args(parser):
    """Add segmentation training argument groups to a parser. Returns the parser."""
    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience steps for EarlyStopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)
    hparams_group.add_argument('--accumulate_grad_batches', help='Number of gradient accumulation steps', type=int, default=1)
    hparams_group.add_argument('--find_unused_parameters', help='find_unused_parameters', type=int, default=0)

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--nn', help='Type of neural network', type=str, default='TTUNet')
    input_group.add_argument('--model', help='Model to continue training', type=str, default=None)
    input_group.add_argument('--data_module', help='Type of data module to use', type=str, default='TTDataModuleSegPklGPUTrans')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default='./')
    output_group.add_argument('--monitor', help='Metric to monitor to save checkpoints', type=str, default='val_loss')
    output_group.add_argument('--monitor_mode', help='Monitor mode (min or max)', type=str, default='min')

    log_group = parser.add_argument_group('Logger')
    log_group.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)
    log_group.add_argument('--mlflow_tags', help='MLFlow tags', type=str, nargs='+', default=None)
    log_group.add_argument('--mlflow_experiment_name', help='MLFlow experiment name', type=str, default='trachoma/segmentation')
    return parser


def main(args):

    if args.out and not os.path.exists(args.out):
        os.makedirs(args.out)

    NN = getattr(segmentation, args.nn)
    if args.model:
        model = NN.load_from_checkpoint(args.model, strict=False, **vars(args))
    else:
        model = NN(**vars(args))

    DM = getattr(tt_dataset, args.data_module)
    datamodule = DM(**vars(args))

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{' + args.monitor + ':.2f}',
        save_top_k=2,
        monitor=args.monitor,
        mode=args.monitor_mode,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor=args.monitor,
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode=args.monitor_mode,
    )

    callbacks = [early_stop_callback, checkpoint_callback]

    if args.mlflow_tags:
        logger_mlflow = MLFlowLogger(
            experiment_name=args.mlflow_experiment_name,
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            tags={str(i): tag for i, tag in enumerate(args.mlflow_tags)},
        )
    else:
        logger_mlflow = None

    trainer = Trainer(
        logger=logger_mlflow,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=callbacks,
        devices=torch.cuda.device_count(),
        accelerator='gpu',
        strategy=DDPStrategy(find_unused_parameters=bool(args.find_unused_parameters)) if torch.cuda.device_count() > 1 else 'auto',
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.model)
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TT segmentation Training', add_help=False)
    add_train_args(parser)

    args, unknownargs = parser.parse_known_args()

    NN = getattr(segmentation, args.nn)
    NN.add_model_specific_args(parser)

    data_module = getattr(tt_dataset, args.data_module)
    parser = data_module.add_data_specific_args(parser)

    parser = argparse.ArgumentParser(parents=[parser])
    args = parser.parse_args()

    main(args)
