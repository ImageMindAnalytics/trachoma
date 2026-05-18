#!/usr/bin/env python3
"""Optuna hyperparameter optimization driver for segmentation training.

Calls segmentation_train_yin.py as a subprocess (required for DDPStrategy to work correctly).
Uses Optuna with RDB storage. Pruning is not supported with subprocess.
"""
import argparse
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import optuna

from loaders import tt_dataset
from nets import segmentation

from segmentation_train_yin import add_train_args


def _args_to_argv(args, exclude=None):
    """Convert namespace to argv list for subprocess. Skips None and excluded keys."""
    exclude = exclude or set()
    exclude |= {'optuna_n_trials', 'optuna_study_name', 'optuna_storage', 'segmentation_train_script'}
    argv = []
    for k, v in vars(args).items():
        if k in exclude or v is None:
            continue
        key = '--' + k
        if isinstance(v, bool):
            if v:
                argv.append(key)
        elif isinstance(v, (list, tuple)):
            argv.append(key)
            argv.extend(str(x) for x in v)
        elif isinstance(v, dict):
            argv.append(key)
            argv.append(json.dumps(v))
        else:
            argv.append(key)
            argv.append(str(v))
    return argv


def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation Optuna hyperparameter optimization', add_help=False)
    add_train_args(parser)

    optuna_group = parser.add_argument_group('Optuna')
    optuna_group.add_argument('--optuna_n_trials', help='Number of Optuna trials', type=int, required=True)
    optuna_group.add_argument('--optuna_study_name', help='Optuna study name', type=str, default='segmentation_optuna')
    optuna_group.add_argument('--optuna_storage', help='Optuna storage name in the output directory', type=str, default=None)
    optuna_group.add_argument('--segmentation_train_script', help='Path to segmentation_train_yin.py', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'segmentation_train_yin.py'))

    args, _ = parser.parse_known_args()

    NN = getattr(segmentation, args.nn)
    NN.add_model_specific_args(parser)

    data_module = getattr(tt_dataset, args.data_module)
    parser = data_module.add_data_specific_args(parser)

    parser = argparse.ArgumentParser(parents=[parser])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    NN = getattr(segmentation, args.nn)
    suggest_fn = getattr(NN, 'suggest_hyper_params', None)
    if suggest_fn is None:
        raise ValueError(f"Model {args.nn} has no suggest_hyper_params; cannot run Optuna.")

    segmentation_train_script = args.segmentation_train_script
    if not segmentation_train_script:
        segmentation_train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'segmentation_train_yin.py')

    def objective(trial):
        suggested = suggest_fn(trial)
        trial_args = SimpleNamespace(**{**vars(args), **suggested})
        trial_args.model = None  # do not resume from checkpoint during hyperopt
        trial_args.out = os.path.join(args.out, f"trial_{trial.number}")
        os.makedirs(trial_args.out, exist_ok=True)

        metric_file = os.path.join(trial_args.out, 'best_metric.json')

        try:
            trial_args.write_metric = metric_file
            argv = [sys.executable, segmentation_train_script] + _args_to_argv(trial_args)

            result = subprocess.run(argv, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"segmentation_train_yin subprocess exited with code {result.returncode}")

            with open(metric_file, 'r') as f:
                metrics = json.load(f)
            return float(metrics[args.monitor])

        except Exception as e:
            print(f"Error running segmentation_train_yin: {e}")
            if args.monitor_mode == "min":
                return float('inf')
            elif args.monitor_mode == "max":
                return float('-inf')
            else:
                raise ValueError(f"Invalid monitor mode: {args.monitor_mode}")

    study_kw = {
        "direction": "minimize" if args.monitor_mode == "min" else "maximize",
        "study_name": args.optuna_study_name,
    }

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    if args.optuna_storage:
        storage_path = os.path.join(args.out, args.optuna_storage)
        if "://" in args.optuna_storage:
            study_kw["storage"] = args.optuna_storage
        else:
            storage_path = os.path.join(args.out, args.optuna_storage)
            study_kw["storage"] = f"sqlite:///{storage_path}"
        study_kw["load_if_exists"] = True

    study = optuna.create_study(**study_kw)
    study.optimize(objective, n_trials=args.optuna_n_trials, show_progress_bar=True)

    best_params_path = os.path.join(args.out, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=2)

    print(f"Best parameters saved to {best_params_path}")
    print("Best trial:", study.best_trial.params)
    print("Best value:", study.best_value)
