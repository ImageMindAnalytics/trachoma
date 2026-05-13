from __future__ import print_function

import argparse
import itertools
import os
import pickle

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".3f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    return cm


def read_true_seg(seg_path, csv_root=None):
    if csv_root and not os.path.isabs(seg_path):
        seg_path = os.path.join(csv_root, seg_path)

    if seg_path.endswith(".pkl"):
        with open(seg_path, "rb") as f:
            d = pickle.load(f)

        y_true = d["seg"]

        if hasattr(y_true, "detach"):
            y_true = y_true.detach().cpu().numpy()

        y_true = np.squeeze(y_true)
        return y_true

    return sitk.GetArrayFromImage(sitk.ReadImage(seg_path))


def read_pred_seg(pred_path):
    y_pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))

    if y_pred.ndim >= 3 and y_pred.shape[-1] > 1:
        y_pred = np.argmax(y_pred, axis=-1)

    y_pred = np.squeeze(y_pred)
    return y_pred


def match_shapes(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        print("Shape mismatch:", y_true.shape, y_pred.shape)

        h = min(y_true.shape[0], y_pred.shape[0])
        w = min(y_true.shape[1], y_pred.shape[1])

        y_true = y_true[:h, :w]
        y_pred = y_pred[:h, :w]

    return y_true, y_pred


def main(args):
    df = pd.read_csv(args.csv)
    labels = list(range(args.num_classes))

    dice_arr = []
    cnf_matrix_arr = []
    cl_report_arr = []

    for i, row in df.iterrows():
        print("Reading:", row[args.seg_column])
        y_true = read_true_seg(row[args.seg_column], csv_root=args.csv_root)

        print("Reading:", row[args.pred_column])
        y_pred = read_pred_seg(row[args.pred_column])

        y_true, y_pred = match_shapes(y_true, y_pred)

        y_true = np.reshape(y_true, -1).astype(int)
        y_pred = np.reshape(y_pred, -1).astype(int)

        cnf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

        row_sums = cnf_matrix.sum(axis=1, keepdims=True)
        cnf_matrix_norm = np.divide(
            cnf_matrix.astype(float),
            row_sums,
            out=np.zeros_like(cnf_matrix, dtype=float),
            where=row_sums != 0,
        )

        cnf_matrix_arr.append(cnf_matrix_norm)
        print(cnf_matrix_norm)

        cl_report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        cl_report_arr.append(cl_report)
        print(cl_report)

        dice = np.full(args.num_classes, np.nan)

        for c in labels:
            true_c = (y_true == c)
            pred_c = (y_pred == c)
        
            true_sum = true_c.sum()
            pred_sum = pred_c.sum()
        
            # no class i in both GT and prediction: do not count
            if true_sum == 0 and pred_sum == 0:
                dice[c] = np.nan
            else:
                intersection = np.logical_and(true_c, pred_c).sum()
                dice[c] = 2.0 * intersection / (true_sum + pred_sum)
        
        print(dice)
        dice_arr.append(dice)

    out_base = os.path.splitext(args.csv)[0]

    pickle.dump(
        cl_report_arr,
        open(out_base + "_classification_report.pickle", "wb"),
    )
    pickle.dump(
        cnf_matrix_arr,
        open(out_base + "_confusion_matrix.pickle", "wb"),
    )

    dice_arr = np.array(dice_arr)
    dice_arr_plot = pd.DataFrame(dice_arr, columns=[f"class_{i}" for i in labels])
    print(dice_arr.shape)

    np.save(out_base + "_violin_plot.npy", dice_arr)

    fig3 = plt.figure(figsize=(8, 5))
    try:
        s = sns.violinplot(data=dice_arr_plot, cut=0, density_norm="count")
    except TypeError:
        s = sns.violinplot(data=dice_arr_plot, cut=0, scale="count")

    s.set_title("Dice coefficients")
    s.set_xlabel("Class")
    s.set_ylabel("Dice")
    s.set_xticklabels([str(i) for i in labels])

    violin_filename = out_base + "_violin_plot.png"
    fig3.savefig(violin_filename, bbox_inches="tight", dpi=200)

    dice_df = pd.DataFrame(
        dice_arr,
        columns=[f"dice_class_{i}" for i in labels],
    )

    df_out = pd.concat([df.reset_index(drop=True), dice_df], axis=1)
    df_out.to_csv(args.csv.replace(".csv", "_dice.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation predictions for pkl ground truth and nrrd predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_param_group = parser.add_argument_group("Input")
    input_param_group.add_argument("--csv", type=str, help="CSV file with segmentation and prediction columns", required=True)
    input_param_group.add_argument("--seg_column", type=str, help="column name for ground-truth segmentation/pkl", default="seg")
    input_param_group.add_argument("--pred_column", type=str, help="column name for prediction", default="pred")
    input_param_group.add_argument("--csv_root", type=str, help="root path for relative pkl paths", default=None)
    input_param_group.add_argument("--num_classes", type=int, help="number of segmentation classes", default=7)

    args = parser.parse_args()
    main(args)