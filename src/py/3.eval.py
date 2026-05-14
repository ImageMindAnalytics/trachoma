import argparse
import os

import pandas as pd


def main(args):

    df = pd.read_csv(args.test_csv)

    df["pred"] = df["img"].apply(
        lambda x: os.path.join(
            args.pred_root,
            os.path.splitext(x)[0] + ".nrrd"
        )
    )

    df.to_csv(args.out_csv, index=False)

    print(args.out_csv)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="path to original test csv"
    )

    parser.add_argument(
        "--pred_root",
        type=str,
        required=True,
        help="root directory containing predicted nrrd files"
    )

    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="output evaluation csv"
    )

    args = parser.parse_args()

    main(args)