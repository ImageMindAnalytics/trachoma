import pandas as pd, os

test_csv = "/home/yin816/tt_training/practice/test.csv"
pred_root = "/home/yin816/tt_training/output/seg_run_gpu_trans_dicece_sqrt/pred_test"
out_csv = "/home/yin816/tt_training/output/seg_run_gpu_trans_dicece_sqrt/test_eval.csv"

df = pd.read_csv(test_csv)

df["pred"] = df["img"].apply(
    lambda x: os.path.join(pred_root, os.path.splitext(x)[0] + ".nrrd")
)

df.to_csv(out_csv, index=False)
print(out_csv)