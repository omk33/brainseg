import os
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import mlflow

from model import Att3DUNet
from hyperparameters import NUM_WORKERS, BATCH_SIZE, IN_CH, OUT_CH, NUM_FILTERS, NUM_HEADS
from utils import npy_dataset, one_hot_from_logits

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

def main():

    print("starting test loop...")

    # hyperparameters
    num_workers = NUM_WORKERS
    batch_size = BATCH_SIZE
    in_ch = IN_CH
    out_ch = OUT_CH
    num_filters = NUM_FILTERS
    num_heads = NUM_HEADS

    # experiment setup
    mlflow.set_experiment("brainseg")
    mlflow.start_run(run_name="brainseg_test")
    mlflow.log_params(
        {
            "dataset": "BraTS2020",
            "num_workers": num_workers,
            "batch_size": batch_size,
            "model": "Att3DUNet",
            "in_channels": in_ch,
            "out_channels": out_ch,
            "num_filters": num_filters,
            "num_heads": num_heads,
        }
    )

    # data setup
    data_dir = '/home/omkos333/projects/brainseg/data/processed'

    test_ds = npy_dataset(
        os.path.join(data_dir, 'test', 'images'),
        os.path.join(data_dir, 'test', 'masks'),
        data_augmentation=False
    )

    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # model setup
    model = Att3DUNet(
        in_channels=in_ch, out_channels=out_ch, num_filters=num_filters, num_heads=num_heads, dropout=False
    )

    model.to(device)

    model.load_state_dict(torch.load('weights/best_weights.pt', map_location=device))
    mlflow.log_artifact('weights/best_weights.pt')

    # define metrics
    dice_WT = DiceMetric(include_background=False, reduction="mean_batch")
    dice_TC = DiceMetric(include_background=False, reduction="mean_batch")
    dice_ET = DiceMetric(include_background=False, reduction="mean_batch")

    hd95_WT = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")
    hd95_TC = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")
    hd95_ET = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")

    # testing loop
    with torch.no_grad():
        model.eval()

        for image, true_mask in test_loader:
            image, true_mask = image.to(device), true_mask.to(device)

            with autocast(device_str):
                logits = model(image)
                pred_mask = one_hot_from_logits(logits, out_ch)

            # whole tumor (WT) = ED + NCR + ET
            wt_pred = (pred_mask[:, 1:, ...].sum(dim=1, keepdim=True) > 0).float()
            wt_true = (true_mask[:, 1:, ...].sum(dim=1, keepdim=True) > 0).float()

            # tumor core (TC) = NCR + ET
            tc_pred = ((pred_mask[:, 1, ...] + pred_mask[:, 3, ...]) > 0).unsqueeze(1).float()
            tc_true = ((true_mask[:, 1, ...] + true_mask[:, 3, ...]) > 0).unsqueeze(1).float()

            # enhancing tumor (ET) = ET only
            et_pred = pred_mask[:, 3:4, ...]
            et_true = true_mask[:, 3:4, ...]

            # update metrics
            dice_WT(wt_pred, wt_true)
            dice_TC(tc_pred, tc_true)
            dice_ET(et_pred, et_true)

            hd95_WT(wt_pred, wt_true)
            hd95_TC(tc_pred, tc_true)
            hd95_ET(et_pred, et_true)

    # aggregate metrics
    dice_wt = dice_WT.aggregate().item()
    dice_tc = dice_TC.aggregate().item()
    dice_et = dice_ET.aggregate().item()

    hd95_wt = hd95_WT.aggregate().item()
    hd95_tc = hd95_TC.aggregate().item()
    hd95_et = hd95_ET.aggregate().item()

    mean_dice = (dice_wt + dice_tc + dice_et) / 3
    mean_hd95 = (hd95_wt + hd95_tc + hd95_et) / 3

    # print metrics
    print(f"dice - WT: {dice_wt:.4f}, TC: {dice_tc:.4f}, ET: {dice_et:.4f}, mean: {mean_dice:.4f}")
    print(f"hd95 - WT: {hd95_wt:.4f}, TC: {hd95_tc:.4f}, ET: {hd95_et:.4f}, mean: {mean_hd95:.4f}")

    # log metrics
    mlflow.log_metrics(
        {
            "dice_WT": dice_wt,
            "dice_TC": dice_tc,
            "dice_ET": dice_et,
            "mean_dice": mean_dice,
            "hd95_WT": hd95_wt,
            "hd95_TC": hd95_tc,
            "hd95_ET": hd95_et,
            "mean_hd95": mean_hd95,
        }
    )

    mlflow.end_run()

    print("done")

if __name__ == "__main__":
    main()
