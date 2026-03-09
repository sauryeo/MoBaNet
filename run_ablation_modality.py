import csv
import argparse
import sys
from pathlib import Path

_SCRIPT_ARG_PARSER = argparse.ArgumentParser(add_help=False)
_SCRIPT_ARG_PARSER.add_argument("-weights", dest="full_weights", default=None)
_SCRIPT_ARG_PARSER.add_argument("-base_weights", dest="baseline_weights", default=None)
_SCRIPT_ARG_PARSER.add_argument("-eval_modalities", dest="eval_modalities", default=None)
_SCRIPT_ARG_PARSER.add_argument(
    "--only_model",
    choices=["baseline", "full", "both"],
    default="both",
)
SCRIPT_ARGS, CFG_ARGV = _SCRIPT_ARG_PARSER.parse_known_args(sys.argv[1:])
ORIGINAL_ARGV = list(sys.argv)
sys.argv = [sys.argv[0], *CFG_ARGV]

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from skimage import io
from tqdm import tqdm

import Model.cfg as cfg
from MMNet import MMNet as MFNet
from utils import *
from utils import _potsdam_dsm_id

sys.argv = ORIGINAL_ARGV


MODALITY_LABELS = {
    "rgbd": "RGB+DSM",
    "rgb": "RGB-only",
    "dsm": "DSM-only",
}

MODALITY_DIR_NAMES = {
    "rgbd": "rgbd",
    "rgb": "rgb_only",
    "dsm": "dsm_only",
}

MODEL_LABELS = {
    "baseline": "w/o MCRC",
    "full": "Full",
}

MODEL_DIR_NAMES = {
    "baseline": "w_o_mcrc",
    "full": "full",
}

# Default checkpoints used when no CLI weights are provided.
BASELINE_WEIGHTS_PATH = "/data1/lihaocheng/deeplearning/paper1/MoBaSF-dinov2-ablation/resultsp/MMNet_ablation_base_epoch25_0.8494706829369975"
FULL_WEIGHTS_PATH = "/data1/lihaocheng/deeplearning/paper1/MoBaSF-dinov2/resultsp/MMNet_epoch50_0.8748094843563392"

DEFAULT_MODALITIES = ["rgbd", "rgb", "dsm"]
DEFAULT_TEST_STRIDE = 32


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def parse_eval_modalities(value):
    text = "" if value is None else str(value).strip().lower()
    if not text or text == "rgbd":
        return list(DEFAULT_MODALITIES)

    modes = []
    for token in text.split(","):
        mode = token.strip().lower()
        if not mode:
            continue
        if mode not in MODALITY_LABELS:
            raise ValueError(
                f"Unsupported eval modality '{mode}'. Expected one of: {', '.join(MODALITY_LABELS.keys())}"
            )
        if mode not in modes:
            modes.append(mode)
    return modes or list(DEFAULT_MODALITIES)


def compute_metrics_summary(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(gts, predictions, labels=range(len(label_values)))

    total = int(np.sum(cm))
    diag = np.diag(cm).astype(np.float64)
    row_sum = np.sum(cm, axis=1).astype(np.float64)
    col_sum = np.sum(cm, axis=0).astype(np.float64)
    summary_count = min(5, len(label_values))

    oa = 100.0 * float(np.sum(diag)) / float(total) if total > 0 else 0.0
    f1_denom = row_sum + col_sum
    f1 = np.divide(2.0 * diag, f1_denom, out=np.zeros_like(diag), where=f1_denom != 0)
    iou_denom = row_sum + col_sum - diag
    iou = np.divide(diag, iou_denom, out=np.zeros_like(diag), where=iou_denom != 0)

    return {
        "oa": oa,
        "mf1": 100.0 * float(np.nanmean(f1[:summary_count])) if summary_count > 0 else 0.0,
        "miou": 100.0 * float(np.nanmean(iou[:summary_count])) if summary_count > 0 else 0.0,
        "cm": cm,
    }


def print_modality_table(results):
    print("\nTable-ready results (%):")
    print("Modality\tOA\tmF1\tmIoU")
    for mode, summary in results:
        print(
            f"{MODALITY_LABELS[mode]}\t{summary['oa']:.2f}\t{summary['mf1']:.2f}\t{summary['miou']:.2f}"
        )

    print("\nLaTeX rows:")
    for mode, summary in results:
        print(
            f"{MODALITY_LABELS[mode]} & {summary['oa']:.2f} & {summary['mf1']:.2f} & {summary['miou']:.2f} \\\\"
        )

    baseline = next((summary for mode, summary in results if mode == "rgbd"), None)
    if baseline is not None:
        print("\nDrops vs RGB+DSM (mIoU points):")
        for mode, summary in results:
            if mode == "rgbd":
                continue
            delta = baseline["miou"] - summary["miou"]
            print(f"{MODALITY_LABELS[mode]}: -{delta:.2f}")


def save_summary_files(output_root, results):
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Modality", "OA", "mF1", "mIoU"])
        for mode, summary in results:
            writer.writerow(
                [
                    MODALITY_LABELS[mode],
                    f"{summary['oa']:.2f}",
                    f"{summary['mf1']:.2f}",
                    f"{summary['miou']:.2f}",
                ]
            )

    latex_path = output_root / "summary_latex.txt"
    with latex_path.open("w", encoding="utf-8") as f:
        for mode, summary in results:
            f.write(
                f"{MODALITY_LABELS[mode]} & {summary['oa']:.2f} & {summary['mf1']:.2f} & {summary['miou']:.2f} \\\\\n"
            )


def checkpoint_tag(weights_path):
    return Path(str(weights_path)).name


def resolve_model_specs(script_args):
    baseline_weights = BASELINE_WEIGHTS_PATH
    full_weights = FULL_WEIGHTS_PATH

    if script_args.baseline_weights:
        baseline_weights = script_args.baseline_weights
    if script_args.full_weights:
        full_weights = script_args.full_weights

    if not baseline_weights or str(baseline_weights) == "0":
        raise ValueError("Please set BASELINE_WEIGHTS_PATH in run_ablation_modality.py.")
    if not full_weights or str(full_weights) == "0":
        raise ValueError("Please set FULL_WEIGHTS_PATH in run_ablation_modality.py.")
    all_specs = [
        ("baseline", baseline_weights),
        ("full", full_weights),
    ]
    if script_args.only_model == "baseline":
        return [all_specs[0]]
    if script_args.only_model == "full":
        return [all_specs[1]]
    return all_specs


def print_checkpoint_comparison(model_results, modality_modes):
    baseline_results = model_results["baseline"]
    full_results = model_results["full"]

    print("\nDirect comparison (%):")
    print("Modality\tw/o MCRC mIoU\tFull mIoU\tDelta")
    for mode in modality_modes:
        baseline_summary = baseline_results[mode]
        full_summary = full_results[mode]
        delta = full_summary["miou"] - baseline_summary["miou"]
        print(
            f"{MODALITY_LABELS[mode]}\t"
            f"{baseline_summary['miou']:.2f}\t"
            f"{full_summary['miou']:.2f}\t"
            f"{delta:+.2f}"
        )


def save_comparison_files(output_root, model_results, modality_modes):
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Modality",
                "w/o MCRC OA",
                "w/o MCRC mF1",
                "w/o MCRC mIoU",
                "Full OA",
                "Full mF1",
                "Full mIoU",
                "Delta OA",
                "Delta mF1",
                "Delta mIoU",
            ]
        )
        for mode in modality_modes:
            baseline_summary = model_results["baseline"][mode]
            full_summary = model_results["full"][mode]
            writer.writerow(
                [
                    MODALITY_LABELS[mode],
                    f"{baseline_summary['oa']:.2f}",
                    f"{baseline_summary['mf1']:.2f}",
                    f"{baseline_summary['miou']:.2f}",
                    f"{full_summary['oa']:.2f}",
                    f"{full_summary['mf1']:.2f}",
                    f"{full_summary['miou']:.2f}",
                    f"{full_summary['oa'] - baseline_summary['oa']:+.2f}",
                    f"{full_summary['mf1'] - baseline_summary['mf1']:+.2f}",
                    f"{full_summary['miou'] - baseline_summary['miou']:+.2f}",
                ]
            )

    latex_path = output_root / "comparison_latex.txt"
    with latex_path.open("w", encoding="utf-8") as f:
        for mode in modality_modes:
            baseline_summary = model_results["baseline"][mode]
            full_summary = model_results["full"][mode]
            f.write(
                f"{MODALITY_LABELS[mode]} & "
                f"{baseline_summary['oa']:.2f} & {baseline_summary['mf1']:.2f} & {baseline_summary['miou']:.2f} & "
                f"{full_summary['oa']:.2f} & {full_summary['mf1']:.2f} & {full_summary['miou']:.2f} \\\\\n"
            )


def load_model(weights_path):
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0], *CFG_ARGV]
    try:
        net = MFNet(num_classes=N_CLASSES).cuda()
    finally:
        sys.argv = original_argv
    state_dict = torch.load(weights_path, map_location="cpu")
    incompatible = net.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    print(
        f"Loaded checkpoint: {weights_path}\n"
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("Using DataParallel with {} GPUs".format(torch.cuda.device_count()))

    net.eval()
    return net


def iter_test_data(ids):
    if DATASET == "Potsdam":
        test_images = (
            normalize_rgb(1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id_))[:, :, :3], dtype="float32"))
            for id_ in ids
        )
        test_dsms = (
            np.asarray(io.imread(DSM_FOLDER.format(_potsdam_dsm_id(id_))), dtype="float32")
            for id_ in ids
        )
    else:
        test_images = (
            normalize_rgb(1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id_)), dtype="float32"))
            for id_ in ids
        )
        test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id_)), dtype="float32") for id_ in ids)

    eval_labels = (convert_from_color(io.imread(LABEL_FOLDER.format(id_))) for id_ in ids)
    return zip(ids, test_images, test_dsms, eval_labels)


def run_test_mode(net, ids, modality_mode, stride=DEFAULT_TEST_STRIDE, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for tile_id, img, dsm, gt_e in tqdm(iter_test_data(ids), total=len(ids), leave=False):
            if dsm.shape[:2] != img.shape[:2]:
                dsm = match_spatial_shape(dsm, img.shape)

            dsm_min = np.min(dsm)
            dsm_max = np.max(dsm)
            if dsm_max > dsm_min:
                dsm = (dsm - dsm_min) / (dsm_max - dsm_min)
            else:
                dsm = np.zeros_like(dsm)

            pred = np.zeros(img.shape[:2] + (N_CLASSES,))
            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size

            for coords in tqdm(
                grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)),
                total=total,
                leave=False,
            ):
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)

                if modality_mode == "rgb":
                    dsm_patches = np.zeros_like(dsm_patches)
                elif modality_mode == "dsm":
                    image_patches = np.zeros_like(image_patches)

                image_patches, dsm_patches, pad_hw = pad_batch_to_multiple(image_patches, dsm_patches)
                image_tensor = torch.from_numpy(image_patches).cuda()
                dsm_tensor = torch.from_numpy(dsm_patches).cuda()

                outs = net(image_tensor, dsm_tensor, mode="Test")
                if isinstance(outs, (tuple, list)):
                    outs = outs[0]
                outs = outs.data.cpu().numpy()
                if pad_hw != (0, 0):
                    outs = outs[:, :, :window_size[0], :window_size[1]]

                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)

    summary = compute_metrics_summary(
        np.concatenate([p.ravel() for p in all_preds]),
        np.concatenate([p.ravel() for p in all_gts]).ravel(),
    )
    return summary, all_preds


def save_predictions(preds, ids, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for pred, tile_id in zip(preds, ids):
        img = convert_to_color(pred)
        io.imsave(str(output_dir / f"tile_{tile_id}.png"), img)


def main():
    script_args = SCRIPT_ARGS
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0], *CFG_ARGV]
    try:
        cfg_args = cfg.parse_args()
    finally:
        sys.argv = original_argv

    model_specs = resolve_model_specs(script_args)
    eval_modalities_arg = (
        script_args.eval_modalities
        if script_args.eval_modalities is not None
        else getattr(cfg_args, "eval_modalities", None)
    )
    modality_modes = parse_eval_modalities(eval_modalities_arg)
    print(
        f"Running modality ablation on {DATASET} with modes: {', '.join(MODALITY_LABELS[m] for m in modality_modes)}"
    )
    print(
        f"ENCODER: {ENCODER}, WINDOW_SIZE: {WINDOW_SIZE}, STRIDE: {DEFAULT_TEST_STRIDE}, "
        f"BATCH_SIZE: {BATCH_SIZE}"
    )

    root_dir = Path("./resultsp_modality_compare" if DATASET == "Potsdam" else "./resultsv_modality_compare")
    if len(model_specs) == 2:
        run_dir = root_dir / f"{checkpoint_tag(model_specs[0][1])}__vs__{checkpoint_tag(model_specs[1][1])}"
    else:
        run_dir = root_dir / checkpoint_tag(model_specs[0][1])
    model_results = {}

    for model_key, weights_path in model_specs:
        print(f"\n##### {MODEL_LABELS[model_key]} #####")
        net = load_model(weights_path)
        results = []
        model_results[model_key] = {}
        model_dir = run_dir / MODEL_DIR_NAMES[model_key]

        for modality_mode in modality_modes:
            print(f"\n=== {MODEL_LABELS[model_key]} | {MODALITY_LABELS[modality_mode]} ===")
            summary, preds = run_test_mode(net, test_ids, modality_mode, stride=DEFAULT_TEST_STRIDE)
            mode_dir = model_dir / MODALITY_DIR_NAMES[modality_mode]
            save_predictions(preds, test_ids, mode_dir)
            results.append((modality_mode, summary))
            model_results[model_key][modality_mode] = summary

        print_modality_table(results)
        save_summary_files(model_dir, results)
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(model_specs) == 2:
        print_checkpoint_comparison(model_results, modality_modes)
        save_comparison_files(run_dir, model_results, modality_modes)
    print(f"\nSaved comparison results to {run_dir}")


if __name__ == "__main__":
    main()
