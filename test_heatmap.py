import os
import copy

import cv2
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

import Model.cfg as cfg
from Net_heatmap import Net
from utils import *
from utils import _potsdam_dsm_id

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_X = 65
TARGET_Y = 100
# Target class index for Grad-CAM-style heatmap generation.
# 0: roads
# 1: buildings
# 2: low veg.
# 3: trees
# 4: cars
# 5: clutter
TARGET_CLASS = 1
OUTPUT_DIR = os.path.join("heatmap_compare", DATASET.lower())
OUTPUT_DPI = 300
BASE_WEIGHTS_PATH = "/data1/lihaocheng/deeplearning/paper1/MoBaSF-dinov2-ablation/resultsp/Net_ablation_base_epoch25_0.8494706829369975"
FULL_WEIGHTS_PATH = "/data1/lihaocheng/deeplearning/paper1/MoBaSF-dinov2/resultsp/Net_epoch50_0.8748094843563392"
MAX_EXPORTS = None
SHOW_TARGET_MARKER = False
HEATMAP_STRIDE = 256
HEATMAP_WINDOW_SIZE = (512, 512)
DISPLAY_SIZE = 512
BASE_USE_CPIA = False
BASE_USE_BAGF = False
BASE_USE_MCRC = False
FULL_USE_CPIA = True
FULL_USE_BAGF = True
FULL_USE_MCRC = True

os.environ["HEATMAP_TARGET_X"] = str(TARGET_X)
os.environ["HEATMAP_TARGET_Y"] = str(TARGET_Y)
os.environ["HEATMAP_TARGET_CLASS"] = str(TARGET_CLASS)


def normalize_dsm(dsm):
    if dsm.shape[:2] == (0, 0):
        return dsm
    dsm_min = float(np.min(dsm))
    dsm_max = float(np.max(dsm))
    if dsm_max > dsm_min:
        return (dsm - dsm_min) / (dsm_max - dsm_min)
    return np.zeros_like(dsm, dtype=np.float32)


def to_display_rgb(image_patch):
    image_patch = np.asarray(np.clip(image_patch * 255.0, 0, 255), dtype=np.uint8)
    return cv2.resize(image_patch, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_LINEAR)


def to_display_dsm(dsm_patch):
    dsm_patch = np.asarray(np.clip(dsm_patch * 255.0, 0, 255), dtype=np.uint8)
    return cv2.resize(dsm_patch, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)


def combine_branch_heatmaps(heatmap_rgb, heatmap_dsm):
    combined = 0.5 * (np.nan_to_num(heatmap_rgb) + np.nan_to_num(heatmap_dsm))
    combined = np.maximum(combined, 0)
    peak = float(np.max(combined))
    if peak > 0:
        combined = combined / peak
    return combined


def get_padded_window_size(window_size):
    pad_h = (PATCH_MULTIPLE - window_size[0] % PATCH_MULTIPLE) % PATCH_MULTIPLE
    pad_w = (PATCH_MULTIPLE - window_size[1] % PATCH_MULTIPLE) % PATCH_MULTIPLE
    return window_size[0] + pad_h, window_size[1] + pad_w


def heatmap_to_rgb(heatmap, output_size=(DISPLAY_SIZE, DISPLAY_SIZE)):
    heatmap = cv2.resize(heatmap, output_size[::-1])
    heatmap = np.asarray(np.clip(heatmap * 255.0, 0, 255), dtype=np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap[:, :, (2, 1, 0)]


def build_runtime_args(use_cpia, use_bagf, mcrc, window_size):
    args = copy.deepcopy(cfg.parse_args())
    args.use_cpia = use_cpia
    args.use_bagf = use_bagf
    args.mcrc = mcrc
    padded_h, padded_w = get_padded_window_size(window_size)
    if padded_h != padded_w:
        raise ValueError("DINOv2 heatmap export expects a square window size after padding.")
    args.image_size = padded_h
    args.out_size = window_size[0]
    return args


def adapt_state_dict_for_window(state_dict, model_state_dict, label):
    adapted_state = {}
    resized_keys = []
    dropped_keys = []

    for key, value in state_dict.items():
        if key not in model_state_dict:
            dropped_keys.append((key, "unexpected"))
            continue

        target_value = model_state_dict[key]
        if value.shape == target_value.shape:
            adapted_state[key] = value
            continue

        if key.endswith("rpe_table") and value.ndim == 3 and target_value.ndim == 3:
            resized = F.interpolate(
                value.unsqueeze(0),
                size=target_value.shape[-2:],
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
            adapted_state[key] = resized.to(dtype=target_value.dtype)
            resized_keys.append((key, tuple(value.shape), tuple(target_value.shape)))
            continue

        dropped_keys.append((key, f"shape {tuple(value.shape)} -> {tuple(target_value.shape)}"))

    if resized_keys:
        print(f"[{label}] resized keys:")
        for key, src_shape, dst_shape in resized_keys:
            print(f"  - {key}: {src_shape} -> {dst_shape}")
    if dropped_keys:
        print(f"[{label}] dropped keys: {len(dropped_keys)}")
        for key, reason in dropped_keys[:10]:
            print(f"  - {key}: {reason}")
        if len(dropped_keys) > 10:
            print(f"  ... and {len(dropped_keys) - 10} more")

    return adapted_state


def load_model(weights_path, label, runtime_args):
    if not weights_path or str(weights_path) == "0":
        raise ValueError(f"{label} checkpoint is required. Please set {label.upper()}_WEIGHTS_PATH in test_heatmap.py.")
    net = Net(num_classes=N_CLASSES, runtime_args=runtime_args).cuda()
    state_dict = torch.load(weights_path, map_location="cpu")
    model_state_dict = net.state_dict()
    state_dict = adapt_state_dict_for_window(state_dict, model_state_dict, label)
    incompatible = net.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    print(
        f"[{label}] checkpoint={weights_path}\n"
        f"[{label}] use_cpia={runtime_args.use_cpia}, use_bagf={runtime_args.use_bagf}, mcrc={runtime_args.mcrc}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    net.eval()
    return net


def read_test_tile(tile_id):
    if DATASET == "Potsdam":
        image = 1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(tile_id))[:, :, :3], dtype="float32")
        dsm_id = _potsdam_dsm_id(tile_id)
        dsm = np.asarray(io.imread(DSM_FOLDER.format(dsm_id)), dtype="float32")
    else:
        image = 1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(tile_id)), dtype="float32")
        dsm = np.asarray(io.imread(DSM_FOLDER.format(tile_id)), dtype="float32")
    if dsm.shape[:2] != image.shape[:2]:
        dsm = match_spatial_shape(dsm, image.shape)
    return image, normalize_rgb(image), normalize_dsm(dsm)


def build_patch_tensors(image_norm, dsm_norm, coords):
    x, y, w, h = coords
    image_patch = np.copy(image_norm[x:x + w, y:y + h]).transpose((2, 0, 1))[None, ...]
    dsm_patch = np.copy(dsm_norm[x:x + w, y:y + h])[None, ...]
    image_patch, dsm_patch, pad_hw = pad_batch_to_multiple(image_patch, dsm_patch)
    image_tensor = torch.from_numpy(image_patch).cuda()
    dsm_tensor = torch.from_numpy(dsm_patch).cuda()
    return image_tensor, dsm_tensor, pad_hw


def infer_combined_heatmap(net, image_tensor, dsm_tensor):
    net.zero_grad(set_to_none=True)
    _, heatmap_rgb, heatmap_dsm = net(image_tensor, dsm_tensor, mode="Test")
    return combine_branch_heatmaps(heatmap_rgb, heatmap_dsm)


def add_focus_box(axis):
    axis.add_patch(
        plt.Rectangle(
            (TARGET_X - 2, TARGET_Y - 2),
            2,
            2,
            color="red",
            fill=False,
            linewidth=1,
        )
    )


def save_comparison(tile_id, patch_index, rgb_patch, dsm_patch, base_heatmap, full_heatmap, coords, output_dir):
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    panels = [
        (rgb_patch, None),
        (dsm_patch, "gray"),
        (base_heatmap, None),
        (full_heatmap, None),
    ]
    for axis, (panel, cmap) in zip(axes, panels):
        axis.imshow(panel, cmap=cmap)
        axis.axis("off")
        if SHOW_TARGET_MARKER:
            add_focus_box(axis)
    figure.tight_layout(pad=0.6)
    x, y, _, _ = coords
    save_name = f"{DATASET.lower()}_{tile_id}_patch{patch_index:04d}_x{x}_y{y}.png"
    save_path = os.path.join(output_dir, save_name)
    figure.savefig(save_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(figure)
    return save_path


def export_heatmap_comparisons(base_net, full_net, tile_ids, stride, batch_size=1, window_size=HEATMAP_WINDOW_SIZE, max_exports=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    export_count = 0
    saved_paths = []

    for tile_id in tqdm(tile_ids, total=len(tile_ids), leave=False):
        image_raw, image_norm, dsm_norm = read_test_tile(tile_id)
        patch_total = count_sliding_window(image_raw, step=stride, window_size=window_size) // batch_size

        for coords_group in tqdm(
            grouper(batch_size, sliding_window(image_raw, step=stride, window_size=window_size)),
            total=patch_total,
            leave=False,
        ):
            coords = coords_group[0]
            x, y, w, h = coords
            rgb_patch = to_display_rgb(image_raw[x:x + w, y:y + h])
            dsm_patch = to_display_dsm(dsm_norm[x:x + w, y:y + h])
            image_tensor, dsm_tensor, _ = build_patch_tensors(image_norm, dsm_norm, coords)

            with torch.enable_grad():
                base_heatmap = infer_combined_heatmap(base_net, image_tensor, dsm_tensor)
                full_heatmap = infer_combined_heatmap(full_net, image_tensor, dsm_tensor)

            base_panel = heatmap_to_rgb(base_heatmap)
            full_panel = heatmap_to_rgb(full_heatmap)
            save_path = save_comparison(
                tile_id=tile_id,
                patch_index=export_count,
                rgb_patch=rgb_patch,
                dsm_patch=dsm_patch,
                base_heatmap=base_panel,
                full_heatmap=full_panel,
                coords=coords,
                output_dir=OUTPUT_DIR,
            )
            saved_paths.append(save_path)
            export_count += 1

            if max_exports is not None and export_count >= max_exports:
                return saved_paths

    return saved_paths


def main():
    print(
        f"Heatmap window size: {HEATMAP_WINDOW_SIZE}, stride: {HEATMAP_STRIDE}, "
        f"padded input: {get_padded_window_size(HEATMAP_WINDOW_SIZE)}"
    )
    base_args = build_runtime_args(BASE_USE_CPIA, BASE_USE_BAGF, BASE_USE_MCRC, HEATMAP_WINDOW_SIZE)
    full_args = build_runtime_args(FULL_USE_CPIA, FULL_USE_BAGF, FULL_USE_MCRC, HEATMAP_WINDOW_SIZE)
    base_net = load_model(BASE_WEIGHTS_PATH, "Base", base_args)
    full_net = load_model(FULL_WEIGHTS_PATH, "Full", full_args)

    saved_paths = export_heatmap_comparisons(
        base_net=base_net,
        full_net=full_net,
        tile_ids=test_ids,
        stride=HEATMAP_STRIDE,
        batch_size=1,
        window_size=HEATMAP_WINDOW_SIZE,
        max_exports=MAX_EXPORTS,
    )
    print(f"Saved {len(saved_paths)} comparison PNG files to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
