from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader, SequentialSampler

from datasets import build_dataset
from models import build_model
from utils.ap_calculator import get_ap_config_dict, parse_predictions
from utils.dist import batch_dict_to_cuda
from utils.misc import my_worker_init_fn


def load_args(config_path: Path, dataset_root: Path, checkpoint: Path, workers: int, conf: float) -> SimpleNamespace:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["test_only"] = True
    cfg["auto_test"] = False
    cfg["test_ckpt"] = str(checkpoint)
    cfg["dataset_root_dir"] = str(dataset_root)
    cfg["dataset_num_workers"] = workers
    cfg["ngpus"] = 1
    cfg["wandb_activate"] = False
    cfg["conf_thresh"] = conf
    return SimpleNamespace(**cfg)


def corners_to_axis_box(corners) -> tuple[float, float, float, float, float, float, float]:
    mins = corners.min(axis=0)
    maxs = corners.max(axis=0)
    center = (mins + maxs) * 0.5
    size = maxs - mins
    return (
        float(center[0]),
        float(center[1]),
        float(center[2]),
        float(size[0]),
        float(size[1]),
        float(size[2]),
        0.0,
    )


@torch.no_grad()
def export_predictions(args: SimpleNamespace, out_tsv: Path) -> None:
    torch.cuda.set_device(0)
    datasets, dataset_config = build_dataset(args)
    dataset = datasets["test"]
    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=args.batchsize_per_gpu,
        num_workers=args.dataset_num_workers,
        worker_init_fn=my_worker_init_fn,
        collate_fn=dataset.collate_fn,
    )

    model = build_model(args, dataset_config)
    checkpoint = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_state = model.state_dict()
    filtered = {
        key: value
        for key, value in checkpoint["model"].items()
        if key in model_state and value.shape == model_state[key].shape
    }
    model.load_state_dict(filtered, strict=False)
    model.cuda(0)
    model.eval()

    ap_config = get_ap_config_dict(
        dataset_config=dataset_config,
        remove_empty_box=False,
        no_nms=args.test_no_nms,
        use_3d_nms=not args.no_3d_nms,
        nms_iou=args.nms_iou,
        empty_pt_thre=args.empty_pt_thre,
        conf_thresh=args.conf_thresh,
        rotated_nms=args.rotated_nms,
        angle_nms=args.angle_nms,
        angle_conf=args.angle_conf,
        use_old_type_nms=args.use_old_type_nms,
        cls_nms=not args.no_cls_nms,
        per_class_proposal=not args.no_per_class_proposal,
        use_cls_confidence_only=args.use_cls_confidence_only,
    )

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["slice_id", "class_id", "score", "x", "y", "z", "width", "length", "height", "yaw"])
        for batch_idx, batch in enumerate(loader):
            file_names = batch["file_names"]
            batch = batch_dict_to_cuda(batch, local_rank=torch.device("cuda:0"))
            outputs = model(
                {
                    "point_clouds": batch["point_clouds"],
                    "point_cloud_dims_min": batch["point_cloud_dims_min"],
                    "point_cloud_dims_max": batch["point_cloud_dims_max"],
                }
            )["outputs"]
            if args.cls_loss.split("_")[0] == "focalloss":
                outputs["sem_cls_prob"] = outputs["sem_cls_prob"].sigmoid()
            predicted_box_csa = torch.cat(
                (
                    outputs["center_unnormalized"].detach(),
                    outputs["size_unnormalized"].detach(),
                    outputs["angle_continuous"].detach().unsqueeze(-1),
                ),
                dim=-1,
            )
            pred_lists = parse_predictions(
                outputs["box_corners"],
                outputs["sem_cls_prob"],
                outputs["objectness_prob"],
                outputs["angle_prob"],
                batch["point_clouds"],
                ap_config,
                predicted_box_csa,
            )
            for file_name, pred_list in zip(file_names, pred_lists):
                slice_id = str(file_name).replace(".ply", "")
                for class_id, corners, score in pred_list:
                    x, y, z, width, length, height, yaw = corners_to_axis_box(corners)
                    writer.writerow(
                        [
                            slice_id,
                            int(class_id),
                            f"{float(score):.6f}",
                            f"{x:.6f}",
                            f"{y:.6f}",
                            f"{z:.6f}",
                            f"{width:.6f}",
                            f"{length:.6f}",
                            f"{height:.6f}",
                            f"{yaw:.6f}",
                        ]
                    )
            if batch_idx % 100 == 0:
                print(f"exported_batch={batch_idx}/{len(loader)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export V-DETR per-slice predictions to TSV.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    args = load_args(cli.config, cli.dataset_root, cli.checkpoint, cli.workers, cli.conf)
    export_predictions(args, cli.out_tsv)


if __name__ == "__main__":
    main()
