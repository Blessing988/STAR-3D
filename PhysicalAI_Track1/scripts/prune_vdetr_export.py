from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def frame_key(slice_id: str) -> tuple[str, int]:
    parts = slice_id.split('_')
    if len(parts) < 5:
        raise ValueError(f'bad slice_id={slice_id!r}')
    scene_name = '_'.join(parts[:-3])
    frame_id = int(parts[-3])
    return scene_name, frame_id


def prune_rows(rows: list[dict[str, str]], *, topk_per_group: int, topk_total: int) -> list[dict[str, str]]:
    if not rows:
        return []
    if topk_per_group > 0:
        grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[int(row['class_id'])].append(row)
        rows = []
        for group_rows in grouped.values():
            rows.extend(sorted(group_rows, key=lambda item: float(item['score']), reverse=True)[:topk_per_group])
    if topk_total > 0 and len(rows) > topk_total:
        rows = sorted(rows, key=lambda item: float(item['score']), reverse=True)[:topk_total]
    return sorted(rows, key=lambda item: (int(item['class_id']), -float(item['score'])))


def flush(writer: csv.DictWriter, rows: list[dict[str, str]], *, topk_per_group: int, topk_total: int) -> int:
    kept = prune_rows(rows, topk_per_group=topk_per_group, topk_total=topk_total)
    for row in kept:
        writer.writerow(row)
    return len(kept)


def run(args: argparse.Namespace) -> dict:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    allowed_groups = {int(x) for x in args.allowed_groups.split(',') if x.strip()}

    total_rows = 0
    score_rows = 0
    kept_rows = 0
    frame_groups = 0
    current_key: tuple[str, int] | None = None
    current_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None

    with args.input.open('r', encoding='utf-8', newline='') as src, args.out.open('w', encoding='utf-8', newline='') as dst:
        reader = csv.DictReader(src, delimiter='\t')
        if reader.fieldnames is None:
            raise ValueError(f'empty TSV: {args.input}')
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(dst, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
        writer.writeheader()

        for row in reader:
            total_rows += 1
            score = float(row['score'])
            group_id = int(row['class_id'])
            if score < args.min_score or group_id not in allowed_groups:
                continue
            score_rows += 1
            key = frame_key(row['slice_id'])
            if current_key is not None and key != current_key:
                kept_rows += flush(
                    writer,
                    current_rows,
                    topk_per_group=args.topk_per_group,
                    topk_total=args.topk_total,
                )
                frame_groups += 1
                current_rows = []
            current_key = key
            current_rows.append(row)

        if current_key is not None:
            kept_rows += flush(
                writer,
                current_rows,
                topk_per_group=args.topk_per_group,
                topk_total=args.topk_total,
            )
            frame_groups += 1

    stats = {
        'input': str(args.input),
        'out': str(args.out),
        'fieldnames': fieldnames,
        'total_rows': total_rows,
        'rows_after_score_group_filter': score_rows,
        'kept_rows': kept_rows,
        'frame_groups': frame_groups,
        'min_score': args.min_score,
        'allowed_groups': sorted(allowed_groups),
        'topk_per_group': args.topk_per_group,
        'topk_total': args.topk_total,
    }
    args.out.with_suffix(args.out.suffix + '.json').write_text(json.dumps(stats, indent=2, sort_keys=True), encoding='utf-8')
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prune huge V-DETR export TSVs to per-frame top-k candidates.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--min-score', type=float, default=0.025)
    parser.add_argument('--allowed-groups', default='0,1,2')
    parser.add_argument('--topk-per-group', type=int, default=80)
    parser.add_argument('--topk-total', type=int, default=320)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
