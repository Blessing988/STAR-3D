from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect V-DETR TSV columns/classes/scores.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-rows", type=int, default=5)
    args = parser.parse_args()

    scores = []
    classes = Counter()
    first_rows = []
    with args.path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        for idx, row in enumerate(reader):
            if idx < args.max_rows:
                first_rows.append(row)
            if "score" in row and row["score"] not in {"", None}:
                scores.append(float(row["score"]))
            if "class_id" in row and row["class_id"] not in {"", None}:
                classes[str(row["class_id"])] += 1

    scores_sorted = sorted(scores)
    def quantile(q: float):
        if not scores_sorted:
            return None
        pos = min(len(scores_sorted) - 1, max(0, int(round(q * (len(scores_sorted) - 1)))))
        return scores_sorted[pos]

    print(json.dumps({
        "path": str(args.path),
        "fieldnames": fieldnames,
        "rows": len(scores),
        "classes": dict(classes.most_common()),
        "score_min": scores_sorted[0] if scores_sorted else None,
        "score_p25": quantile(0.25),
        "score_p50": quantile(0.50),
        "score_p75": quantile(0.75),
        "score_p90": quantile(0.90),
        "score_p99": quantile(0.99),
        "score_max": scores_sorted[-1] if scores_sorted else None,
        "first_rows": first_rows,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
