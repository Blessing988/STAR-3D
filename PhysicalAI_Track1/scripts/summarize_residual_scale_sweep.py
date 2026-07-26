from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    results = []
    for eval_path in sorted(run_root.glob("*/eval.json")):
        if eval_path.stat().st_size == 0:
            continue
        metrics = json.loads(eval_path.read_text(encoding="utf-8"))
        results.append(
            {
                "run": eval_path.parent.name,
                "eval": str(eval_path),
                "hota_like": float(metrics["hota_like"]),
                "deta": float(metrics["deta"]),
                "assa": float(metrics["assa"]),
                "loca": float(metrics["loca"]),
            }
        )
    if not results:
        raise ValueError(f"No completed eval.json files found under {run_root}")
    results.sort(key=lambda item: item["hota_like"], reverse=True)
    payload = {"best": results[0], "trials": len(results), "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
