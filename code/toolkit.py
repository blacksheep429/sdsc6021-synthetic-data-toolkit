"""Single project CLI for environment, generation and G2 development verification."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from a_protocol import ROOT, atomic_json, candidate_plan, config_hash, inside, load_config, training_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "plan", "generate", "smoke", "freeze", "select", "features", "train", "evaluate", "summarize"])
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--models", action="store_true", help="doctor: download/load SD1.5, CLIP and ResNet, and test inference")
    parser.add_argument("--smoke", action="store_true", help="generate: isolated small generation check, not the G2 full-pipeline smoke")
    parser.add_argument("--limit", type=int, default=1, help="generate --smoke: maximum 10")
    parser.add_argument("--per-class", type=int, help="generate --smoke: balanced 1-3 images per class")
    parser.add_argument("--run-dir", help="freeze/summarize: a completed local G2 smoke directory")
    parser.add_argument("--final", action="store_true", help="reserved for the final evaluation gate")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "plan":
            fingerprint = config_hash(config)
            plan = {"schema_version": 1, "config_hash": fingerprint, "status": "planned",
                    "protocol_status": config["protocol_status"], "config": config,
                    "candidates": candidate_plan(config), "training_runs": training_plan(config)}
            path = ROOT / "runs/plans" / fingerprint / "plan.json"
            atomic_json(path, plan)
            result = {"status": "planned", "candidate_images": len(plan["candidates"]),
                      "classifier_runs": len(plan["training_runs"]), "config_hash": fingerprint,
                      "plan": str(path.relative_to(ROOT)), "gpu_used": False, "models_downloaded": False}
        elif args.command in {"doctor", "generate"}:
            from a_runtime import doctor, generate
            if args.per_class is not None and not args.smoke:
                parser.error("--per-class requires generate --smoke")
            result = doctor(config, models=args.models) if args.command == "doctor" else generate(config, smoke=args.smoke, limit=args.limit,per_class=args.per_class)
        elif args.command == "smoke":
            from smoke import smoke
            result = smoke(config)
        elif args.command in {"freeze","summarize"}:
            if not args.run_dir:
                parser.error("--run-dir is required")
            from smoke import freeze, export_smoke
            directory = inside(ROOT,args.run_dir)
            config_path = inside(ROOT,args.config)
            result = freeze(config,config_path,directory) if args.command == "freeze" else export_smoke(config,directory)
        else:
            parser.error(f"{args.command} formal-stage integration is pending; use smoke for G2 development verification.")
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        atomic_json(ROOT / "runs/failures" / f"{stamp}.json", {
            "status": "failed", "command": args.command, "config": args.config,
            "error_type": type(exc).__name__, "error": str(exc)})
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
