import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Glyde reproducible research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--raw", default="data/raw")
    prep.add_argument("--output", default="data/prepared")
    embed = sub.add_parser("embed")
    embed.add_argument("--prepared", default="data/prepared")
    train = sub.add_parser("train")
    train.add_argument("--prepared", default="data/prepared")
    train.add_argument("--output", default="artifacts/multimodal")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--profile", choices=["quick", "scaled"], default="quick")
    train.add_argument("--max-minutes", type=float, default=45)
    train.add_argument("--diabetes-flag", action="store_true")
    status = sub.add_parser("train-status")
    status.add_argument("--prepared", default="data/prepared")
    status.add_argument("--output", default="artifacts/status")
    status.add_argument("--seed", type=int, default=42)
    fast = sub.add_parser("train-fast")
    fast.add_argument("--prepared", default="data/prepared")
    fast.add_argument("--output", default="artifacts/fast")
    fast.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare
        result = prepare(args.raw, args.output)
    elif args.command == "embed":
        from .embeddings import embed_meals
        import torch
        torch.set_num_threads(4)
        embed_meals(args.prepared)
        return
    elif args.command == "train-fast":
        from .fast import train_fast
        result = train_fast(args.prepared, args.output, args.seed)
    elif args.command == "train-status":
        from .status import train_status
        result = train_status(args.prepared, args.output, args.seed)
    else:
        from .experiment import run
        if args.epochs < 1 or args.max_minutes <= 0:
            parser.error("epochs and max-minutes must be positive")
        if args.diabetes_flag and args.output == "artifacts/multimodal":
            args.output = "artifacts/multimodal-status"
        result = run(args.prepared, args.output, args.epochs, args.seed, args.profile, args.max_minutes, args.diabetes_flag)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
