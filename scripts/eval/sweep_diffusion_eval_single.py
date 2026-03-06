"""Sweep evaluation across multiple Diffusion Policy checkpoints using single-env eval.

Runs eval_diffusion_policy.py sequentially for each checkpoint,
collects results into a summary table and JSON.

Use this instead of sweep_diffusion_eval.py when parallel eval has issues.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def discover_checkpoints(base_dir: Path) -> list[int]:
    """Find all checkpoint_NNNNN directories, return sorted step numbers."""
    steps = []
    for d in base_dir.iterdir():
        if d.is_dir():
            m = re.match(r"checkpoint_(\d+)$", d.name)
            if m:
                steps.append(int(m.group(1)))
    return sorted(steps)


def validate_checkpoint(base_dir: Path, label: str) -> str | None:
    """Check if checkpoint has model files. Returns error message or None."""
    if label == "best":
        path = base_dir / "best"
    elif label == "latest":
        path = base_dir
    else:
        path = base_dir / label
    if not path.is_dir():
        return f"directory not found: {path}"
    if not (path / "model.safetensors").exists():
        return f"model.safetensors not found in {path}"
    return None


def build_eval_command(
    args: argparse.Namespace,
    checkpoint_label: str,
    output_dir: Path,
) -> list[str]:
    """Build subprocess command for a single checkpoint eval."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "eval_diffusion_policy.py"),
        "--checkpoint", str(args.checkpoint),
        "--output", str(output_dir),
        "--episodes", str(args.episodes),
        "--max-steps", str(args.max_steps),
        "--env", str(args.env),
        "--physics-hz", str(args.physics_hz),
        "--policy-hz", str(args.policy_hz),
        "--render-hz", str(args.render_hz),
    ]

    # Checkpoint selection
    if checkpoint_label == "best":
        cmd.append("--use-best")
    elif checkpoint_label == "latest":
        cmd.append("--use-latest")
    else:
        step = int(checkpoint_label.replace("checkpoint_", ""))
        cmd.extend(["--step", str(step)])

    # Optional args
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.n_action_steps is not None:
        cmd.extend(["--n-action-steps", str(args.n_action_steps)])
    if args.num_inference_steps is not None:
        cmd.extend(["--num-inference-steps", str(args.num_inference_steps)])
    if args.randomize_light:
        cmd.append("--randomize-light")

    return cmd


def extract_step(label: str) -> int | None:
    m = re.match(r"checkpoint_(\d+)$", label)
    return int(m.group(1)) if m else None


def print_table(results: list[dict]) -> None:
    headers = ["Checkpoint", "Episodes", "Success%", "Avg Steps", "Avg Steps(ok)", "Avg Steps(fail)", "Eval Time(s)"]
    widths = [20, 8, 10, 10, 14, 15, 12]

    def fmt_val(v, w):
        if v is None:
            return "-".center(w)
        if isinstance(v, float):
            return f"{v:.1f}".rjust(w)
        return str(v).ljust(w)

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_line = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, widths)) + "|"

    print(sep)
    print(header_line)
    print(sep)

    for r in results:
        if r.get("status") == "SKIPPED":
            row = [r["checkpoint"].ljust(widths[0]), "".center(widths[1]), "SKIPPED".center(widths[2])]
            row += ["".center(w) for w in widths[3:]]
        elif r.get("error"):
            row = [r["checkpoint"].ljust(widths[0]), "".center(widths[1]), "ERROR".center(widths[2])]
            row += ["".center(w) for w in widths[3:]]
        else:
            row = [
                fmt_val(r["checkpoint"], widths[0]),
                fmt_val(r.get("total_episodes"), widths[1]),
                fmt_val(r.get("success_rate"), widths[2]),
                fmt_val(r.get("avg_steps"), widths[3]),
                fmt_val(r.get("avg_steps_success"), widths[4]),
                fmt_val(r.get("avg_steps_fail"), widths[5]),
                fmt_val(r.get("total_eval_time_s", r.get("eval_time_s")), widths[6]),
            ]
        print("|" + "|".join(f" {c} " for c in row) + "|")

    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Diffusion Policy checkpoints (single-env eval)",
    )

    # Sweep selection
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to training output directory")
    parser.add_argument("--best", action="store_true",
                        help="Evaluate best/ checkpoint")
    parser.add_argument("--latest", action="store_true",
                        help="Evaluate latest checkpoint (root dir)")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[],
                        help="Specific step numbers to evaluate")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all checkpoint_* dirs + best")

    # Eval parameters
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--env", type=str, default="figure_shape_placement")
    parser.add_argument("--seed", type=str, default=None,
                        help="Shared seed (default: auto). Use 'random' for independent seeds")
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Override diffusion denoising steps (default: from model config)")
    parser.add_argument("--randomize-light", action="store_true")
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument("--policy-hz", type=int, default=30)
    parser.add_argument("--render-hz", type=int, default=30)

    # Output
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    # Seed handling
    if args.seed is None:
        import numpy as np
        args.seed = int(np.random.default_rng().integers(0, 2**31))
        print(f"Generated shared seed: {args.seed}")
    elif args.seed == "random":
        args.seed = None
        print("Using independent random seeds per checkpoint")
    else:
        args.seed = int(args.seed)

    base_dir = Path(args.checkpoint)
    if not base_dir.is_dir():
        print(f"Error: checkpoint directory not found: {base_dir}")
        sys.exit(1)

    # Determine which checkpoints to evaluate
    labels: list[str] = []
    if args.all:
        if (base_dir / "best").is_dir():
            labels.append("best")
        for step in discover_checkpoints(base_dir):
            labels.append(f"checkpoint_{step}")
    else:
        if args.best:
            labels.append("best")
        if args.latest:
            labels.append("latest")
        for step in sorted(args.checkpoints):
            labels.append(f"checkpoint_{step}")

    if not labels:
        labels.append("best")

    # Pre-validate
    skipped: dict[str, str] = {}
    valid_labels: list[str] = []
    for label in labels:
        err = validate_checkpoint(base_dir, label)
        if err:
            print(f"SKIP: {label} — {err}")
            skipped[label] = err
        else:
            valid_labels.append(label)

    if not valid_labels:
        print("Error: no valid checkpoints found")
        sys.exit(1)

    # Output directory
    if args.output:
        sweep_dir = Path(args.output)
    else:
        model_name = base_dir.name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sweep_dir = Path("outputs/eval_sweeps") / f"{model_name}_single_{timestamp}"

    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep (single-env): {len(valid_labels)} checkpoint(s) from {base_dir}")
    if skipped:
        print(f"Skipped: {len(skipped)} ({', '.join(skipped)})")
    print(f"Output: {sweep_dir}")
    print(f"Checkpoints: {', '.join(valid_labels)}")
    print()

    # Run evaluations
    results: list[dict] = [
        {"checkpoint": label, "step": extract_step(label), "status": "SKIPPED", "skip_reason": reason}
        for label, reason in skipped.items()
    ]

    for i, label in enumerate(valid_labels):
        print(f"=== [{i+1}/{len(valid_labels)}] Evaluating: {label} ===")
        eval_output = sweep_dir / label
        cmd = build_eval_command(args, label, eval_output)

        t0 = time.time()
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: eval failed for {label} (exit code {e.returncode})")
            results.append({"checkpoint": label, "step": extract_step(label), "error": True})
            print()
            print_table(results)
            print()
            continue
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving partial results...")
            break

        elapsed = time.time() - t0

        summary_path = eval_output / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            result = {
                "checkpoint": label,
                "step": extract_step(label),
                "total_eval_time_s": round(elapsed, 2),
                **summary,
            }
        else:
            print(f"WARNING: no summary.json found at {summary_path}")
            result = {"checkpoint": label, "step": extract_step(label), "error": True}

        results.append(result)
        print()
        print_table(results)
        print()

    # Save results
    sweep_output = {
        "sweep_config": {
            "checkpoint": str(base_dir),
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "env": args.env,
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "mode": "single-env",
        },
        "results": results,
    }

    output_path = sweep_dir / "sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(sweep_output, f, indent=2)

    print(f"Results saved to {output_path}")
    print()
    print("=== Final Results ===")
    print_table(results)


if __name__ == "__main__":
    main()
