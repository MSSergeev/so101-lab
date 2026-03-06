"""Sweep evaluation across multiple VLA (SmolVLA/Pi0/GR00T) checkpoints.

Runs eval_vla_policy.py sequentially for each checkpoint,
collects results into a summary table and JSON.

LeRobot checkpoints use NNNNNN/pretrained_model/ format (not checkpoint_NNNNN/).

The policy server (smolvla_server.py) is started automatically in lerobot-env.
LEROBOT_ENV must be set in .env.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add repo root to path for so101_lab imports
sys.path.insert(0, str(Path(__file__).parents[2]))
from so101_lab.utils.policy_server import start_policy_server  # noqa: E402


def discover_checkpoints(base_dir: Path) -> list[int]:
    """Find all NNNNNN/pretrained_model/ directories, return sorted step numbers."""
    steps = []
    for d in base_dir.iterdir():
        if d.is_dir() and re.match(r"\d{6}$", d.name):
            if (d / "pretrained_model").is_dir():
                steps.append(int(d.name))
    return sorted(steps)


def resolve_checkpoint_path(base_dir: Path, label: str) -> Path:
    """Resolve pretrained_model path for a given label."""
    if label == "last":
        return base_dir / "last" / "pretrained_model"
    else:
        # NNNNNN
        return base_dir / label / "pretrained_model"


def validate_checkpoint(base_dir: Path, label: str) -> str | None:
    """Check if checkpoint has model files. Returns error message or None."""
    path = resolve_checkpoint_path(base_dir, label)
    if not path.is_dir():
        return f"directory not found: {path}"
    if not (path / "model.safetensors").exists():
        return f"model.safetensors not found in {path}"
    return None



def build_eval_command(
    args: argparse.Namespace,
    checkpoint_path: Path,
    output_dir: Path,
    port: int,
) -> list[str]:
    """Build subprocess command for a single checkpoint eval."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "eval_vla_policy.py"),
        "--checkpoint", str(checkpoint_path),
        "--output", str(output_dir),
        "--episodes", str(args.episodes),
        "--max-steps", str(args.max_steps),
        "--env", str(args.env),
        "--port", str(port),
    ]

    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.n_action_steps is not None:
        cmd.extend(["--n-action-steps", str(args.n_action_steps)])
    if args.no_domain_rand:
        cmd.append("--no-domain-rand")
    if args.noise_prior:
        cmd.extend(["--noise-prior", str(args.noise_prior)])

    return cmd


def extract_step(label: str) -> int | None:
    """Extract step number from label."""
    m = re.match(r"(\d{6})$", label)
    return int(m.group(1)) if m else None


def print_table(results: list[dict]) -> None:
    """Print results as a formatted table."""
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
                fmt_val(r.get("eval_time_s"), widths[6]),
            ]
        print("|" + "|".join(f" {c} " for c in row) + "|")

    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep evaluation across multiple VLA checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Sweep selection
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoints/ directory")
    parser.add_argument("--last", action="store_true",
                        help="Evaluate last/ checkpoint")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[],
                        help="Specific step numbers to evaluate (e.g. 5000 10000)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all checkpoints")

    # Eval parameters
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--env", type=str, default="figure_shape_placement")
    parser.add_argument("--seed", type=str, default=None,
                        help="Shared seed (default: auto). Use 'random' for independent seeds")
    parser.add_argument("--n-action-steps", type=int, default=None,
                        help="Override n_action_steps (how many actions from chunk to use)")
    parser.add_argument("--no-domain-rand", action="store_true")
    parser.add_argument("--noise-prior", type=str, default=None)

    # Server
    parser.add_argument("--port", type=int, default=8080,
                        help="Policy server port (default: 8080)")

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
        print(f"Error: directory not found: {base_dir}")
        sys.exit(1)

    # Determine which checkpoints to evaluate
    labels: list[str] = []

    if args.all:
        for step in discover_checkpoints(base_dir):
            labels.append(f"{step:06d}")
        if (base_dir / "last" / "pretrained_model").is_dir():
            labels.append("last")
    else:
        for step in sorted(args.checkpoints):
            labels.append(f"{step:06d}")
        if args.last:
            labels.append("last")

    if not labels:
        # Default: all
        for step in discover_checkpoints(base_dir):
            labels.append(f"{step:06d}")

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
        model_name = base_dir.parent.name if base_dir.name == "checkpoints" else base_dir.name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sweep_dir = Path("outputs/eval_sweeps") / f"{model_name}_{timestamp}"

    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep: {len(valid_labels)} checkpoint(s) from {base_dir}")
    if skipped:
        print(f"Skipped: {len(skipped)} ({', '.join(skipped)})")
    print(f"Output: {sweep_dir}")
    print(f"Checkpoints: {', '.join(valid_labels)}")
    print()

    # Start policy server
    server_proc = start_policy_server(port=args.port)

    # Run evaluations
    results: list[dict] = [
        {"checkpoint": label, "step": extract_step(label), "status": "SKIPPED", "skip_reason": reason}
        for label, reason in skipped.items()
    ]

    for i, label in enumerate(valid_labels):
        print(f"=== [{i+1}/{len(valid_labels)}] Evaluating: {label} ===")
        checkpoint_path = resolve_checkpoint_path(base_dir, label)
        eval_output = sweep_dir / label
        cmd = build_eval_command(args, checkpoint_path, eval_output, args.port)

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
            print("\nInterrupted. Saving partial results...")
            break

        elapsed = time.time() - t0

        summary_path = eval_output / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            result = {
                **summary,
                "checkpoint": label,
                "step": extract_step(label),
                "eval_time_s": round(elapsed, 1),
            }
        else:
            print(f"WARNING: no summary.json at {summary_path}")
            result = {"checkpoint": label, "step": extract_step(label), "error": True}

        results.append(result)
        print()
        print_table(results)
        print()

    # Save sweep results
    sweep_output = {
        "sweep_config": {
            "checkpoint": str(base_dir),
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "env": args.env,
            "seed": args.seed,
            "no_domain_rand": args.no_domain_rand,
            "noise_prior": args.noise_prior,
        },
        "results": results,
    }

    output_path = sweep_dir / "sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(sweep_output, f, indent=2)

    # Stop policy server
    server_proc.terminate()
    server_proc.wait()
    print("Policy server stopped")

    print(f"Results saved to {output_path}")
    print()
    print("=== Final Results ===")
    print_table(results)


if __name__ == "__main__":
    main()
