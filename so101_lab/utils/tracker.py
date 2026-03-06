"""Experiment tracker setup (trackio / wandb / none).

Usage in train scripts:

    from so101_lab.utils.tracker import add_tracker_args, setup_tracker, cleanup_tracker

    # In parse_args():
    add_tracker_args(parser, default_project="so101-act")

    # In main():
    tracker, sys_monitor = setup_tracker(args, run_name, config={"lr": args.lr, ...})
    ...
    if tracker:
        tracker.log({"loss": loss.item()}, step=step)
    ...
    cleanup_tracker(tracker, sys_monitor)
"""

from __future__ import annotations

import argparse
from typing import Any


def add_tracker_args(parser: argparse.ArgumentParser, default_project: str = "so101") -> None:
    """Add --tracker, --tracker-project, --system-stats, --wandb args."""
    parser.add_argument(
        "--tracker", type=str, default="none",
        choices=["trackio", "wandb", "none"],
        help="Experiment tracker: trackio (local), wandb (cloud), none",
    )
    parser.add_argument(
        "--tracker-project", type=str, default=default_project,
        help="Tracker project name",
    )
    parser.add_argument(
        "--system-stats", action="store_true",
        help="Log CPU/RAM metrics (requires trackio)",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="(deprecated) Use --tracker wandb",
    )


def setup_tracker(
    args: argparse.Namespace,
    run_name: str,
    config: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Initialize tracker and optional system monitor.

    Returns (tracker_module, sys_monitor). Both can be None.
    tracker_module exposes .log() and .finish().
    """
    tracker = None
    sys_monitor = None

    # Resolve deprecated --wandb flag
    tracker_type = args.tracker
    if tracker_type == "none" and getattr(args, "wandb", False):
        tracker_type = "wandb"

    if config is None:
        config = vars(args)

    if tracker_type == "trackio":
        import trackio
        trackio.init(
            project=args.tracker_project,
            name=run_name,
            resume="allow",
            config=config,
            auto_log_gpu=True,
        )
        tracker = trackio
        print(f"Using trackio (project='{args.tracker_project}', run='{run_name}')")

        if args.system_stats:
            from so101_lab.utils.system_monitor import SystemMonitor
            sys_monitor = SystemMonitor()
            sys_monitor.start()

    elif tracker_type == "wandb":
        import wandb
        wandb.init(project=args.tracker_project, name=run_name, config=config)
        tracker = wandb
        print(f"Using wandb (project='{args.tracker_project}', run='{run_name}')")

    return tracker, sys_monitor


def cleanup_tracker(tracker: Any | None, sys_monitor: Any | None) -> None:
    """Stop system monitor and finish tracker."""
    if sys_monitor:
        sys_monitor.stop()
    if tracker:
        tracker.finish()
