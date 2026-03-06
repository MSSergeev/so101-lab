"""Checkpoint path resolution utilities."""

from pathlib import Path


def resolve_checkpoint_path(base_path: str, use_best: bool, use_latest: bool, step: int | None) -> Path:
    """Resolve checkpoint path based on flags.

    Args:
        base_path: Base training output directory
        use_best: Load best/ subdirectory
        use_latest: Load from root (latest saved)
        step: Load specific checkpoint_N/ subdirectory

    Returns:
        Resolved path to checkpoint directory
    """
    base = Path(base_path)

    # If path already points to best/ or checkpoint_N/, use it directly
    if base.name == "best" or base.name.startswith("checkpoint_"):
        return base

    if step is not None:
        checkpoint_dir = base / f"checkpoint_{step}"
        if not checkpoint_dir.exists():
            available = [d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")]
            raise ValueError(f"Checkpoint step {step} not found. Available: {available}")
        return checkpoint_dir

    if use_best:
        best_dir = base / "best"
        if best_dir.exists():
            return best_dir
        print(f"[WARNING] best/ not found, falling back to latest")

    # use_latest or fallback
    return base
