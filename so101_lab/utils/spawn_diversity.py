"""Hex-grid based spawn diversity checker.

Prevents clustering of object spawn positions when recording teleop episodes
by tracking spatial distribution and suggesting re-rolls for overcrowded cells.
"""

import json
import math
from pathlib import Path

import numpy as np


class SpawnDiversityChecker:
    """Check spawn position diversity using hex-grid binning.

    Loads existing episode metadata, bins spawn coordinates into a hex grid,
    and flags overcrowded cells for re-rolling.

    Args:
        dataset_dir: Path to dataset (containing meta/episode_metadata.json).
        coord_keys: Keys in initial_state for (x, y) coordinates.
        max_ratio: Cell is overcrowded if count > mean * max_ratio.
        max_rerolls: Maximum re-roll attempts before accepting anyway.
    """

    def __init__(
        self,
        dataset_dir: str,
        coord_keys: list[str],
        max_ratio: float = 2.0,
        max_rerolls: int = 5,
        target_per_cell: float = 5.0,
    ):
        self.coord_keys = coord_keys
        self.max_ratio = max_ratio
        self.max_rerolls = max_rerolls
        self.enabled = False
        self.step = 1.0
        self.grid: dict[tuple[int, int], int] = {}
        self.mean_count = 0.0
        self.total_points = 0
        self.reroll_count = 0

        metadata_path = Path(dataset_dir) / "meta" / "episode_metadata.json"
        if not metadata_path.exists():
            print(f"[DIVERSITY] No episode_metadata.json found, disabled")
            return

        with open(metadata_path) as f:
            metadata = json.load(f)

        points = []
        for ep_data in metadata.values():
            state = ep_data.get("initial_state", {})
            if all(k in state for k in coord_keys):
                points.append([state[coord_keys[0]], state[coord_keys[1]]])

        if len(points) < 10:
            print(f"[DIVERSITY] Only {len(points)} points, need >= 10, disabled")
            return

        pts = np.array(points)
        self.step = self._compute_step(pts, target_per_cell)
        self.enabled = True

        # Bin existing points
        for p in points:
            cell = self._cell(p[0], p[1])
            self.grid[cell] = self.grid.get(cell, 0) + 1
        self.total_points = len(points)
        self._update_mean()

        print(f"[DIVERSITY] Loaded {len(points)} points, step={self.step:.4f}, "
              f"cells={len(self.grid)}, mean={self.mean_count:.1f}")

    @staticmethod
    def _compute_step(pts: np.ndarray, target_per_cell: float) -> float:
        """Compute hex cell size from bounding box."""
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        area = max((maxs[0] - mins[0]) * (maxs[1] - mins[1]), 1e-10)
        n_cells = len(pts) / target_per_cell
        cell_area = area / max(n_cells, 1.0)
        # Hex cell: area = step * (step * sqrt(3)/2) = step^2 * sqrt(3)/2
        return float(math.sqrt(cell_area / (math.sqrt(3) / 2)))

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        """Map (x, y) to hex grid cell."""
        row = int(math.floor(y / (self.step * math.sqrt(3) / 2)))
        if row % 2 == 0:
            col = int(math.floor(x / self.step))
        else:
            col = int(math.floor((x - self.step / 2) / self.step))
        return (row, col)

    def _update_mean(self) -> None:
        """Recompute mean count across non-empty cells."""
        if self.grid:
            self.mean_count = sum(self.grid.values()) / len(self.grid)
        else:
            self.mean_count = 0.0

    def should_reroll(self, initial_state: dict) -> bool:
        """Check if spawn position is in an overcrowded cell."""
        if not self.enabled:
            return False
        if not all(k in initial_state for k in self.coord_keys):
            return False
        x = initial_state[self.coord_keys[0]]
        y = initial_state[self.coord_keys[1]]
        cell = self._cell(x, y)
        count = self.grid.get(cell, 0)
        return count >= self.mean_count * self.max_ratio

    def accept(self, initial_state: dict) -> None:
        """Register accepted spawn position."""
        if not self.enabled:
            return
        if not all(k in initial_state for k in self.coord_keys):
            return
        x = initial_state[self.coord_keys[0]]
        y = initial_state[self.coord_keys[1]]
        cell = self._cell(x, y)
        count_before = self.grid.get(cell, 0)
        threshold = self.mean_count * self.max_ratio
        status = "CROWDED" if count_before >= threshold else "ok"
        self.grid[cell] = count_before + 1
        self.total_points += 1
        self._update_mean()
        print(f"[DIVERSITY] accept {cell}: {count_before}/{threshold:.1f} ({status})")

    def stats(self) -> str:
        """Return coverage statistics string."""
        if not self.enabled:
            return "[DIVERSITY] Disabled (insufficient data)"
        counts = list(self.grid.values())
        return (
            f"[DIVERSITY] points={self.total_points}, cells={len(self.grid)}, "
            f"mean={self.mean_count:.1f}, min={min(counts)}, max={max(counts)}, "
            f"rerolls={self.reroll_count}"
        )

    def report(self) -> str:
        """Return detailed diversity report with histogram."""
        if not self.enabled:
            return "[DIVERSITY] Disabled (insufficient data)"

        from collections import Counter
        counts = sorted(self.grid.values())
        hist = Counter(counts)
        threshold = self.mean_count * self.max_ratio
        crowded = sum(1 for v in self.grid.values() if v >= threshold)
        would_reroll = sum(v for v in self.grid.values() if v >= threshold)

        lines = [
            f"Spawn Diversity Report",
            f"  points: {self.total_points}, cells: {len(self.grid)}, "
            f"step: {self.step:.4f}",
            f"  mean: {self.mean_count:.1f}, threshold (mean*{self.max_ratio}): "
            f"{threshold:.1f}",
            f"  crowded cells: {crowded}/{len(self.grid)}, "
            f"points in crowded: {would_reroll}/{self.total_points} "
            f"({100*would_reroll/self.total_points:.0f}%)",
            f"  distribution:",
        ]
        for c in sorted(hist):
            marker = " <-- crowded" if c >= threshold else ""
            bar = "#" * hist[c]
            lines.append(f"    {c:3d} pts: {hist[c]:2d} cells {bar}{marker}")
        return "\n".join(lines)
