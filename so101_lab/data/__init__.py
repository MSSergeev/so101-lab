"""Data collection and dataset utilities for SO-101 Lab."""

__all__ = ["LeRobotDatasetWriter", "RecordingManager"]


def __getattr__(name: str):
    if name == "LeRobotDatasetWriter":
        from so101_lab.data.lerobot_dataset import LeRobotDatasetWriter
        return LeRobotDatasetWriter
    if name == "RecordingManager":
        from so101_lab.data.collector import RecordingManager
        return RecordingManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
