"""Background system metrics logger for trackio.

Logs CPU, RAM, and per-process metrics via trackio.log_system() on a background thread.
GPU metrics are handled by trackio's built-in auto_log_gpu (requires nvidia-ml-py).

Usage:
    import trackio
    from so101_lab.utils.system_monitor import SystemMonitor

    trackio.init(project="my-project", auto_log_gpu=True)
    monitor = SystemMonitor(interval=10)
    monitor.start()
    # ... training loop ...
    monitor.stop()
"""

import threading
import time

import psutil


class SystemMonitor:
    """Background thread that logs CPU/RAM metrics to trackio."""

    def __init__(self, interval: float = 10.0, log_disk: bool = False):
        self.interval = interval
        self.log_disk = log_disk
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._process = psutil.Process()

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None

    def _loop(self):
        import trackio

        # Warm up CPU percent (first call always returns 0)
        psutil.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)

        while not self._stop_event.wait(self.interval):
            try:
                metrics = self._collect()
                trackio.log_system(metrics)
            except Exception:
                pass

    def _collect(self) -> dict:
        vm = psutil.virtual_memory()
        metrics = {
            "system/cpu_percent": psutil.cpu_percent(interval=None),
            "system/ram_percent": vm.percent,
            "system/ram_used_gb": vm.used / (1024**3),
            "system/ram_available_gb": vm.available / (1024**3),
        }

        # Per-process metrics (training process)
        try:
            proc_mem = self._process.memory_info()
            metrics["process/rss_gb"] = proc_mem.rss / (1024**3)
            metrics["process/cpu_percent"] = self._process.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        if self.log_disk:
            disk = psutil.disk_usage("/")
            metrics["system/disk_percent"] = disk.percent

        return metrics
