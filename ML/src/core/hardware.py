"""
hardware.py — Serial data acquisition for Cercus-Calibrator.

Expected MCU line format (comma-separated):
    sys_time, x_dx, x_dy, y_dx, y_dy

Runs in a daemon thread; all public buffers are thread-safe.
"""

import threading
import time
from collections import deque
from typing import Optional, List, Tuple

import serial
import serial.tools.list_ports


class SerialReader:
    """High-frequency serial reader running in a daemon thread.

    Collects (x_dx, x_dy, y_dx, y_dy) tuples into a thread-safe deque.
    The timestamp from the MCU is used only for live display, not buffered.
    """

    def __init__(self, port: str, baudrate: int = 115200, max_samples: int = 100_000):
        self._port = port
        self._baudrate = baudrate
        self._buffer: deque = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[float, float, float, float]] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ser: Optional[serial.Serial] = None
        self._start_time: float = 0.0

    # ------------------------------------------------------------------ public
    def start(self):
        self._ser = serial.Serial(self._port, self._baudrate, timeout=0.5)
        self._ser.reset_input_buffer()
        self._stop_event.clear()
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[Tuple[float, float, float, float]]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.5)
        with self._lock:
            return list(self._buffer)

    def clear_buffer(self):
        with self._lock:
            self._buffer.clear()

    def snapshot_and_clear(self) -> List[Tuple[float, float, float, float]]:
        with self._lock:
            data = list(self._buffer)
            self._buffer.clear()
            return data

    @property
    def latest(self) -> Optional[Tuple[float, float, float, float]]:
        with self._lock:
            return self._latest

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def elapsed(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    # --------------------------------------------------------------- internal
    def _read_loop(self):
        try:
            while not self._stop_event.is_set():
                try:
                    raw = self._ser.readline()
                    if not raw:
                        self._stop_event.wait(timeout=0.05)
                        continue
                    text = raw.decode("utf-8", errors="ignore").strip()
                    if not text:
                        continue
                    parts = text.split(",")
                    if len(parts) < 5:
                        continue
                    # Format: t, x_dx, x_dy, y_dx, y_dy
                    x_dx, x_dy, y_dx, y_dy = (
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    )
                    with self._lock:
                        self._latest = (x_dx, x_dy, y_dx, y_dy)
                        self._buffer.append(self._latest)
                except (serial.SerialException, OSError):
                    self._stop_event.set()
                    break
                except (ValueError, IndexError):
                    continue
        finally:
            if self._ser and self._ser.is_open:
                self._ser.close()


def list_serial_ports() -> List[str]:
    """Return a sorted list of available serial port device names."""
    return sorted(p.device for p in serial.tools.list_ports.comports())
