"""
ui.py — CustomTkinter GUI for Cercus-Calibrator.

State machine:
    IDLE  →  COLLECTING  →  TRAINING  →  DONE
"""

import queue
import threading
import time
from typing import Optional, List
import customtkinter as ctk

from src.core.hardware import SerialReader, list_serial_ports
from src.model.optimizer import Calibrator, build_matrix_3x3, save_json


class CercusCalibratorUI(ctk.CTk):
    # states
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    TRAINING = "TRAINING"
    DONE = "DONE"

    # polling intervals (ms)
    DATA_POLL_MS = 100
    TIMER_POLL_MS = 1000
    TRAIN_POLL_MS = 100

    # styling
    COLORS = {"green": "#2ecc71", "red": "#e74c3c", "yellow": "#f39c12"}

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Cercus-Calibrator")
        self.geometry("640x720")
        self.minsize(560, 640)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # runtime state
        self._state = self.IDLE
        self._reader: Optional[SerialReader] = None
        self._timer_after_id: Optional[str] = None
        self._data_after_id: Optional[str] = None
        self._train_after_id: Optional[str] = None
        self._queue: queue.Queue = queue.Queue()
        self._samples: List = []
        self._collect_start_time: float = 0.0
        self._noise_threshold: Optional[float] = None

        self._build_ui()
        self._refresh_ports()
        self._update_controls()

    # ───────────────────────────────────────────── UI construction
    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────
        top = ctk.CTkFrame(self, height=48)
        top.pack(fill="x", padx=10, pady=(10, 5))
        top.pack_propagate(False)

        ctk.CTkLabel(top, text="Serial Port:").pack(side="left", padx=(8, 4))
        self._port_var = ctk.StringVar()
        self._port_menu = ctk.CTkOptionMenu(
            top, variable=self._port_var, values=["(no ports)"], width=160
        )
        self._port_menu.pack(side="left", padx=4)

        ctk.CTkButton(
            top, text="Refresh", width=60, command=self._refresh_ports
        ).pack(side="left", padx=4)

        self._connect_btn = ctk.CTkButton(
            top, text="Connect", width=72, command=self._toggle_connect
        )
        self._connect_btn.pack(side="left", padx=4)

        self._status_dot = ctk.CTkLabel(top, text="●", text_color=self.COLORS["red"])
        self._status_dot.pack(side="right", padx=(4, 10))
        ctk.CTkLabel(top, text="Status:").pack(side="right")

        # ── Data panel ───────────────────────────────────────────
        mid = ctk.CTkFrame(self)
        mid.pack(fill="x", padx=10, pady=5)

        row_a = ctk.CTkFrame(mid, fg_color="transparent")
        row_a.pack(fill="x", padx=10, pady=(10, 2))
        ctk.CTkLabel(row_a, text="S_A  (dx, dy):", font=("", 13)).pack(
            side="left", padx=(0, 8)
        )
        self._lbl_sa = ctk.CTkLabel(
            row_a, text="0.00 , 0.00", font=("Consolas", 18, "bold")
        )
        self._lbl_sa.pack(side="left")

        row_b = ctk.CTkFrame(mid, fg_color="transparent")
        row_b.pack(fill="x", padx=10, pady=(2, 10))
        ctk.CTkLabel(row_b, text="S_B  (dx, dy):", font=("", 13)).pack(
            side="left", padx=(0, 8)
        )
        self._lbl_sb = ctk.CTkLabel(
            row_b, text="0.00 , 0.00", font=("Consolas", 18, "bold")
        )
        self._lbl_sb.pack(side="left")

        timer_frame = ctk.CTkFrame(mid, fg_color="transparent")
        timer_frame.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(timer_frame, text="Timer:").pack(side="left", padx=(0, 8))
        self._lbl_timer = ctk.CTkLabel(
            timer_frame, text="00:00", font=("Consolas", 32, "bold")
        )
        self._lbl_timer.pack(side="left")

        # ── Bottom: controls + log ───────────────────────────────
        bot = ctk.CTkFrame(self)
        bot.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        ctrl = ctk.CTkFrame(bot, fg_color="transparent")
        ctrl.pack(fill="x", padx=8, pady=(8, 4))

        self._btn_start = ctk.CTkButton(
            ctrl,
            text="Start Data Collection",
            height=36,
            command=self._start_collect,
        )
        self._btn_start.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self._btn_stop = ctk.CTkButton(
            ctrl,
            text="Stop & Start Training",
            height=36,
            command=self._stop_and_train,
        )
        self._btn_stop.pack(side="left", expand=True, fill="x", padx=(4, 0))

        self._pbar = ctk.CTkProgressBar(bot)
        self._pbar.pack(fill="x", padx=8, pady=4)
        self._pbar.set(0)

        self._logbox = ctk.CTkTextbox(bot, font=("Consolas", 11), state="disabled")
        self._logbox.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ────────────────────────────────────────── port management
    def _refresh_ports(self):
        ports = list_serial_ports()
        if ports:
            self._port_menu.configure(values=ports)
            self._port_var.set(ports[0])
        else:
            self._port_menu.configure(values=["(no ports)"])
            self._port_var.set("(no ports)")

    def _toggle_connect(self):
        if self._reader and self._reader.is_running:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self._port_var.get()
        if not port or port.startswith("("):
            self._log("⚠  No serial port selected.")
            return
        try:
            self._reader = SerialReader(port, baudrate=115200)
            self._reader.start()
            self._status_dot.configure(text_color=self.COLORS["green"])
            self._connect_btn.configure(text="Disconnect")
            self._log(f"✔  Connected to {port}")
            self._start_data_poll()
        except Exception as e:
            self._log(f"✘  Connection failed: {e}")
            self._reader = None

    def _disconnect(self):
        if self._reader:
            try:
                self._reader.stop()
            except Exception as e:
                self._log(f"⚠  Hardware cleanup error: {e}")
            finally:
                self._reader = None
        self._status_dot.configure(text_color=self.COLORS["red"])
        self._connect_btn.configure(text="Connect")
        self._cancel_data_poll()
        self._lbl_sa.configure(text="0.00 , 0.00")
        self._lbl_sb.configure(text="0.00 , 0.00")
        self._update_controls()

    # ────────────────────────────────────────── state machine
    @property
    def state(self):
        return self._state

    def _enter(self, new_state: str):
        self._state = new_state
        self._update_controls()

    def _update_controls(self):
        s = self._state
        serial_ok = self._reader is not None and self._reader.is_running

        self._btn_start.configure(
            state="normal" if (s in (self.IDLE, self.DONE) and serial_ok) else "disabled"
        )
        self._btn_stop.configure(
            state="normal" if s == self.COLLECTING else "disabled"
        )

    # ────────────────────────────────────────── collection
    def _start_collect(self):
        if self._state not in (self.IDLE, self.DONE) or not self._reader:
            return
        self._reader.clear_buffer()
        self._enter(self.COLLECTING)
        self._collect_start_time = time.monotonic()
        self._start_timer()
        self._log("▸  Data collection started …")

    def _stop_and_train(self):
        if self._state != self.COLLECTING:
            return
        self._stop_timer()

        self._samples = self._reader.snapshot_and_clear()

        n = len(self._samples)
        self._log(f"■  Collection stopped — {n} samples")
        if n < 10:
            self._log("✘  Too few samples to calibrate.")
            self._show_insufficient_dialog(n)
            self._enter(self.IDLE)
            return
        self._launch_training()

    def _show_insufficient_dialog(self, sample_count: int):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Insufficient Data")
        dlg.geometry("380x200")
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()

        dlg.protocol("WM_DELETE_WINDOW", lambda: None)

        frame = ctk.CTkFrame(dlg, fg_color="transparent")
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkLabel(
            frame,
            text="⚠",
            font=("", 40),
            text_color=self.COLORS["yellow"],
        ).pack(pady=(0, 6))

        ctk.CTkLabel(
            frame,
            text=f"Only {sample_count} sample(s) collected.\n"
                 f"At least 10 are required for calibration.",
            font=("", 14),
            justify="center",
        ).pack(pady=(0, 12))

        ctk.CTkButton(
            frame,
            text="OK",
            width=100,
            command=dlg.destroy,
        ).pack()

    # ────────────────────────────────────────── timer
    def _start_timer(self):
        self._timer_tick()

    def _stop_timer(self):
        if self._timer_after_id is not None:
            self.after_cancel(self._timer_after_id)
            self._timer_after_id = None

    def _timer_tick(self):
        if self._state != self.COLLECTING:
            return
        t = int(time.monotonic() - self._collect_start_time)
        self._lbl_timer.configure(text=f"{t // 60:02d}:{t % 60:02d}")
        self._timer_after_id = self.after(self.TIMER_POLL_MS, self._timer_tick)

    # ────────────────────────────────────────── data polling
    def _start_data_poll(self):
        self._data_tick()

    def _cancel_data_poll(self):
        if self._data_after_id is not None:
            self.after_cancel(self._data_after_id)
            self._data_after_id = None

    def _data_tick(self):
        if self._reader and not self._reader.is_running:
            self._log("⚠  Hardware disconnected unexpectedly.")
            self._disconnect()
            if self._state == self.COLLECTING:
                self._stop_timer()
                self._enter(self.IDLE)
            return

        if not self._reader:
            return

        reading = self._reader.latest
        if reading:
            dx1, dy1, dx2, dy2 = reading
            self._lbl_sa.configure(text=f"{dx1:+.2f} , {dy1:+.2f}")
            self._lbl_sb.configure(text=f"{dx2:+.2f} , {dy2:+.2f}")
        self._data_after_id = self.after(self.DATA_POLL_MS, self._data_tick)

    # ────────────────────────────────────────── training
    def _launch_training(self):
        self._enter(self.TRAINING)
        self._pbar.set(0)
        self._log("▸  Training started (1000 epochs) …")

        cal = Calibrator()
        thread = threading.Thread(
            target=self._train_worker,
            args=(cal, self._samples, self._queue, self._noise_threshold),
            daemon=True,
        )
        thread.start()
        self._train_tick()

    @staticmethod
    def _train_worker(
        cal: Calibrator,
        data: list,
        q: queue.Queue,
        noise_threshold: Optional[float] = None,
    ):
        try:
            def cb(epoch, total, loss_val):
                q.put(("progress", epoch / total, epoch, loss_val))

            W_a, final_loss = cal.run(
                data, epochs=1000, progress_cb=cb, noise_threshold=noise_threshold,
            )
            mat = build_matrix_3x3(W_a)
            save_json(mat)
            q.put(("done", final_loss))
        except Exception as e:
            q.put(("error", str(e)))

    def _train_tick(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, frac, epoch, loss_val = msg
                    self._pbar.set(frac)
                    self._log(f"  Epoch {epoch:4d}  Loss: {loss_val:.6f}")
                elif kind == "done":
                    _, final_loss = msg
                    self._pbar.set(1.0)
                    self._log(f"✔  Training complete — Loss: {final_loss:.6f}")
                    self._log("✔  Calibration Saved Successfully → calibration_cfg.json")
                    self._samples = []
                    self._enter(self.DONE)
                    return
                elif kind == "error":
                    _, err = msg
                    self._log(f"✘  Training error: {err}")
                    self._enter(self.IDLE)
                    return
        except queue.Empty:
            pass
        self._train_after_id = self.after(self.TRAIN_POLL_MS, self._train_tick)

    # ────────────────────────────────────────── logging
    def _log(self, text: str):
        self._logbox.configure(state="normal")
        self._logbox.insert("end", text + "\n")
        self._logbox.see("end")
        self._logbox.configure(state="disabled")

    # ────────────────────────────────────────── cleanup
    def _on_close(self):
        self._stop_timer()
        self._cancel_data_poll()
        if self._train_after_id:
            self.after_cancel(self._train_after_id)
        if self._reader:
            self._disconnect()
        self.destroy()
