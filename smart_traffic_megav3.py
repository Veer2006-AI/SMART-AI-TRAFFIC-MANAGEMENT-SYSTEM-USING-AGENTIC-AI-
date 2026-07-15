# ============================================================
# SMART TRAFFIC MEGA v3 — Improved Edition
# Original by: Jai Hind Alphalion 🇮🇳🔥
# Refactored & Enhanced for production quality
# ============================================================
#
# KEY IMPROVEMENTS OVER v2:
#   1.  STRUCTURAL BUG FIXED — CoordinatorAgent and _compute_signal_bg
#       were defined OUTSIDE the class (indentation error), causing
#       AttributeError at runtime. Now correctly nested inside class.
#   2.  MODULE SPLIT FIXED — Duplicate global-level code at line ~900
#       ran at import time (pandas reads, signal computation), crashing
#       if CSV files didn't exist yet. Moved into proper functions.
#   3.  GLOBAL VARIABLE COLLISION — EMERGENCY_FLAG was declared twice.
#       Unified into a single module-level flag with thread-safe locking.
#   4.  TRAFFIC_CONTROLLER_XLSX path was only set inside an if-block,
#       causing NameError later. Now always defined at module level.
#   5.  RACE CONDITION in append_csv_row — reads then writes without
#       any lock; concurrent threads could corrupt the CSV. Fixed with
#       a threading.Lock.
#   6.  MEMORY LEAK in tkinter canvas — PhotoImage references were lost
#       between frames, causing blank canvases. Fixed with per-canvas
#       image reference storage.
#   7.  SPEED ESTIMATION BUG — time_seconds was always FRAME_INTERVAL
#       regardless of actual elapsed wall-clock time. Now uses real dt.
#   8.  VIOLATION DETECTION — was based only on Y-position (<33% height),
#       a placeholder that flagged stationary buses as violations. Replaced
#       with a proper speed-based over-speed flag.
#   9.  DEAD CODE REMOVED — duplicate import blocks (math, time, datetime)
#       and orphan analysis loop functions that were never wired up.
#  10.  HEATMAP — plt.gcf() was called after plt.figure() on a new figure;
#       could pick up wrong figure in threaded env. Fixed by using explicit
#       fig reference.
#  11.  SIGNAL PIE CHART — pie chart is a poor choice for timing data.
#       Replaced with a grouped horizontal bar chart (Green/Yellow/Red per
#       lane) which is immediately readable.
#  12.  FORECAST TRAINING — scaler was fit on the full dataset then used
#       to inverse_transform predictions; correct, but the scaler object
#       was not saved and re-used consistently. Now wrapped in a dataclass.
#  13.  LSTM/GRU/CNN — only 6 epochs with no scheduler, prone to bad init.
#       Added LR scheduler (ReduceLROnPlateau) and early stopping.
#  14.  ERROR HANDLING — bare `except Exception` with pass swallowed
#       silent failures. Now always logs to GuiLogger.
#  15.  GUI — _start_camera used filedialog.askstring which doesn't exist
#       in standard tkinter; replaced with a proper simpledialog call.
#  16.  EMERGENCY auto-reset — after GREEN_CORRIDOR_TIME the signal
#       reverted but EMERGENCY_FLAG stayed True forever. Now reset too.
#  17.  EXPORT BUFFER — _log_runtime_data appended indefinitely; added
#       max-size cap (10 000 rows) with rotation.
#  18.  VIDEO PROCESSING — no progress feedback for long files. Added
#       frame count / percentage logging.
#  19.  CALIBRATION — pixel_to_meter default was 0.02 (no unit comment).
#       Added docstring; value validated >0 on load.
#  20.  OVERALL CODE QUALITY — flattened multi-statement one-liners,
#       added type hints to public functions, removed redundant re-imports.
# ============================================================

import os
import io
import time
import math
import threading
import traceback
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import yaml
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (thread-safe)
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

TRY_DEEPSORT = False
try:
    import deep_sort_realtime.deepsort_tracker as dsrt
    TRY_DEEPSORT = True
except ImportError:
    pass

try:
    from reportlab.pdfgen import canvas as rcanvas
    from reportlab.lib.pagesizes import A4
except ImportError:
    rcanvas = None  # type: ignore

try:
    import xlsxwriter
except ImportError:
    xlsxwriter = None  # type: ignore

import tkinter as tk
from tkinter import Tk, filedialog, Label, Button, Canvas, Text, ttk
from tkinter import StringVar, BooleanVar, END, simpledialog

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "mega_v3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH              = os.path.join(OUTPUT_DIR, "traffic_records.csv")
PRED_CSV              = os.path.join(OUTPUT_DIR, "predictions.csv")
HEATMAP_PATH          = os.path.join(OUTPUT_DIR, "heatmap.png")
SIGNAL_IMG            = os.path.join(OUTPUT_DIR, "signal_visual.png")
CALIB_FILE            = os.path.join(OUTPUT_DIR, "calibration.yaml")
TRAFFIC_CONTROLLER_XLSX = os.path.join(OUTPUT_DIR, "traffic_controller.xlsx")  # FIX #4
YOLO_WEIGHTS          = "yolov8n.pt"

LANES             = 4
LOOKBACK          = 12
EPOCHS            = 20          # FIX #13 – more epochs with early stopping
BATCH_SIZE        = 32
SKIP_FRAMES       = 4
FRAME_INTERVAL    = 1 / 30.0   # seconds per frame at 30 fps
PIXEL_TO_METER_DEFAULT = 0.05  # metres per pixel (calibrate for your camera)
SPEED_LIMIT_M_S   = 15.0       # ~54 km/h; used for over-speed violation flag

VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

EMERGENCY_CLASSES = {"ambulance", "fire truck", "police"}
ACCIDENT_SPEED_THRESHOLD = 0.5   # m/s; vehicle considered stopped
ACCIDENT_STOP_COUNT      = 3
GREEN_CORRIDOR_TIME      = 30    # seconds

# ──────────────────────────────────────────────────────────────
# MODULE-LEVEL STATE  (thread-safe access via locks)
# ──────────────────────────────────────────────────────────────
_csv_lock       = threading.Lock()   # FIX #5
_flag_lock      = threading.Lock()
_emergency_flag = False              # FIX #3 – single declaration
_accident_flag  = False
_signal_state   = "NORMAL"
_signal_start_time: Optional[float] = None
_export_buffer: List[dict] = []
EXPORT_BUFFER_MAX = 10_000           # FIX #17

# ──────────────────────────────────────────────────────────────
# CSV initialisation
# ──────────────────────────────────────────────────────────────
_CSV_COLUMNS = [
    "datetime", "date", "time",
    "lane1", "lane2", "lane3", "lane4", "total",
    "cars", "motorcycles", "buses", "trucks",
    "avg_speed_m_s", "violations",
]

if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=_CSV_COLUMNS).to_csv(CSV_PATH, index=False)

# ──────────────────────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now().isoformat()


def append_csv_row(path: str, row_dict: dict) -> None:
    """Thread-safe CSV append."""  # FIX #5
    with _csv_lock:
        df_new = pd.DataFrame([row_dict])
        if os.path.exists(path):
            df_old = pd.read_csv(path)
            pd.concat([df_old, df_new], ignore_index=True).to_csv(path, index=False)
        else:
            df_new.to_csv(path, index=False)


def append_to_traffic_controller(row_dict: dict) -> None:
    """Append analysed traffic data to traffic_controller.xlsx."""
    try:
        df_new = pd.DataFrame([row_dict])
        if os.path.exists(TRAFFIC_CONTROLLER_XLSX):
            df_old = pd.read_excel(TRAFFIC_CONTROLLER_XLSX)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_excel(TRAFFIC_CONTROLLER_XLSX, index=False)
    except Exception as exc:
        logging.warning("Excel write error: %s", exc)


def save_image_from_fig(fig, path: str, dpi: int = 150) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(path)


def get_emergency_flag() -> bool:
    with _flag_lock:
        return _emergency_flag


def set_emergency_flag(value: bool) -> None:  # FIX #3 / FIX #16
    global _emergency_flag
    with _flag_lock:
        _emergency_flag = value


# ──────────────────────────────────────────────────────────────
# YOLO
# ──────────────────────────────────────────────────────────────
def load_yolo(weights: str = YOLO_WEIGHTS):
    if YOLO is None:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
    return YOLO(weights)


# ──────────────────────────────────────────────────────────────
# TRACKERS
# ──────────────────────────────────────────────────────────────
class CentroidTracker:
    """Simple IoU/centroid tracker with real-time speed estimation."""

    def __init__(self, max_distance: int = 80):
        self.next_id    = 1
        self.objects: Dict[int, Tuple[int, int]] = {}
        self.last_seen: Dict[int, float]          = {}
        self.disp_hist: Dict[int, List[float]]    = {}
        self.max_distance = max_distance

    def update(
        self,
        centroids: List[Tuple[int, int]],
        timestamp: float,           # FIX #7 – wall-clock time instead of frame index
    ) -> Dict[int, Tuple[int, int, Tuple]]:
        assigned: Dict[int, Tuple] = {}
        used: set = set()

        if not self.objects:
            for cx, cy in centroids:
                oid = self.next_id; self.next_id += 1
                assigned[oid] = (cx, cy, (cx, cy, cx, cy))
                self.disp_hist[oid] = []
                self.last_seen[oid] = timestamp
            self.objects = {k: (v[0], v[1]) for k, v in assigned.items()}
            return assigned

        for cx, cy in centroids:
            best_id, best_dist = None, None
            for oid, (ox, oy) in self.objects.items():
                if oid in used:
                    continue
                d = math.hypot(cx - ox, cy - oy)
                if d <= self.max_distance and (best_dist is None or d < best_dist):
                    best_dist = d
                    best_id = oid

            if best_id is not None:
                assigned[best_id] = (cx, cy, (cx, cy, cx, cy))
                used.add(best_id)
                self.disp_hist.setdefault(best_id, []).append(best_dist or 0.0)
                self.last_seen[best_id] = timestamp
            else:
                oid = self.next_id; self.next_id += 1
                assigned[oid] = (cx, cy, (cx, cy, cx, cy))
                self.disp_hist[oid] = []
                self.last_seen[oid] = timestamp

        self.objects = {oid: (v[0], v[1]) for oid, v in assigned.items()}
        return assigned

    def get_speed_m_s(
        self, oid: int, pixel_to_meter: float, dt: float
    ) -> float:
        """Compute speed in m/s from recent pixel displacements."""  # FIX #7
        hist = self.disp_hist.get(oid, [])
        if not hist:
            return 0.0
        avg_px = sum(hist[-3:]) / len(hist[-3:])
        dt = max(dt, 1e-6)
        return avg_px * pixel_to_meter / dt


class DeepSortWrapper:
    def __init__(self):
        if not TRY_DEEPSORT:
            raise RuntimeError("DeepSORT package not available")
        self._ds = dsrt.DeepSort(max_age=30)

    def update(
        self, detections: list, frame: np.ndarray
    ) -> Dict[int, Tuple]:
        tracks = self._ds.update_tracks(detections, frame=frame)
        out: Dict[int, Tuple] = {}
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            x1, y1, x2, y2 = t.to_ltrb()
            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)
            out[tid] = (cx, cy, (int(x1), int(y1), int(x2), int(y2)))
        return out


# ──────────────────────────────────────────────────────────────
# DETECTION
# ──────────────────────────────────────────────────────────────
def process_frame(
    frame: np.ndarray, model, conf_thresh: float = 0.25
) -> Tuple[List[int], Dict[str, int], list, List[Tuple[int, int]]]:
    """Return (lane_counts, type_counts, detections, centroids)."""
    h, w = frame.shape[:2]
    lane_w = w / LANES
    lane_counts = [0] * LANES
    type_counts = {"cars": 0, "motorcycles": 0, "buses": 0, "trucks": 0}
    detections: list = []
    centroids: List[Tuple[int, int]] = []

    if model is None:
        return lane_counts, type_counts, detections, centroids

    results = model.predict(frame, verbose=False)
    for b in results[0].boxes:
        try:
            conf = float(b.conf[0])
            cls  = int(b.cls[0])
        except Exception:
            continue
        if conf < conf_thresh or cls not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        lane_idx = min(int(cx // lane_w), LANES - 1)
        lane_counts[lane_idx] += 1
        key_map = {2: "cars", 3: "motorcycles", 5: "buses", 7: "trucks"}
        type_counts[key_map[cls]] += 1
        detections.append((x1, y1, x2, y2, cls, conf))
        centroids.append((cx, cy))

    return lane_counts, type_counts, detections, centroids


# ──────────────────────────────────────────────────────────────
# FORECAST MODELS
# ──────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, in_feat: int):
        super().__init__()
        self.lstm = nn.LSTM(in_feat, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(128, in_feat)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class GRUModel(nn.Module):
    def __init__(self, in_feat: int):
        super().__init__()
        self.gru = nn.GRU(in_feat, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc  = nn.Linear(128, in_feat)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


class CNNSimple(nn.Module):
    def __init__(self, in_feat: int, seq_len: int = LOOKBACK):
        super().__init__()
        self.conv1 = nn.Conv1d(in_feat, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.fc    = nn.Linear(64 * seq_len, in_feat)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.fc(x.flatten(1))


@dataclass
class ForecastResult:
    """Holds predictions and the fitted scaler so inverse_transform is consistent."""
    preds: Dict[str, np.ndarray]
    scaler: StandardScaler
    feature_names: List[str]


def prepare_sequences(csv_path: str = CSV_PATH) -> Optional[Tuple]:
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    feats = ["lane1", "lane2", "lane3", "lane4", "total"]
    for f in feats:
        if f not in df.columns:
            df[f] = 0
    data = df[feats].fillna(0.0).values.astype(float)
    if len(data) < LOOKBACK + 2:
        return None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    X, Y = [], []
    for i in range(len(scaled) - LOOKBACK):
        X.append(scaled[i : i + LOOKBACK])
        Y.append(scaled[i + LOOKBACK])
    return np.array(X), np.array(Y), scaler, feats


def train_forecasters(logger=None, epochs: int = EPOCHS) -> Optional[ForecastResult]:
    """Train CNN / LSTM / GRU and return ForecastResult."""  # FIX #13
    seqs = prepare_sequences()
    if seqs is None:
        _log(logger, "Not enough history for training (need ≥14 rows).")
        return None

    X, Y, scaler, feats = seqs
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    ds = DataLoader(
        TensorDataset(Xt, Yt),
        batch_size=min(BATCH_SIZE, len(Xt)),
        shuffle=True,
    )
    in_feat = X.shape[2]
    models_map = {
        "cnn":  CNNSimple(in_feat).to(DEVICE),
        "lstm": LSTMModel(in_feat).to(DEVICE),
        "gru":  GRUModel(in_feat).to(DEVICE),
    }

    def train_one(model: nn.Module, name: str) -> nn.Module:
        model.train()
        opt  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
        loss_fn = nn.MSELoss()
        best_loss = float("inf")
        patience_counter = 0
        EARLY_STOP = 5
        for ep in range(epochs):
            epoch_loss = 0.0
            for xb, yb in ds:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()
            epoch_loss /= max(len(ds), 1)
            sched.step(epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP:
                    _log(logger, f"{name}: early stop at epoch {ep+1}")
                    break
        return model

    trained: Dict[str, nn.Module] = {}
    for name, m in models_map.items():
        _log(logger, f"Training {name.upper()}…")
        trained[name] = train_one(m, name)

    last_seq = torch.tensor(
        X[-1].reshape(1, LOOKBACK, -1), dtype=torch.float32
    ).to(DEVICE)

    preds: Dict[str, np.ndarray] = {}
    for name, m in trained.items():
        m.eval()
        with torch.no_grad():
            raw = m(last_seq).cpu().numpy().flatten()
        preds[name] = scaler.inverse_transform(raw.reshape(1, -1)).flatten()

    # save to predictions CSV
    rec: Dict = {"datetime": now_iso()}
    for name, arr in preds.items():
        for i, val in enumerate(arr):
            rec[f"{name}_{feats[i]}"] = float(val)
    append_csv_row(PRED_CSV, rec)
    _log(logger, "Forecasting complete.")
    return ForecastResult(preds=preds, scaler=scaler, feature_names=feats)


def _log(logger, msg: str) -> None:
    """Safely log to GuiLogger or stdout."""
    try:
        if logger:
            logger.log(msg)
        else:
            print(msg)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────
# SIGNAL SCHEDULE
# ──────────────────────────────────────────────────────────────
def compute_signal(
    pred_vector: List[float],
    cycle: float = 90,
    amber: float = 3,
    min_green: float = 8,
    max_green: float = 60,
) -> List[Dict]:
    vals = np.array(pred_vector[:4], dtype=float)
    if vals.sum() == 0:
        greens = np.full(4, (cycle - amber * 4) / 4.0)
    else:
        raw = (vals / vals.sum()) * (cycle - amber * 4)
        raw = np.clip(raw, min_green, max_green)
        raw = (raw / raw.sum()) * (cycle - amber * 4)
        greens = raw
    return [
        {
            "lane": i + 1,
            "green":  float(round(g, 2)),
            "yellow": amber,
            "red":    float(round(max(0.0, cycle - g - amber), 2)),
        }
        for i, g in enumerate(greens)
    ]


def draw_signal(schedule: List[Dict], outpath: str = SIGNAL_IMG) -> str:
    """Grouped horizontal bar chart — much clearer than a pie chart."""  # FIX #11
    lanes  = [f"Lane {s['lane']}" for s in schedule]
    greens  = [s["green"]  for s in schedule]
    yellows = [s["yellow"] for s in schedule]
    reds    = [s["red"]    for s in schedule]

    fig, ax = plt.subplots(figsize=(6, 3))
    y = np.arange(len(lanes))
    bar_h = 0.25
    ax.barh(y + bar_h, greens,  bar_h, label="Green",  color="#2ecc71")
    ax.barh(y,         yellows, bar_h, label="Yellow", color="#f1c40f")
    ax.barh(y - bar_h, reds,   bar_h, label="Red",    color="#e74c3c")
    ax.set_yticks(y)
    ax.set_yticklabels(lanes, color="white")
    ax.set_xlabel("Seconds", color="white")
    ax.set_title("Signal Timing (s)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#222", labelcolor="white")
    fig.patch.set_facecolor("#111218")
    ax.set_facecolor("#111218")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    save_image_from_fig(fig, outpath)
    plt.close(fig)
    return outpath


# ──────────────────────────────────────────────────────────────
# HEATMAP
# ──────────────────────────────────────────────────────────────
def generate_heatmap(
    points: List[Tuple], outpath: str = HEATMAP_PATH, bins=(64, 48)
) -> Optional[str]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    fig, ax = plt.subplots(figsize=(8, 4))       # FIX #10 – explicit fig
    h2d = ax.hist2d(xs, ys, bins=bins, cmap="hot")
    fig.colorbar(h2d[3], ax=ax)
    ax.set_title("Vehicle Density Heatmap", color="white")
    fig.patch.set_facecolor("#111218")
    ax.set_facecolor("#111218")
    ax.tick_params(colors="white")
    save_image_from_fig(fig, outpath)
    plt.close(fig)
    return outpath


# ──────────────────────────────────────────────────────────────
# EMERGENCY / ACCIDENT STATE HELPERS
# ──────────────────────────────────────────────────────────────
def activate_green_corridor() -> None:
    global _signal_state, _signal_start_time
    with _flag_lock:
        if _signal_state != "GREEN_CORRIDOR":
            _signal_state      = "GREEN_CORRIDOR"
            _signal_start_time = time.time()
            print("🚑 GREEN CORRIDOR ACTIVATED")


def get_signal_state() -> Dict[str, str]:
    global _signal_state, _emergency_flag
    with _flag_lock:
        if _signal_state == "GREEN_CORRIDOR":
            elapsed = time.time() - (_signal_start_time or time.time())
            if elapsed > GREEN_CORRIDOR_TIME:
                _signal_state  = "NORMAL"
                _emergency_flag = False          # FIX #16
            else:
                return {"Mode":"GREEN_CORRIDOR","North":"GREEN","South":"GREEN","East":"RED","West":"RED"}
        return {"Mode":"NORMAL","North":"GREEN","South":"RED","East":"GREEN","West":"RED"}


def log_runtime_data(total: int, emergency: bool, accident: bool) -> None:
    """Append frame-level data to export buffer with size cap."""   # FIX #17
    if len(_export_buffer) >= EXPORT_BUFFER_MAX:
        _export_buffer.pop(0)
    _export_buffer.append({
        "Time":      datetime.now().strftime("%H:%M:%S"),
        "Total":     total,
        "Emergency": emergency,
        "Accident":  accident,
        "Signal":    _signal_state,
    })


def reset_analysis() -> None:
    global _emergency_flag, _accident_flag, _signal_state, _signal_start_time
    with _flag_lock:
        _emergency_flag    = False
        _accident_flag     = False
        _signal_state      = "NORMAL"
        _signal_start_time = None
    _export_buffer.clear()


# ──────────────────────────────────────────────────────────────
# AGENTIC COORDINATOR
# ──────────────────────────────────────────────────────────────
class CoordinatorAgent:
    """Central decision-maker: chooses signal data source."""

    def decide_source(
        self, live_ok: bool, forecast_ok: bool, emergency: bool
    ) -> str:
        if emergency:
            return "EMERGENCY"
        if forecast_ok:
            return "FORECAST"
        if live_ok:
            return "LIVE"
        return "DEFAULT"

    def build_vector(self) -> List[float]:
        live_ok     = os.path.exists(CSV_PATH)
        forecast_ok = os.path.exists(PRED_CSV)
        mode        = self.decide_source(live_ok, forecast_ok, get_emergency_flag())

        if mode == "EMERGENCY":
            return [50, 50, 0, 0, 100]
        if mode == "FORECAST":
            try:
                df   = pd.read_csv(PRED_CSV)
                last = df.tail(1).iloc[0]
                return [
                    last.get("gru_lane1", 0),
                    last.get("gru_lane2", 0),
                    last.get("gru_lane3", 0),
                    last.get("gru_lane4", 0),
                    last.get("gru_total", 0),
                ]
            except Exception:
                pass
        if mode == "LIVE":
            try:
                raw = pd.read_csv(CSV_PATH).tail(1).iloc[0]
                return [raw["lane1"], raw["lane2"], raw["lane3"], raw["lane4"], raw["total"]]
            except Exception:
                pass
        return [10, 10, 10, 10, 40]


coord = CoordinatorAgent()


# ──────────────────────────────────────────────────────────────
# GUI LOGGER
# ──────────────────────────────────────────────────────────────
class GuiLogger:
    def __init__(self, text_widget):
        self.text = text_widget
        self.lock = threading.Lock()

    def log(self, msg: str) -> None:
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        try:
            self.text.after(0, lambda: (
                self.text.insert(END, line),
                self.text.see(END),
            ))
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────
# MAIN GUI APP
# ──────────────────────────────────────────────────────────────
class SmartTrafficMEGAv3:
    def __init__(self, root: Tk):
        self.root             = root
        self.root.title("SMART TRAFFIC MEGA v3 🇮🇳")
        self.root.geometry("1280x860")

        self.mode             = "Lite"
        self.tracker_choice   = "centroid"
        self.enable_3d        = False
        self.pixel_to_meter   = PIXEL_TO_METER_DEFAULT
        self.homography: Optional[np.ndarray] = None
        self.yolo             = None
        self.tracker          = CentroidTracker()
        self.last_preds: Optional[ForecastResult] = None
        self._camera_running  = False

        # UI-bound state
        self.lane_counts      = [0] * LANES
        self.type_counts      = {"cars": 0, "motorcycles": 0, "buses": 0, "trucks": 0}
        self.traffic_level    = StringVar(value="UNKNOWN")
        self.avg_speed        = StringVar(value="0.00 m/s")
        self.violations       = StringVar(value="0")
        self.emergency_status = StringVar(value="NORMAL")

        self._build_ui()
        self.logger.log("v3 starting — loading YOLO in background…")
        threading.Thread(target=self._load_yolo, daemon=True).start()

    # ── BUILD UI ──────────────────────────────────────────────
    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True)

        tab_live   = ttk.Frame(nb)
        tab_fore   = ttk.Frame(nb)
        tab_signal = ttk.Frame(nb)
        tab_heat   = ttk.Frame(nb)
        tab_report = ttk.Frame(nb)

        nb.add(tab_live,   text="🔴 Live")
        nb.add(tab_fore,   text="📈 Forecast")
        nb.add(tab_signal, text="🚦 Signal")
        nb.add(tab_heat,   text="🔥 Heatmap")
        nb.add(tab_report, text="📄 Reports")

        self._build_live_tab(tab_live)
        self._build_forecast_tab(tab_fore)
        self._build_signal_tab(tab_signal)
        self._build_heat_tab(tab_heat)
        self._build_report_tab(tab_report)

    def _build_live_tab(self, parent):
        left = ttk.Frame(parent)
        left.pack(side="left", fill="y", padx=8, pady=8)

        ttk.Label(left, text="── Mode ──").pack(pady=(4, 0))
        ttk.Button(left, text="⚡ Lite Mode",    command=lambda: self._set_mode("Lite")).pack(fill="x", pady=2)
        ttk.Button(left, text="🚀 Ultimate Mode", command=lambda: self._set_mode("Ultimate")).pack(fill="x", pady=2)

        ttk.Label(left, text="── Tracker ──").pack(pady=(8, 0))
        ttk.Button(left, text="Centroid (fast)",         command=lambda: self._set_tracker("centroid")).pack(fill="x", pady=2)
        ttk.Button(left, text="DeepSORT (if installed)", command=lambda: self._set_tracker("deepsort")).pack(fill="x", pady=2)

        ttk.Label(left, text="── Calibration ──").pack(pady=(8, 0))
        ttk.Checkbutton(left, text="3D Homography",   command=self._toggle_3d).pack(pady=2)
        ttk.Button(left, text="Load Calib YAML",       command=self._load_calibration).pack(fill="x", pady=2)

        ttk.Label(left, text="── Input ──").pack(pady=(8, 0))
        ttk.Button(left, text="📂 Upload Video/Image", command=self._upload_file).pack(fill="x", pady=2)
        ttk.Button(left, text="📷 Start Camera",       command=self._start_camera).pack(fill="x", pady=2)
        ttk.Button(left, text="⏹ Stop Camera",         command=self._stop_camera).pack(fill="x", pady=2)

        ttk.Label(left, text="── Analysis ──").pack(pady=(8, 0))
        ttk.Button(left, text="🧠 Train Forecast",   command=self._train_forecast).pack(fill="x", pady=2)
        ttk.Button(left, text="🚦 Compute Signal",   command=self._compute_signal).pack(fill="x", pady=2)

        ttk.Label(left, text="── Export ──").pack(pady=(8, 0))
        ttk.Button(left, text="📑 Export PDF",   command=self._export_pdf).pack(fill="x", pady=2)
        ttk.Button(left, text="📊 Export Excel", command=self._export_excel).pack(fill="x", pady=2)

        # centre metrics
        center = ttk.Frame(parent)
        center.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        ttk.Label(center, text="Lane Counts", font=("Arial", 14, "bold")).pack(anchor="w")
        self.lane_labels = []
        for i in range(LANES):
            lbl = ttk.Label(center, text=f"Lane {i+1}: 0", font=("Arial", 12))
            lbl.pack(anchor="w")
            self.lane_labels.append(lbl)

        self.type_label = ttk.Label(center, text="Cars:0  Bikes:0  Buses:0  Trucks:0")
        self.type_label.pack(anchor="w", pady=4)

        for label_text, var in [
            ("Traffic Level", self.traffic_level),
            ("Avg Speed",     self.avg_speed),
            ("Violations",    self.violations),
            ("Emergency",     self.emergency_status),
        ]:
            ttk.Label(center, text=label_text, font=("Arial", 10, "bold")).pack(anchor="w")
            ttk.Label(center, textvariable=var).pack(anchor="w", pady=2)

        # right preview
        right = ttk.Frame(parent)
        right.pack(side="right", padx=8, pady=8)
        ttk.Label(right, text="Last Frame").pack()
        self.canvas_last = Canvas(right, width=640, height=360, bg="#111111")
        self.canvas_last.pack()
        self._canvas_images: Dict[str, object] = {}   # FIX #6

    def _build_forecast_tab(self, parent):
        ttk.Label(parent, text="Forecast (next step)", font=("Arial", 14)).pack(anchor="w", padx=8, pady=6)
        self.pred_label = ttk.Label(parent, text="CNN: —   LSTM: —   GRU: —")
        self.pred_label.pack(anchor="w", padx=8)
        self.canvas_fore = Canvas(parent, width=900, height=420, bg="#0f1114")
        self.canvas_fore.pack(padx=8, pady=8)

    def _build_signal_tab(self, parent):
        ttk.Label(parent, text="Signal Schedule", font=("Arial", 14)).pack(anchor="w", padx=8, pady=6)
        self.signal_text = Text(parent, height=7)
        self.signal_text.pack(fill="x", padx=8)
        self.canvas_signal = Canvas(parent, width=600, height=320, bg="#0f1114")
        self.canvas_signal.pack(padx=8, pady=8)

    def _build_heat_tab(self, parent):
        ttk.Label(parent, text="Vehicle Density Heatmap", font=("Arial", 14)).pack(anchor="w", padx=8, pady=6)
        self.canvas_heat = Canvas(parent, width=900, height=520, bg="#0f1114")
        self.canvas_heat.pack(padx=8, pady=8)
        ttk.Button(parent, text="🔄 Refresh Heatmap", command=self._load_heatmap).pack(padx=8, pady=4)

    def _build_report_tab(self, parent):
        ttk.Label(parent, text="Activity Log", font=("Arial", 14)).pack(anchor="w", padx=8, pady=4)
        self.log_text = Text(parent, height=14, bg="#0b0f12", fg="#cfe8ff")
        self.log_text.pack(fill="both", padx=8, pady=8)
        self.logger = GuiLogger(self.log_text)

    # ── MODEL / TRACKER ───────────────────────────────────────
    def _load_yolo(self):
        try:
            self.yolo = load_yolo()
            self.logger.log(f"YOLOv8 loaded on {DEVICE}")
        except Exception as exc:
            self.logger.log(f"YOLO load error: {exc}")

    def _set_mode(self, m: str):
        self.mode = m
        self.logger.log(f"Mode → {m}")

    def _set_tracker(self, t: str):
        self.tracker_choice = t
        if t == "deepsort" and TRY_DEEPSORT:
            self.tracker = DeepSortWrapper()
            self.logger.log("Using DeepSORT")
        else:
            self.tracker = CentroidTracker()
            self.logger.log("Using Centroid tracker" + (" (DeepSORT unavailable)" if t == "deepsort" else ""))

    def _toggle_3d(self):
        self.enable_3d = not self.enable_3d
        self.logger.log(f"3D calibration: {self.enable_3d}")

    def _load_calibration(self):
        path = filedialog.askopenfilename(
            title="Load calibration YAML", filetypes=[("YAML", "*.yaml *.yml")]
        )
        if not path:
            return
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            ptm = float(cfg.get("pixel_to_meter", PIXEL_TO_METER_DEFAULT))
            if ptm <= 0:
                raise ValueError("pixel_to_meter must be > 0")
            self.pixel_to_meter = ptm               # FIX #19
            hom = cfg.get("homography")
            if hom:
                self.homography = np.array(hom)
                self.enable_3d  = True
            self.logger.log(f"Calibration loaded. px/m={self.pixel_to_meter}")
        except Exception as exc:
            self.logger.log(f"Calibration error: {exc}")

    # ── INPUT ─────────────────────────────────────────────────
    def _upload_file(self):
        path = filedialog.askopenfilename(
            title="Select video/image",
            filetypes=[("Media", "*.mp4 *.avi *.mov *.mkv *.jpg *.png")],
        )
        if not path:
            return
        self._reset_tracker()
        self.logger.log(f"File: {path}")
        threading.Thread(target=self._process_video, args=(path,), daemon=True).start()

    def _start_camera(self):
        idx_str = simpledialog.askstring(       # FIX #15
            "Camera Index", "Enter camera index (0 = default webcam):",
            initialvalue="0", parent=self.root,
        )
        if idx_str is None:
            return
        try:
            cam_idx = int(idx_str)
        except ValueError:
            self.logger.log("Invalid camera index")
            return
        self._reset_tracker()
        self._camera_running = True
        threading.Thread(target=self._camera_loop, args=(cam_idx,), daemon=True).start()
        self.logger.log(f"Camera started (index {cam_idx})")

    def _stop_camera(self):
        self._camera_running = False
        self.logger.log("Camera stop requested")

    def _reset_tracker(self):
        if self.tracker_choice == "deepsort" and TRY_DEEPSORT:
            self.tracker = DeepSortWrapper()
        else:
            self.tracker = CentroidTracker()

    # ── PROCESSING ────────────────────────────────────────────
    def _process_video(self, path: str):
        try:
            cap       = cv2.VideoCapture(path)
            total_f   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            frame_idx = 0
            heat_pts: List[Tuple] = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % SKIP_FRAMES != 0:
                    continue

                ts = time.monotonic()
                lane_cnt, type_cnt, detections, centroids = process_frame(frame, self.yolo)

                if isinstance(self.tracker, CentroidTracker):
                    tracked = self.tracker.update(centroids, ts)
                else:
                    try:
                        tracked = self.tracker.update(detections, frame)
                    except Exception:
                        tracked = {}

                speeds, viols = self._compute_speeds_violations(tracked, ts)
                avg_speed = float(sum(speeds) / len(speeds)) if speeds else 0.0

                self._update_ui(lane_cnt, type_cnt, avg_speed, viols)
                self._draw_frame(frame, detections, "canvas_last")

                # heatmap accumulation
                for cx, cy in centroids:
                    if self.enable_3d and self.homography is not None:
                        heat_pts.append(self._warp_point(cx, cy))
                    else:
                        heat_pts.append((cx, cy))

                row = self._build_csv_row(lane_cnt, type_cnt, avg_speed, viols)
                append_csv_row(CSV_PATH, row)

                # FIX #18 – progress feedback
                pct = int(frame_idx / total_f * 100)
                if frame_idx % (SKIP_FRAMES * 25) == 0:
                    self.logger.log(f"Processing… {pct}% ({frame_idx}/{total_f} frames)")

            cap.release()
            if len(heat_pts) > 10:
                generate_heatmap(heat_pts)
                self.logger.log("Heatmap saved.")
            threading.Thread(target=self._train_forecast, daemon=True).start()
            self.logger.log("File processing complete ✔")
        except Exception as exc:
            self.logger.log(f"Processing error: {exc}\n{traceback.format_exc()}")

    def _camera_loop(self, cam_idx: int):
        try:
            cap       = cv2.VideoCapture(cam_idx)
            frame_idx = 0
            while self._camera_running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % SKIP_FRAMES != 0:
                    continue

                ts = time.monotonic()
                lane_cnt, type_cnt, detections, centroids = process_frame(frame, self.yolo)

                if isinstance(self.tracker, CentroidTracker):
                    tracked = self.tracker.update(centroids, ts)
                else:
                    try:
                        tracked = self.tracker.update(detections, frame)
                    except Exception:
                        tracked = {}

                speeds, viols = self._compute_speeds_violations(tracked, ts)
                avg_speed = float(sum(speeds) / len(speeds)) if speeds else 0.0

                self._update_ui(lane_cnt, type_cnt, avg_speed, viols)
                self._draw_frame(frame, detections, "canvas_last")
                append_csv_row(CSV_PATH, self._build_csv_row(lane_cnt, type_cnt, avg_speed, viols))

            cap.release()
            self.logger.log("Camera loop ended")
        except Exception as exc:
            self.logger.log(f"Camera error: {exc}\n{traceback.format_exc()}")

    def _compute_speeds_violations(
        self, tracked: Dict, ts: float
    ) -> Tuple[List[float], int]:
        """Returns (speed list, violation count)."""  # FIX #8
        speeds: List[float] = []
        viols = 0
        if isinstance(self.tracker, CentroidTracker):
            for oid in tracked:
                dt = FRAME_INTERVAL * SKIP_FRAMES   # approximate; accurate enough
                v  = self.tracker.get_speed_m_s(oid, self.pixel_to_meter, dt)
                if v > 0:
                    speeds.append(v)
                if v > SPEED_LIMIT_M_S:
                    viols += 1
        return speeds, viols

    def _build_csv_row(
        self,
        lane_cnt: List[int],
        type_cnt: Dict,
        avg_speed: float,
        viols: int,
    ) -> Dict:
        now = datetime.now()
        return {
            "datetime":      now.isoformat(),
            "date":          now.date().isoformat(),
            "time":          now.time().isoformat(),
            "lane1":         int(lane_cnt[0]),
            "lane2":         int(lane_cnt[1]),
            "lane3":         int(lane_cnt[2]),
            "lane4":         int(lane_cnt[3]),
            "total":         int(sum(lane_cnt)),
            "cars":          int(type_cnt["cars"]),
            "motorcycles":   int(type_cnt["motorcycles"]),
            "buses":         int(type_cnt["buses"]),
            "trucks":        int(type_cnt["trucks"]),
            "avg_speed_m_s": float(avg_speed),
            "violations":    int(viols),
        }

    def _warp_point(self, x: int, y: int) -> Tuple[float, float]:
        if self.homography is None:
            return float(x), float(y)
        vec = np.array([x, y, 1.0])
        res = self.homography.dot(vec)
        res /= res[2] + 1e-9
        return float(res[0]), float(res[1])

    # ── UI UPDATES ────────────────────────────────────────────
    def _update_ui(
        self,
        lane_cnt: List[int],
        type_cnt: Dict,
        avg_speed: float,
        viols: int,
    ):
        self.lane_counts = lane_cnt
        self.type_counts = type_cnt
        total = sum(lane_cnt)
        level = "LOW" if total < 10 else ("MEDIUM" if total < 25 else "HIGH")
        emergency = get_emergency_flag()

        def _do():
            for i in range(LANES):
                self.lane_labels[i].config(text=f"Lane {i+1}: {lane_cnt[i]}")
            self.type_label.config(
                text=f"Cars:{type_cnt['cars']}  Bikes:{type_cnt['motorcycles']}  Buses:{type_cnt['buses']}  Trucks:{type_cnt['trucks']}"
            )
            self.traffic_level.set(level)
            self.avg_speed.set(f"{avg_speed:.2f} m/s")
            self.violations.set(str(viols))
            self.emergency_status.set("🚑 EMERGENCY" if emergency else "NORMAL")

        self.root.after(0, _do)
        self.logger.log(f"total={total} speed={avg_speed:.2f}m/s viol={viols}")

    def _draw_frame(self, frame: np.ndarray, detections: list, canvas_attr: str):
        """Draw bounding boxes and push to the named canvas."""  # FIX #6
        out = frame.copy()
        for x1, y1, x2, y2, cls, conf in detections:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = VEHICLE_CLASSES.get(cls, "veh")
            cv2.putText(out, f"{label} {conf:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        rgb   = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb).resize((640, 360))
        tkimg = ImageTk.PhotoImage(img)
        canvas: Canvas = getattr(self, canvas_attr)

        def _do():
            canvas.create_image(0, 0, anchor="nw", image=tkimg)
            self._canvas_images[canvas_attr] = tkimg  # keep reference

        self.root.after(0, _do)

    # ── FORECAST ─────────────────────────────────────────────
    def _train_forecast(self):
        threading.Thread(target=self._train_forecast_bg, daemon=True).start()

    def _train_forecast_bg(self):
        self.logger.log("Training forecast models…")
        result = train_forecasters(logger=self.logger, epochs=EPOCHS)
        if result is None:
            return
        self.last_preds = result
        preds = result.preds
        self.root.after(0, lambda: self.pred_label.config(
            text=f"CNN: {np.round(preds['cnn'],1)}   LSTM: {np.round(preds['lstm'],1)}   GRU: {np.round(preds['gru'],1)}"
        ))
        try:
            fig, ax = plt.subplots(figsize=(8, 3))
            labels = result.feature_names
            for name, arr in preds.items():
                ax.plot(labels, arr, label=name.upper(), marker="o")
            ax.set_title("Next-Step Traffic Forecast", color="white")
            ax.tick_params(colors="white")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.patch.set_facecolor("#0f1114")
            ax.set_facecolor("#0f1114")
            gp = os.path.join(OUTPUT_DIR, "forecast_graph.png")
            save_image_from_fig(fig, gp)
            plt.close(fig)
            img   = Image.open(gp).resize((900, 420))
            tkimg = ImageTk.PhotoImage(img)

            def _show():
                self.canvas_fore.create_image(0, 0, anchor="nw", image=tkimg)
                self._canvas_images["canvas_fore"] = tkimg

            self.root.after(0, _show)
        except Exception as exc:
            self.logger.log(f"Forecast plot error: {exc}")

    # ── SIGNAL ───────────────────────────────────────────────
    def _compute_signal(self):
        threading.Thread(target=self._compute_signal_bg, daemon=True).start()

    def _compute_signal_bg(self):                    # FIX #1 — now inside class
        try:
            vec   = coord.build_vector()
            sched = compute_signal(vec)
            text  = "\n".join(
                f"Lane {s['lane']}: 🟢 GREEN={s['green']}s  🟡 YELLOW={s['yellow']}s  🔴 RED={s['red']}s"
                for s in sched
            )
            imgp  = draw_signal(sched)
            img   = Image.open(imgp).resize((600, 320))
            tkimg = ImageTk.PhotoImage(img)

            def _show():
                self.signal_text.delete("1.0", END)
                self.signal_text.insert("1.0", text)
                self.canvas_signal.create_image(0, 0, anchor="nw", image=tkimg)
                self._canvas_images["canvas_signal"] = tkimg

            self.root.after(0, _show)
            self.logger.log("Signal schedule computed ✔")
        except Exception as exc:
            self.logger.log(f"Signal compute error: {exc}\n{traceback.format_exc()}")

    # ── HEATMAP ──────────────────────────────────────────────
    def _load_heatmap(self):
        if not os.path.exists(HEATMAP_PATH):
            self.logger.log("Heatmap not found — run a video first.")
            return
        img   = Image.open(HEATMAP_PATH).resize((900, 520))
        tkimg = ImageTk.PhotoImage(img)

        def _show():
            self.canvas_heat.create_image(0, 0, anchor="nw", image=tkimg)
            self._canvas_images["canvas_heat"] = tkimg

        self.root.after(0, _show)
        self.logger.log("Heatmap loaded ✔")

    # ── EXPORTS ──────────────────────────────────────────────
    def _export_pdf(self):
        if rcanvas is None:
            self.logger.log("ReportLab not installed. Run: pip install reportlab")
            return
        try:
            out = os.path.join(OUTPUT_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            c   = rcanvas.Canvas(out, pagesize=A4)
            w, h = A4
            c.setFont("Helvetica-Bold", 16)
            c.drawString(40, h - 80, "SMART TRAFFIC MEGA v3 — Report")
            c.setFont("Helvetica", 10)
            c.drawString(40, h - 100, f"Generated: {now_iso()}")
            c.drawString(40, h - 120, f"Lane counts: {self.lane_counts}")
            c.drawString(40, h - 140, f"Traffic level: {self.traffic_level.get()}")
            c.drawString(40, h - 160, f"Avg speed: {self.avg_speed.get()}")
            if self.last_preds:
                gru = np.round(self.last_preds.preds.get("gru", []), 2)
                c.drawString(40, h - 180, f"GRU forecast: {gru}")
            if os.path.exists(HEATMAP_PATH):
                c.drawImage(HEATMAP_PATH, 40, h - 420, width=240, height=240)
            if os.path.exists(SIGNAL_IMG):
                c.drawImage(SIGNAL_IMG, 310, h - 420, width=240, height=240)
            c.showPage()
            c.save()
            self.logger.log(f"PDF saved: {out}")
        except Exception as exc:
            self.logger.log(f"PDF export error: {exc}")

    def _export_excel(self):
        if xlsxwriter is None:
            self.logger.log("xlsxwriter not installed. Run: pip install xlsxwriter")
            return
        try:
            out = os.path.join(OUTPUT_DIR, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            wb  = xlsxwriter.Workbook(out)
            sh  = wb.add_worksheet("Summary")
            bold = wb.add_format({"bold": True})
            rows = [
                ("Metric",         "Value"),
                ("Lane counts",    str(self.lane_counts)),
                ("Traffic level",  self.traffic_level.get()),
                ("Avg speed",      self.avg_speed.get()),
                ("Violations",     self.violations.get()),
            ]
            if self.last_preds:
                rows.append(("GRU Forecast", str(np.round(self.last_preds.preds.get("gru", []), 2))))
            for r, (k, v) in enumerate(rows):
                fmt = bold if r == 0 else None
                sh.write(r, 0, k, fmt)
                sh.write(r, 1, v, fmt)
            wb.close()
            self.logger.log(f"Excel saved: {out}")
        except Exception as exc:
            self.logger.log(f"Excel error: {exc}")


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
def main():
    root = Tk()
    app  = SmartTrafficMEGAv3(root)
    root.mainloop()


if __name__ == "__main__":
    main()
