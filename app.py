# ============================================================
# SMART TRAFFIC ULTIMATE DASHBOARD (Option-D Optimized Version)
# Flask + YOLOv8 + DeepSORT + LSTM/GRU/CNN Forecasting
# Clean Console (no TF warnings) ✦ No retracing ✦ Fast inference
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")  # Suppress all python warnings

# Silence TensorFlow logs (NO WARNINGS, NO INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import time
import math
import threading
import traceback
from datetime import datetime

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import cv2
import numpy as np
import pandas as pd
import openpyxl

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, Flatten, Input
from sklearn.preprocessing import StandardScaler

# ============================================================
# DIRECTORIES & GLOBAL SETTINGS
# ============================================================
PROJECT_ROOT = os.path.abspath(".")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "smart_v3_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_CSV = os.path.join(OUTPUT_DIR, "traffic_records.csv")
PRED_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")
EXCEL_PATH = os.path.join(PROJECT_ROOT, "traffic_output.xlsx")

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8n.pt")

NUM_LANES = 4
PIXEL_TO_METER = 0.05
LOOKBACK = 12
FORECAST_EPOCHS = 5
BATCH = 32

VEHICLE_CLASSES = {2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}

STATE_LOCK = threading.Lock()
STATE = {
    "lane_counts":[0]*NUM_LANES,
    "type_counts":{"car":0,"motorcycle":0,"bus":0,"truck":0,"other":0},
    "avg_speed_mps":0.0,
    "predictions":{"cnn":[0]*5,"lstm":[0]*5,"gru":[0]*5},
    "signal":{},
    "last_frame_image":None,
    "heatmap_image":None,
    "tracking_ids":{}
}

app = Flask(__name__)
CORS(app)

# ============================================================
# INITIALIZATION
# ============================================================

DETECTOR = YOLO(MODEL_PATH if os.path.exists(MODEL_PATH) else "yolov8n.pt")
TRACKER = DeepSort(max_age=30, n_init=2)

# ============================================================
# ENSURE FILE STRUCTURE
# ============================================================
def ensure_outputs():
    if not os.path.exists(DATA_CSV):
        pd.DataFrame([], columns=["datetime","date","time"] +
                     [f"lane{i+1}" for i in range(NUM_LANES)] +
                     ["total","car","motorcycle","bus","truck","other","avg_speed"]
                    ).to_csv(DATA_CSV, index=False)

    if not os.path.exists(PRED_CSV):
        pd.DataFrame([], columns=["datetime","model","lane1_pred",
                                  "lane2_pred","lane3_pred","lane4_pred",
                                  "total_pred"]).to_csv(PRED_CSV, index=False)

    if not os.path.exists(EXCEL_PATH):
        wb = openpyxl.Workbook()
        wb.active.title = "Traffic Summary"
        for sheet in ["Lane Count","Signal Timing","Timestamp Data",
                      "Vehicle Types","Forecasts"]:
            wb.create_sheet(sheet)
        wb.save(EXCEL_PATH)

ensure_outputs()

# ============================================================
# EXCEL APPEND FUNCTION (Option-D Safe Version)
# ============================================================
def safe_append_excel(forecast_rows=None):
    try:
        with pd.ExcelWriter(EXCEL_PATH, mode="a", engine="openpyxl",
                            if_sheet_exists="overlay") as writer:

            if forecast_rows:
                df = pd.DataFrame(forecast_rows)
                try:
                    old = pd.read_excel(EXCEL_PATH, sheet_name="Forecasts")
                    df = pd.concat([old, df], ignore_index=True)
                except:
                    pass
                df.to_excel(writer, sheet_name="Forecasts", index=False)
        return True

    except Exception as e:
        print("Excel Error:", e)
        return False

# ============================================================
# FRAME PROCESSING
# ============================================================
def process_frame(frame):
    h, w = frame.shape[:2]
    lane_width = w / NUM_LANES

    lane_counts = [0]*NUM_LANES
    type_counts = {"car":0,"motorcycle":0,"bus":0,"truck":0,"other":0}
    detections = []

    results = DETECTOR.predict(frame, verbose=False)
    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls not in VEHICLE_CLASSES:
            continue

        x1,y1,x2,y2 = map(float, box.xyxy[0])
        cx = (x1+x2)/2

        lane_idx = min(int(cx // lane_width), NUM_LANES-1)
        lane_counts[lane_idx] += 1

        type_counts[VEHICLE_CLASSES[cls]] += 1
        detections.append(([x1,y1,x2,y2], 1.0, cls))

    # ---------- DeepSORT Tracking + Speed ----------
    speeds = []
    tracks = TRACKER.update_tracks(detections, frame=frame)
    now = time.time()

    for tr in tracks:
        if not tr.is_confirmed():
            continue

        x1,y1,x2,y2 = tr.to_ltrb()
        cx, cy = (x1+x2)/2, (y1+y2)/2

        if tr.track_id in STATE["tracking_ids"]:
            prev_x, prev_y, prev_t = STATE["tracking_ids"][tr.track_id]
            dt = now - prev_t
            dist = math.dist([cx,cy],[prev_x,prev_y])
            speed = (dist * PIXEL_TO_METER) / max(dt,0.0001)
            speeds.append(speed)

        STATE["tracking_ids"][tr.track_id] = (cx,cy,now)

    avg_speed = np.mean(speeds) if speeds else 0.0

    # Annotated frame
    annotated = frame.copy()
    for tr in TRACKER._tracker.tracks:
        if tr.is_confirmed():
            x1,y1,x2,y2 = map(int, tr.to_ltrb())
            cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(annotated, f"ID:{tr.track_id}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,0),2)

    return lane_counts, type_counts, avg_speed, annotated

# ============================================================
# FORECASTING MODELS (Option-D optimized)
# ============================================================

@tf.function(reduce_retracing=True)
def tf_predict(model, x):
    return model(x, training=False)

def prepare_data():
    df = pd.read_csv(DATA_CSV)
    feats = [f"lane{i+1}" for i in range(NUM_LANES)] + ["total"]
    data = df[feats].values.astype(float)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    X, Y = [], []
    for i in range(len(scaled)-LOOKBACK):
        X.append(scaled[i:i+LOOKBACK])
        Y.append(scaled[i+LOOKBACK])

    return np.array(X), np.array(Y), scaler

def build_models(X, Y):
    feats = X.shape[2]

    model_cnn = Sequential([
        Input(shape=(LOOKBACK, feats)),
        Conv1D(32,3,activation="relu",padding="same"),
        Flatten(),
        Dense(32,activation="relu"),
        Dense(feats)
    ])
    model_cnn.compile(optimizer="adam", loss="mse")
    model_cnn.fit(X, Y, epochs=FORECAST_EPOCHS, verbose=False)

    model_lstm = Sequential([
        Input(shape=(LOOKBACK,feats)),
        LSTM(64),
        Dense(feats)
    ])
    model_lstm.compile(optimizer="adam", loss="mse")
    model_lstm.fit(X, Y, epochs=FORECAST_EPOCHS, verbose=False)

    model_gru = Sequential([
        Input(shape=(LOOKBACK,feats)),
        GRU(64),
        Dense(feats)
    ])
    model_gru.compile(optimizer="adam", loss="mse")
    model_gru.fit(X, Y, epochs=FORECAST_EPOCHS, verbose=False)

    return model_cnn, model_lstm, model_gru

def forecast(models, scaler):
    model_cnn, model_lstm, model_gru = models
    X, _, _ = prepare_data()
    last_seq = X[-1].reshape(1,LOOKBACK,-1)

    pred_cnn = tf_predict(model_cnn, last_seq)[0].numpy()
    pred_lstm = tf_predict(model_lstm, last_seq)[0].numpy()
    pred_gru = tf_predict(model_gru, last_seq)[0].numpy()

    cnn = scaler.inverse_transform(pred_cnn.reshape(1,-1))[0]
    lstm = scaler.inverse_transform(pred_lstm.reshape(1,-1))[0]
    gru = scaler.inverse_transform(pred_gru.reshape(1,-1))[0]

    return {"cnn":cnn,"lstm":lstm,"gru":gru}

# ============================================================
# TRAINER THREAD (TRAIN ONCE → PREDICT FOREVER)
# ============================================================
def trainer_loop():
    print("[Trainer] Training forecasting models once...")
    try:
        X, Y, scaler = prepare_data()
        models = build_models(X, Y)
        print("[Trainer] Training complete.")
    except Exception as e:
        print("[Trainer] Not enough data yet:", e)
        return

    while True:
        try:
            preds = forecast(models, scaler)
            gru_vals = preds["gru"]

            now = datetime.now().isoformat()

            pred_row = {
                "datetime":now,
                "model":"GRU",
                "lane1_pred":float(gru_vals[0]),
                "lane2_pred":float(gru_vals[1]),
                "lane3_pred":float(gru_vals[2]),
                "lane4_pred":float(gru_vals[3]),
                "total_pred":float(gru_vals[4])
            }

            with STATE_LOCK:
                STATE["predictions"] = preds
                STATE["signal"] = compute_signal(gru_vals)

            pd.DataFrame([pred_row]).to_csv(PRED_CSV, mode="a",
                                            header=False, index=False)
            safe_append_excel(forecast_rows=[pred_row])

        except Exception as e:
            print("Prediction error:", e)

        time.sleep(8)

# ============================================================
# SIGNAL COMPUTATION
# ============================================================
def compute_signal(gru_vals):
    cycle = 90
    amber = 3

    lane_load = np.array(gru_vals[:NUM_LANES])
    if lane_load.sum() == 0:
        lane_load += 1

    greens = (lane_load / lane_load.sum()) * (cycle - amber*NUM_LANES)
    greens = np.clip(greens, 8, 60)

    signal = {}
    for i,g in enumerate(greens):
        signal[f"Lane {i+1}"] = {
            "GREEN": round(float(g),2),
            "YELLOW": amber,
            "RED": round(float(cycle-g-amber),2)
        }
    return signal

# ============================================================
# VIDEO PROCESSING
# ============================================================
def video_worker(path, run_once=False):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 12
    skip = max(1, int(fps//4))
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % skip != 0:
            continue

        lane_counts, type_counts, speed, annotated = process_frame(frame)

        now = datetime.now()
        row = {
            "datetime": now.isoformat(),
            "date": now.date().isoformat(),
            "time": now.time().isoformat()
        }

        for i in range(NUM_LANES):
            row[f"lane{i+1}"] = lane_counts[i]

        row["total"] = sum(lane_counts)

        for k in type_counts:
            row[k] = type_counts[k]

        row["avg_speed"] = speed

        pd.DataFrame([row]).to_csv(DATA_CSV, mode="a", header=False, index=False)

        with STATE_LOCK:
            STATE["lane_counts"] = lane_counts
            STATE["type_counts"] = type_counts
            STATE["avg_speed_mps"] = speed
            _,buf = cv2.imencode(".jpg", annotated)
            STATE["last_frame_image"] = buf.tobytes()

        if run_once:
            break

    cap.release()

# ============================================================
# FLASK ROUTES
# ============================================================
@app.route("/")
def home():
    return """
    <h1>SMART TRAFFIC ULTIMATE DASHBOARD</h1>
    <p>Upload video using /upload</p>
    <p>Live stats → /stats</p>
    """

@app.route("/stats")
def stats():
    with STATE_LOCK:
        return jsonify({
            "lanes": STATE["lane_counts"],
            "types": STATE["type_counts"],
            "avg_speed": STATE["avg_speed_mps"],
            "predictions": {k:v.tolist() for k,v in STATE["predictions"].items()},
            "signal": STATE["signal"]
        })

@app.route("/last_frame")
def last_frame():
    with STATE_LOCK:
        img = STATE["last_frame_image"]
    if img is None:
        blank = np.zeros((200,200,3), dtype=np.uint8)
        _,buf = cv2.imencode(".jpg", blank)
        img = buf.tobytes()
    return Response(img, mimetype="image/jpeg")

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files['file']
    path = os.path.join("uploads", f.filename)
    os.makedirs("uploads", exist_ok=True)
    f.save(path)

    threading.Thread(target=video_worker, args=(path, True), daemon=True).start()
    return jsonify({"status":"processing", "file":f.filename})

# ============================================================
# START TRAINING THREAD & FLASK
# ============================================================

threading.Thread(target=trainer_loop, daemon=True).start()

if __name__ == "__main__":
    print("▶ Starting SMART TRAFFIC ULTIMATE (Option-D optimized)...")
    print("▶ Open dashboard at: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)