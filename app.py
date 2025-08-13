import os, json, time, uuid
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, flash

from ultralytics import YOLO

# ---------------- Config ----------------
BASE_DIR = Path(__file__).parent
WEIGHTS = str(BASE_DIR / "weights" / "best.pt")
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ROI_PATH = BASE_DIR / "roi.json"       # shared ROI file (optional)
ACCEPT_CLASSES = {"car", "bus", "motorcycle", "autorickshaw", "truck"}

# Default ROI if roi.json not present (edit these to your camera’s geometry)
DEFAULT_ROI = [(100, 300), (540, 300), (620, 430), (60, 430)]

# Green time rule
MIN_GREEN, MAX_GREEN = 8, 60
def calculate_green_time(queue_count, base=10, k=1.8, min_t=MIN_GREEN, max_t=MAX_GREEN, bonus=0):
    return int(max(min_t, min(max_t, base + k*queue_count + bonus)))

# Queue estimator: median of recent counts while RED
class QueueEstimator:
    def __init__(self, fps, window_sec=2.5):
        self.n = max(1, int(fps*window_sec))
        self.buf = deque(maxlen=self.n)
    def add(self, count):
        self.buf.append(count)
    def median(self):
        return int(np.median(self.buf)) if self.buf else 0
    def clear(self):
        self.buf.clear()

# Simple signal state machine
class LightSM:
    def __init__(self, fps, red_fixed=10, yellow_sec=3):
        self.fps = fps
        self.red_fixed = red_fixed
        self.yellow_sec = yellow_sec
        self.state = "RED"
        self.t_end = time.time() + red_fixed
        self.qest = QueueEstimator(fps)
        self.wait_time = 0
        self.pending_green = 12
        self.last_queue_est = 0

    def update(self, per_frame_queue_count):
        now = time.time()
        if self.state == "RED":
            self.qest.add(per_frame_queue_count)
            self.wait_time += 1.0 / self.fps
            if now >= self.t_end:
                q = self.qest.median()
                self.last_queue_est = q
                fairness_bonus = min(10, int(self.wait_time / 30))  # mild anti-starvation
                self.pending_green = calculate_green_time(q, bonus=fairness_bonus)
                self.state = "GREEN"
                self.t_end = now + self.pending_green
                self.qest.clear(); self.wait_time = 0
        elif self.state == "GREEN":
            if now >= self.t_end:
                self.state = "YELLOW"
                self.t_end = now + self.yellow_sec
        elif self.state == "YELLOW":
            if now >= self.t_end:
                self.state = "RED"
                self.t_end = now + self.red_fixed

    def time_left(self):
        return max(0, int(self.t_end - time.time()))

# Helpers
def load_roi():
    if ROI_PATH.exists():
        try:
            pts = json.loads(ROI_PATH.read_text())["points"]
            return np.array(pts, dtype=np.int32)
        except Exception:
            pass
    return np.array(DEFAULT_ROI, dtype=np.int32)

def save_roi_points(points):
    ROI_PATH.write_text(json.dumps({"points": points}))

def inside_polygon(pt, poly):
    return cv2.pointPolygonTest(poly.reshape((-1,1,2)), pt, False) >= 0

def ensure_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if writer.isOpened():
        return writer, path
    base, _ = os.path.splitext(path)
    alt = base + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(alt, fourcc, fps, size)
    if writer.isOpened():
        return writer, alt
    return None, None

# Load YOLO once
model = YOLO(WEIGHTS)
CLASS_NAMES = model.model.names if hasattr(model, "model") else model.names

app = Flask(__name__)
app.secret_key = "dev-secret"  # for flash messages

# --------------- Pages ------------------
@app.route("/")
def index():
    # Small ROI string helper for manual edits
    roi = load_roi().tolist()
    roi_str = ";".join([f"{x},{y}" for x,y in roi])
    return render_template("index.html", roi_str=roi_str)

@app.route("/camera")
def camera_page():
    return render_template("camera.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/play/<vid_id>")
def play_page(vid_id):
    return render_template("play.html", vid_id=vid_id)

# ------------- ROI handling -------------
@app.route("/set_roi", methods=["POST"])
def set_roi():
    roi_str = request.form.get("roi_str", "").strip()
    try:
        pts = []
        for pair in roi_str.split(";"):
            x, y = pair.split(",")
            pts.append((int(float(x)), int(float(y))))
        if len(pts) >= 3:
            save_roi_points(pts)
            flash("ROI updated ✅", "success")
        else:
            flash("ROI must have at least 3 points.", "error")
    except Exception as e:
        flash(f"Invalid ROI string: {e}", "error")
    return redirect(url_for("index"))

# ------------- Camera stream ------------
@app.route("/camera_feed")
def camera_feed():
    def gen():
        cap = cv2.VideoCapture(0)  # default webcam
        if not cap.isOpened():
            yield b""
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sm  = LightSM(fps=fps)
        roi = load_roi()

        while True:
            ok, frame = cap.read()
            if not ok:
                continue  # webcams sometimes drop a frame

            # Draw ROI
            cv2.polylines(frame, [roi], True, (0,255,0), 2)

            # Inference
            res = model.predict(source=frame, conf=0.25, iou=0.5, imgsz=640, verbose=False)[0]
            boxes = getattr(res, "boxes", None)

            qcount = 0
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls  = boxes.cls.cpu().numpy().astype(int)
                for (x1,y1,x2,y2), c in zip(xyxy, cls):
                    cname = CLASS_NAMES.get(int(c), str(c)) if isinstance(CLASS_NAMES, dict) else CLASS_NAMES[int(c)]
                    if cname not in ACCEPT_CLASSES:
                        continue
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    if inside_polygon((cx,cy), roi):
                        qcount += 1
                        cv2.circle(frame, (cx,cy), 4, (255,0,0), -1)
                    cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,140,255), 2)
                    cv2.putText(frame, cname, (int(x1), max(0,int(y1)-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,255), 2)

            # Update state machine
            sm.update(qcount)

            # HUD
            cv2.putText(frame, f"Queue in ROI: {qcount}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(frame, f"State: {sm.state}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(frame, f"Time left: {sm.time_left()}s", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            # Light widget
            h, w = frame.shape[:2]
            cx0, cy0, r = w-60, 70, 16
            red, yellow, green = (50,50,50), (50,50,50), (50,50,50)
            if   sm.state == "RED":    red    = (0,0,255)
            elif sm.state == "YELLOW": yellow = (0,255,255)
            elif sm.state == "GREEN":  green  = (0,200,0)
            cv2.circle(frame, (cx0, cy0+0), r, red,   -1)
            cv2.circle(frame, (cx0, cy0+40), r, yellow,-1)
            cv2.circle(frame, (cx0, cy0+80), r, green, -1)

            # JPEG encode and stream
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        cap.release()

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ------------- Upload handling ----------
@app.route("/upload_video", methods=["POST"])
def upload_video():
    f = request.files.get("video")
    if not f or f.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("upload_page"))
    vid_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{vid_id}_{f.filename}"
    f.save(str(save_path))
    return redirect(url_for("play_page", vid_id=f"{vid_id}_{f.filename}"))

# ------------- Uploaded video stream ----
@app.route("/video_feed/<vid_id>")
def video_feed(vid_id):
    video_path = str(UPLOAD_DIR / vid_id)

    def gen():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield b""
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sm  = LightSM(fps=fps)
        roi = load_roi()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            cv2.polylines(frame, [roi], True, (0,255,0), 2)

            res = model.predict(source=frame, conf=0.25, iou=0.5, imgsz=640, verbose=False)[0]
            boxes = getattr(res, "boxes", None)

            qcount = 0
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls  = boxes.cls.cpu().numpy().astype(int)
                for (x1,y1,x2,y2), c in zip(xyxy, cls):
                    cname = CLASS_NAMES.get(int(c), str(c)) if isinstance(CLASS_NAMES, dict) else CLASS_NAMES[int(c)]
                    if cname not in ACCEPT_CLASSES:
                        continue
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    if inside_polygon((cx,cy), roi):
                        qcount += 1
                        cv2.circle(frame, (cx,cy), 4, (255,0,0), -1)
                    cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,140,255), 2)
                    cv2.putText(frame, cname, (int(x1), max(0,int(y1)-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,255), 2)

            sm.update(qcount)

            cv2.putText(frame, f"Queue in ROI: {qcount}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(frame, f"State: {sm.state}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(frame, f"Time left: {sm.time_left()}s", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            h, w = frame.shape[:2]
            cx0, cy0, r = w-60, 70, 16
            red, yellow, green = (50,50,50), (50,50,50), (50,50,50)
            if   sm.state == "RED":    red    = (0,0,255)
            elif sm.state == "YELLOW": yellow = (0,255,255)
            elif sm.state == "GREEN":  green  = (0,200,0)
            cv2.circle(frame, (cx0, cy0+0), r, red,   -1)
            cv2.circle(frame, (cx0, cy0+40), r, yellow,-1)
            cv2.circle(frame, (cx0, cy0+80), r, green, -1)

            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

        cap.release()

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ------------- Run ----------------------
if __name__ == "__main__":
    # For dev only; use a proper WSGI server in production
    app.run(host="0.0.0.0", port=5000, debug=True)
