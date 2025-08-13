# Adaptive Traffic Control (YOLOv8 + Flask)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python\&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000.svg?logo=flask\&logoColor=white)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-8A2BE2.svg)](https://docs.ultralytics.com/)

> **AI-powered signal timing** that counts stopped vehicles near the stop-line and dynamically picks the **next green time**. Works with **live webcam** or **uploaded videos**. Built with **YOLOv8**, **OpenCV**, and **Flask**.

---

## 🔥 TL;DR

* **Two modes**:

  1. **Camera Mode** – live webcam stream with detection + queue counting + adaptive timer
  2. **Upload Mode** – upload a clip and watch processed frames in-browser
* **Queue logic**: Estimates **stopped vehicles** inside a polygon **ROI** during **RED**, then computes **GREEN** time via `base + k*queue` (clamped).
* **Stable visuals**: Tiny tracker + smoothing to reduce jitter (“flying boxes”).
* **India-centric**: Trained initially on **IITM-HeTra** (car/bus). Ready to mix **IRUVD** for motorcycles/autorickshaws/trucks.

---

## 📽️ Live & Upload Demos

<img src="https://github.com/FR34KY-CODER/Adaptive-Traffic-Control-System/blob/main/Screenshots/1.png?raw=true" align = center>
<br>
<img src = "https://github.com/FR34KY-CODER/Adaptive-Traffic-Control-System/blob/main/Screenshots/2.png?raw=true" align = center>

---

## 🧠 How It Works

```mermaid
flowchart LR;
  A[Camera / Uploaded Video] -->|Frame| B[YOLOv8 Detector]
  B --> C{Accept classes? <br/>car, bus, moto, auto, truck}
  C -->|Yes| D[ROI center test]
  D -->|Inside| E[Queue count]
  E --> F[Queue buffer (2.5s) while RED]
  F --> G[Median queue]
  G --> H[GREEN time = clamp(min, base + k*q + bonus(wait), max)]
  H --> I[RED→GREEN→YELLOW state machine]
  I --> J[Overlay HUD + Light widget]
  J --> K[Stream to browser]
  C -->|No| J
```

**Highlights**

* **ROI** = polygon around the **stopped area** just before the stop-line.
* **Median of last W seconds** during **RED** = robust queue estimate.
* **Fairness bonus** grows with wait time to avoid starvation.
* **TinyTrack** = lightweight IoU tracker + EMA smoothing → fewer jumpy boxes.

---

## ✨ Features

* ✅ Live MJPEG stream to the browser (camera & uploaded video)
* ✅ ROI editor on the home page (string format `x,y;x,y;...`)
* ✅ Adaptive traffic light widget + countdown
* ✅ Tiny tracker for visual stability
* ✅ Dark, glassy UI (Bootstrap 5) with pause/resume & snapshot
* ✅ Kaggle-ready training notebook flow

---

## 🗂️ Project Structure

```
adaptive-traffic/
├─ app.py                  # Flask web app (two modes, tracker, UI)
├─ weights/
│  └─ best.pt              # YOLOv8 weights (download from Kaggle training)
├─ uploads/                # uploaded videos (auto-created)
├─ templates/              # HTML templates (Bootstrap 5)
│  ├─ base.html
│  ├─ index.html
│  ├─ camera.html
│  ├─ upload.html
│  └─ play.html
├─ static/
│  ├─ style.css            # dark theme, glass cards, skeleton loaders
│  └─ app.js               # pause/resume, snapshot, ROI helpers
├─ roi.json                # saved ROI polygon (shared across modes)
└─ docs/
   └─ media/               # put your GIFs here (hero_demo.gif, etc.)
```

---

## ⚙️ Setup

1. **Python deps**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install flask ultralytics==8.3.178 opencv-python numpy
```

2. **Weights**
   Put your trained weights at `weights/best.pt`. (See “Training” below.)

3. **Run**

```bash
python app.py
# open http://localhost:5000
```

---

## 🎛️ Configuration

* **ROI**: Home page → paste/edit polygon string like:

  ```
  100,300;540,300;620,430;60,430
  ```

  Save once; both modes use the same ROI.

* **Timing rule** (`app.py`):

  ```python
  MIN_GREEN, MAX_GREEN = 8, 60
  def calculate_green_time(queue_count, base=10, k=1.8, min_t=MIN_GREEN, max_t=MAX_GREEN, bonus=0):
      return int(max(min_t, min(max_t, base + k*queue_count + bonus)))
  ```

  Tweak `base`, `k`, `min/max`, and the bonus logic to match your intersection.

* **Detection thresholds** (in generators):

  ```python
  conf = 0.40   # raise to reduce jitter
  iou  = 0.50
  imgsz = 512   # live; 896-960 for uploaded video
  ```

---

## 🧪 Training (Kaggle-first)

1. **Datasets**

   * Start: **IITM-HeTra** → strong for **car/bus**.
   * Add later: **IRUVD** → expands to **motorcycle / autorickshaw / truck**.

2. **Notebook**

   * Attach dataset(s) via Kaggle “Add data”.
   * Convert XML→YOLO (car/bus first), write `data.yaml`, split train/val/test.
   * Train:

     ```python
     from ultralytics import YOLO
     model = YOLO("yolov8s.pt")       # 's' model balances accuracy vs speed
     model.train(data="data.yaml", epochs=60, imgsz=960, batch=-1, patience=15, workers=0, seed=42, cos_lr=True)
     model.val(data="data.yaml")
     ```
   * Download `best.pt` to `weights/`.

3. **Next**

   * Merge **IRUVD** labels → re-train → expect better detection of **two-wheelers & autos**, especially in dense Indian traffic.

---

## 🧰 Make GIFs (for README)

Using ffmpeg:

```bash
# 1) Create a short MP4 clip (10s)
ffmpeg -y -i outputs/demo.mp4 -t 10 docs/media/hero_clip.mp4

# 2) Convert to GIF (reasonable size)
ffmpeg -y -i docs/media/hero_clip.mp4 -vf "fps=12,scale=960:-1:flags=lanczos" -r 12 docs/media/hero_demo.gif
```

---

## 🤝 UI Highlights

* Dark, glassy cards + skeleton shimmer for streams
* **Camera page**: pause/resume & snapshot
* **Upload page**: nicer form and animated progress
* **Playback**: clean stream container, responsive

*Screenshots (add yours):*

<p align="center">
  <img src="docs/media/home.png" width="45%" />
  <img src="docs/media/camera.png" width="45%" />
</p>

---

## 📈 Tips for Better Results

* **Raise `conf`** to 0.40–0.50 to cut spurious boxes.
* **Increase `imgsz`** for offline uploads (896–960).
* **Good ROI** = fewer false counts (cover only the stopped area).
* **Retrain with IRUVD** for motorcycles/autorickshaws/trucks in Indian scenes.
* On CPU: use `yolov8n.pt` for tests or `imgsz=512` for speed.

---

## 🐛 Troubleshooting

* **“Video not smooth”** → use `imgsz=512` live; try GPU; limit frame rate; streaming is MJPEG.
* **“Boxes flying”** → ensure tracker patch is enabled + raise `conf`.
* **“Can’t open camera”** → try source `1` or `2`; external webcams may be `1`.
* **“Writer failed”** → app falls back to `AVI/XVID`; or install a full ffmpeg build.

---

## 🗺️ Roadmap

* [ ] Multi-approach simulator (2–4 videos cycling phases)
* [ ] Interactive in-browser ROI drawer overlay
* [ ] ByteTrack/BoT-Sort option for stronger tracking
* [ ] Per-class weighting (e.g., bus > car) in queue → green time
* [ ] Export CSV of queue estimates + chosen greens for analytics

---

## 🙏 Acknowledgements

* **Ultralytics YOLOv8** for a great training/inference API.
* **IITM-HeTra / IRUVD** datasets for India-centric visuals.

---

### ⭐ If this helped, consider starring the repo!

Have ideas or want the multi-approach simulator added? Open an issue and I’ll jump in.
