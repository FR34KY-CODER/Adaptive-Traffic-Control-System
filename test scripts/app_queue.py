import os, json, time, argparse
import numpy as np
import cv2
from ultralytics import YOLO

ACCEPT_CLASSES = {"car","bus","motorcycle","autorickshaw","truck"}

def parse_roi(roi_arg):
    if roi_arg.endswith(".json") and os.path.exists(roi_arg):
        with open(roi_arg) as f:
            pts = json.load(f)["points"]
            return np.array(pts, dtype=np.int32)
    # also support "x1,y1;x2,y2;..."
    pts = []
    for p in roi_arg.split(";"):
        x,y = p.split(","); pts.append((int(float(x)), int(float(y))))
    return np.array(pts, dtype=np.int32)

def inside_polygon(pt, poly):
    return cv2.pointPolygonTest(poly.reshape((-1,1,2)), pt, False) >= 0

def calculate_green_time(queue_count, base=10, k=1.8, min_t=8, max_t=60, bonus=0):
    return int(max(min_t, min(max_t, base + k*queue_count + bonus)))

class QueueEstimator:
    def __init__(self, fps, window_sec=2.5):
        self.n = max(1, int(fps*window_sec))
        self.buf = []
    def add(self, count):
        self.buf.append(count)
        if len(self.buf) > self.n: self.buf.pop(0)
    def median(self):
        return int(np.median(self.buf)) if self.buf else 0
    def clear(self):
        self.buf.clear()

class LightSM:
    # Simple RED->GREEN->YELLOW loop for a single approach demo
    def __init__(self, fps, red_fixed=10, yellow_sec=3):
        self.fps = fps
        self.state = "RED"
        self.red_fixed = red_fixed
        self.yellow_sec = yellow_sec
        self.t_end = time.time() + red_fixed
        self.qest = QueueEstimator(fps)
        self.wait_time = 0
        self.pending_green = 12

    def update(self, per_frame_queue_count):
        now = time.time()
        if self.state == "RED":
            self.qest.add(per_frame_queue_count)
            self.wait_time += 1.0 / self.fps
            if now >= self.t_end:
                q = self.qest.median()
                fairness_bonus = min(10, int(self.wait_time/30))  # small anti-starvation
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

def run(weights, video, roi, out_path, conf=0.25, iou=0.5, headless=False):
    model = YOLO(weights)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    poly = parse_roi(roi)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))

    sm = LightSM(fps=fps)

    while True:
        ok, frame = cap.read()
        if not ok: break

        # Draw ROI
        cv2.polylines(frame, [poly], True, (0,255,0), 2)

        # Inference
        res = model.predict(source=frame, conf=conf, iou=iou, verbose=False, max_det=300)[0]
        boxes = getattr(res, "boxes", None)

        qcount = 0
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls  = boxes.cls.cpu().numpy().astype(int)
            names = res.names  # id->name from the trained model
            for (x1,y1,x2,y2), c in zip(xyxy, cls):
                cname = names.get(int(c), str(c)) if isinstance(names, dict) else names[int(c)]
                if cname not in ACCEPT_CLASSES:
                    continue
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                # count only if center is within ROI (stopped vehicles at stop-line)
                if inside_polygon((cx,cy), poly):
                    qcount += 1
                    cv2.circle(frame, (cx,cy), 4, (255,0,0), -1)
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,140,255), 2)
                cv2.putText(frame, cname, (int(x1), max(0,int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,255), 2)

        # Update light state using queue count measured during RED window
        sm.update(qcount)

        # HUD
        cv2.putText(frame, f"Queue in ROI: {qcount}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, f"State: {sm.state}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(frame, f"Time left: {sm.time_left()}s", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # simple traffic-light widget
        cx0, cy0, r = w-60, 70, 16
        red, yellow, green = (50,50,50), (50,50,50), (50,50,50)
        if   sm.state == "RED":    red    = (0,0,255)
        elif sm.state == "YELLOW": yellow = (0,255,255)
        elif sm.state == "GREEN":  green  = (0,200,0)
        cv2.circle(frame,(cx0, cy0+0), r, red,   -1)
        cv2.circle(frame,(cx0, cy0+40), r, yellow,-1)
        cv2.circle(frame,(cx0, cy0+80), r, green, -1)

        writer.write(frame)
        if not headless:
            cv2.imshow("Adaptive TL Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release(); writer.release()
    if not headless:
        cv2.destroyAllWindows()
    print(f"Saved demo to {out_path}")

if __name__ == "__main__":
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="path to best.pt")
    parser.add_argument("--video",   required=True, help="input video path")
    parser.add_argument("--roi",     required=True, help="roi.json or 'x1,y1;...'")
    parser.add_argument("--out",     default="outputs/sample_output.mp4")
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.5)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # auto-headless on Kaggle
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        args.headless = True

    run(args.weights, args.video, args.roi, args.out, args.conf, args.iou, headless=args.headless)
