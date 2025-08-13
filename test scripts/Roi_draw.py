# Extract the very first frame to an image:
import cv2, os
os.makedirs("roi", exist_ok=True)
cap = cv2.VideoCapture("videos/test.mp4")
ok, frame = cap.read()
cap.release()
print("First frame read:", ok, "shape:" if ok else "", frame.shape if ok else "")
if ok: cv2.imwrite("roi/frame0.jpeg", frame)