import cv2
cap = cv2.VideoCapture("videos/test.mp4")  # change path if needed
print("cap opened:", cap.isOpened())
ok, frame = cap.read()
print("first frame ok:", ok)
if ok: print("frame size:", frame.shape)
cap.release()