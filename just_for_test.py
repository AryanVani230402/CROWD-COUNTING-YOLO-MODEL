
from ultralytics import YOLO
import cv2

cap=cv2.VideoCapture(0)
img=cap.read()
model=YOLO('yolov8l.pt')
result = model(img,show=True)
cv2.waitKey(0)
