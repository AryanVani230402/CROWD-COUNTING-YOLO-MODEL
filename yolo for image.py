from pygame import math
from ultralytics import YOLO
import cvzone
import cv2

img = "bus.jpg"
model = YOLO("yolov8n.pt")
result=model(img)
for r in result:
    boxes=r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        if(cls==0):
            result=model(show=True)

# while True:
#     result=model(img)
#     for r in result:
#         boxes = r.boxes
#         for box in boxes:
#             cls = int(box.cls[0])
#             if(cls==0):
#                 # Bounding Box
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#                 # w, h = x2 - x1, y2 - y1
#                 # cvzone.cornerRect(img, (x1, y1, w, h))
#                 # Confidence
#                 conf = math.ceil((box.conf[0] * 100)) / 100
#                 # Class Name
#
#
#                 cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
#     cv2.waitKey(0)
