from ultralytics import YOLO
import cv2

cerebro = YOLO("yolov8x.pt")
frame = "frame_0031.jpg"

usando_cerebro = cerebro(frame)

img = cv2.imread(frame)
coordenadas_objetos_no_frame = list()
cls = ""
for result in usando_cerebro:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        coordenadas_objetos_no_frame.append([x1, y1, x2, y2])
        conf = float(box.conf)
        cls = int(box.cls)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img, 'O', (x1-20, y1 -50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        #cv2.putText(img, f'{cls} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ((255,255,0)), 2)


cv2.imshow("Mostrar a imagem",img)
cv2.waitKey(0)

