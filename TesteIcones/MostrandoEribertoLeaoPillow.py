from ultralytics import YOLO
import supervision as sv
import numpy as np
from PIL import Image, ImageDraw
import cv2

# Carregar o modelo YOLO
model = YOLO('yolov8n.pt')

# Abrir o vídeo de entrada
cap = cv2.VideoCapture("carriata de beto do brasil.mp4")
icon_annotator = sv.IconAnnotator()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converter frame do formato BGR do OpenCV para RGB para usar com Pillow
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Aplicar a detecção de objetos
    results = model(frame)
    car_positions = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if model.names[class_id] == 'car':
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                car_positions.append((x1, y1 - 50, x2, y2))

    if car_positions:
        detections = sv.Detections(
            xyxy=np.array(car_positions),
            confidence=np.array([0.6] * len(car_positions)),
            class_id=np.array([0] * len(car_positions))  # Classe 'car'
        )

        # Anotar a imagem com o ícone usando supervision
        annotated_image = icon_annotator.annotate(
            scene=pil_image,
            detections=detections,
            icon_path="EribertoLeao.jpeg"
        )

        # Converter a imagem anotada de volta para OpenCV (BGR) para exibir
        annotated_frame = np.array(annotated_image)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Exibir o frame anotado
        cv2.imshow("Annotated Video Frame", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
