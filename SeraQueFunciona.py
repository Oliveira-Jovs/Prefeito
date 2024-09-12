from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Carregar o modelo YOLO pré-treinado (v8)
model = YOLO('yolov8n.pt')  

# Lista para armazenar as posições dos carros
car_positions = []

# Inicializar o vídeo
cap = cv2.VideoCapture("carriata de beto do brasil.mp4")  # Substitua pelo caminho do seu vídeo

# Caminho para o ícone a ser usado nas anotações
icon_path = "EribertoLeao.jpeg"

# Inicializar o IconAnnotator da biblioteca Supervision
icon_annotator = sv.IconAnnotator()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção de objetos com o YOLOv8
    results = model(frame)
    
    # Iterar sobre as detecções e filtrar os carros
    car_positions.clear()  # Limpar a lista de posições de carros para o novo frame
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            
            if class_name == 'car':  # Verificar se o objeto detectado é um carro
                # Coordenadas da caixa delimitadora
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                car_positions.append((x1, y1, x2, y2))  # Adicionar a posição à lista

    # Se houver carros detectados, aplicar o IconAnnotator
    if car_positions:
        # Converter para numpy arrays esperados pelo supervision
        detections = sv.Detections(
            xyxy=np.array(car_positions),
            confidence=np.array([0.9] * len(car_positions)),  # Ajustar a confiança conforme necessário
            class_id=np.array([0] * len(car_positions))  # Classe 'car' como 0
        )

        # Anotar o frame com o ícone
        annotated_frame = icon_annotator.annotate(
            scene=frame.copy(),
            detections=detections,
            icon_path=icon_path
        )

        # Exibir o frame anotado
        cv2.imshow("Annotated Video Frame", annotated_frame)

    # Finalizar se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
