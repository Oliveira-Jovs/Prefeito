from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

# Carregar o modelo YOLO
model = YOLO('yolov8x.pt')

# Abrir o vídeo de entrada
cap = cv2.VideoCapture("carriata de beto do brasil.mp4")
icon_path = "EribertoLeao.jpeg"
icon_annotator = sv.IconAnnotator()

# Definir as propriedades do vídeo de saída
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para o arquivo de saída
out = cv2.VideoWriter('CarriataAnotacao.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicar a detecção de objetos
    results = model(frame)

    car_positions = []  # Limpar a lista de posições de carros e motos para o novo frame
    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls)]

            # Verificar se o objeto é um carro ou uma moto
            if class_name in ['car', 'motorcycle']:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                car_positions.append((x1, y1 - 50, x2, y2))

    if car_positions:
        # Converter para numpy arrays esperados pelo supervision
        detections = sv.Detections(
            xyxy=np.array(car_positions),
            confidence=np.array([0.6] * len(car_positions)),
            class_id=np.array([0] * len(car_positions))  # Classe 'car' ou 'motorcycle'
        )

        # Anotar o frame com o ícone
        annotated_frame = icon_annotator.annotate(
            scene=frame,
            detections=detections,
            icon_path=icon_path
        )

        # Gravar o frame anotado no vídeo de saída
        out.write(annotated_frame)

        # Exibir o frame anotado (opcional)
        cv2.imshow("Annotated Video Frame", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
out.release()  # Fechar o arquivo de vídeo de saída
cv2.destroyAllWindows()
