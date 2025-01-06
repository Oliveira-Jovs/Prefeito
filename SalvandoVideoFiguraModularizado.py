from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

# Carregar o modelo YOLO
model = YOLO('yolov8x.pt')

# Abrir o vídeo de entrada
cap = cv2.VideoCapture("CarreataReduzida.mp4")
if not cap.isOpened():
    print("Erro ao abrir o vídeo de entrada!")
    exit()

icon_path = "ThiagoPatricioImagemReduzida385.pngk"
icon_annotator = sv.IconAnnotator()

# Obter dimensões do vídeo de entrada
width = int(cap.get(3))
height = int(cap.get(4))
print(f"Largura: {width}, Altura: {height}")

# Definir as propriedades do vídeo de saída
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para o arquivo de saída
out = cv2.VideoWriter('CarriataReduzidaAnotacaoPatricio.mp4', fourcc, 30.0, (width, height))

# Verificar se o arquivo de saída foi aberto corretamente
if not out.isOpened():
    print("Erro ao abrir o arquivo de vídeo para gravação!")
    cap.release()
    cv2.destroyAllWindows()
    exit()

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
            confidence=np.array([0.7] * len(car_positions)),
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

# Liberar os recursos
cap.release()
out.release()  # Fechar o arquivo de vídeo de saída
cv2.destroyAllWindows()
