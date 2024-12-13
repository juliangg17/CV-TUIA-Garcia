import cv2
import ultralytics
import tensorflow as tf
import joblib
import numpy as np
from ultralytics import YOLO
import os
from tensorflow.keras.models import load_model

model_k = YOLO("model_k.pt")
model_p = load_model("model_p.h5")
scaler = joblib.load("scaler.pkl")


# Lista de índices de keypoints que el scaler y el modelo de clasificación esperan
lesskeypoints_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]

def predict_pose_from_video(model_k, model_p, scaler, video_path, lesskeypoints_list, threshold=0.5):
    """
    Procesa un video cuadro a cuadro, detecta los keypoints y clasifica la pose.
    Muestra la etiqueta de la clase predicha y la probabilidad sobre el video
    solo si la probabilidad supera un umbral. Guarda el video anotado automáticamente
    agregando una 'p' al final del nombre original.

    Argumentos:
        model_k: Modelo YOLO para detectar keypoints.
        model_p: Modelo para clasificar la pose.
        scaler: Scaler usado para normalizar las coordenadas durante el entrenamiento del modelo de clasificación.
        video_path: Ruta del video de entrada.
        threshold: Umbral mínimo de probabilidad para mostrar la clase (por defecto 0.5).

    Retorna:
        None
    """
    # Verificar si el archivo de video existe
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"El archivo de video no existe: {video_path}")

    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video.")

    # Crear el nombre del archivo de salida automáticamente
    base_name, ext = os.path.splitext(video_path)
    output_path = f"{base_name}p{ext}"

    # Configurar el video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para guardar el video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Etiquetas de las clases
    class_labels = ["Tumbado", "Sentado", "Parado"]

    while True:
        # Leer un cuadro del video
        ret, frame = cap.read()
        if not ret:
            break  # Fin del video

        # Realizar la predicción de keypoints
        results = model_k(frame)

        # Obtener keypoints de la predicción
        coordinates_example = []
        if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
            keypoints_data = results[0].keypoints.data.cpu().numpy()  # Convertir a numpy array
            for row in keypoints_data[0]:  # Suponiendo que es un array 3D [1, N, 3]
                coordinates_example.append((float(row[0]), float(row[1])))
        else:
            coordinates_example = [(0.00, 0.00)] * 24

        # Filtrar los keypoints
        # Filtrar solo los índices válidos
        filtered_coordinates = [
            coordinates_example[i] if i < len(coordinates_example) else (0.0, 0.0)
            for i in lesskeypoints_list
        ]

        expected_features = len(lesskeypoints_list)
        filtered_coordinates = filtered_coordinates[:expected_features]
        filtered_coordinates.extend([(0.0, 0.0)] * (expected_features - len(filtered_coordinates)))

        # Validar dimensiones antes de la normalización
        input_data = np.array(filtered_coordinates).flatten().reshape(1, -1)
        if input_data.shape[1] != scaler.n_features_in_:
            raise ValueError(f"El número de características ({input_data.shape[1]}) no coincide con el escalador ({scaler.n_features_in_}).")

        # Normalizar las coordenadas
        input_data_normalized = scaler.transform(input_data)

        # Realizar la predicción de la pose
        probabilities = model_p.predict(input_data_normalized)
        predicted_class_index = np.argmax(probabilities)
        predicted_class_label = class_labels[predicted_class_index]
        predicted_class_probability = probabilities[0][predicted_class_index]  # Obtener la probabilidad

        # Dibujar los keypoints y las conexiones en el cuadro
        for i, (x, y) in enumerate(coordinates_example):
            if (x, y) != (0.00, 0.00):  # Dibujar solo keypoints válidos
                cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)

        # Mostrar la etiqueta de la pose y la probabilidad sobre el cuadro solo si supera el umbral
        if predicted_class_probability >= threshold:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"{predicted_class_label}: {predicted_class_probability:.2f}"
            text_position = (20, 50)  # Posición del texto
            cv2.putText(frame, text, text_position, font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar el cuadro usando OpenCV
        cv2.imshow("Video con Predicciones", frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Guardar el cuadro en el video de salida
        out.write(frame)

    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Llamar la función
predict_pose_from_video(
    model_k=model_k,
    model_p=model_p,
    scaler=scaler,
    video_path= "videos/perro3_r.mp4",
    lesskeypoints_list=lesskeypoints_list,
    threshold=0.6  # Ajusta el umbral según sea necesario
)



import os
import cv2
import numpy as np

def predict_pose_and_dog_from_video(model_k, model_p, scaler, video_path, lesskeypoints_list, pose_threshold=0.5, bbox_threshold=0.5):
    """
    Procesa un video cuadro a cuadro, detecta keypoints y bounding boxes de la clase "perro",
    y clasifica la pose con un modelo de clasificación.
    Muestra la etiqueta de la pose, la probabilidad, el número de keypoints válidos,
    y dibuja los bounding boxes de los perros detectados.

    Argumentos:
        model_k: Modelo YOLO para detectar keypoints y bounding boxes.
        model_p: Modelo para clasificar la pose.
        scaler: Scaler usado para normalizar las coordenadas durante el entrenamiento del modelo de clasificación.
        video_path: Ruta del video de entrada.
        lesskeypoints_list: Lista de índices de keypoints relevantes.
        pose_threshold: Umbral mínimo de probabilidad para mostrar la pose.
        bbox_threshold: Umbral mínimo de probabilidad para mostrar los bounding boxes.

    Retorna:
        None
    """
    # Verificar si el archivo de video existe
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"El archivo de video no existe: {video_path}")

    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video.")

    # Crear el nombre del archivo de salida automáticamente
    base_name, ext = os.path.splitext(video_path)
    output_path = f"{base_name}_pose_dog{ext}"

    # Configurar el video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para guardar el video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Etiquetas de las clases
    class_labels = ["Tumbado", "Sentado", "Parado"]

    while True:
        # Leer un cuadro del video
        ret, frame = cap.read()
        if not ret:
            break  # Fin del video

        # Realizar la predicción de keypoints y bounding boxes
        results = model_k(frame)

        # Inicializar variable para etiqueta "dog"
        dog_label = None

        # Dibujar bounding boxes para "perros"
        if results[0].boxes is not None:  # Verificar que existan bounding boxes
            for box in results[0].boxes:  # Iterar sobre cada bounding box
                conf = box.conf.item()  # Obtener la probabilidad de confianza
                if conf >= bbox_threshold:  # Filtrar por el umbral de bounding boxes
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Coordenadas del bounding box
                    cls = int(box.cls.item())  # Clase predicha como índice
                    label = f"{results[0].names[cls]} {conf:.2f}"  # Etiqueta con clase y probabilidad

                    if results[0].names[cls] == "dog":  # Solo considerar bounding boxes de perros
                        # Dibujar el bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)  # Rectángulo azul
                        # Guardar la etiqueta "dog" para dibujarla después
                        dog_label = label

        # Dibujar la etiqueta "dog" en la esquina inferior izquierda
        if dog_label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_x = 10  # Coordenada x fija en la esquina inferior izquierda
            label_y = height - 10  # Coordenada y cerca del borde inferior
            cv2.putText(frame, dog_label, (label_x, label_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Obtener keypoints de la predicción
        coordinates_example = []
        if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
            keypoints_data = results[0].keypoints.data.cpu().numpy()  # Convertir a numpy array
            for row in keypoints_data[0]:  # Suponiendo que es un array 3D [1, N, 3]
                coordinates_example.append((float(row[0]), float(row[1])))
        else:
            coordinates_example = [(0.00, 0.00)] * 24

        # Filtrar los keypoints
        filtered_coordinates = [
            coordinates_example[i] if i < len(coordinates_example) else (0.0, 0.0)
            for i in lesskeypoints_list
        ]

        expected_features = len(lesskeypoints_list)
        filtered_coordinates = filtered_coordinates[:expected_features]
        filtered_coordinates.extend([(0.0, 0.0)] * (expected_features - len(filtered_coordinates)))

        # Contar keypoints válidos (distintos de (0.0, 0.0))
        valid_keypoints_count = sum(1 for x, y in coordinates_example if (x, y) != (0.0, 0.0))

        # Validar dimensiones antes de la normalización
        input_data = np.array(filtered_coordinates).flatten().reshape(1, -1)
        if input_data.shape[1] != scaler.n_features_in_:
            raise ValueError(f"El número de características ({input_data.shape[1]}) no coincide con el escalador ({scaler.n_features_in_}).")

        # Normalizar las coordenadas
        input_data_normalized = scaler.transform(input_data)

        # Realizar la predicción de la pose
        probabilities = model_p.predict(input_data_normalized)
        predicted_class_index = np.argmax(probabilities)
        predicted_class_label = class_labels[predicted_class_index]
        predicted_class_probability = probabilities[0][predicted_class_index]  # Obtener la probabilidad

        # Mostrar la etiqueta de la pose y la probabilidad sobre el cuadro solo si supera el umbral
        if predicted_class_probability >= pose_threshold:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"{predicted_class_label}: {predicted_class_probability:.2f}"
            cv2.putText(frame, text, (20, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Dibujar el número de keypoints válidos en la esquina superior derecha
        keypoints_text = f"Keypoints: {valid_keypoints_count}"
        (font_width, font_height), _ = cv2.getTextSize(keypoints_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = max(10, width - font_width - 10)
        text_y = max(30, 50)
        cv2.putText(frame, keypoints_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Mostrar el cuadro usando OpenCV
        cv2.imshow("Video con Predicciones", frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Guardar el cuadro en el video de salida
        out.write(frame)

    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()


predict_pose_and_dog_from_video(
    model_k=model_k,
    model_p=model_p,
    scaler=scaler,
    video_path= "videos/perro2.mp4",
    lesskeypoints_list=lesskeypoints_list,
    pose_threshold=0.6,
    bbox_threshold=0.5
)