import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Inicialización de mediapipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuración del módulo de manos de mediapipe
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directorio de datos
DATA_DIR = './imgs'

data = []  # Lista para almacenar datos procesados
labels = []  # Lista para almacenar etiquetas de clases

# Recorre los directorios dentro de DATA_DIR (cada uno representa una clase)
for dir_ in os.listdir(DATA_DIR):
    # Recorre las imágenes dentro del directorio de cada clase
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Lista auxiliar para almacenar coordenadas normalizadas
        x_ = []  # Lista para almacenar coordenadas x de los puntos de referencia
        y_ = []  # Lista para almacenar coordenadas y de los puntos de referencia

        # Lee la imagen y la convierte a RGB
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Procesa la imagen para detectar manos
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Si se detectan manos, se procesan los puntos de referencia
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normaliza las coordenadas de los puntos de referencia
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)  # Añade los datos procesados a la lista principal
            labels.append(dir_)  # Añade la etiqueta de la clase correspondiente

# Guarda los datos y etiquetas en un archivo pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
