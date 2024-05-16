# Proyecto # 4 - Vision por computadora
# @ buildDataSet.py - encargado de construir el .pickle con los datos.

import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Inicialización de mediapipe para detección de manos
mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles

# Configuración del módulo de manos de mediapipe
handsDetector = mpHands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directorio de datos
dataDirectory = './imgs'

handLandmarksData = []  # Lista para almacenar datos procesados
classLabels = []  # Lista para almacenar etiquetas de clases

# Recorre los directorios dentro de dataDirectory (cada uno representa una clase)
for classDir in os.listdir(dataDirectory):
    # Recorre las imágenes dentro del directorio de cada clase
    for imageFilename in os.listdir(os.path.join(dataDirectory, classDir)):
        landmarksNormalized = []  # Lista auxiliar para almacenar coordenadas normalizadas
        xCoordinates = []  # Lista para almacenar coordenadas x de los puntos de referencia
        yCoordinates = []  # Lista para almacenar coordenadas y de los puntos de referencia

        # Lee la imagen y la convierte a RGB
        imagePath = os.path.join(dataDirectory, classDir, imageFilename)
        image = cv2.imread(imagePath)
        if image is None:
            print(f"Error: No se pudo leer la imagen {imagePath}")
            continue
        imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesa la imagen para detectar manos
        results = handsDetector.process(imageRgb)
        if results.multi_hand_landmarks:
            # Si se detectan manos, se procesan los puntos de referencia
            for handLandmarks in results.multi_hand_landmarks:
                for landmark in handLandmarks.landmark:
                    xCoordinates.append(landmark.x)
                    yCoordinates.append(landmark.y)

                # Normaliza las coordenadas de los puntos de referencia
                for landmark in handLandmarks.landmark:
                    landmarksNormalized.append(landmark.x - min(xCoordinates))
                    landmarksNormalized.append(landmark.y - min(yCoordinates))

            handLandmarksData.append(landmarksNormalized)  # Añade los datos procesados a la lista principal
            classLabels.append(classDir)  # Añade la etiqueta de la clase correspondiente

# Guarda los datos y etiquetas en un archivo pickle
with open('data.pickle', 'wb') as pickleFile:
    pickle.dump({'data': handLandmarksData, 'labels': classLabels}, pickleFile)
