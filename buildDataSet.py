# ----------------------------------------------------------------------

# Visión por Computadora
# Proyecto 4: Detección de Alfabeto de LENSEGUA

# Archivo: buildDataSet.py
# Descripción: Este programa se encarga de procesar las imágenes capturadas con addData.py para extraer los puntos de referencia de las manos detectadas.
#              Los puntos de referencia se normalizan y se guardan en un archivo pickle para su uso en el entrenamiento del modelo.

# Autores:
# Stefano Alberto Aragoni Maldonado - 20261
# Carol Andreé Arévalo Estrada - 20461
# José Miguel González González - 20335
# Luis Diego Santos Cuellar - 20226

# ----------------------------------------------------------------------
# IMPORTAR LIBRERÍAS

import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# VARIABLES GLOBALES

mpHands = mp.solutions.hands                                            # Módulo de manos de mediapipe
mpDrawing = mp.solutions.drawing_utils                                  # Módulo de dibujo de mediapipe
mpDrawingStyles = mp.solutions.drawing_styles                           # Estilos de dibujo de mediapipe

handsDetector = mpHands.Hands(static_image_mode=True, min_detection_confidence=0.3)     # Inicialización de detector de manos

dataDirectory = './imgs'                                                # Directorio de imágenes a procesar

handLandmarksData = []                                                  # Lista para almacenar datos de puntos de referencia
classLabels = []                                                        # Lista para almacenar etiquetas de las clases

# ----------------------------------------------------------------------


for classDir in os.listdir(dataDirectory):                                      # Recorre las clases dentro del directorio de imágenes
    for imageFilename in os.listdir(os.path.join(dataDirectory, classDir)):     # Recorre las imágenes dentro de cada clase

        imagePath = os.path.join(dataDirectory, classDir, imageFilename)        # Ruta de la imagen a procesar
        image = cv2.imread(imagePath)                                           # Lee la imagen            
        if image is None:                                                       # Verifica si la imagen se pudo leer correctamente
            print(f"Error: No se pudo leer la imagen {imagePath}")
            continue


        imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                       # Convierte la imagen a RGB       
        r, g, b = cv2.split(imageRgb)                                           # Separa los canales de la imagen

        r_eq = cv2.equalizeHist(r)                                              # Aplica histogram equalization a cada canal       
        g_eq = cv2.equalizeHist(g)
        b_eq = cv2.equalizeHist(b)

        imageRgb = cv2.merge((r_eq, g_eq, b_eq))                                # Une los canales ecualizados


        landmarksNormalized = []                                                # Lista para almacenar puntos de referencia normalizados
        xCoordinates = []                                                       # Lista para almacenar coordenadas x de los puntos de referencia           
        yCoordinates = []                                                       # Lista para almacenar coordenadas y de los puntos de referencia

        results = handsDetector.process(imageRgb)                               # Procesa la imagen con el detector de manos
        if results.multi_hand_landmarks:                                        # Verifica si se detectaron manos en la imagen

            for handLandmarks in results.multi_hand_landmarks:                      # Recorre los puntos de referencia de cada mano detectada
                xCoordinates = [landmark.x for landmark in handLandmarks.landmark]  # Obtiene todas las coordenadas x
                yCoordinates = [landmark.y for landmark in handLandmarks.landmark]  # Obtiene todas las coordenadas y

                min_x = min(xCoordinates)                                       # Encuentra el valor mínimo de x
                min_y = min(yCoordinates)                                       # Encuentra el valor mínimo de y

                for landmark in handLandmarks.landmark:                         # Recorre los puntos de referencia de la mano
                    landmarksNormalized.append(landmark.x - min_x)              # Normaliza la coordenada x del punto de referencia
                    landmarksNormalized.append(landmark.y - min_y)              # Normaliza la coordenada y del punto de referencia

            handLandmarksData.append(landmarksNormalized)                       # Añade los puntos de referencia normalizados a la lista de datos
            classLabels.append(classDir)                                        # Añade la etiqueta de la clase correspondiente

# ----------------------------------------------------------------------

with open('data.pickle', 'wb') as pickleFile:                                   # Guarda los datos de puntos de referencia y las etiquetas en un archivo pickle
    pickle.dump({'data': handLandmarksData, 'labels': classLabels}, pickleFile) 
