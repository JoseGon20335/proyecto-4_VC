# ----------------------------------------------------------------------

# Visión por Computadora
# Proyecto 4: Detección de Alfabeto de LENSEGUA

# Archivo: inference.py
# Descripción: Este programa carga un modelo de red neuronal entrenado con trainer.py y realiza predicciones en tiempo real
#              con la cámara del dispositivo. Se utiliza la librería Mediapipe para detectar las manos y extraer las coordenadas.

# Autores:
# Stefano Alberto Aragoni Maldonado - 20261
# Carol Andreé Arévalo Estrada - 20461
# José Miguel González González - 20335
# Luis Diego Santos Cuellar - 20226

# ----------------------------------------------------------------------
# IMPORTAR LIBRERÍAS

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------
# CARGA DEL MODELO Y CONFIGURACIÓN DE LA CÁMARA

try:
    model = load_model('./model.h5')                                    # Carga el modelo de red neuronal entrenado
except Exception as e:                                                  # En caso de error, imprime un mensaje y termina el programa
    print(f"Error al cargar el modelo: {e}")                            
    exit()

cap = cv2.VideoCapture(0)                                               # Inicializa la cámara del dispositivo
if not cap.isOpened():                                                  # En caso de error al abrir la cámara, imprime un mensaje y termina el programa
    print("Error: No se pudo abrir la cámara.")
    exit()

# ----------------------------------------------------------------------
# VARIABLES GLOBALES

mpHands = mp.solutions.hands                                            # Módulo de manos de mediapipe
mpDrawing = mp.solutions.drawing_utils                                  # Módulo de dibujo de mediapipe
mpDrawingStyles = mp.solutions.drawing_styles                           # Estilos de dibujo de mediapipe

handsDetector = mpHands.Hands(static_image_mode=True, min_detection_confidence=0.3)     # Inicialización de detector de manos

labelsDict = {                                                                          # Diccionario de etiquetas            
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}

# ----------------------------------------------------------------------
# CAPTURA DE VIDEO Y PROCESAMIENTO DE IMÁGENES

while True:                                                             # Mientras la cámara esté abierta            
    dataAux = []                                                        # Lista auxiliar para almacenar coordenadas normalizadas
    xCoordinates = []                                                   # Lista para almacenar coordenadas x de los puntos de referencia
    yCoordinates = []                                                   # Lista para almacenar coordenadas y de los puntos de referencia

    ret, frame = cap.read()                                             # Captura un frame de la cámara
    if not ret:                                                         # En caso de error al capturar el frame, imprime un mensaje y termina el programa            
        print("Error: No se pudo capturar el frame.")
        break

    H, W, _ = frame.shape                                               # Obtiene las dimensiones del frame


    frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                   # Convierte el frame a RGB
    r, g, b = cv2.split(frameRgb)                                       # Separa los canales de la imagen

    r_eq = cv2.equalizeHist(r)                                          # Aplica histogram equalization a cada canal       
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)

    frameRgb = cv2.merge((r_eq, g_eq, b_eq))                            # Une los canales ecualizados


    results = handsDetector.process(frameRgb)                           # Procesa el frame con el detector de manos
    if results.multi_hand_landmarks:                                    # Verifica si se detectaron manos en el frame

        for handLandmarks in results.multi_hand_landmarks:              # Recorre los puntos de referencia de cada mano detectada
            mpDrawing.draw_landmarks(                                   # Dibuja los puntos de referencia y conexiones en el frame 
                frame,                                                      # frame de salida
                handLandmarks,                                              # salida del modelo
                mpHands.HAND_CONNECTIONS,                                   # conexiones de la mano
                mpDrawingStyles.get_default_hand_landmarks_style(),         # estilo de los puntos de referencia
                mpDrawingStyles.get_default_hand_connections_style())       # estilo de las conexiones

        for handLandmarks in results.multi_hand_landmarks:              # Recorre los puntos de referencia de cada mano detectada
            for landmark in handLandmarks.landmark:                     # Recorre los puntos de referencia de la mano
                xCoordinates.append(landmark.x)                         # Añade la coordenada x del punto de referencia
                yCoordinates.append(landmark.y)                         # Añade la coordenada y del punto de referencia

            for landmark in handLandmarks.landmark:                     # Recorre los puntos de referencia de la mano
                dataAux.append(landmark.x - min(xCoordinates))          # Normaliza la coordenada x del punto de referencia
                dataAux.append(landmark.y - min(yCoordinates))          # Normaliza la coordenada y del punto de referencia

        while len(dataAux) < 84:                                        # En caso de que no se detecten suficientes puntos de referencia, añade ceros
            dataAux.append(0)                                           # Añade ceros a la lista auxiliar (42 por mano, 84 en total)

        if len(dataAux) == 84:                                          # En caso de que se detecten suficientes puntos de referencia
            x1 = int(min(xCoordinates) * W) - 10                        # Calcula las coordenadas del rectángulo que encierra la mano
            y1 = int(min(yCoordinates) * H) - 10                         
            x2 = int(max(xCoordinates) * W) - 10
            y2 = int(max(yCoordinates) * H) - 10

            try:
                dataAux = np.asarray(dataAux).reshape(1, -1) / np.max(dataAux)      # Normaliza los datos y los convierte a un arreglo de numpy

                prediction = model.predict(dataAux)                     # Realiza la predicción con el modelo
                predictedIndex = np.argmax(prediction)                  # Obtiene el índice de la predicción
                print(f"Predicción: {prediction}")                      # Imprime la predicción

                predictedCharacter = labelsDict[predictedIndex]         # Obtiene el carácter predicho
                print(f"Carácter predicho: {predictedCharacter}")       # Imprime el carácter predicho

                probability = prediction[0][predictedIndex]             # Obtiene la probabilidad de la predicción
                print(f"Probabilidad: {probability}")                   # Imprime la probabilidad

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)  # Dibuja un rectángulo alrededor de la mano
                cv2.putText(frame, predictedCharacter, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)     # Muestra el carácter predicho en el frame
                cv2.putText(frame, f"{probability:.2f}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1, cv2.LINE_AA)   # Muestra la probabilidad en el frame


            except Exception as e:                                      # En caso de error en la predicción, imprime un mensaje
                print(f"Error en la predicción: {e}")

    cv2.imshow('frame', frame)                                          # Muestra el frame en una ventana
    if cv2.waitKey(1) & 0xFF == ord('q'):                               # En caso de presionar la tecla 'q', termina el programa
        break

# ----------------------------------------------------------------------

cap.release()                                                           # Libera la cámara
cv2.destroyAllWindows()                                                 # Cierra todas las ventanas
