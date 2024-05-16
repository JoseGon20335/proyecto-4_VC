# Proyecto # 4 - Vision por computadora
# @ inference.py - encargado de correr el modelo.

import pickle
import cv2
import mediapipe as mp
import numpy as np

# Carga el modelo entrenado desde un archivo pickle
try:
    modelDict = pickle.load(open('./model.p', 'rb'))
    model = modelDict['model']
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Inicializa la captura de video desde la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Inicialización de mediapipe para detección de manos
mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles

# Configuración del módulo de manos de mediapipe
handsDetector = mpHands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Diccionario de etiquetas para las predicciones
labelsDict = {0: 'A', 1: 'B', 2: 'L'}

while True:
    dataAux = []  # Lista auxiliar para almacenar coordenadas normalizadas
    xCoordinates = []  # Lista para almacenar coordenadas x de los puntos de referencia
    yCoordinates = []  # Lista para almacenar coordenadas y de los puntos de referencia

    # Captura un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    H, W, _ = frame.shape  # Obtiene las dimensiones del frame

    # Convierte el frame a RGB
    frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesa el frame para detectar manos
    results = handsDetector.process(frameRgb)
    if results.multi_hand_landmarks:
        # Dibuja las conexiones de las manos en el frame
        for handLandmarks in results.multi_hand_landmarks:
            mpDrawing.draw_landmarks(
                frame,  # imagen para dibujar
                handLandmarks,  # salida del modelo
                mpHands.HAND_CONNECTIONS,  # conexiones de la mano
                mpDrawingStyles.get_default_hand_landmarks_style(),
                mpDrawingStyles.get_default_hand_connections_style())

        # Procesa los puntos de referencia de las manos detectadas
        for handLandmarks in results.multi_hand_landmarks:
            for landmark in handLandmarks.landmark:
                xCoordinates.append(landmark.x)
                yCoordinates.append(landmark.y)

            # Normaliza las coordenadas de los puntos de referencia
            for landmark in handLandmarks.landmark:
                dataAux.append(landmark.x - min(xCoordinates))
                dataAux.append(landmark.y - min(yCoordinates))

        if dataAux:
            # Calcula las coordenadas del rectángulo delimitador
            x1 = int(min(xCoordinates) * W) - 10
            y1 = int(min(yCoordinates) * H) - 10
            x2 = int(max(xCoordinates) * W) - 10
            y2 = int(max(yCoordinates) * H) - 10

            try:
                # Realiza una predicción con el modelo
                prediction = model.predict([np.asarray(dataAux)])
                print(f"Predicción: {prediction}")

                # Obtiene el carácter predicho
                predictedCharacter = labelsDict[int(prediction[0])]
                print(f"Carácter predicho: {predictedCharacter}")

                # Dibuja el rectángulo y la etiqueta en el frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predictedCharacter, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            except Exception as e:
                print(f"Error en la predicción: {e}")

    # Muestra el frame con las anotaciones
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()
