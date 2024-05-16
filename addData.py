# Proyecto # 4 - Vision por computadora
# @ addData.py - encargado de crear y cargar data dentro de la carpeta de ./imgs

import os
import cv2

# Directorio donde se almacenarán los datos
dataDirectory = './imgs'

# Número de clases y tamaño del conjunto de datos
numClasses = 3
imagesPerClass = 100

# Índice de la cámara a utilizar
cameraIndex = 2

def createDirectory(path):
    """
    Crea un directorio si no existe.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def captureImagesForClass(classId, imagesCount, camera):
    """
    Captura imágenes para una clase específica y las guarda en el directorio correspondiente.
    """
    classDirectory = os.path.join(dataDirectory, str(classId))
    createDirectory(classDirectory)
    print(f'Recolectando datos para la clase {classId}')

    # Muestra el mensaje de preparación hasta que el usuario presione 'x'
    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            print("Error: No se pudo capturar la imagen.")
            continue
        cv2.putText(frame, 'Presiona "X" para empezar.', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('x'):
            break

    # Captura y guarda las imágenes
    for counter in range(imagesCount):
        ret, frame = camera.read()
        if not ret or frame is None:
            print("Error: No se pudo capturar la imagen.")
            continue
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(classDirectory, f'{counter}.jpg'), frame)

def main():
    # Crea el directorio principal de datos
    createDirectory(dataDirectory)
    
    # Abre la cámara
    camera = cv2.VideoCapture(cameraIndex)
    if not camera.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    # Captura imágenes para cada clase
    for classId in range(numClasses):
        captureImagesForClass(classId, imagesPerClass, camera)

    # Libera la cámara y cierra todas las ventanas
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
