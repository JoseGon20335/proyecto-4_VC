# ----------------------------------------------------------------------

# Visión por Computadora
# Proyecto 4: Detección de Alfabeto de LENSEGUA

# Archivo: addData.py
# Descripción: Este programa se encarga de recolectar datos para el entrenamiento del modelo. 
#              Se enciende la cámara y se capturan imágenes de las manos realizando los signos del 
#              alfabeto de LENSEGUA.

# Autores:
# Stefano Alberto Aragoni Maldonado - 20261
# Carol Andreé Arévalo Estrada - 20461
# José Miguel González González - 20335
# Luis Diego Santos Cuellar - 20226

# ----------------------------------------------------------------------
# IMPORTAR LIBRERÍAS

import os
import cv2

# ------------------------------------------------------------------
# VARIABLES GLOBALES

dataDirectory = './imgs'                            # Directorio donde se guardarán las imágenes

numClasses = 26                                     # Número de clases (letras del alfabeto)
imagesPerClass = 100                                # Número de imágenes a capturar por clase

# ------------------------------------------------------------------
# FUNCIONES

"""
Función: createDirectory
Descripción: Crea un directorio en la ruta especificada si no existe.
Parámetros:
     path: ruta del directorio que se desea crear.
"""
def createDirectory(path):
    if not os.path.exists(path):                    # Si el directorio no existe, lo crea
        os.makedirs(path)                           # Crea el directorio       

"""
Función: captureImagesForClass
Descripción: Captura imágenes para una clase específica y las guarda en el directorio correspondiente.
Parámetros:
     classId: identificador de la clase para la que se capturan las imágenes.
     imagesCount: número de imágenes a capturar.
     camera: objeto de captura de video (cv2.VideoCapture).
"""
def captureImagesForClass(classId, imagesCount, camera):
    classDirectory = os.path.join(dataDirectory, str(classId))              # Se define el directorio de la clase
    createDirectory(classDirectory)                                         # Crea el directorio de la clase
    print(f'\n----- CLASE {classId}:')                                      # Muestra un mensaje con el identificador de la clase                    

    print("Presione enter para empezar a capturar las imágenes.")           # Muestra un mensaje para indicar al usuario que presione enter para empezar a capturar las imágenes
    input()                                                                 # Espera a que el usuario presione enter para empezar a capturar las imágenes

    for counter in range(imagesCount):                                      # Captura el número de imágenes especificado
        ret, frame = camera.read()                                          # Captura un frame de la cámara
        if not ret or frame is None:                                        # Si no se pudo capturar el frame, muestra un mensaje de error
            print("Error: No se pudo capturar la imagen.")
            continue
        
        cv2.imshow('frame', frame)                                          # Muestra el frame en una ventana
        cv2.waitKey(25)                                                     # Espera 25 ms

        cv2.imwrite(os.path.join(classDirectory, f'{counter}.jpg'), frame)  # Guarda la imagen en el directorio de la clase


"""
Función: main
Descripción: Función principal que configura el entorno, abre la cámara, captura imágenes para cada clase y cierra la cámara.
Parámetros: Ninguno.
"""
def main():
    createDirectory(dataDirectory)              # Crea el directorio de datos

    camera = cv2.VideoCapture(0)                # Abre la cámara
    if not camera.isOpened():                   # Si no se pudo abrir la cámara, se termina el programa
        return

    for classId in range(numClasses):           # Captura imágenes para cada clase
        captureImagesForClass(classId, imagesPerClass, camera)  

    camera.release()                            # Cierra la cámara
    cv2.destroyAllWindows()                     # Cierra todas las ventanas

if __name__ == '__main__':
    main()
