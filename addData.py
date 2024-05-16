# Proyecto # 4 - Vision por computadora
# @ addData.py - encargado de crear y cargar data dentro de la carpeta de ./imgs

import os
import cv2

# Directorio donde se almacenarán los datos
DATA_DIR = './imgs'

# Número de clases y tamaño del conjunto de datos
NUMBER_OF_CLASSES = 3
DATASET_SIZE = 100

# Índice de la cámara a utilizar
CAMERA_INDEX = 2

def create_directory(path):
    """
    Crea un directorio si no existe.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def capture_images_from_class(class_id, dataset_size, cap):
    """
    Captura imágenes para una clase específica y las guarda en el directorio correspondiente.
    """
    class_dir = os.path.join(DATA_DIR, str(class_id))
    create_directory(class_dir)
    print(f'Collecting data for class {class_id}')

    # Muestra el mensaje de preparación hasta que el usuario presione 'x'
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            continue
        cv2.putText(frame, 'Preciona "X" para empezar.', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('x'):
            break

    # Captura y guarda las imágenes
    for counter in range(dataset_size):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            continue
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)

def main():
    # Crea el directorio principal de datos
    create_directory(DATA_DIR)
    
    # Abre la cámara
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Captura imágenes para cada clase
    for class_id in range(NUMBER_OF_CLASSES):
        capture_images_from_class(class_id, DATASET_SIZE, cap)

    # Libera la cámara y cierra todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
