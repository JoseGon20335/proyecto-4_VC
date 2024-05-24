# ----------------------------------------------------------------------

# Visión por Computadora
# Proyecto 4: Detección de Alfabeto de LENSEGUA

# Archivo: trainer.py
# Descripción: Este programa se encarga de entrenar un modelo de red neuronal con los datos procesados en buildDataSet.py.
#              El modelo se guarda en un archivo H5 para su uso en la detección de letras en tiempo real.

# Autores:
# Stefano Alberto Aragoni Maldonado - 20261
# Carol Andreé Arévalo Estrada - 20461
# José Miguel González González - 20335
# Luis Diego Santos Cuellar - 20226

# ----------------------------------------------------------------------
# IMPORTAR LIBRERÍAS

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# ----------------------------------------------------------------------
# VARIABLES GLOBALES

dataDict = pickle.load(open('./data.pickle', 'rb'))                                                 # Carga los datos procesados en buildDataSet.py

max_length = max(len(seq) for seq in dataDict['data'])                                              # Se define la longitud máxima de las secuencias

data = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in dataDict['data']])  # Se rellenan las secuencias con ceros
data = data / np.max(data)                                                                          # Se normalizan los datos

labels = np.asarray(dataDict['labels'])                                                             # Se obtienen las etiquetas de las secuencias

num_classes = len(np.unique(labels))                                                                # Se obtiene el número de clases
labels = to_categorical(labels, num_classes)                                                        # Se convierten las etiquetas a un formato categórico

# ----------------------------------------------------------------------
# ENTRENAMIENTO DE LA RED NEURONAL

xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) # Se dividen los datos en entrenamiento y prueba


model = Sequential()                                                        # Se crea un modelo de red neuronal secuencial / feedforward
model.add(Dense(128, input_dim=xTrain.shape[1], activation='relu'))         # Capa densa de 128 neuronas con función de activación ReLU
model.add(Dense(64, activation='relu'))                                     # Capa densa de 64 neuronas con función de activación ReLU
model.add(Dense(num_classes, activation='softmax'))                         # Capa densa de salida con función de activación Softmax (clasificación multiclase)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Se compila el modelo con función de pérdida para clasificación multiclase

model.fit(xTrain, yTrain, epochs=20, batch_size=32, validation_split=0.2)               # Se entrena el modelo con los datos de entrenamiento

score = model.evaluate(xTest, yTest, verbose=0)                                         # Se evalúa el modelo con los datos de prueba
print('Test loss:', score[0])                                                           # Se imprime la pérdida
print('Test accuracy:', score[1])                                                       # Se imprime la precisión

# ----------------------------------------------------------------------
# GUARDAR EL MODELO

model.save('model.h5')                                                                  # Se guarda el modelo en un archivo H5
