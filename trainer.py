import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Carga los datos procesados desde un archivo pickle
dataDict = pickle.load(open('./data.pickle', 'rb'))

# Verifica la longitud máxima de las secuencias en los datos
max_length = max(len(seq) for seq in dataDict['data'])

# Rellena las secuencias más cortas con ceros (o un valor de relleno adecuado)
data = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in dataDict['data']])
labels = np.asarray(dataDict['labels'])

# Normaliza los datos
data = data / np.max(data)

# Convierte las etiquetas a one-hot encoding
num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes)

# Divide los datos en conjuntos de entrenamiento y prueba, manteniendo la proporción de clases
xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define la arquitectura de la red neuronal
model = Sequential()
model.add(Dense(128, input_dim=xTrain.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compila el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrena el modelo con los datos de entrenamiento
model.fit(xTrain, yTrain, epochs=20, batch_size=32, validation_split=0.2)

# Evalúa el modelo con los datos de prueba
score = model.evaluate(xTest, yTest, verbose=0)
print('{}% de la data ha sido procesada con éxito.'.format(score[1] * 100))

# Guarda el modelo entrenado en un archivo HDF5
model.save('model.h5')
