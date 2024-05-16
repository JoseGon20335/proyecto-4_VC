# Proyecto # 4 - Vision por computadora
# @ trainer.py - encargado de hacer el entrenamiento a partir del pickle generado.

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Carga los datos procesados desde un archivo pickle
dataDict = pickle.load(open('./data.pickle', 'rb'))

# Convierte los datos y las etiquetas a arrays de numpy
data = np.asarray(dataDict['data'])
labels = np.asarray(dataDict['labels'])

# Divide los datos en conjuntos de entrenamiento y prueba, manteniendo la proporción de clases
xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Inicializa el modelo de Random Forest
model = RandomForestClassifier()

# Entrena el modelo con los datos de entrenamiento
model.fit(xTrain, yTrain)

# Realiza predicciones con los datos de prueba
yPredict = model.predict(xTest)

# Calcula la precisión de las predicciones
score = accuracy_score(yPredict, yTest)

# Imprime el porcentaje de muestras clasificadas correctamente
print('{}% de la data ha sido procesada con exito.'.format(score * 100))

# Guarda el modelo entrenado en un archivo pickle
with open('model.p', 'wb') as pickleFile:
    pickle.dump({'model': model}, pickleFile)
