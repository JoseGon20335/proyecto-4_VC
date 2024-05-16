import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Carga los datos procesados desde un archivo pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convierte los datos y las etiquetas a arrays de numpy
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Divide los datos en conjuntos de entrenamiento y prueba, manteniendo la proporción de clases
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Inicializa el modelo de Random Forest
model = RandomForestClassifier()

# Entrena el modelo con los datos de entrenamiento
model.fit(x_train, y_train)

# Realiza predicciones con los datos de prueba
y_predict = model.predict(x_test)

# Calcula la precisión de las predicciones
score = accuracy_score(y_predict, y_test)

# Imprime el porcentaje de muestras clasificadas correctamente
print('{}% of samples were classified correctly !'.format(score * 100))

# Guarda el modelo entrenado en un archivo pickle
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
