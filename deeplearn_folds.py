import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Cargar los datos densos desde un archivo CSV
data = pd.read_csv("./19x19_dl/classif_18x18_callout.csv")

# Separar características (X) y etiquetas (y)
X = data.iloc[:, 2:-1].values  # Todas las columnas excepto la última
y = data.iloc[:, -1].values   # Última columna como etiquetas
cellid = data.iloc[:, 1].values  # Columna 'cellid'


# Configuración de la red neuronal
layers = [20, 8, 10]  # Capas ocultas, ajustables según el problema

# Init predictions dataset
all_predictions = pd.DataFrame(columns=["cellid", "prediction"])

# Configuración de 10-Fold Cross Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
accuracies = []

# Validación cruzada
for train_index, test_index in skf.split(X, y):
    # Dividir los datos en train y test para este fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    cellid_train, cellid_test = cellid[train_index], cellid[test_index]

    # Convertir las etiquetas a one-hot encoding si es un problema multiclase
    y_train_categorical = to_categorical(y_train, num_classes=len(np.unique(y)))

    # Crear el modelo
    model = Sequential()
    # Definir la entrada del modelo usando Input
    model.add(Input(shape=(X.shape[1],)))  # X.shape[1] es el número de características

    # warning: model.add(Dense(layers[0], activation='tanh', input_shape=(X.shape[1],)))
    for nodes in layers:
        model.add(Dense(nodes, activation='tanh'))
    model.add(Dense(len(np.unique(y)), activation='sigmoid'))

    # Compilar el modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(X_train, y_train_categorical, epochs=100, batch_size=1, verbose=0)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convertir probabilidades a clases
    accuracy = accuracy_score(y_test, y_pred_classes)
    accuracies.append(accuracy)

    # Crear un DataFrame con cellid, predicción y probabilidades
    fold_predictions = pd.DataFrame({
        "cellid": cellid_test,
        "prediction": y_pred_classes,
    })

    # Agregar las predicciones de este fold al DataFrame total
    all_predictions = pd.concat([all_predictions, fold_predictions])

    print(f"Fold Accuracy: {accuracy:.4f}")

all_predictions.to_csv("./19x19_dl/predictions_callout.csv", index=False)
# Promediar las métricas finales
print(f"\nMean Accuracy: {np.mean(accuracies):.4f}")
print(f"Standard Deviation: {np.std(accuracies):.4f}")
