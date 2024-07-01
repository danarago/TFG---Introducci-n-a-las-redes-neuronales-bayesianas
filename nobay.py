# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
import arviz as az
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from time import perf_counter


# Parámetros de programa.
RUTA_DATOS = "datos_entrenamiento"
RUTA_RESULTADOS = "resultados_nobay"
TAM_LOTES = 64
N_EPOCAS = 150
GRAFICAR = True
az.style.use("arviz-darkgrid")


# Función para leer los datos de entrenamiento y prueba de la carpeta "ruta"
# generados por el archivo generador_datos.py.
def leer_datos(ruta: str = "") -> tuple[np.array]:
    X_train, X_test = [], []
    
    with open(path.join(ruta, "datos_x_train.txt")) as archivo:
        for linea in archivo:
            X_train.append(np.array(list(map(int, linea.split()))))
    with open(path.join(ruta, "datos_y_train.txt")) as archivo:
        Y_train = np.array(list(map(int, archivo.read().split())))
    
    with open(path.join(ruta, "datos_x_test.txt")) as archivo:
        for linea in archivo:
            X_test.append(np.array(list(map(int, linea.split()))))
    with open(path.join(ruta, "datos_y_test.txt")) as archivo:
        Y_test = np.array(list(map(int, archivo.read().split())))
    
    return np.array(X_train), Y_train, np.array(X_test), Y_test

# Función que evalúa los resultados obtenidos.
def evaluar(model, X_test, Y_true) -> tuple[float]:
    Y_prob = model.predict(X_test)
    Y_pred = (Y_prob > 0.5).astype(int)
    
    pos_c, pos_i, neg_c, neg_i = 0, 0, 0, 0
    prob_pc, prob_pi, prob_nc, prob_ni = 0., 0., 0., 0.
    for i in range(len(Y_true)):
        prob, pred, real = Y_prob[i][0], Y_pred[i][0], Y_true[i]
        if pred == 1 and pred == real:
            pos_c += 1
            prob_pc += prob
        elif pred == 1 and pred != real:
            pos_i += 1
            prob_pi += prob
        elif pred == real:
            neg_c += 1
            prob_nc += prob
        else:
            neg_i += 1
            prob_ni += prob
    
    prob_pc = prob_pc / pos_c if pos_c > 0 else 0.
    prob_pi = prob_pi / pos_i if pos_i > 0 else 0.
    prob_nc = 1 - prob_nc / neg_c if neg_c > 0 else 0.
    prob_ni = 1 - prob_ni / neg_i if neg_i > 0 else 0.
    
    prec_p = (pos_c) / (pos_c + pos_i) * 100 if pos_c + pos_i > 0 else 0
    prec_n = (neg_c) / (neg_c + neg_i) * 100 if neg_c + neg_i > 0 else 0
    prec_t = (pos_c + neg_c) / len(Y_true) * 100
    
    print(f"Positivos correctos: {pos_c}. Probabilidad media: {prob_pc}.")
    print(f"Falsos positivos: {pos_i}. Probabilidad media: {prob_pi}.")
    print(f"Negativos correctos: {neg_c}. Probabilidad media: {prob_nc}.")
    print(f"Falsos negativos: {neg_i}. Probabilidad media: {prob_ni}.")
    print(f"Precisión cuando predice que hay mina: {prec_p:.4f}%.")
    print(f"Precisión cuando predice que no hay mina: {prec_n:.4f}%.")
    print(f"Precisión total: {prec_t:.4f}%.")
    
    return (pos_c, pos_i, neg_c, neg_i, prob_pc, prob_pi, prob_nc, prob_ni,
            prec_p, prec_n, prec_t)

def escribir_resultados(tupla: tuple, ruta: str, test: int) -> None:
    current_dir = path.dirname(path.abspath(__file__))
    ruta = path.join(current_dir, ruta)
    # Crear la carpeta si no existe.
    if not path.exists(ruta):
        makedirs(ruta)
    with open(path.join(ruta, f"resultados{test}.txt"), "w") as archivo:
        for r in tupla:
            archivo.write(f"{r} ")


# Programa principal.
if __name__ == "__main__":
    # Lectura y normalización de datos.
    X_train, Y_train, X_test, Y_test = leer_datos(RUTA_DATOS)
    X_train = X_train.astype("float32") / 10
    Y_train = Y_train.astype("float32")
    X_test = X_test.astype("float32") / 10
    Y_test = Y_test.astype("float32")
    
    # Separación de los datos de prueba.
    n_test = len(Y_test)
    n_test1 = n_test // 10
    X_test1 = X_test[:n_test1]
    Y_test1 = Y_test[:n_test1]
    X_test2 = X_test[n_test1:]
    Y_test2 = Y_test[n_test1:]
    
    # Definición de nuestro perceptrón multicapa.
    model = Sequential([
        Dense(32, input_dim=48, activation="relu",
              kernel_initializer="he_normal"),
        Dense(1, input_dim=32, activation="sigmoid",
              kernel_initializer="glorot_normal")
    ])
    
    # Compilar y entrenar el modelo y guardar el historial.
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    
    t1 = perf_counter()
    history = model.fit(X_train, Y_train, epochs=N_EPOCAS,
                        batch_size=TAM_LOTES,
                        validation_data=(X_test2, Y_test2))
    t2 = perf_counter()
    
    # Precisión.
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Precisión de la red: {accuracy * 100}%.")
    
    # Medición del tiempo.
    print(f"\nTiempo de entrenamiento: {t2 - t1:.4f} segundos.")
    
    # Gráficas de las curvas de pérdida y precisión.
    if GRAFICAR:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Pérdida del entrenamiento")
        plt.plot(history.history['val_loss'], label="Pérdida de la validación")
        plt.xlabel("Épocas")
        plt.ylabel('Pérdida')
        plt.legend()
        plt.title("Pérdida a lo largo de las épocas")
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"],
                 label="Precisión del entrenamiento")
        plt.plot(history.history["val_accuracy"],
                 label="Precisión de la validación")
        plt.xlabel("Épocas")
        plt.ylabel("Precisión")
        plt.legend()
        plt.title("Precisión a lo largo de las épocas")
        
        plt.show()
    
    # Evaluación y análisis de los dos conjuntos de prueba.
    print("\nTest 1:")
    t1 = perf_counter()
    tupla1 = evaluar(model, X_test1, Y_test1)
    t2 = perf_counter()
    print(f"Tiempo de prueba: {t2 - t1:.4f} segundos.")
    
    print("\nTest 2:")
    t1 = perf_counter()
    tupla2 = evaluar(model, X_test2, Y_test2)
    t2 = perf_counter()
    print(f"Tiempo de prueba: {t2 - t1:.4f} segundos.")
    
    # Guardar los datos para su posterior representación en gráficas.
    escribir_resultados(tupla1, RUTA_RESULTADOS, 1)
    escribir_resultados(tupla2, RUTA_RESULTADOS, 2)
