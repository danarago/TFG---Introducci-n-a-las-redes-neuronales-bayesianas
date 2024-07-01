# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import numpy as np
from os import path, makedirs
from torch.utils.data import DataLoader, TensorDataset
from time import perf_counter


# Parámetros del programa.
RUTA_DATOS = "datos_entrenamiento"
RUTA_RESULTADOS = "resultados_bay"
TAM_LOTES = 64
N_EPOCAS = 200


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

# Tamaño de las capas.
N0, N1, N2 = 48, 32, 1

# Inicializaciones de la distribución a priori.
def he_init(n_in: int) -> float:
    return np.sqrt(2 / n_in)

def glorot(n_in: int, n_out: int) -> float:
    return np.sqrt(2 / (n_in + n_out))

# Definición de nuestro perceptrón multicapa.
class BayesianNN(nn.Module):
    def __init__(self):
        super(BayesianNN, self).__init__()
        self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=he_init(N0),
                                   in_features=N0, out_features=N1)
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=glorot(N1, N2),
                                   in_features=N1, out_features=N2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Función que realiza una época de entrenamiento.
def entrenar(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.view(-1, 1)
        optimizer.zero_grad()
        output = model(data)
        classification_loss = criterion(output, target)
        kl_loss = sum(layer.kl_loss() for layer in model.modules()
                      if hasattr(layer, "kl_loss"))
        loss = classification_loss + kl_loss / len(train_loader)
        loss.backward()
        optimizer.step()
    print(f"Época: {epoch}. Pérdida: {loss.item()}")

# Función de predicción con muestreo.
def predecir(model, data_loader, num_samples=1000):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            batch_preds = []
            for data, _ in data_loader:
                output = model(data)
                batch_preds.append(output)
            predictions.append(torch.cat(batch_preds, dim=0))
    return torch.mean(torch.stack(predictions), dim=0)

# Función que evalúa los resultados obtenidos.
def evaluar(model, test_loader, Y_true, num_samples=1000) -> tuple[float]:
    Y_prob = predecir(model, test_loader, num_samples).numpy()
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
    
    # Convertir los datos en objetos de pytorch para que la red pueda trabajar
    # con ellos.
    train_set = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    test1_set = TensorDataset(torch.tensor(X_test1), torch.tensor(Y_test1))
    test2_set = TensorDataset(torch.tensor(X_test2), torch.tensor(Y_test2))
    
    train = DataLoader(train_set, batch_size=TAM_LOTES, shuffle=False)
    test1 = DataLoader(test1_set, batch_size=200, shuffle=False)
    test2 = DataLoader(test2_set, batch_size=1800, shuffle=False)
    
    # Inicializar el modelo, el optimizador y la función de pérdida.
    model = BayesianNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Entrenar el modelo.
    t1 = perf_counter()
    for epoch in range(1, N_EPOCAS + 1):
        entrenar(model, train, optimizer, epoch)
    t2 = perf_counter()
    
    # Medición del tiempo.
    print(f"\nTiempo de entrenamiento: {t2 - t1:.4f} segundos.")
    
    # Evaluación y análisis de los dos conjuntos de prueba.
    print("\nTest 1:")
    t1 = perf_counter()
    tupla1 = evaluar(model, test1, Y_test1, num_samples=1000)
    t2 = perf_counter()
    print(f"Tiempo de prueba: {t2 - t1:.4f} segundos.")
    
    print("\nTest 2:")
    t1 = perf_counter()
    tupla2 = evaluar(model, test2, Y_test2, num_samples=1000)
    t2 = perf_counter()
    print(f"Tiempo de prueba: {t2 - t1:.4f} segundos.")
    
    # Guardar los datos para su posterior representación en gráficas.
    escribir_resultados(tupla1, RUTA_RESULTADOS, 1)
    escribir_resultados(tupla2, RUTA_RESULTADOS, 2)
