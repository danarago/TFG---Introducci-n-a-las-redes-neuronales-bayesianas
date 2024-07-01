# -*- coding: utf-8 -*-

from random import randint
import numpy as np
from os import path, makedirs


# Parámetros del programa.
RUTA = "datos_entrenamiento"
N_TRAIN = 10000
N_TEST = 2000
TAMAÑO = 7
MINAS = 7
EJECUTAR = True
EJECUTAR2 = True
IMPRIMIR = False


# Genera un tablero de buscaminas cuadrado de lado "tamaño" con "minas" minas.
# En realidad genera un tablero más grande para la siguiente función.
def generar_tablero(tamaño: int, minas: float) -> list[list[bool]]:
    # Inicializamos un tablero sin minas del tamaño adecuado.
    n_minas, tablero = 0, [[False] * (tamaño + 2) for _ in range(tamaño + 2)]
    
    # Ponemos el número de minas aleatoriamente.
    while n_minas < minas:
        i, j = randint(1, tamaño), randint(1, tamaño)
        # Si sale una casilla que ya había salido, la ignoramos.
        if not tablero[i][j]:
            tablero[i][j] = True
            n_minas += 1
    
    return tablero

# Convierte el tablero t en un mapa de juego. Si en una casilla hay un número
# del 0 al 8, eso indica que hay ese número de minas en las casillas de
# alrededor, mientras que si hay un 10, en esa casilla hay una mina.
def tablero_a_mapa(t: list[list[bool]]) -> list[list[int]]:
    # Inicializamos el mapa con minas únicamente.
    n = len(t) - 2
    mapa = [[10] * n for _ in range(n)]
    
    # Recorremos con un bucle todas las casillas y contamos la cantidad de
    # minas en las casillas adyacentes si no hay mina.
    for i in range(n):
        for j in range(n):
            if not t[i + 1][j + 1]:
                contador = t[i][j] + t[i + 1][j] + t[i + 2][j] + \
                           t[i][j + 1] + t[i + 2][j + 1] + \
                           t[i][j + 2] + t[i + 1][j + 2] + t[i + 2][j + 2]
                mapa[i][j] = contador
    
    return mapa

# Vectores que definen las coordenadas relativas de una casilla adyacente
# respecto a una dada.
D_X, D_Y = [-1, 0, 1, 1, 1, 0, -1, -1], [-1, -1, -1, 0, 1, 1, 1, 0]

# Función que genera un dato de la red neuronal sin normalizar en la que se
# destapan aleatoriamente "cas_desc" casillas en el tablero m y que se
# convierte en un posible mapa de buscaminas durante una partida. Las casillas
# en las que no se sabe si hay una mina se indican con 9. La casilla central
# siempre se mantiene con un 9, ya que es la que debe adivinar la red neuronal.
# La función también devuelve si hay mina (1) o no (0) en la casilla central.
def mapa_a_input(m: list[list[int]], cas_desc: int,
                 imprimir: bool = False) -> tuple[np.array, int]:
    
    # Inicializamos el mapa con nueves.
    n = len(m)
    casilla = (n >> 1, n >> 1)
    partida = np.array([[9] * n for _ in range(n)])
    
    # Descubrimos "cas_desc" casillas manteniendo la central cubierta.
    descubiertas = {casilla}
    while len(descubiertas) <= cas_desc:
        descubiertas.add((randint(0, n - 1), randint(0, n - 1)))
    descubiertas.remove(casilla)
    descubiertas = list(descubiertas)
    
    # Lo hacemos efectivo en el mapa.
    for i, j in descubiertas:
        partida[i][j] = m[i][j]
    
    # Si en una casilla hay un 0, en el juego original se destapan todas las
    # casillas adyacentes. Esta parte del código se encarga de hacer esto,
    # pudiendo obtenerse más casillas descubiertas de las pedidas.
    i_cas = 0
    while i_cas < len(descubiertas) < n ** 2 - 1:
        
        i, j = descubiertas[i_cas]
        
        if not m[i][j]:
            for k in range(8):
                x, y = i + D_X[k], j + D_Y[k]
                
                if 0 <= x < n and 0 <= y < n:
                    if partida[x][y] == 9:
                        partida[x][y] = m[x][y]
                        descubiertas.append((x, y))
        i_cas += 1
    
    if imprimir:
        print(partida)
        print(np.array(m))
    
    # Convertimos el mapa a un formato correcto, eliminando la casilla central.
    partida = list(np.concatenate(partida))
    del partida[n ** 2 >> 1]
    
    return np.array(partida), int(m[casilla[0]][casilla[1]] == 10)

# Genera un único dato a partir de las funciones anteriores.
def generar_dato(tamaño: int, minas: float, cas_desc: float,
                 imprimir: bool = False) -> tuple[np.array, int]:
    return mapa_a_input(tablero_a_mapa(generar_tablero(tamaño, minas)),
                        cas_desc, imprimir)

# Genera "N" datos de entrenamiento de tamaño "tamaño" y "minas" minas.
def generar_train(N: int, tamaño: int, minas: float,
                          imprimir: bool) -> tuple[np.array]:
    
    X, Y = [0] * N, np.array([0] * N)
    
    for i in range(N):
        # Destapamos aleatoriamente entre al menos 9 y 48 minas.
        X[i], Y[i] = generar_dato(tamaño, minas, randint(9, 48), imprimir)
    
    return np.array(X), Y

# Genera "N" datos de prueba de tamaño "tamaño" y "minas" minas.
def generar_test(N: int, tamaño: int, minas: float,
                 imprimir: bool) -> tuple[np.array]:
    
    X, Y, N_10 = [0] * N, np.array([0] * N), N // 10
    
    # La décima parte de los datos de prueba solo tienen una casilla destapada
    # para comprobar la incertidumbre.
    i = 0
    while i < N_10:
        X[i], Y[i] = generar_dato(tamaño, minas, 1, imprimir)
        if np.count_nonzero(X[i] == 9) == 47:
            i += 1
    
    # El resto son iguales que los datos de entrenamiento.
    for i in range(N_10, N):
        X[i], Y[i] = generar_dato(tamaño, minas, randint(9, 48), imprimir)
    
    return np.array(X), Y


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


# Programa principal.
if __name__ == "__main__":
    
    # Generación de datos.
    if EJECUTAR:
        X_train, Y_train = generar_train(N_TRAIN, TAMAÑO, MINAS, IMPRIMIR)
        X_test, Y_test = generar_test(N_TEST, TAMAÑO, MINAS, IMPRIMIR) 
        
        current_dir = path.dirname(path.abspath(__file__))
        ruta = path.join(current_dir, RUTA)
        # Crear la carpeta si no existe.
        if not path.exists(ruta):
            makedirs(ruta)
        
        with open(path.join(RUTA, "datos_x_train.txt"), "w") as archivo:
            for x in X_train:
                for i in x:
                    archivo.write(str(i) + " ")
                archivo.write("\n")
        with open(path.join(RUTA, "datos_y_train.txt"), "w") as archivo:
            for y in Y_train:
                archivo.write(str(y) + "\n")
                
        with open(path.join(RUTA, "datos_x_test.txt"), "w") as archivo:
            for x in X_test:
                for i in x:
                    archivo.write(str(i) + " ")
                archivo.write("\n")
        with open(path.join(RUTA, "datos_y_test.txt"), "w") as archivo:
            for y in Y_test:
                archivo.write(str(y) + "\n")
    
    # Algunas características de los datos obtenidos.
    if EJECUTAR2:
        # Leer datos.
        X_train, Y_train, X_test, Y_test = leer_datos(RUTA)
        
        cas_test = np.array([np.count_nonzero(x == 9) for x in X_test])
        print(np.count_nonzero(cas_test == 47))
        cas_test = np.array([np.count_nonzero(x == 9)
                             for x in X_test[:N_TEST // 10]])
        print(np.count_nonzero(cas_test == 47))
        n_positivos_train = np.count_nonzero(Y_train == 1)
        print(n_positivos_train)
        n_positivos_test = np.count_nonzero(Y_test == 1)
        print(n_positivos_test)
        
