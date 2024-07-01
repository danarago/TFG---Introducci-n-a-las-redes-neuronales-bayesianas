# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


# Parámetros del programa.
RUTA_BAY = "resultados_bay/resultados2.txt"
RUTA_NOBAY = "resultados_nobay/resultados2.txt"


# Función para leer los resultados obtenidos por los códigos "nobay.py" y
# "bay.py".
def leer_resultados(ruta):
    with open(ruta, "r") as archivo:
        datos = archivo.read().strip().split()
        datos = list(map(float, datos[:8]))
    return datos


# Programa principal.
if __name__ == "__main__":
    
    # Leer los resultados de los archivos.
    datos_bay = leer_resultados(RUTA_BAY)
    datos_nobay = leer_resultados(RUTA_NOBAY)
    
    pos_c_bay, pos_i_bay = int(datos_bay[0]), int(datos_bay[1])
    neg_c_bay, neg_i_bay = int(datos_bay[2]), int(datos_bay[3])
    prob_pc_bay, prob_pi_bay = datos_bay[4], datos_bay[5]
    prob_nc_bay, prob_ni_bay = datos_bay[6], datos_bay[7]
    
    pos_c_nobay, pos_i_nobay = int(datos_nobay[0]), int(datos_nobay[1])
    neg_c_nobay, neg_i_nobay = int(datos_nobay[2]), int(datos_nobay[3])
    prob_pc_nobay, prob_pi_nobay = datos_nobay[4], datos_nobay[5]
    prob_nc_nobay, prob_ni_nobay = datos_nobay[6], datos_nobay[7]
    
    # Resultados de la red bayesiana.
    resultados_bay = {
        "Positivos Correctos": {"count": pos_c_bay, "mean_prob": prob_pc_bay},
        "Falsos Positivos": {"count": pos_i_bay, "mean_prob": prob_pi_bay},
        "Negativos Correctos": {"count": neg_c_bay, "mean_prob": prob_nc_bay},
        "Falsos Negativos": {"count": neg_i_bay, "mean_prob": prob_ni_bay}
    }
    
    # Resultados de la red clásica.
    resultados_nobay = {
        "Positivos Correctos": {"count": pos_c_nobay,
                                "mean_prob": prob_pc_nobay},
        "Falsos Positivos": {"count": pos_i_nobay, "mean_prob": prob_pi_nobay},
        "Negativos Correctos": {"count": neg_c_nobay,
                                "mean_prob": prob_nc_nobay},
        "Falsos Negativos": {"count": neg_i_nobay, "mean_prob": prob_ni_nobay}
    }
    
    casos = list(resultados_bay.keys())
    
    # Gráfico de barras para contar positivos y negativos.
    counts_bayesian = [resultados_bay[c]["count"] for c in casos]
    counts_non_bayesian = [resultados_nobay[c]["count"] for c in casos]
    
    x = np.arange(len(casos))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, counts_bayesian, width, label="Red bayesiana")
    bars2 = ax.bar(x + width/2, counts_non_bayesian, width,
                   label="Red no bayesiana")
    
    ax.set_xlabel("Casos")
    ax.set_ylabel("Cantidades")
    ax.set_title("Comparación de las redes bayesiana y no bayesiana")
    ax.set_xticks(x)
    ax.set_xticklabels(casos)
    ax.legend()
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate("{}".format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom")
    
    plt.show()
    
    # Gráfico de barras para la probabilidad media.
    mean_prob_bayesian = [resultados_bay[c]["mean_prob"] for c in casos]
    mean_prob_non_bayesian = [resultados_nobay[c]["mean_prob"] for c in casos]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, mean_prob_bayesian, width, color="blue",
                   label="Red bayesiana")
    bars2 = ax.bar(x + width/2, mean_prob_non_bayesian, width, color="red",
                   label="Red no bayesiana")
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom")
    
    ax.set_xlabel("Casos")
    ax.set_ylabel("Probabilidad media")
    ax.set_title("Probabilidad media de las redes bayesiana y no bayesiana")
    ax.set_xticks(x)
    ax.set_xticklabels(casos)
    ax.legend()
    
    plt.show()
