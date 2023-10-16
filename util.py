import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_decision_boundary(model,X,y,normalize=False):
    #código obtido em: https://markd87.github.io/articles/ml.html
    padding=0.15
    res=0.01

    x_axis_list = X.to_numpy()[:,0] if type(X) == pd.DataFrame else X[:,0]
    y_axis_list = X.to_numpy()[:,1] if type(X) == pd.DataFrame else X[:,1]

    #valores máximos e mínimos de x e y do dataset
    x_min,x_max=x_axis_list.min(), x_axis_list.max()
    y_min,y_max=y_axis_list.min(), y_axis_list.max()

    #normaliza os valores de X e Y (se necessário)
    if(normalize):
        x_axis_list /= x_max
        y_axis_list /= y_max

    #range dos x e dos y
    x_range=x_max-x_min
    y_range=y_max-y_min

    #adiciona padding nos ranges
    x_min -= x_range * padding
    y_min -= y_range * padding
    x_max += x_range * padding
    y_max += y_range * padding

    #cria um meshgrid (grade) dos pontos fora dos ranges
    xx,yy=np.meshgrid(np.arange(x_min,x_max,res),np.arange(y_min,y_max,res))

    #usa o modelo para prever classe em cada ponto da grid (grade)
    #método ravel trasnforma os arrays 2d em vetores
    #método c_ concaceta os vetores para criar um grande vetor no qual realiza a predição
    #finalmente, o vetor de predição passa um reshape(é reformulado) para o formato original dos dados
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    #plota os contornos da grade
    plt.figure(figsize=(8,6))
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

    #plota os dados originais e os labels
    plt.scatter(x_axis_list, y_axis_list, s=35, c=y, cmap=plt.cm.Spectral)
