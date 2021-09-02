#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:20:07 2020

Geração de funções de forma polinomiais completas para elementos bidimensionais em função da ordem:
    - 4 nós: ordem 1
    - 9 nós: ordem 2
    - 16 nós: ordem 3
    - 25 nós: ordem 4
    - 36 nós: ordem 5
    - 49 nós: ordem 6 (demora 4841.194 s para gerar as funções, 80.687 min, 1.345 h)

Faz qualquer ordem, além da 6!
Basta mudar a variável ordem, que altera as funções automaticamente!

Funções: OK!


@author: markinho
"""

import sympy as sp
import numpy as np
import time

inicioTempo = time.time()
print('Inicializando...')

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

x = sp.Symbol('x')
y = sp.Symbol('y')

for ordem in range(1, 7): #gerando todas as funções de forma de 1 a 6!!
    #termos de polinômios completos de ordem "ordem", sem os coeficientes
    # ordem = 1
    print("Gerando os polinômios completos de ordem %d" % ordem)
    parcialTempo = time.time()
    termos = []
    for i in range(0, ordem + 1):
        for j in range(0, ordem + 1):
            termos.append(x**i*y**j)
    nnos = len(termos) #número total de nós
    nnosD = int(np.sqrt(nnos)) #número de nós por dimensão
    print("Polinômios gerados em %.3f s (total: %.3f s)" % (time.time() - parcialTempo, time.time() - inicioTempo))
    
    print("Construindo a matriz de termos.")
    #definindo o tamanho do lado do elemento genérico l, quadrado
    l = sp.Symbol('l')
    
    #considerando o eixo cartesiano x, y com origem no centro do elemento quadrado: NÓS NO SENTIDO ANTI-HORÁRIO INICIANDO PELO NÓ COM x e y POSITIVOS!!!
    posicaoNos_x = sp.nsimplify(np.linspace(0.5, -0.5, nnosD)*l) #ou genérico com símbolos sp.symbols('x0:%d' % nnos)
    posicaoNos_y = sp.nsimplify(np.linspace(0.5, -0.5, nnosD)*l) #ou genérico com símbolos sp.symbols('y0:%d' % nnos)
    
    #construção da matriz de termos nas coordenadas dos nós
    matrizTermos = []
    ordemNos = []
    for j in range(nnosD):
        for i in range(nnosD):
            matrizTermos.append([expr.subs({x:posicaoNos_x[i], y:posicaoNos_y[j]}) for expr in termos])
            ordemNos.append([posicaoNos_x[i], posicaoNos_y[j]])
    matrizTermos = sp.Matrix(matrizTermos)
    print("Matriz de termos construída em %.3f s (total: %.3f s)" % (time.time() - parcialTempo, time.time() - inicioTempo))
    
    print("Determinando as funções de forma.")
    parcialTempo = time.time()
    #variávels para solução
    grauLiberdadeNos = sp.Matrix(sp.symbols('u0:%d' % len(termos)))
    
    #calculando os coeficientes
    Coefs = matrizTermos.inv() * grauLiberdadeNos
    
    #determinação das funções de forma
    Ncompleto = Coefs.T*sp.Matrix(termos)
    Ncompleto = sp.expand(Ncompleto[0])
    
    #extraindo os coeficientes relativos a cada grau de liberdade ou a cada nó
    N = []
    for i in range(nnos):
        N.append(sp.collect(Ncompleto, grauLiberdadeNos[i], evaluate=False)[grauLiberdadeNos[i]])
    N = sp.Matrix(N)
    print("Funções de forma determinadas em %.3f s (total: %.3f s)" % (time.time() - parcialTempo, time.time() - inicioTempo))
    
    np.save("funcoesN%d.npy" % (nnos), N) #já salva em array do numpy
    print("Funções de forma salvas! \n\n")

# ##grafico das funcoes de forma -----------------------------------------------------------------------------------
# #convertendo as equações da funções de forma para funções numpy
# print("Converterndo para valores numéricos para os gráficos.")
# parcialTempo = time.time()
# numericoN = []
# for i in range(nnos):
#     numericoN.append(sp.utilities.lambdify([x, y, l], N[i], "numpy"))

# #definido valores para x, y e l:
# numerico_l = 1.0 #já que a variação inicial foi de -0.5 a 0.5
# numerico_xy = np.linspace(-0.5*numerico_l, 0.5*numerico_l, 50) #valores para x e y numéricos, são iguais pois o elemento é quadrado

# #criando o grid para o gráfico
# grid_x, grid_y = np.meshgrid(numerico_xy, numerico_xy)
# print("Convertido para valores numéricos em %.3f s (total: %.3f s)" % (time.time() - parcialTempo, time.time() - inicioTempo))

# print("Iniciando a geração dos gráficos...")
# #geração do gráfico com o matplotlib
# fig = plt.figure(figsize=(16, 12), dpi=100)
# for i in range(nnos):
#     # ax = fig.add_subplot(nnosD, nnosD, i + 1, projection='3d') #todos em um gráfico
#     ax = fig.add_subplot(1, 1, 1, projection='3d') #todos em um gráfico
#     surf = ax.plot_surface(grid_x, grid_y, numericoN[i](grid_x, grid_y, numerico_l), cmap=cm.jet, linewidth=0, antialiased=False)
#     fig.colorbar(surf, shrink=0.7)
#     fig.savefig("Ns%d/funcaoN%d.png" % (nnos, i))
# # plt.show()
# print("Gráficos prontos em %.3f s (total: %.3f s)" % (time.time() - parcialTempo, time.time() - inicioTempo))
# #------------------------------------------------------------------------------------------------------------------

