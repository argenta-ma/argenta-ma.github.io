#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:14:23 2020

Elemento de placa de Reissner-Mildlin com 4 nós e interpolação linear


Resultados coerentes dos deslocamentos, mas reações de apoio estranhas...
Momentos e cortes muito estranhos... ??!?!?!?!

@author: markinho
"""

import sympy as sp
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

nosElemento = 4
grausLiberdadeNo = 3
grausLiberdade = grausLiberdadeNo*nosElemento

x, y, l = sp.symbols('x, y, l')
E, G, nu = sp.symbols('E, G, nu')
t = sp.Symbol('t') #espessura da placa

N = sp.Matrix(np.load('funcoesN%d.npy' % nosElemento, allow_pickle=True))

Blx = sp.diff(N, x)
Bly = sp.diff(N, y)

####!!!!! RESOLVENDO AINDA SEM O JACOBIANO!!!
u = sp.Matrix(sp.symbols('u0:%d' % grausLiberdade))

Bb = []
Bs = []
for i in range(nosElemento):
    BB = np.array([ [0, Blx[i],      0],
                    [0,      0, Bly[i]],
                    [0, Bly[i], Blx[i]] ])
    BS = np.array([ [Blx[i], -N[i],     0],
                    [Bly[i],      0, -N[i]] ])
    Bb.append(np.transpose(BB))
    Bs.append(np.transpose(BS))
Bb = np.transpose(np.array(Bb).reshape(12, 3))
Bs = np.transpose(np.array(Bs).reshape(12, 2))

Db = E/(1 - nu**2)*np.array([ [1, nu, 0],
                              [nu, 1, 0],
                              [0, 0, (1 - nu)/2]])
Ds = G*np.eye(2)

Bb_mult = sp.Matrix(np.matmul(np.matmul(Bb.T, Db), Bb))
Bs_mult = sp.Matrix(np.matmul(np.matmul(Bs.T, Ds), Bs))

#integrando e calculando as matrizes de rigidez !!!!! será que tem que multiplicar pelo determinante do jabociano L/2?????
Kb = t*sp.integrate( sp.integrate( Bb_mult, (x, -l*sp.Rational(1, 2), l*sp.Rational(1, 2)) ), (y, -l*sp.Rational(1, 2), l*sp.Rational(1, 2)) )#*L*sp.Rational(1, 2)
Ks = t*sp.integrate( sp.integrate( Bs_mult, (x, -l*sp.Rational(1, 2), l*sp.Rational(1, 2)) ), (y, -l*sp.Rational(1, 2), l*sp.Rational(1, 2)) )#*L*sp.Rational(1, 2)


#Usando valores numéricos
Ep = 20000.
Gp = 7700.
nup = 0.3
L = 1. #precisa ser 1 pois não foi usado o Jacobiano???
tp = 50. #cm

Kbp = np.array(Kb.subs({E: Ep, nu: nup, l: L, t: tp }), dtype = float) #### usar o lambdify!!!
Ksp = np.array(Ks.subs({G: Gp, nu: nup, l: L, t: tp }), dtype = float)

F = np.array([-10., 0., 0., -10., 0., 0. ])

#restringido a placa, engastada na esquerda (negativos x)
Ku = np.delete(np.delete(Kbp, [3, 4, 5, 9, 10, 11], axis=0), [3, 4, 5, 9, 10, 11], axis=1) + np.delete(np.delete(Ksp, [3, 4, 5, 9, 10, 11], axis=0), [3, 4, 5, 9, 10, 11], axis=1)
Kr = np.delete(np.delete(Kbp, [3, 4, 5, 9, 10, 11], axis=0), [0, 1, 2, 6, 7, 8], axis=1) + np.delete(np.delete(Ksp, [3, 4, 5, 9, 10, 11], axis=0), [0, 1, 2, 6, 7, 8], axis=1)

U = np.linalg.solve(Ku, F)
Ra = np.matmul(Kr, U)

Ug = np.zeros(12)
Ug[[3, 4, 5, 9, 10, 11]] = U

epsilon_b = sp.Matrix(np.matmul(Bb, Ug)).subs({E: Ep, nu: nup, l: L, t: tp })
epsilon_s = sp.Matrix(np.matmul(Bs, Ug)).subs({E: Ep, nu: nup, l: L, t: tp })

Ms = (t**3/12*Db*epsilon_b).subs({E: Ep, nu: nup, l: L, t: tp })
Qs = (5/6*t*Ds*epsilon_s).subs({G: Gp, nu: nup, l: L, t: tp })

#gerando os gráficos de momentos e cortantes-----------------------------------------------------------------------------
#definido valores para x, y:
numerico_xy = np.linspace(-0.5*L, 0.5*L, 50) #valores para x e y numéricos, são iguais pois o elemento é quadrado

#criando funções numéricas para Ms e Qs
Mx = sp.utilities.lambdify([x, y], Ms[0], "numpy")
My = sp.utilities.lambdify([x, y], Ms[1], "numpy")
Mxy = sp.utilities.lambdify([x, y], Ms[2], "numpy")
Qx = sp.utilities.lambdify([x, y], Qs[0], "numpy")
Qy = sp.utilities.lambdify([x, y], Qs[1], "numpy")

#criando o grid para o gráfico
grid_x, grid_y = np.meshgrid(numerico_xy, numerico_xy)

#geração do gráfico com o matplotlib
fig = plt.figure(figsize=(16, 12), dpi=100)
ax = fig.add_subplot(1, 1, 1, projection='3d') #todos em um gráfico
surfMx = ax.plot_surface(grid_x, grid_y, Mx(grid_x, grid_y), cmap=cm.jet, linewidth=0, antialiased=False)
fig.colorbar(surfMx, shrink=0.7)
plt.show()

fig = plt.figure(figsize=(16, 12), dpi=100)
ax = fig.add_subplot(1, 1, 1, projection='3d') #todos em um gráfico
surfMy = ax.plot_surface(grid_x, grid_y, My(grid_x, grid_y), cmap=cm.jet, linewidth=0, antialiased=False)
fig.colorbar(surfMy, shrink=0.7)
plt.show()

fig = plt.figure(figsize=(16, 12), dpi=100)
ax = fig.add_subplot(1, 1, 1, projection='3d') #todos em um gráfico
surfMxy = ax.plot_surface(grid_x, grid_y, Mxy(grid_x, grid_y), cmap=cm.jet, linewidth=0, antialiased=False)
fig.colorbar(surfMxy, shrink=0.7)
plt.show()

fig = plt.figure(figsize=(16, 12), dpi=100)
ax = fig.add_subplot(1, 1, 1, projection='3d') #todos em um gráfico
surfQx = ax.plot_surface(grid_x, grid_y, Qx(grid_x, grid_y), cmap=cm.jet, linewidth=0, antialiased=False)
fig.colorbar(surfQx, shrink=0.7)
plt.show()

fig = plt.figure(figsize=(16, 12), dpi=100)
ax = fig.add_subplot(1, 1, 1, projection='3d') #todos em um gráfico
surfQy = ax.plot_surface(grid_x, grid_y, Qy(grid_x, grid_y), cmap=cm.jet, linewidth=0, antialiased=False)
fig.colorbar(surfQy, shrink=0.7)
plt.show()
#------------------------------------------------------------------------------------------------------------------
