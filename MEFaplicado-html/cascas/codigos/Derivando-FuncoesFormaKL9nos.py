#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Na plotagem: PORQUE N2, N3, N6, N11, N15 e N23 NEGATIVOS???



O elemento padrão:

    2 -- 5 -- 1
    |         |
    6    9    8
    |         |
    3 -- 7 -- 4

@author: markinho
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
#import meshio

import time

tempo_inicio = time.process_time()
print('Processo iniciado, gerando as funções de forma.')

#elemento padrão ------------------------------------------------------------------------------------------------------------------------------------
rs = np.array([[1, 1],
               [-1, 1],
               [-1, -1],
               [1, -1],
               [0, 1],
               [-1, 0],
               [0, -1],
               [1, 0],
               [0, 0]])

u1 = sp.Symbol('u1')
u2 = sp.Symbol('u2')
u3 = sp.Symbol('u3')
u4 = sp.Symbol('u4')
u5 = sp.Symbol('u5')
u6 = sp.Symbol('u6')
u7 = sp.Symbol('u7')
u8 = sp.Symbol('u8')
u9 = sp.Symbol('u9')
u10 = sp.Symbol('u10')
u11 = sp.Symbol('u11')
u12 = sp.Symbol('u12')
u13 = sp.Symbol('u13')
u14 = sp.Symbol('u14')
u15 = sp.Symbol('u15')
u16 = sp.Symbol('u16')
u17 = sp.Symbol('u17')
u18 = sp.Symbol('u18')
u19 = sp.Symbol('u19')
u20 = sp.Symbol('u20')
u21 = sp.Symbol('u21')
u22 = sp.Symbol('u22')
u23 = sp.Symbol('u23')
u24 = sp.Symbol('u24')
u25 = sp.Symbol('u25')
u26 = sp.Symbol('u26')
u27 = sp.Symbol('u27')

##polinomio maluco incompleto para montagem da matriz dos coeficientes
x = sp.Symbol('x')
y = sp.Symbol('y')
pmi = sp.Matrix([1, x, x*y, y, 
                 x**2, x**2*y, x**2*y**2, x*y**2, y**2, 
                 x**3, x**3*y, x**3*y**2, x**2*y**3, x*y**3, y**3, 
                 x**4, x**4*y, x**4*y**2, x**2*y**4, x*y**4, y**4, 
                 x**5, x**5*y, x**5*y**2, x**2*y**5, x*y**5, y**5])

pmi = pmi.T
dpmidx = sp.diff(pmi, x)
dpmidy = sp.diff(pmi, y)

pmiN = []
for coord in rs:
    pmiN.append(pmi.subs({x: coord[0], y: coord[1]}))
    pmiN.append(dpmidy.subs({x: coord[0], y: coord[1]}))
    pmiN.append(-dpmidx.subs({x: coord[0], y: coord[1]}))
Mat_Coef = sp.Matrix(pmiN)

ue = sp.Matrix([u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18, u19, u20, u21, u22, u23, u24, u25, u26, u27])

print('Calculando os coeficientes...')
Coefs = Mat_Coef.inv() * ue

Ac = Coefs[0]
Bc = Coefs[1]
Cc = Coefs[2]
Dc = Coefs[3]
Ec = Coefs[4]
Fc = Coefs[5]
Gc = Coefs[6]
Hc = Coefs[7]
Ic = Coefs[8]
Jc = Coefs[9]
Kc = Coefs[10]
Lc = Coefs[11]
Mc = Coefs[12]
Nc = Coefs[13]
Oc = Coefs[14]
Pc = Coefs[15]
Qc = Coefs[16]
Rc = Coefs[17]
Sc = Coefs[18]
Tc = Coefs[19]
Uc = Coefs[20]
Wc = Coefs[21]
Xc = Coefs[22]
Yc = Coefs[23]
Zc = Coefs[24]
Acc = Coefs[25]
Bcc = Coefs[26]

r = sp.Symbol('r')
s = sp.Symbol('s')

Ns = sp.expand(Ac + Bc*r + Cc*r*s + Dc*s + Ec*r**2 + Fc*r**2*s + Gc*r**2*s**2 + Hc*r*s**2 + Ic*s**2 + Jc*r**3 + 
               Kc*r**3*s + Lc*r**3*s**2 + Mc*r**2*s**3 + Nc*r*s**3 + Oc*s**3 + Pc*r**4 + Qc*r**4*s + Rc*r**4*s**2 + 
               Sc*r**2*s**4 + Tc*r*s**4 + Uc*s**4 + Wc*r**5 + Xc*r**5*s + Yc*r**5*s**2 + Zc*r**2*s**5 + Acc*r*s**5 + Bcc*s**5)

N1 = sp.Add(*[argi for argi in Ns.args if argi.has(u1)]).subs(u1, 1)
N2 = sp.Add(*[argi for argi in Ns.args if argi.has(u2)]).subs(u2, 1)
N3 = sp.Add(*[argi for argi in Ns.args if argi.has(u3)]).subs(u3, 1)
N4 = sp.Add(*[argi for argi in Ns.args if argi.has(u4)]).subs(u4, 1)
N5 = sp.Add(*[argi for argi in Ns.args if argi.has(u5)]).subs(u5, 1)
N6 = sp.Add(*[argi for argi in Ns.args if argi.has(u6)]).subs(u6, 1)
N7 = sp.Add(*[argi for argi in Ns.args if argi.has(u7)]).subs(u7, 1)
N8 = sp.Add(*[argi for argi in Ns.args if argi.has(u8)]).subs(u8, 1)
N9 = sp.Add(*[argi for argi in Ns.args if argi.has(u9)]).subs(u9, 1)
N10 = sp.Add(*[argi for argi in Ns.args if argi.has(u10)]).subs(u10, 1)
N11 = sp.Add(*[argi for argi in Ns.args if argi.has(u11)]).subs(u11, 1)
N12 = sp.Add(*[argi for argi in Ns.args if argi.has(u12)]).subs(u12, 1)
N13 = sp.Add(*[argi for argi in Ns.args if argi.has(u13)]).subs(u13, 1)
N14 = sp.Add(*[argi for argi in Ns.args if argi.has(u14)]).subs(u14, 1)
N15 = sp.Add(*[argi for argi in Ns.args if argi.has(u15)]).subs(u15, 1)
N16 = sp.Add(*[argi for argi in Ns.args if argi.has(u16)]).subs(u16, 1)
N17 = sp.Add(*[argi for argi in Ns.args if argi.has(u17)]).subs(u17, 1)
N18 = sp.Add(*[argi for argi in Ns.args if argi.has(u18)]).subs(u18, 1)
N19 = sp.Add(*[argi for argi in Ns.args if argi.has(u19)]).subs(u19, 1)
N20 = sp.Add(*[argi for argi in Ns.args if argi.has(u20)]).subs(u20, 1)
N21 = sp.Add(*[argi for argi in Ns.args if argi.has(u21)]).subs(u21, 1)
N22 = sp.Add(*[argi for argi in Ns.args if argi.has(u22)]).subs(u22, 1)
N23 = sp.Add(*[argi for argi in Ns.args if argi.has(u23)]).subs(u23, 1)
N24 = sp.Add(*[argi for argi in Ns.args if argi.has(u24)]).subs(u24, 1)
N25 = sp.Add(*[argi for argi in Ns.args if argi.has(u25)]).subs(u25, 1)
N26 = sp.Add(*[argi for argi in Ns.args if argi.has(u26)]).subs(u26, 1)
N27 = sp.Add(*[argi for argi in Ns.args if argi.has(u27)]).subs(u27, 1)

N = sp.Matrix([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17, N18, N19, N20, N21, N22, N23, N24, N25, N26, N27])

##Na plotagem: PORQUE N3, N6, N9, N12, N15, N18, N21, N24 e N27 NEGATIVOS??? corrigindo abaixo
#N = sp.Matrix([N1, N2, -N3, N4, N5, -N6, N7, N8, -N9, N10, N11, -N12, N13, N14, -N15, N16, N17, -N18, N19, N20, -N21, N22, N23, -N24, N25, N26, -N27])
tempo_parcial = time.process_time()
print('Geração das funções de forma completa! ' + str(time.process_time() - tempo_inicio))
#-----------------------------------------------------------------------------------------------------------------------------------------------------

##plotagem com o matplotlib --------------------------------------------------------------------------------------------------------------------------
#nN1 = sp.utilities.lambdify([r, s], N1, "numpy")
#nN2 = sp.utilities.lambdify([r, s], N2, "numpy")
#nN3 = sp.utilities.lambdify([r, s], N3, "numpy")
#nN4 = sp.utilities.lambdify([r, s], N4, "numpy")
#nN5 = sp.utilities.lambdify([r, s], N5, "numpy")
#nN6 = sp.utilities.lambdify([r, s], N6, "numpy")
#nN7 = sp.utilities.lambdify([r, s], N7, "numpy")
#nN8 = sp.utilities.lambdify([r, s], N8, "numpy")
#nN9 = sp.utilities.lambdify([r, s], N9, "numpy")
#nN10 = sp.utilities.lambdify([r, s], N10, "numpy")
#nN11 = sp.utilities.lambdify([r, s], N11, "numpy")
#nN12 = sp.utilities.lambdify([r, s], N12, "numpy")
#nN13 = sp.utilities.lambdify([r, s], N13, "numpy")
#nN14 = sp.utilities.lambdify([r, s], N14, "numpy")
#nN15 = sp.utilities.lambdify([r, s], N15, "numpy")
#nN16 = sp.utilities.lambdify([r, s], N16, "numpy")
#nN17 = sp.utilities.lambdify([r, s], N17, "numpy")
#nN18 = sp.utilities.lambdify([r, s], N18, "numpy")
#nN19 = sp.utilities.lambdify([r, s], N19, "numpy")
#nN20 = sp.utilities.lambdify([r, s], N20, "numpy")
#nN21 = sp.utilities.lambdify([r, s], N21, "numpy")
#nN22 = sp.utilities.lambdify([r, s], N22, "numpy")
#nN23 = sp.utilities.lambdify([r, s], N23, "numpy")
#nN24 = sp.utilities.lambdify([r, s], N24, "numpy")
#nN25 = sp.utilities.lambdify([r, s], N25, "numpy")
#nN26 = sp.utilities.lambdify([r, s], N26, "numpy")
#nN27 = sp.utilities.lambdify([r, s], N27, "numpy")
#
#rl = np.linspace(-1., 1., 100)
#sl = np.linspace(-1., 1., 100)
#
#rm, sm = np.meshgrid(rl, sl)
#
##para o nó 1, 2, 3 e 4
#fig = plt.figure()
##ax = Axes3D(fig)
#
#ax = fig.add_subplot(4, 3, 1, projection='3d')
#ax.set_title('N1')
#surf = ax.plot_surface(rm, sm, nN1(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 2, projection='3d')
#ax.set_title('N2')
#surf = ax.plot_surface(rm, sm, nN2(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 3, projection='3d')
#ax.set_title('N3')
#surf = ax.plot_surface(rm, sm, nN3(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(4, 3, 4, projection='3d')
#ax.set_title('N4')
#surf = ax.plot_surface(rm, sm, nN4(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 5, projection='3d')
#ax.set_title('N5')
#surf = ax.plot_surface(rm, sm, nN5(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 6, projection='3d')
#ax.set_title('N6')
#surf = ax.plot_surface(rm, sm, nN6(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(4, 3, 7, projection='3d')
#ax.set_title('N7')
#surf = ax.plot_surface(rm, sm, nN7(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 8, projection='3d')
#ax.set_title('N8')
#surf = ax.plot_surface(rm, sm, nN8(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 9, projection='3d')
#ax.set_title('N9')
#surf = ax.plot_surface(rm, sm, nN9(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(4, 3, 10, projection='3d')
#ax.set_title('N10')
#surf = ax.plot_surface(rm, sm, nN10(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 11, projection='3d')
#ax.set_title('N11')
#surf = ax.plot_surface(rm, sm, nN11(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 12, projection='3d')
#ax.set_title('N12')
#surf = ax.plot_surface(rm, sm, nN12(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#plt.show()
#
##para o nó 5, 6, 7, 8 e 9
#fig = plt.figure()
##ax = Axes3D(fig)
#
#ax = fig.add_subplot(5, 3, 1, projection='3d')
#ax.set_title('N13')
#surf = ax.plot_surface(rm, sm, nN13(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 2, projection='3d')
#ax.set_title('N14')
#surf = ax.plot_surface(rm, sm, nN14(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 3, projection='3d')
#ax.set_title('N15')
#surf = ax.plot_surface(rm, sm, nN15(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(5, 3, 4, projection='3d')
#ax.set_title('N16')
#surf = ax.plot_surface(rm, sm, nN16(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 5, projection='3d')
#ax.set_title('N17')
#surf = ax.plot_surface(rm, sm, nN17(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 6, projection='3d')
#ax.set_title('N18')
#surf = ax.plot_surface(rm, sm, nN18(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(5, 3, 7, projection='3d')
#ax.set_title('N19')
#surf = ax.plot_surface(rm, sm, nN19(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 8, projection='3d')
#ax.set_title('N20')
#surf = ax.plot_surface(rm, sm, nN20(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 9, projection='3d')
#ax.set_title('N21')
#surf = ax.plot_surface(rm, sm, nN21(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(5, 3, 10, projection='3d')
#ax.set_title('N22')
#surf = ax.plot_surface(rm, sm, nN22(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 11, projection='3d')
#ax.set_title('N23')
#surf = ax.plot_surface(rm, sm, nN23(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 12, projection='3d')
#ax.set_title('N24')
#surf = ax.plot_surface(rm, sm, nN24(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(5, 3, 13, projection='3d')
#ax.set_title('N25')
#surf = ax.plot_surface(rm, sm, nN25(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 14, projection='3d')
#ax.set_title('N26')
#surf = ax.plot_surface(rm, sm, nN26(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 15, projection='3d')
#ax.set_title('N27')
#surf = ax.plot_surface(rm, sm, nN27(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#plt.show()
##---------------------------------------------------------------------------------------------------------------------------------------------------

##resolvendo o equilíbrio minimizando o funcional de energia potencial total --------------------------------------------------------------------------
###NÃO FUNCIONOU!!!!!
#z = sp.Symbol('z')
#t = sp.Symbol('t')
#Ee = sp.Symbol('Ee')
#nu = sp.Symbol('nu')
#
##w = (Nc * ue)[0]
##
##epsilon = sp.Matrix([ [sp.diff(w, r, r)],
##                      [sp.diff(w, s, s)],
##                      [2*sp.diff(w, r, s)]])
#
#tempo_parcial = time.process_time()
#print('Criando epsilon... ' + str(time.process_time() - tempo_inicio))
#epsilonN = sp.Matrix([ sp.expand(sp.diff(Nc, r, r)), sp.expand(sp.diff(Nc, s, s)), 2 * sp.expand(sp.diff(Nc, r, s))])
#print('Epsilon criado!')
#
#De = sp.Matrix([[1, nu, 0], 
#                [nu, 1, 0], 
#                [0,  0, (1 - nu)/2]])
#Ep = 1/12 * Ee * t**3/(1 - nu**2) 
#
#tempo_parcial = time.process_time()
#print('Multiplicando epsilon.T De epsilon... ' + str(time.process_time() - tempo_inicio))
#integrando = sp.expand(epsilonN.T * De * epsilonN)
#print('Epsilon.T De epsilon multiplicado! '  + str(time.process_time() - tempo_inicio) + ' -> ' + str(time.process_time() - tempo_parcial))
#
#tempo_parcial = time.process_time()
#print('Integrando analiticamente em r e s... ' + str(time.process_time() - tempo_inicio))
#PI = sp.integrate( sp.integrate(integrando, (r, -1, 1) ), (s, -1, 1) )
#print('Integração concluída! ' + str(time.process_time() - tempo_inicio) + ' -> ' + str(time.process_time() - tempo_parcial))
#
#tempo_parcial = time.process_time()
#print('Multiplicando pelos deslocamentos... ' + str(time.process_time() - tempo_inicio))
#PIc = ue.T * PI * ue
#print('Multiplicação concluída! ' + str(time.process_time() - tempo_inicio) + ' -> ' + str(time.process_time() - tempo_parcial))
#
#ku_diffs = []
#
#tempo_parcial = time.process_time()
#print('Variando nos deslocamentos... ' + str(time.process_time() - tempo_inicio))
#for i in range(27):
#    ku_diffs.append(sp.diff(sp.expand(PIc[0,0]), ue[i]))
#print('Variação concluída! ' + str(time.process_time() - tempo_inicio) + ' -> ' + str(time.process_time() - tempo_parcial))
#
#tempo_parcial = time.process_time()
#print('Montando a matriz de rigidez... ' + str(time.process_time() - tempo_inicio))
#K = sp.zeros(27, 27)
#for i in range(27):
#    for j in range(27):
#        K[i, j] = sp.Add(*[argi for argi in ku_diffs[j].args if argi.has(ue[i])]).subs(ue[i], 1)
#print('Matriz de rigidez concluída! ' + str(time.process_time() - tempo_inicio) + ' -> ' + str(time.process_time() - tempo_parcial))

#ku1 = sp.diff(sp.expand(PIc[0,0]), u1)
#k1_1 = sp.Add(*[argi for argi in ku1.args if argi.has(u1)]).subs(u1, 1)
#k1_2 = sp.Add(*[argi for argi in ku1.args if argi.has(u2)]).subs(u2, 1)
#k1_3 = sp.Add(*[argi for argi in ku1.args if argi.has(u3)]).subs(u3, 1)
#k1_4 = sp.Add(*[argi for argi in ku1.args if argi.has(u4)]).subs(u4, 1)
#k1_5 = sp.Add(*[argi for argi in ku1.args if argi.has(u5)]).subs(u5, 1)
#k1_6 = sp.Add(*[argi for argi in ku1.args if argi.has(u6)]).subs(u6, 1)
#k1_7 = sp.Add(*[argi for argi in ku1.args if argi.has(u7)]).subs(u7, 1)
#k1_8 = sp.Add(*[argi for argi in ku1.args if argi.has(u8)]).subs(u8, 1)
#k1_9 = sp.Add(*[argi for argi in ku1.args if argi.has(u9)]).subs(u9, 1)
#k1_10 = sp.Add(*[argi for argi in ku1.args if argi.has(u10)]).subs(u10, 1)
#k1_11 = sp.Add(*[argi for argi in ku1.args if argi.has(u11)]).subs(u11, 1)
#k1_12 = sp.Add(*[argi for argi in ku1.args if argi.has(u12)]).subs(u12, 1)
#k1_13 = sp.Add(*[argi for argi in ku1.args if argi.has(u13)]).subs(u13, 1)
#k1_14 = sp.Add(*[argi for argi in ku1.args if argi.has(u14)]).subs(u14, 1)
#k1_15 = sp.Add(*[argi for argi in ku1.args if argi.has(u15)]).subs(u15, 1)
#k1_16 = sp.Add(*[argi for argi in ku1.args if argi.has(u16)]).subs(u16, 1)
#k1_17 = sp.Add(*[argi for argi in ku1.args if argi.has(u17)]).subs(u17, 1)
#k1_18 = sp.Add(*[argi for argi in ku1.args if argi.has(u18)]).subs(u18, 1)
#k1_19 = sp.Add(*[argi for argi in ku1.args if argi.has(u19)]).subs(u19, 1)
#k1_20 = sp.Add(*[argi for argi in ku1.args if argi.has(u20)]).subs(u20, 1)
#k1_21 = sp.Add(*[argi for argi in ku1.args if argi.has(u21)]).subs(u21, 1)
#k1_22 = sp.Add(*[argi for argi in ku1.args if argi.has(u22)]).subs(u22, 1)
#k1_23 = sp.Add(*[argi for argi in ku1.args if argi.has(u23)]).subs(u23, 1)
#k1_24 = sp.Add(*[argi for argi in ku1.args if argi.has(u24)]).subs(u24, 1)
#k1_25 = sp.Add(*[argi for argi in ku1.args if argi.has(u25)]).subs(u25, 1)
#k1_26 = sp.Add(*[argi for argi in ku1.args if argi.has(u26)]).subs(u26, 1)
#k1_27 = sp.Add(*[argi for argi in ku1.args if argi.has(u27)]).subs(u27, 1)
#-----------------------------------------------------------------------------------------------------------------------------------------------------

##procedimento pela integração de Gauss ---------------------------------------------------------------------------------------------------------------
#NÃO FUNCIONOU, VOLUME MUITO GRANDE DE INFORMAÇÃO!!!
tempo_parcial = time.process_time()
print('Iniciando as derivadas das funções de interpolação... ' + str(time.process_time() - tempo_inicio))
#primeira derivada em r
dNr = sp.diff(N, r).T
#segunda derivada em r
dNrr = sp.diff(N, r, r).T

#primeira derivada em s
dNs = sp.diff(N, s).T
#segunda derivada em s
dNss = sp.diff(N, s, s).T

#derivada em r e s
dNrs = sp.diff(N, r, s).T

#gerando o Jacobiano analítico
x1 = sp.Symbol('x1')
y1 = sp.Symbol('y1')
x2 = sp.Symbol('x2')
y2 = sp.Symbol('y2')
x3 = sp.Symbol('x3')
y3 = sp.Symbol('y3')
x4 = sp.Symbol('x4')
y4 = sp.Symbol('y4')
x5 = sp.Symbol('x5')
y5 = sp.Symbol('y5')
x6 = sp.Symbol('x6')
y6 = sp.Symbol('y6')
x7 = sp.Symbol('x7')
y7 = sp.Symbol('y7')
x8 = sp.Symbol('x8')
y8 = sp.Symbol('y8')
x9 = sp.Symbol('x9')
y9 = sp.Symbol('y9')

#Matriz dos nós de um elemento
Xe = sp.Matrix([[x1, y1],[x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6], [x7, y7], [x8, y8], [x9, y9]])

#Matriz das derivadas, segundas derivadas e derivadas em r e s das funções de interpolação do elemento padrão no sistema r s
dNdr = np.concatenate((dNr, dNs), axis=0)
d2Ndr2 = np.concatenate((dNrr, dNss, dNrs), axis=0)

#indices para a seleção das funções de interpolação somente do w
indice_w = [0, 3, 6, 9, 12, 15, 18, 21, 24]
dN_w = dNdr[:,indice_w]
dN2_w = d2Ndr2[:2,indice_w]
dNrs_w = dNrs[:,indice_w]

#print('Cálculo das derivadas das funções de interpolação completo! ' + str(time.process_time() - tempo_parcial) + ' -> ' + str(time.process_time() - tempo_inicio))
#
#tempo_parcial = time.process_time()
#print('Iniciando o cálculo dos Jacobianos... ' + str(time.process_time() - tempo_inicio))




##Jacobiano analítico
#J = sp.expand(Xe.T * dN_w.T)
#dJ = sp.expand(Xe.T * dN2_w.T)
#dJrs = sp.expand(Xe.T * dNrs_w.T)
#
#J23 = sp.Matrix([ [dJ[0,0], dJ[1,0]],
#                  [dJ[0,1], dJ[1,1]],
#                  [dJrs[0], dJrs[1]] ])
#
##jacobiano expandido
#Jex = sp.Matrix([ [     J[0,0]**2,     J[1,0]**2,               2*J[0,0]*J[1,0] ],
#                  [     J[0,1]**2,     J[1,1]**2,               2*J[0,1]*J[1,1] ],
#                  [ J[0,0]*J[0,1], J[1,0]*J[1,1], J[1,0]*J[0,1] + J[0,0]*J[1,1] ]])
#print('Invertendo o Jacobiano e o Jacobiano expandido...')
#JI = J.inv()
#JexI = sp.expand(Jex).inv()
#
#print('Cálculo dos Jacobianos completo! ' + str(time.process_time() - tempo_parcial) + ' -> ' + str(time.process_time() - tempo_inicio))
#
#tempo_parcial = time.process_time()
#print('Iniciando o cálculo das derivadas das funções de interpolação para as deformações... ' + str(time.process_time() - tempo_inicio))
###derivadas das funções de interpolação do elemento no sistema local x y para placas
#dNdx = dNdr.T * JI
#dNdxP = JexI * ( d2Ndr2.T - J23 * dNdx )
#print('Cálculo das derivadas das funções de interpolação para as deformações completo! ' + str(time.process_time() - tempo_parcial) + ' -> ' + str(time.process_time() - tempo_inicio))
#-----------------------------------------------------------------------------------------------------------------------------------------------------

