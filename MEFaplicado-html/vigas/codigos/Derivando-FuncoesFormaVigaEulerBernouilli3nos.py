#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Funções de forma para a viga de 3 nós de Euler-Bernouilli

Completo!

@author: markinho
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


#para viga
L = sp.Symbol('L')
x1 = -L/2
x2 = 0
x3 = L/2
u1 = sp.Symbol('u1')
u2 = sp.Symbol('u2')
u3 = sp.Symbol('u3')
u4 = sp.Symbol('u4')
u5 = sp.Symbol('u5')
u6 = sp.Symbol('u6')

#Mat_Coef = sp.Matrix([[1, -L/2, L**2/4, -L**3/8, L**4/16, -L**5/32],
#                      [0, 1, -L, 3*L**2/4, -L**3/2, 5*L**4/16],
#                      [1, 0, 0, 0, 0, 0],
#                      [0, 1, 0, 0, 0, 0],
#                      [1, L/2, L**2/4, L**3/8, L**4/16, L**5/32],
#                      [0, 1, L, 3*L**2/4, L**3/2, 5*L**4/16]])

Mat_Coef = sp.Matrix([[1, x1, x1**2, x1**3, x1**4, x1**5],
                      [0, 1, 2*x1, 3*x1**2, 4*x1**3, 5*x1**4],
                      [1, x2, x2**2, x2**3, x2**4, x2**5],
                      [0, 1, 2*x2, 3*x2**2, 4*x2**3, 5*x2**4],
                      [1, x3, x3**2, x3**3, x3**4, x3**5],
                      [0, 1, 2*x3, 3*x3**2, 4*x3**3, 5*x3**4]])

U = sp.Matrix([u1, u2, u3, u4, u5, u6])

Coefs = Mat_Coef.inv() * U

A = Coefs[0]
B = Coefs[1]
C = Coefs[2]
D = Coefs[3]
E = Coefs[4]
F = Coefs[5]

x = sp.Symbol('x')

Ns = sp.expand(A + B*x + C*x**2 + D*x**3 + E*x**4 + F*x**5)

N1 = sp.Add(*[argi for argi in Ns.args if argi.has(u1)]).subs(u1, 1)
N2 = sp.Add(*[argi for argi in Ns.args if argi.has(u2)]).subs(u2, 1)
N3 = sp.Add(*[argi for argi in Ns.args if argi.has(u3)]).subs(u3, 1)
N4 = sp.Add(*[argi for argi in Ns.args if argi.has(u4)]).subs(u4, 1)
N5 = sp.Add(*[argi for argi in Ns.args if argi.has(u5)]).subs(u5, 1)
N6 = sp.Add(*[argi for argi in Ns.args if argi.has(u6)]).subs(u6, 1)

Nn = sp.Matrix([N1, N2, N3, N4, N5, N6])

##geração do grafico ---------------------------------------------------------------------
##convertendo para função python
#nN1 = sp.utilities.lambdify([x, L], N1, "numpy")
#nN2 = sp.utilities.lambdify([x, L], N2, "numpy")
#
#nN3 = sp.utilities.lambdify([x, L], N3, "numpy")
#nN4 = sp.utilities.lambdify([x, L], N4, "numpy")
#
#nN5 = sp.utilities.lambdify([x, L], N5, "numpy")
#nN6 = sp.utilities.lambdify([x, L], N6, "numpy")
#
#L = 2.
#x = np.linspace(-L/2., L/2, 100)
#
#plt.plot(x, nN1(x, L), label="N1")
#plt.plot(x, nN3(x, L), label="N3")
#plt.plot(x, nN5(x, L), label="N5")
#plt.title('Deslocamentos')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#
#plt.plot(x, nN2(x, L), label="N2")
#plt.plot(x, nN4(x, L), label="N4")
#plt.plot(x, nN6(x, L), label="N6")
#plt.title('Rotações')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()

#primeira derivada
dN1 = sp.diff(N1, x)
dN2 = sp.diff(N2, x)
dN3 = sp.diff(N3, x)
dN4 = sp.diff(N4, x)
dN5 = sp.diff(N5, x)
dN6 = sp.diff(N6, x)

#segunda derivada
ddN1 = sp.diff(dN1, x)
ddN2 = sp.diff(dN2, x)
ddN3 = sp.diff(dN3, x)
ddN4 = sp.diff(dN4, x)
ddN5 = sp.diff(dN5, x)
ddN6 = sp.diff(dN6, x)

#terceira derivada
dddN1 = sp.diff(ddN1, x)
dddN2 = sp.diff(ddN2, x)
dddN3 = sp.diff(ddN3, x)
dddN4 = sp.diff(ddN4, x)
dddN5 = sp.diff(ddN5, x)
dddN6 = sp.diff(ddN6, x)

#quarta derivada
ddddN1 = sp.diff(dddN1, x)
ddddN2 = sp.diff(dddN2, x)
ddddN3 = sp.diff(dddN3, x)
ddddN4 = sp.diff(dddN4, x)
ddddN5 = sp.diff(dddN5, x)
ddddN6 = sp.diff(dddN6, x)

#cálculo da matriz de rigidez
B = sp.Matrix([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6])
BB = B * B.T

E = sp.Symbol('E')
I = sp.Symbol('I')

Ke = E*I*sp.integrate( BB, (x, x1, x3) )

#viga em balanço com 1000kN na extremidade, E = 20000 kN/cm2, nu=0.3, b=2cm, h=10cm e L = 6m ---------------------
b = 2
h = 10
F = 1000

Kvb = Ke[2:,2:]
Kvb = np.array(Kvb.subs({L:6, I:b*h**3/12, E:20000})).astype(np.float64)
Fvb = np.array([0, 0, -F, 0])

UvbEB = np.linalg.solve(Kvb,Fvb)
UgvbEB = np.array([0, 0, UvbEB[0], UvbEB[1], UvbEB[2], UvbEB[3]])

#viga biapoiada com 1000kN no meio, E = 20000 kN/cm2, nu=0.3, b=2cm, h=10cm e L = 6m ---------------------
b = 2
h = 10
F = 1000

Kvba = Ke[1:5,1:5]
Kvba = np.array(Kvba.subs({L:6, I:b*h**3/12, E:20000})).astype(np.float64)
Fvba = np.array([0, -F, 0, 0])

UvbaEB = np.linalg.solve(Kvba,Fvba)
UgvbaEB = np.array([0, UvbEB[0], UvbEB[1], UvbEB[2], UvbEB[3], 0])


# 160 elementos do site ------------------------------------------------------------------------------------------------ 160!!!
t_w = 3
h = 139
b_f = 50
t_f = 3
I_z = (t_w*h**3)/(12) + 2 * (b_f * t_f**3)/(12) + 2 * b_f * t_f * ( t_f/2 + h/2 )**2
Ke2 = np.array(Ke.subs({L:200, I:I_z, E:20000})).astype(np.float64)
Ke7 = np.array(Ke.subs({L:700, I:I_z, E:20000})).astype(np.float64)

#calculo do vetor de forças nodais equivalentes
g = sp.Symbol('g')
q = sp.Symbol('q')

Feg = -g * sp.integrate( Nn, (x, x1, x3) )
Fegq = -(g+q) * sp.integrate( Nn, (x, x1, x3) )

Fe1 = np.array(Feg.subs({L:200, g:0.528})).astype(np.float64)
Fe2 = np.array(Fegq.subs({L:700, g:0.528, q:2.11})).astype(np.float64)
Fe3 = np.array(Feg.subs({L:700, g:0.528})).astype(np.float64)

#correspondencia
ID1 = np.array([12, 0, 1, 2, 3, 4])
ID2 = np.array([3, 4, 5, 6, 7, 8])
ID3 = np.array([7, 8, 9, 10, 13, 11])

#matriz de rigidez global
K = np.zeros((14,14))

for i in range(0, 6):
    for j in range(0,6):
        K[ ID1[i], ID1[j] ] += Ke2[i,j]
        K[ ID2[i], ID2[j] ] += Ke7[i,j]
        K[ ID3[i], ID3[j] ] += Ke7[i,j]

#vetor de forças global
F = np.zeros(14)

for i in range(0, 6):
    F[ ID1[i] ] += Fe1[i]
    F[ ID2[i] ] += Fe2[i]
    F[ ID3[i] ] += Fe3[i]

Ku = K[:-2, :-2]
Fu = F[:-2]

Kr = K[-2:, :-2]

#usando o numpy
U_np = np.linalg.solve(Ku, Fu)

U = np.zeros(14)
U[:-2] = U_np

#cálculo das reações de apoio
Frapo = np.zeros(2)
Frapo = F[-2:]
Rapo = np.dot(Kr, U_np) - Frapo

#deformações no sistema local
u1 = np.zeros(6)
u1 = U[ ID1 ]

u2 = np.zeros(6)
u2 = U[ ID2 ]

u3 = np.zeros(6)
u3 = U[ ID3 ]


#deslocamentos no elemento ---------------------------------------------------------------------------------
N_EL = sp.Matrix([[N1],
                  [N2],
                  [N3],
                  [N4],
                  [N5],
                  [N6]])
#para cálculo das rotações
dN_ES = sp.Matrix([[dN1],
                  [dN2],
                  [dN3],
                  [dN4],
                  [dN5],
                  [dN6]])
#para o cálculo do momento
dN_M = sp.Matrix([[ddN1],
                  [ddN2],
                  [ddN3],
                  [ddN4],
                  [ddN5],
                  [ddN6]])
#para o cálculo do cortante
dN_C = sp.Matrix([[dddN1],
                  [dddN2],
                  [dddN3],
                  [dddN4],
                  [dddN5],
                  [dddN6]])

##vetor de deformações genérico
#ug1, ug2, ug3, ug4, ug5, ug6 = sp.symbols('ug1 ug2 ug3 ug4 ug5 ug6')
#Ug = sp.Matrix([ug1, ug2, ug3, ug4, ug5, ug6])

#Ug = sp.Matrix(UgvbEB)
#Ug = sp.Matrix(UgvbaEB)
Ug1 = sp.Matrix(u1)
Ug2 = sp.Matrix(u2)
Ug3 = sp.Matrix(u3)


# analítico --------------------------------------------------------------------------------------------------------

Ra = 2725/8*g + 3675/8*q
Rb = 4475/8*g + 1925/8*q

Ms1 = Ra*x - g*x**2/2
Ms2 = Ra*(200 + x) - g*200*(100 + x) - q*x**2/2
Ms3 = Rb*x - g*x**2/2

Vs1 = sp.diff(Ms1, x)
Vs2 = sp.diff(Ms2, x)
Vs3 = -sp.diff(Ms3, x)

# para viga em balanço com 1 elemento --------------------------------------------------------------------------

#deslocamentos = (N_EL.transpose() * Ug)[0]
#rotacoes = (dN_ES.transpose() * Ug)[0]
#momento = - (20000 * b*h**3/12) * (dN_M.transpose() * Ug)[0]
#cortante = (20000 * b*h**3/12) * (dN_C.transpose() * Ug)[0]
#
#
#deslocamentos_f = sp.utilities.lambdify([x, L], deslocamentos, "numpy")
#rotacoes_f = sp.utilities.lambdify([x, L], rotacoes, "numpy")
#momento_f = sp.utilities.lambdify([x, L], momento, "numpy")
#cortante_f = sp.utilities.lambdify([x, L], cortante, "numpy")
#
#L = 6.
#x = np.linspace(-L/2, L/2, 100)
#
#plt.plot(x, deslocamentos_f(x, L), label="deslocamentos")
#plt.plot(x, np.zeros(100), label="zero")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#
#plt.plot(x, rotacoes_f(x, L), label="rotacoes")
#plt.plot(x, np.zeros(100), label="zero")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#
#plt.plot(x, momento_f(x, L), label="momento")
#plt.plot(x, np.zeros(100), label="zero")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#
#plt.plot(x, cortante_f(x, L), label="cortante")
#plt.plot(x, np.zeros(100), label="zero")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()

# para viga do trem do material com 3 elementos --------------------------------------------------------------------

deslocamentos1 = (N_EL.transpose() * Ug1)[0]
deslocamentos2 = (N_EL.transpose() * Ug2)[0]
deslocamentos3 = (N_EL.transpose() * Ug3)[0]
rotacoes1 = (dN_ES.transpose() * Ug1)[0]
rotacoes2 = (dN_ES.transpose() * Ug2)[0]
rotacoes3 = (dN_ES.transpose() * Ug3)[0]
momento1 = - (20000 * I_z) * (dN_M.transpose() * Ug1)[0]
momento2 = - (20000 * I_z) * (dN_M.transpose() * Ug2)[0]
momento3 = - (20000 * I_z) * (dN_M.transpose() * Ug3)[0]
cortante1 = -(20000 * I_z) * (dN_C.transpose() * Ug1)[0]
cortante2 = -(20000 * I_z) * (dN_C.transpose() * Ug2)[0]
cortante3 = -(20000 * I_z) * (dN_C.transpose() * Ug3)[0]

Ms1f = sp.utilities.lambdify([x, g, q], -Ms1, "numpy")
Ms2f = sp.utilities.lambdify([x, g, q], -Ms2, "numpy")
Ms3f = sp.utilities.lambdify([x, g, q], -Ms3, "numpy")

Vs1f = sp.utilities.lambdify([x, g, q], Vs1, "numpy")
Vs2f = sp.utilities.lambdify([x, g, q], Vs2, "numpy")
Vs3f = sp.utilities.lambdify([x, g, q], Vs3, "numpy")

deslocamentos_f1 = sp.utilities.lambdify([x, L], deslocamentos1, "numpy")
deslocamentos_f2 = sp.utilities.lambdify([x, L], deslocamentos2, "numpy")
deslocamentos_f3 = sp.utilities.lambdify([x, L], deslocamentos3, "numpy")
rotacoes_f1 = sp.utilities.lambdify([x, L], rotacoes1, "numpy")
rotacoes_f2 = sp.utilities.lambdify([x, L], rotacoes2, "numpy")
rotacoes_f3 = sp.utilities.lambdify([x, L], rotacoes3, "numpy")
momento_f1 = sp.utilities.lambdify([x, L], momento1, "numpy")
momento_f2 = sp.utilities.lambdify([x, L], momento2, "numpy")
momento_f3 = sp.utilities.lambdify([x, L], momento3, "numpy")
cortante_f1 = sp.utilities.lambdify([x, L], cortante1, "numpy")
cortante_f2 = sp.utilities.lambdify([x, L], cortante2, "numpy")
cortante_f3 = sp.utilities.lambdify([x, L], cortante3, "numpy")

x200 = np.linspace(-200/2, 200/2, 100)
x700 = np.linspace(-700/2, 700/2, 100)
x1 = np.linspace(0, 200, 100)
x2 = np.linspace(200, 900, 100)
x3 = np.linspace(900, 1600, 100)
x_20 = np.linspace(0, 700, 100)
x_3i = np.linspace(700, 0, 100)
x = np.linspace(0, 1600, 300)

#plt.plot(x1, deslocamentos_f1(x200, 200), label="deslocamentos")
#plt.plot(x2, deslocamentos_f2(x700, 700))
#plt.plot(x3, deslocamentos_f3(x700, 700))
#plt.plot(x, np.zeros(300), label="zero", color="black")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#
#plt.plot(x1, rotacoes_f1(x200, 200), label="rotacoes")
#plt.plot(x2, rotacoes_f2(x700, 700))
#plt.plot(x3, rotacoes_f3(x700, 700))
#plt.plot(x, np.zeros(300), label="zero", color="black")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()

plt.plot(x1, momento_f1(x200, 200), label="momento")
plt.plot(x2, momento_f2(x700, 700))
plt.plot(x3, momento_f3(x700, 700))
plt.plot(x1, Ms1f(x1, 0.528, 0.528+2.11), "--", color="red", label="Momento analítico")
plt.plot(x2, Ms2f(x_20, 0.528, 0.528+2.11), "--", color="red")
plt.plot(x3, Ms3f(x_3i, 0.528, 0.528+2.11), "--", color="red")
plt.plot(x, np.zeros(300), label="zero", color="black")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

plt.plot(x1, cortante_f1(x200, 200), label="cortante")
plt.plot(x2, cortante_f2(x700, 700))
plt.plot(x3, cortante_f3(x700, 700))
plt.plot(x1, -Vs1f(x1, 0.528, 0.528+2.11), "--", color="red", label="Cortante analítico")
plt.plot(x2, -Vs2f(x_20, 0.528, 0.528+2.11), "--", color="red")
plt.plot(x3, -Vs3f(x_3i, 0.528, 0.528+2.11), "--", color="red")
plt.plot(x, np.zeros(300), label="zero", color="black")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()