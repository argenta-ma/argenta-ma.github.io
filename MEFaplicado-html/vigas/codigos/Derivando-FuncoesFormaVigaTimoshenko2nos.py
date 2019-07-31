#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Funções de forma para a viga de 2 nós de Timoshenko?????

Usando as mesmas intepolações de Euler-Bernouilli com 2 nós mas separando as funções de forma
para deflexões e rotações

ERRADO!!! OU NÃO FUNCIONA... SEI LÁ...

@author: markinho
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


#para viga
L = sp.Symbol('L')
x1 = -L/2
x2 = L/2
u1 = sp.Symbol('u1')
u2 = sp.Symbol('u2')
u3 = sp.Symbol('u3')
u4 = sp.Symbol('u4')

Mat_Coef = sp.Matrix([[1, x1, x1**2, x1**3],
                      [0, 1, 2*x1, 3*x1**2],
                      [1, x2, x2**2, x2**3],
                      [0, 1, 2*x2, 3*x2**2]])

U = sp.Matrix([u1, u2, u3, u4])

Coefs = Mat_Coef.inv() * U

A = Coefs[0]
B = Coefs[1]
C = Coefs[2]
D = Coefs[3]


x = sp.Symbol('x')

Ns = sp.expand(A + B*x + C*x**2 + D*x**3)

N1 = sp.Add(*[argi for argi in Ns.args if argi.has(u1)]).subs(u1, 1)
N2 = sp.Add(*[argi for argi in Ns.args if argi.has(u2)]).subs(u2, 1)
N3 = sp.Add(*[argi for argi in Ns.args if argi.has(u3)]).subs(u3, 1)
N4 = sp.Add(*[argi for argi in Ns.args if argi.has(u4)]).subs(u4, 1)


##geração dos gráficos --------------------------------------------------------------
##convertendo para função python
#nN1 = sp.utilities.lambdify([x, L], N1, "numpy")
#nN2 = sp.utilities.lambdify([x, L], N2, "numpy")
#
#nN3 = sp.utilities.lambdify([x, L], N3, "numpy")
#nN4 = sp.utilities.lambdify([x, L], N4, "numpy")
#
#L = 1.
#x = np.linspace(-L/2., L/2, 100)
#
#plt.plot(x, nN1(x, L), label="N1")
#plt.plot(x, nN3(x, L), label="N3")
#plt.title('Deslocamentos')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#
#plt.plot(x, nN2(x, L), label="N2")
#plt.plot(x, nN4(x, L), label="N4")
#plt.title('Rotações')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()

#primeira derivada
dN1 = sp.diff(N1, x)
dN2 = sp.diff(N2, x)
dN3 = sp.diff(N3, x)
dN4 = sp.diff(N4, x)

#segunda derivada
ddN1 = sp.diff(dN1, x)
ddN2 = sp.diff(dN2, x)
ddN3 = sp.diff(dN3, x)
ddN4 = sp.diff(dN4, x)

#terceira derivada
dddN1 = sp.diff(ddN1, x)
dddN2 = sp.diff(ddN2, x)
dddN3 = sp.diff(ddN3, x)
dddN4 = sp.diff(ddN4, x)

#quarta derivada
ddddN1 = sp.diff(dddN1, x)
ddddN2 = sp.diff(dddN2, x)
ddddN3 = sp.diff(dddN3, x)
ddddN4 = sp.diff(dddN4, x)

#cálculo da matriz de rigidez
Bf = sp.Matrix([0, dN2, 0, dN4])
BBf = Bf * Bf.T

E = sp.Symbol('E')
G = sp.Symbol('G')
I = sp.Symbol('I')

Kef = E*I*sp.integrate( BBf, (x, x1, x2) )

A = sp.Symbol('A')

Bc = sp.Matrix([dN1, -N2, dN3, -N4])
BBc = Bc * Bc.T

Kec = G*A*sp.integrate( BBc, (x, x1, x2) )

Ke = Kef + Kec

#viga em balanço com 1000kN na extremidade, E = 20000 kN/cm2, nu=0.3, b=2cm, h=10cm e L = 6m ---------------------
b = 2
h = 10
F = 1000

Kvb = Ke[2:,2:]
Kvb = np.array(Kvb.subs({A:b*h, G:20000/(2*(1+0.3)), L:6, I:b*h**3/12, E:20000})).astype(np.float64)
Fvb = np.array([-F, 0])

UvbT = np.linalg.solve(Kvb,Fvb)
UgvbT = np.array([0, 0, UvbT[0], UvbT[1]])


###!!!!!!!!!!!!!!!!!!!! ERRADO DAQUI PARA BAIXO!!!!!
#deslocamentos no elemento ---------------------------------------------------------------------------------
N_EL = sp.Matrix([[N1],
                  [0],
                  [N3],
                  [0]])
#para cálculo das rotações
dN_ES = sp.Matrix([[0],
                  [N2],
                  [0],
                  [N4]])
##para o cálculo do momento
#dN_M = sp.Matrix([[0],
#                  [dN2],
#                  [0],
#                  [dN4]])
##para o cálculo do cortante
#dN_C = sp.Matrix([[0],
#                  [ddN2],
#                  [0],
#                  [ddN4]])

##vetor de deformações genérico
#ug1, ug2, ug3, ug4, ug5, ug6 = sp.symbols('ug1 ug2 ug3 ug4 ug5 ug6')
#Ug = sp.Matrix([ug1, ug2, ug3, ug4, ug5, ug6])

Ug = sp.Matrix(UgvbT)
#Ug = sp.Matrix(UgvbaT)

deslocamentos = (N_EL.transpose() * Ug)[0]
rotacoes = (dN_ES.transpose() * Ug)[0]
#momento = - (20000 * b*h**3/12) * (dN_M.transpose() * Ug)[0]
#cortante = (20000 * b*h**3/12) * (dN_C.transpose() * Ug)[0]


deslocamentos_f = sp.utilities.lambdify([x, L], deslocamentos, "numpy")
rotacoes_f = sp.utilities.lambdify([x, L], rotacoes, "numpy")
#momento_f = sp.utilities.lambdify([x, L], momento, "numpy")
#cortante_f = sp.utilities.lambdify([x, L], cortante, "numpy")

L = 6.
x = np.linspace(-L/2, L/2, 100)

plt.plot(x, deslocamentos_f(x, L), label="deslocamentos")
plt.plot(x, np.zeros(100), label="zero")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

plt.plot(x, rotacoes_f(x, L), label="rotacoes")
plt.plot(x, np.zeros(100), label="zero")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

#plt.plot(x, momento_f(x, L), label="momento")
#plt.plot(x, np.zeros(100), label="zero")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#
#plt.plot(x, cortante_f(x, L), label="cortante")
#plt.plot(x, np.zeros(100), label="zero")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()