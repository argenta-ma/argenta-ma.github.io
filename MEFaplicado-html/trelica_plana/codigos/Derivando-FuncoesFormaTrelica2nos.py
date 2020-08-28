#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Funções de forma para a viga de 3 nós de Euler-Bernouilli

INCOMPLETO!!!

@author: markinho
"""

import sympy as sp
import numpy as np

from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
#import matplotlib.pyplot as plt


#para viga
L = sp.Symbol('L')
x1 = -L/2
x2 = L/2
u1 = sp.Symbol('u1')
u2 = sp.Symbol('u2')

Mat_Coef = sp.Matrix([[1, x1],
                      [1, x2]])

U = sp.Matrix([u1, u2])

Coefs = Mat_Coef.inv() * U

A = Coefs[0]
B = Coefs[1]

x = sp.Symbol('x')

Ns = sp.expand(A + B*x)

N1 = sp.Add(*[argi for argi in Ns.args if argi.has(u1)]).subs(u1, 1)
N2 = sp.Add(*[argi for argi in Ns.args if argi.has(u2)]).subs(u2, 1)

###geração do grafico ---------------------------------------------------------------------
##convertendo para função python
#nN1 = sp.utilities.lambdify([x, L], N1, "numpy")
#nN2 = sp.utilities.lambdify([x, L], N2, "numpy")
#
#L = 2.
#x = np.linspace(-L/2., L/2, 100)
#
#plt.figure(figsize=(10, 4), dpi=100)
##plt.figure(figsize=(6, 2), dpi=100)
#
#plt.plot(x, nN1(x, L), label="N1", color='blue', linewidth=2.0)
#plt.plot(x, nN2(x, L), label="N2", color='orange', linewidth=2.0)
#
##plt.title(u'Funções de forma para o elemento de treliça de 2 nós', **fonteFigura)
#ax = plt.gca()  # gca significa 'get current axis'
#ax.spines['right'].set_color('none')
#ax.spines['top'].set_color('none')
#ax.xaxis.set_ticks_position('bottom')
#ax.spines['bottom'].set_position(('data',0))
#ax.yaxis.set_ticks_position('left')
#ax.spines['left'].set_position(('data',0))
#ax.set_xlim(right=1.3)
#ax.set_ylim(top=1.1)
#
#plt.xticks([-1, 0, 1], ha='left', fontsize=12)
#plt.yticks([1], fontsize=12)
#
#
#plt.plot([-1, -1], [0, 1], color='gray', linewidth=1., linestyle="-")
#plt.scatter([-1], [0.98], 30, marker='^', color='gray')
#plt.annotate(r'$u_a$', xy=(-0.97, 0.5), color='blue', xycoords='data', fontsize=16)
#
#plt.plot([1, 1], [0, 1], color='gray', linewidth=1., linestyle="-")
#plt.scatter([1], [0.98], 30, marker='^', color='gray')
#plt.annotate(r'$u_b$', xy=(0.9, 0.5), color='orange', xycoords='data', fontsize=16)
#
#plt.scatter([1.285], [0], 50, marker='>', color='black')
#plt.annotate(r'$r$', xy=(1.23, 0.03), color='black', xycoords='data', fontsize=18)
#
#plt.scatter([0], [1.083], 50, marker='^', color='black')
#plt.annotate(r'$u(r)$', xy=(0.03, 1.05), color='black', xycoords='data', fontsize=18)
#
#plt.legend(bbox_to_anchor=(0.9, 0.9), loc=2, borderaxespad=0.)
#plt.show()

#primeira derivada
dN1 = sp.diff(N1, x)
dN2 = sp.diff(N2, x)

#cálculo da matriz de rigidez
B = sp.Matrix([dN1, dN2])
BB = B * B.T

E = sp.Symbol('E')
A = sp.Symbol('A')

Ke = E*A*sp.integrate( BB, (x, x1, x2) )

#matriz de decomposição D
c = sp.Symbol('c')
s = sp.Symbol('s')

D = sp.Matrix([[c, 0.], [s, 0.], [0., c], [0., s]])

#matriz de rigidez do elemento global
Kg = D*Ke*D.T

#coordenadas das barras
XY = np.array([[1000., 0.],[0., 1000.],[0., 0.]])

#comprimentos das barras
L1 = np.sqrt( (XY[1,0]-XY[0,0])**2 + (XY[1,1]-XY[0,1])**2 )
L2 = np.sqrt( (XY[1,0]-XY[2,0])**2 + (XY[1,1]-XY[2,1])**2 )
L3 = np.sqrt( (XY[2,0]-XY[0,0])**2 + (XY[2,1]-XY[0,1])**2 )

#cossenos e senos das barras
c1 = (XY[1,0]-XY[0,0])/L1; s1 = (XY[1,1]-XY[0,1])/L1
c2 = (XY[1,0]-XY[2,0])/L2; s2 = (XY[1,1]-XY[2,1])/L2
c3 = (XY[0,0]-XY[2,0])/L3; s3 = (XY[0,1]-XY[2,1])/L3

#area da secao transversal
Ast = np.pi*113**2/4

##barra de treliça (diagonal) E = 200000 N/mm2, d=113mm e L = np.sqrt(2)m ---------------------
Kg_d = np.array(Kg.subs({L:L1, A:Ast, E:200000, c:c1, s:s1})).astype(np.float64)
##barras de treliça (vertical) E = 69000 N/mm2, d=113mm e L = 1m ---------------------
Kg_v = np.array(Kg.subs({L:L2, A:Ast, E:69000, c:c2, s:s2})).astype(np.float64)
##barras de treliça (horizontal) E = 69000 N/mm2, d=113mm e L = 1m ---------------------
Kg_h = np.array(Kg.subs({L:L3, A:Ast, E:69000, c:c3, s:s3})).astype(np.float64)

#correspondencia
ID1 = np.array([0, 1, 3, 2])
ID2 = np.array([4, 5, 3, 2])
ID3 = np.array([4, 5, 0, 1])

#matriz de rigidez global
K = np.zeros((6,6))

for i in range(0, 4):
    for j in range(0,4):
        K[ ID1[i], ID1[j] ] += Kg_d[i,j]
        K[ ID2[i], ID2[j] ] += Kg_v[i,j]
        K[ ID3[i], ID3[j] ] += Kg_h[i,j]

Ku = K[:3, :3]
Kr = K[3:, :3]

F = np.array([0., -10000., 0.])

#usando o numpy
U_np = np.linalg.solve(Ku, F)


Kch = Ku.copy()
Uch = F.copy()

def choleski(a):
    '''
    Choleski decomposition: [L][L]transpose = [a]
    '''
    n = len(a)
    for k in range(n):
        try:
            a[k,k] = np.sqrt(a[k,k] - np.dot(a[k,0:k],a[k,0:k]))
        except ValueError:
            print('Matrix is not positive definite')
        for i in range(k+1,n):
            a[i,k] = (a[i,k] - np.dot(a[i,0:k],a[k,0:k]))/a[k,k]
    for k in range(1,n): a[0:k,k] = 0.0
    return a


def choleskiSol(L,b):
    '''
    Solution phase of Choleski's decomposition method
    '''
    n = len(b)
  # Solution of [L]{y} = {b}  
    for k in range(n):
        b[k] = ( b[k] - np.dot(L[k,0:k], b[0:k]) )/L[k,k]
#    print(b)
  # Solution of [L_transpose]{x} = {y}      
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - np.dot(L[k+1:n,k],b[k+1:n]))/L[k,k]
    return b

choleski(Kch)
choleskiSol(Kch, Uch)

U = np.zeros(6)
U[:3] = Uch

#cálculo das reações de apoio
Rapo = np.dot(Kr, Uch)

#deslocamentos no sistema local
U1 = U[ID1]
U2 = U[ID2]
U3 = U[ID3]

u1 = np.dot(np.transpose(np.array(D.subs({c:c1, s:s1})).astype(np.float64)), U1)
u2 = np.dot(np.transpose(np.array(D.subs({c:c2, s:s2})).astype(np.float64)), U2)
u3 = np.dot(np.transpose(np.array(D.subs({c:c3, s:s3})).astype(np.float64)), U3)

#deformações
epsilon1 = np.dot(np.transpose(np.array(B.subs({L:L1})).astype(np.float64)), u1)[0]
epsilon2 = np.dot(np.transpose(np.array(B.subs({L:L2})).astype(np.float64)), u2)[0]
epsilon3 = np.dot(np.transpose(np.array(B.subs({L:L3})).astype(np.float64)), u3)[0]

#tensoes
sigma1 = epsilon1*200000
sigma2 = epsilon2*69000
sigma3 = epsilon3*69000

#esforços
N1 = sigma1 * Ast
N2 = sigma2 * Ast
N3 = sigma3 * Ast