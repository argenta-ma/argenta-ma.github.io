#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Funções de forma para a viga de 2 nós de Euler-Bernouilli

Completo!

@author: markinho
"""

import sympy as sp
import numpy as np
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt


#para viga
L = sp.Symbol('L')
x1 = -L/2
x2 = L/2
u1 = sp.Symbol('u1')
u2 = sp.Symbol('u2')
u3 = sp.Symbol('u3')
u4 = sp.Symbol('u4')

#Mat_Coef = sp.Matrix([[1, -L/2, L**2/4, -L**3/8, L**4/16, -L**5/32],
#                      [0, 1, -L, 3*L**2/4, -L**3/2, 5*L**4/16],
#                      [1, 0, 0, 0, 0, 0],
#                      [0, 1, 0, 0, 0, 0],
#                      [1, L/2, L**2/4, L**3/8, L**4/16, L**5/32],
#                      [0, 1, L, 3*L**2/4, L**3/2, 5*L**4/16]])

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

##geração do grafico ---------------------------------------------------------------------
#convertendo para função python
nN1 = sp.utilities.lambdify([x, L], N1, "numpy")
nN2 = sp.utilities.lambdify([x, L], N2, "numpy")

nN3 = sp.utilities.lambdify([x, L], N3, "numpy")
nN4 = sp.utilities.lambdify([x, L], N4, "numpy")

L = 2.
x = np.linspace(-L/2., L/2, 100)

#deslocamentos
plt.figure(figsize=(10, 4), dpi=100)
#plt.figure(figsize=(6, 2), dpi=100)

plt.plot(x, nN1(x, L), label="N1")
plt.plot(x, nN3(x, L), label="N3")

#plt.title(u'Funções de forma para o elemento de treliça de 2 nós', **fonteFigura)
ax = plt.gca()  # gca significa 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.set_xlim(right=1.3)
ax.set_ylim(top=1.1)

plt.xticks([-1, 0, 1], ha='left', fontsize=12)
plt.yticks([1], fontsize=12)

plt.plot([-1, -1], [0, 1], color='blue', linewidth=1., linestyle="--")
plt.scatter([-1], [0.98], 30, marker='^', color='blue')
plt.annotate(r'$u_a$', xy=(-0.97, 0.5), color='blue', xycoords='data', fontsize=16)

plt.plot([1, 1], [0, 1], color='orange', linewidth=1., linestyle="--")
plt.scatter([1], [0.98], 30, marker='^', color='orange')
plt.annotate(r'$u_c$', xy=(0.9, 0.5), color='orange', xycoords='data', fontsize=16)

plt.scatter([1.285], [0], 50, marker='>', color='black')
plt.annotate(r'$r$', xy=(1.23, 0.03), color='black', xycoords='data', fontsize=18)

plt.scatter([0], [1.083], 50, marker='^', color='black')
plt.annotate(r'$u(r)$', xy=(0.03, 1.05), color='black', xycoords='data', fontsize=18)

#plt.title('Deslocamentos')
plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
plt.show()

#rotacoes
plt.figure(figsize=(10, 4), dpi=100)
#plt.figure(figsize=(6, 2), dpi=100)

plt.plot(x, nN2(x, L), label="N2")
plt.plot(x, nN4(x, L), label="N4")

#plt.title(u'Funções de forma para o elemento de treliça de 2 nós', **fonteFigura)
ax = plt.gca()  # gca significa 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.set_xlim(right=1.3)
ax.set_ylim(top=0.4, bottom=-0.3)

plt.xticks([-1, 0, 1], ha='left', fontsize=12)
plt.yticks([1.0], fontsize=12)

ax.annotate("",xy=(-0.85, 0.13), xycoords='data', xytext=(-0.75, 0.), textcoords='data',
            arrowprops=dict(width=0.5, headwidth=8, headlength=8, color="gray", connectionstyle="arc3,rad=.2"))
plt.annotate(r'$u_b$', xy=(-0.75, 0.07), color='blue', xycoords='data', fontsize=16)

ax.annotate("",xy=(0.85, -0.13), xycoords='data', xytext=(0.75, 0.), textcoords='data',
            arrowprops=dict(width=0.5, headwidth=8, headlength=8, color="gray", connectionstyle="arc3,rad=.2"))
plt.annotate(r'$u_d$', xy=(0.65, -0.1), color='orange', xycoords='data', fontsize=16)

plt.scatter([1.285], [0], 50, marker='>', color='black')
plt.annotate(r'$r$', xy=(1.23, 0.03), color='black', xycoords='data', fontsize=18)

plt.scatter([0], [0.387], 50, marker='^', color='black')
plt.annotate(r'$\theta(r)$', xy=(0.03, 0.33), color='black', xycoords='data', fontsize=18)

#plt.title('Rotações')
plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)
plt.show()

##primeira derivada
#dN1 = sp.diff(N1, x)
#dN2 = sp.diff(N2, x)
#dN3 = sp.diff(N3, x)
#dN4 = sp.diff(N4, x)
#
##segunda derivada
#ddN1 = sp.diff(dN1, x)
#ddN2 = sp.diff(dN2, x)
#ddN3 = sp.diff(dN3, x)
#ddN4 = sp.diff(dN4, x)
#
###terceira derivada
##dddN1 = sp.diff(ddN1, x)
##dddN2 = sp.diff(ddN2, x)
##dddN3 = sp.diff(ddN3, x)
##dddN4 = sp.diff(ddN4, x)
##
###quarta derivada
##ddddN1 = sp.diff(dddN1, x)
##ddddN2 = sp.diff(dddN2, x)
##ddddN3 = sp.diff(dddN3, x)
##ddddN4 = sp.diff(dddN4, x)
#
##cálculo da matriz de rigidez
#B = sp.Matrix([ddN1, ddN2, ddN3, ddN4])
#BB = B * B.T
#
#E = sp.Symbol('E')
#I = sp.Symbol('I')
#
#Ke = E*I*sp.integrate( BB, (x, x1, x2) )
#
##calculo do vetor de forças nodais equivalentes
#g = sp.Symbol('g')
#q = sp.Symbol('q')
#
#Nn = sp.Matrix([N1, N2, N3, N4])
#
#Feg = -g * sp.integrate( Nn, (x, x1, x2) )
#Fegq = -(g+q) * sp.integrate( Nn, (x, x1, x2) )
#
##viga biapoiada com carga distribuída e dividida em 3 elementos ---------------------
#t_w = 3
#h = 139
#b_f = 50
#t_f = 3
#I_z = (t_w*h**3)/(12) + 2 * (b_f * t_f**3)/(12) + 2 * b_f * t_f * ( t_f/2 + h/2 )**2
#Ke2 = np.array(Ke.subs({L:200, I:I_z, E:20000})).astype(np.float64)
#Ke7 = np.array(Ke.subs({L:700, I:I_z, E:20000})).astype(np.float64)
#
#Fe1 = np.array(Feg.subs({L:200, g:0.528})).astype(np.float64)
#Fe2 = np.array(Fegq.subs({L:700, g:0.528, q:2.11})).astype(np.float64)
#Fe3 = np.array(Feg.subs({L:700, g:0.528})).astype(np.float64)
#
##correspondencia
#ID1 = np.array([6, 0, 1, 2])
#ID2 = np.array([1, 2, 3, 4])
#ID3 = np.array([3, 4, 7, 5])
#
##matriz de rigidez global
#K = np.zeros((8,8))
#
#for i in range(0, 4):
#    for j in range(0,4):
#        K[ ID1[i], ID1[j] ] += Ke2[i,j]
#        K[ ID2[i], ID2[j] ] += Ke7[i,j]
#        K[ ID3[i], ID3[j] ] += Ke7[i,j]
#
##vetor de forças global
#F = np.zeros(8)
#
#for i in range(0, 4):
#    F[ ID1[i] ] += Fe1[i]
#    F[ ID2[i] ] += Fe2[i]
#    F[ ID3[i] ] += Fe3[i]
#
#Ku = K[:-2, :-2]
#Fu = F[:-2]
#
#Kr = K[-2:, :-2]
#
##usando o numpy
#U_np = np.linalg.solve(Ku, Fu)
#
#
#Kch = Ku.copy()
#Uch = Fu.copy()
#
#def choleski(a):
#    '''
#    Choleski decomposition: [L][L]transpose = [a]
#    '''
#    n = len(a)
#    for k in range(n):
#        try:
#            a[k,k] = np.sqrt(a[k,k] - np.dot(a[k,0:k],a[k,0:k]))
#        except ValueError:
#            print('Matrix is not positive definite')
#        for i in range(k+1,n):
#            a[i,k] = (a[i,k] - np.dot(a[i,0:k],a[k,0:k]))/a[k,k]
#    for k in range(1,n): a[0:k,k] = 0.0
#    return a
#
#
#def choleskiSol(L,b):
#    '''
#    Solution phase of Choleski's decomposition method
#    '''
#    n = len(b)
#  # Solution of [L]{y} = {b}  
#    for k in range(n):
#        b[k] = ( b[k] - np.dot(L[k,0:k], b[0:k]) )/L[k,k]
##    print(b)
#  # Solution of [L_transpose]{x} = {y}      
#    for k in range(n-1,-1,-1):
#        b[k] = (b[k] - np.dot(L[k+1:n,k],b[k+1:n]))/L[k,k]
#    return b
#
#choleski(Kch)
#choleskiSol(Kch, Uch)
#
#U = np.zeros(8)
#U[:-2] = Uch
#
##cálculo das reações de apoio
#Frapo = np.zeros(2)
#Frapo = F[6:]
#Rapo = np.dot(Kr, Uch) - Frapo
#
##deformações no sistema local
#u1 = np.zeros(4)
#u1 = U[ ID1 ]
#
#u2 = np.zeros(4)
#u2 = U[ ID2 ]
#
#u3 = np.zeros(4)
#u3 = U[ ID3 ]
#
##deformações dos elementos --------------------------------------------------------------------------------------------
#
#s = sp.Symbol('s')
#
#epsilon1 = -s * (B.T.subs({L:200}) * u1[:, np.newaxis])[0]
#epsilon2 = -s * (B.T.subs({L:700}) * u2[:, np.newaxis])[0]
#epsilon3 = -s * (B.T.subs({L:700}) * u3[:, np.newaxis])[0]
#
##deformações nos nós da estrutura (sup: face superior, inf: face inferior da viga)
#
#epsilon_n1_sup = epsilon1.subs({s: h/2+t_f, x: -100})
#epsilon_n1_inf = epsilon1.subs({s: -h/2-t_f, x: -100})
#
#epsilon_n2_sup = ( epsilon1.subs({s: h/2+t_f, x: 100}) + epsilon2.subs({s: h/2+t_f, x: -350}) )/2
#epsilon_n2_inf = ( epsilon1.subs({s: -h/2-t_f, x: 100}) + epsilon2.subs({s: -h/2-t_f, x: -350}) )/2
#
#epsilon_n3_sup = ( epsilon2.subs({s: h/2+t_f, x: 350}) + epsilon3.subs({s: h/2+t_f, x: -350}) )/2
#epsilon_n3_inf = ( epsilon2.subs({s: -h/2-t_f, x: 350}) + epsilon3.subs({s: -h/2-t_f, x: -350}) )/2
#
#epsilon_n4_sup = epsilon3.subs({s: h/2+t_f, x: 350})
#epsilon_n4_inf = epsilon3.subs({s: -h/2-t_f, x: 350})
#
##tensões nos elementos -------------------------------------------------------------------------------------------------
#
#sigma1 = 20000 * epsilon1
#sigma2 = 20000 * epsilon2
#sigma3 = 20000 * epsilon3
#
##tensões nos nós da estrutura (sup: face superior, inf: face inferior da viga)
#
#sigma_n1_sup = epsilon_n1_sup * 20000
#sigma_n1_inf = epsilon_n1_inf * 20000
#
#sigma_n2_sup = epsilon_n2_sup * 20000
#sigma_n2_inf = epsilon_n2_inf * 20000
#
#sigma_n3_sup = epsilon_n3_sup * 20000
#sigma_n3_inf = epsilon_n3_inf * 20000
#
#sigma_n4_sup = epsilon_n4_sup * 20000
#sigma_n4_inf = epsilon_n4_inf * 20000
#
##momento fletor -----------------------------------------------------------------------------------------------------
#
#M1 = t_w * sp.integrate( s * sigma1, (s, -h/2, h/2 ) ) + 2 * b_f * sp.integrate( s * sigma1, (s, h/2, h/2 + t_f ) )
#M2 = t_w * sp.integrate( s * sigma2, (s, -h/2, h/2 ) ) + 2 * b_f * sp.integrate( s * sigma2, (s, h/2, h/2 + t_f ) )
#M3 = t_w * sp.integrate( s * sigma3, (s, -h/2, h/2 ) ) + 2 * b_f * sp.integrate( s * sigma3, (s, h/2, h/2 + t_f ) )
#
##momentos fletores nos nós
#
#M_n1 = M1.subs({x: -100})
#M_n2 = ( M1.subs({x: 100}) + M2.subs({x: -350}) )/2
#M_n3 = ( M2.subs({x: 350}) + M3.subs({x: -350}) )/2
#M_n4 = M3.subs({x: 350})
#
##esforço cortante ---------------------------------------------------------------------------------------------------
#
#V1 = sp.diff(M1, x)
#V2 = sp.diff(M2, x)
#V3 = sp.diff(M3, x)
#
##esforço cortantes nos nós
#
#V_n1 = V1
#V_n2 = ( V1 + V2 )/2
#V_n3 = ( V2 + V3 )/2
#V_n4 = V3
#
## analítico --------------------------------------------------------------------------------------------------------
#
#Ra = 2725/8*g + 3675/8*q
#Rb = 4475/8*g + 1925/8*q
#
#Ms1 = Ra*x - g*x**2/2
#Ms2 = Ra*(200 + x) - g*200*(100 + x) - q*x**2/2
#Ms3 = Rb*x - g*x**2/2
#
#Vs1 = sp.diff(Ms1, x)
#Vs2 = sp.diff(Ms2, x)
#Vs3 = -sp.diff(Ms3, x)
#
## 160 elementos ------------------------------------------------------------------------------------------------ 160!!!
#nelems = 160
#nnos = nelems + 1
#GLs = nnos * 2
#Ke10 = np.array(Ke.subs({L:10, I:I_z, E:20000})).astype(np.float64)
#Feg = np.array(Feg.subs({L:10, g:0.528})).astype(np.float64)
#Feq = np.array(Fegq.subs({L:10, g:0.528, q:2.11})).astype(np.float64)
#
##correspondencia
#ID = np.array([GLs-2] + list(np.arange(0, GLs-3)) + [GLs-1, GLs-3])
#IDi = ID.reshape(nnos, 2)[:-1,:]
#IDj = ID.reshape(nnos, 2)[1:,:]
#ID = np.concatenate((IDi, IDj), axis=1)
#
##matriz de rigidez global
#K160 = np.zeros((GLs,GLs))
#for k in range(0, nelems):
#    for i in range(0, 4):
#        for j in range(0,4):
#            K160[ ID[k, i], ID[k, j] ] += Ke10[i,j]
#
##vetor de forças global
#F160 = np.zeros(GLs)
#
#for k in range(0, nelems):
#    for i in range(0, 4):
#        if 20 < k <= 90:
#            F160[ ID[k, i] ] += Feq[i]
#        else:
#            F160[ ID[k, i] ] += Feg[i]
#
#Ku160 = K160[:-2, :-2]
#Fu160 = F160[:-2]
#
#Kr160 = K160[-2:, :-2]
#
##usando o numpy
#U160_np = np.linalg.solve(Ku160, Fu160)
#
#U160 = np.zeros(GLs)
#U160[:-2] = U160_np
#
##cálculo das reações de apoio
#Frapo160 = np.zeros(2)
#Frapo160 = F160[-2:]
#Rapo160 = np.dot(Kr160, U160_np) - Frapo160
#
##deformações no sistema local
#u160 = np.zeros((nelems, 4))
#for k in range(0, nelems):
#    for i in range(0, 4):
#        u160[k, i] = U160[ ID[k, i] ]
#
#epsilon160 = sp.zeros(nelems, 1)
#for k in range(0, nelems):
#    epsilon160[k] = -s * (B.T.subs({L:10}) * u160[k][:, np.newaxis])[0]
#
#sigma160 = 20000 * epsilon160
#
#M160 = t_w * sp.integrate( s * sigma160, (s, -h/2, h/2 ) ) + 2 * b_f * sp.integrate( s * sigma160, (s, h/2, h/2 + t_f ) )
#V160 = sp.diff(M160, x)
#
##deslocamentos
##plt.plot(np.linspace(0, 1600, nelems), u160[:,[0,2]][:, 0])
#
## graficos dos momentos e cortantes -------------------------------------------------------------------------------
#
#x1 = np.linspace(-200/2, 200/2, 100)
#x23 = np.linspace(-700/2, 700/2, 100)
#
#x_1 = np.linspace(0, 200, 100)
#x_2 = np.linspace(200, 900, 100)
#x_20 = np.linspace(0, 700, 100)
#x_3 = np.linspace(900, 1600, 100)
#x_3i = np.linspace(700, 0, 100)
#
#xe160 = np.linspace(-5, 5, 10)
#
#
#M1f = sp.utilities.lambdify([x], M1, "numpy")
#M2f = sp.utilities.lambdify([x], M2, "numpy")
#M3f = sp.utilities.lambdify([x], M3, "numpy")
#
#Ms1f = sp.utilities.lambdify([x, g, q], -Ms1, "numpy")
#Ms2f = sp.utilities.lambdify([x, g, q], -Ms2, "numpy")
#Ms3f = sp.utilities.lambdify([x, g, q], -Ms3, "numpy")
#
#Ms160 = sp.utilities.lambdify([x, g, q], M160, "numpy")
#
##plt.plot(x_1, M1f(x1), label="Momento elemento 1")
##plt.plot(x_2, M2f(x23), label="Momento elemento 2")
##plt.plot(x_3, M3f(x23), label="Momento elemento 3")
#plt.plot(x_1, Ms1f(x_1, 0.528, 0.528+2.11), "--", color="red", label="Momento analítico")
#plt.plot(x_2, Ms2f(x_20, 0.528, 0.528+2.11), "--", color="red")
#plt.plot(x_3, Ms3f(x_3i, 0.528, 0.528+2.11), "--", color="red")
#for k in range(0, nelems):
#    plt.plot( np.linspace(k*10, k*10 + 10, 10), Ms160(xe160, 0.528, 0.528+2.11)[k][0], color="black")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
#
#V1f = np.ones(100)*V1
#V2f = np.ones(100)*V2
#V3f = np.ones(100)*V3
#
#Vs1f = sp.utilities.lambdify([x, g, q], Vs1, "numpy")
#Vs2f = sp.utilities.lambdify([x, g, q], Vs2, "numpy")
#Vs3f = sp.utilities.lambdify([x, g, q], Vs3, "numpy")
#
#Vs160 = sp.utilities.lambdify([x, g, q], V160, "numpy")
#
##plt.plot(x_1, V1f, label="Cortante elemento 1")
##plt.plot(x_2, V2f, label="Cortante elemento 2")
##plt.plot(x_3, V3f, label="Cortante elemento 3")
#plt.plot(x_1, -Vs1f(x_1, 0.528, 0.528+2.11), "--", color="red", label="Cortante analítico")
#plt.plot(x_2, -Vs2f(x_20, 0.528, 0.528+2.11), "--", color="red")
#plt.plot(x_3, -Vs3f(x_3i, 0.528, 0.528+2.11), "--", color="red")
#for k in range(0, nelems):
#    plt.plot( np.linspace(k*10, k*10 + 10, 10), np.ones(10)*Vs160(xe160, 0.528, 0.528+2.11)[k][0], color="black")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()