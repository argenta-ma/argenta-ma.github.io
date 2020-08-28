#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Elementos de pórtico espacial com 6 graus de liberdade por nó.
Treliça desacoplada da viga e torção desacoplada de ambos.

Eixos locais: i, j, k
Eixos globais: X, Y, Z

Parei nas funções de forma!

@author: markinho
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


#para treliça -------------------------------------------------------------------------------------------------------
def funcoes_trelica(var):
    '''
    Função para a derivação das funções de forma de treliça com interpolação linear
    
    -> * ---------- * ->
    
    eixo longitudinal: r
    
    '''
    l = sp.Symbol('l')
    var1 = -l/2
    var2 = l/2
    u1 = sp.Symbol('u1')
    u2 = sp.Symbol('u2')
    
    Mat_Coef = sp.Matrix([[1, var1],
                          [1, var2]])
    
    U = sp.Matrix([u1, u2])
    
    Coefs = Mat_Coef.inv() * U
    
    A = Coefs[0]
    B = Coefs[1]
    
    Ns = sp.expand(A + B*var)
    
    N1 = sp.Add(*[argi for argi in Ns.args if argi.has(u1)]).subs(u1, 1)
    N2 = sp.Add(*[argi for argi in Ns.args if argi.has(u2)]).subs(u2, 1)
    
    dN1 = sp.diff(N1, var)
    dN2 = sp.diff(N2, var)
    
    return sp.Matrix([N1, N2]), sp.Matrix([dN1, dN2])

#para viga no espaço -------------------------------------------------------------------------------------------------
def funcoes_viga(var):
    '''
    Função para a derivação das funções de forma de viga com interpolação cúbica
    '''
    l = sp.Symbol('l')
    var1 = -l/2
    var2 = l/2
    
    u1 = sp.Symbol('u1')
    u2 = sp.Symbol('u2')
    u3 = sp.Symbol('u3')
    u4 = sp.Symbol('u4')
    
    Mat_Coef = sp.Matrix([[1, var1, var1**2, var1**3],
                          [0, 1, 2*var1, 3*var1**2],
                          [1, var2, var2**2, var2**3],
                          [0, 1, 2*var2, 3*var2**2]])
    
    U = sp.Matrix([u1, u2, u3, u4])
    
    Coefs = Mat_Coef.inv() * U
    
    A = Coefs[0]
    B = Coefs[1]
    C = Coefs[2]
    D = Coefs[3]
    
    Ns = sp.expand(A + B*var + C*var**2 + D*var**3)
    
    N1 = sp.Add(*[argi for argi in Ns.args if argi.has(u1)]).subs(u1, 1)
    N2 = sp.Add(*[argi for argi in Ns.args if argi.has(u2)]).subs(u2, 1)
    N3 = sp.Add(*[argi for argi in Ns.args if argi.has(u3)]).subs(u3, 1)
    N4 = sp.Add(*[argi for argi in Ns.args if argi.has(u4)]).subs(u4, 1)
    
    #primeira derivada
    dN1 = sp.diff(N1, var)
    dN2 = sp.diff(N2, var)
    dN3 = sp.diff(N3, var)
    dN4 = sp.diff(N4, var)
    
    #segunda derivada
    ddN1 = sp.diff(dN1, var)
    ddN2 = sp.diff(dN2, var)
    ddN3 = sp.diff(dN3, var)
    ddN4 = sp.diff(dN4, var)
    
    return sp.Matrix([N1, N2, N3, N4]), sp.Matrix([dN1, dN2, dN3, dN4]), sp.Matrix([ddN1, ddN2, ddN3, ddN4])


#Funções de forma para os graus de liberdade de treliça e torção
i = sp.Symbol('i')
Nt, Bt = funcoes_trelica(i)

#funções de forma para os graus de liberdade de viga em ambos os planos ij e ik
Nv, Bv1, Bv2 = funcoes_viga(i)

#propriedades do material e da geometria utlizadas
E = sp.Symbol('E')
G = sp.Symbol('G')
l = sp.Symbol('l')
A = sp.Symbol('A')
Ii = sp.Symbol('Ii') ##???? inércia a torção???
Ij = sp.Symbol('Ij')
Ik = sp.Symbol('Ik')

#matriz de rigidez de treliça e torção
ketr = A * sp.integrate( Bt * E * Bt.T, (i, -l/2, l/2) )
keto = Ii * sp.integrate( Bt * G * Bt.T, (i, -l/2, l/2) ) # ?????? A??

#matriz de rigidez de viga no plano ij
kev_ij = Ik * sp.integrate( Bv2 * E * Bv2.T, (i, -l/2, l/2) )

#matriz de rigidez de viga no plano ik
kev_ik = Ij * sp.integrate( Bv2 * E * Bv2.T, (i, -l/2, l/2) )

#indexação para a montagem da matriz do pórtico 3D
IDtr = [0, 6]
IDto = [3, 9]
IDv_ij = [1, 5, 7, 11]
IDv_ik = [2, 4, 8, 10]

#matriz de rigidez do elemento de pórtico 3D
Ke = sp.zeros(12,12)
for i in range(2):
    for j in range(2):
        Ke[ IDtr[i], IDtr[j] ] += ketr[i, j]
        Ke[ IDto[i], IDto[j] ] += keto[i, j]
for i in range(4):
    for j in range(4):
        Ke[ IDv_ij[i], IDv_ij[j] ] += kev_ij[i, j]
        Ke[ IDv_ik[i], IDv_ik[j] ] += kev_ik[i, j]

#matriz de rotação 3D
#cossenos diretores
lx, ly, lz = sp.symbols('lx, ly, lz')
mx, my, mz = sp.symbols('mx, my, mz')
nx, ny, nz = sp.symbols('nx, ny, nz')
R3 = sp.Matrix([[ lx, mx, nx,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ ly, my, ny,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ lz, mz, nz,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [  0,  0,  0, lx, mx, nx,  0,  0,  0,  0,  0,  0],
                [  0,  0,  0, ly, my, ny,  0,  0,  0,  0,  0,  0],
                [  0,  0,  0, lz, mz, nz,  0,  0,  0,  0,  0,  0],
                [  0,  0,  0,  0,  0,  0, lx, mx, nx,  0,  0,  0],
                [  0,  0,  0,  0,  0,  0, ly, my, ny,  0,  0,  0],
                [  0,  0,  0,  0,  0,  0, lz, mz, nz,  0,  0,  0],
                [  0,  0,  0,  0,  0,  0,  0,  0,  0, lx, mx, nx],
                [  0,  0,  0,  0,  0,  0,  0,  0,  0, ly, my, ny],
                [  0,  0,  0,  0,  0,  0,  0,  0,  0, lz, mz, nz]])

#matriz de rigidez no global
K = R3.T * Ke
K = K * R3


#matriz N para a interpolação dos deslocamentos ????????? assim??
N = sp.zeros(6,12)
for j in range(2):
    N[ 0, IDtr[j] ] += Nt[j]
    N[ 3, IDto[j] ] += Nt[j]
for j in range(4):
    N[ 1, IDv_ij[j] ] += Nv[j]
    N[ 4, IDv_ij[j] ] += Nv[j]
    N[ 2, IDv_ik[j] ] += Nv[j]
    N[ 5, IDv_ik[j] ] += Nv[j]










##convertendo para função python
#nN1 = sp.utilities.lambdify([x, L], N1, "numpy")
#nN2 = sp.utilities.lambdify([x, L], N2, "numpy")
#nN3 = sp.utilities.lambdify([x, L], N3, "numpy")
#nN4 = sp.utilities.lambdify([x, L], N4, "numpy")
#
#
##L = 2.
##x = np.linspace(-L/2., L/2, 100)
##
##plt.plot(x, nN1(x, L), label="N1")
##plt.plot(x, nN2(x, L), label="N2")
##
##plt.plot(x, nN3(x, L), label="N3")
##plt.plot(x, nN4(x, L), label="N4")
##
##plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
##
##plt.show()


##cálculo da matriz de rigidez
#Bp = sp.Matrix([ddN1, ddN2, ddN3, ddN4])
#Bmulp = Bp * Bp.T
#
#E = sp.Symbol('E')
#I = sp.Symbol('I')
#
#Kep = E*I*sp.integrate( Bmulp, (x, x1, x3) )
#
#A = sp.Symbol('A')
#
#Bt = sp.Matrix([dL1, dL2])
#Bmult = Bt * Bt.T
#
#Ket = E*A*sp.integrate( Bmult, (x, x1, x3) )
#
#Ke = sp.zeros(6, 6)
#
#Ke[1:3,1:3] = Kep[0:2,0:2]
#Ke[1:3,4:] = Kep[0:2,2:4]
#
#Ke[4:,1:3] = Kep[2:4,0:2]
#Ke[4:,4:] = Kep[2:4,2:4]
#
#Ke[0,0] = Ket[0,0]
#Ke[0,3] = Ket[0,1]
#
#Ke[3,0] = Ket[1,0]
#Ke[3,3] = Ket[1,1]
#
##matriz de rotação
#c = sp.Symbol('c')
#s = sp.Symbol('s')
#
#rot = sp.Matrix([ [c, -s, 0], [s, c, 0], [0, 0, 1] ])
#
#R = sp.zeros(6,6)
#R[:3,:3] = rot[:,:]
#R[3:,3:] = rot[:,:]
#
#Keg = R * Ke
#KeG = Keg * R.T
#
#
##deslocamentos no elemento
#N_EL = sp.Matrix([[L1],
#                  [N1],
#                  [N2],
#                  [L2],
#                  [N3],
#                  [N4]])
#
##para cálculo das deformações e tensões
#dN_ES = sp.Matrix([[dL1],
#                  [dN1],
#                  [dN2],
#                  [dL2],
#                  [dN3],
#                  [dN4]])
##para o cálculo do momento
#dN_M = sp.Matrix([[ddN1],
#                  [ddN2],
#                  [ddN3],
#                  [ddN4]])
##para o cálculo do cortante
#dN_C = sp.Matrix([[dddN1],
#                  [dddN2],
#                  [dddN3],
#                  [dddN4]])
##para cálculo da normal
#dN_N = sp.Matrix([[dL1],
#                  [dL2]])
#
##vetor de deformações genérico
#ug0, ug1, ug2, ug3, ug4, ug5 = sp.symbols('ug0 ug1 ug2 ug3 ug4 ug5')
#Ug = sp.Matrix([ug0, ug1, ug2, ug3, ug4, ug5])
#UgM = sp.Matrix([ug1, ug2, ug4, ug5])
#UgN = sp.Matrix([ug0, ug3])
#
#deslocamentos = N_EL.transpose() * Ug
#deformacoes = dN_ES.transpose() * Ug
#tensoes = E * deformacoes
#momento = dN_M.transpose() * UgM
#cortante = dN_C.transpose() * UgM
#normal = dN_N.transpose() * UgN
#
##cargas distribuída constante
#gx = sp.Symbol('gx')
#gy = sp.Symbol('gy')
#
#g_axial = c*gx + s*gy
#g_transv = -s*gx + c*gy
#
#n1 = g_axial*sp.integrate(L1, (x, x1, x3))
#n2 = g_axial*sp.integrate(L2, (x, x1, x3))
#
#f1 = g_transv*sp.integrate(N1, (x, x1, x3))
#f2 = g_transv*sp.integrate(N2, (x, x1, x3))
#f4 = g_transv*sp.integrate(N3, (x, x1, x3))
#f5 = g_transv*sp.integrate(N4, (x, x1, x3))
#
#F = sp.Matrix([n1, f1, f2, n2, f4, f5])
#Fg = R * F
#
###verificando com viga simplesmente apoiada com 1 elemento
##Kvs = Ke[2:7,2:7]
##
##F.row_del(0)
##F.row_del(0)
##F.row_del(-1)
##F.row_del(-1)
##
##U = Kvs.inv()*F
#
##verificando com viga em balanço com carga na extremidade
#F = np.zeros(6)
#F[4] = -10.
#
#Kvb = Ke[3:,3:]
#Fvb = F[3:, np.newaxis]
#
#U = Kvb.inv() * Fvb
#U1 = U[1].subs(L, 4).subs(E, 200. * 1e6).subs(I, 1000. * 0.01**4)
#U2 = U[2].subs(L, 4).subs(E, 200. * 1e6).subs(I, 1000. * 0.01**4)