#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Funções de forma ok!



@author: markinho
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'

import meshio

#elemento mantido no x-y
l = sp.Symbol('l')

x_1, y_1, x_2, y_2, x_3, y_3 = sp.symbols('x_1, y_1, x_2, y_2, x_3, y_3')

u_1 = sp.Symbol('u_1')
u_2 = sp.Symbol('u_2')
u_3 = sp.Symbol('u_3')


#polinomio completo do segundo grau completo
#c0 + c1 x1 + c2 x_2 + c3 x1**2 + c4 x1 x_2 + c5 x_2**2

Mat_Coef = sp.Matrix([[1, x_1, y_1],  #no1
                      [1, x_2, y_2],  #no2
                      [1, x_3, y_3]]) #no3

U = sp.Matrix([u_1, u_2, u_3])

Coefs = Mat_Coef.inv() * U

An, Ad = sp.fraction(sp.simplify(Coefs[0]))
Bn, Bd = sp.fraction(sp.simplify(Coefs[1]))
Cn, Cd = sp.fraction(sp.simplify(Coefs[2]))

x, y = sp.symbols('x, y')
A_t = sp.Symbol('A_t') #área do triângulo

#Ns = 1/(2*A_t) * sp.expand(An + Bn*x + Cn*y)
Ns = sp.expand(An + Bn*x + Cn*y)

N1 = sp.simplify( sp.Add(*[argi for argi in Ns.args if argi.has(u_1)]).subs(u_1, 1) )
N2 = sp.simplify( sp.Add(*[argi for argi in Ns.args if argi.has(u_2)]).subs(u_2, 1) )
N3 = sp.simplify( sp.Add(*[argi for argi in Ns.args if argi.has(u_3)]).subs(u_3, 1) )

N = sp.Matrix([[N1, 0, N2, 0, N3, 0], [0, N1, 0, N2, 0, N3]])

#grafico das funcoes de forma -------------------------------------------------------------------
nN1 = sp.utilities.lambdify([x, y], N1, "numpy")
nN2 = sp.utilities.lambdify([x, y], N2, "numpy")
nN3 = sp.utilities.lambdify([x, y], N3, "numpy")

xl = np.linspace(-1., 1., 30)
yl = np.linspace(-1., 1., 30)

xm, ym = np.meshgrid(xl, yl)

#plotagem com o matplotlib -------------------------------------------------------------------------------
#somente para ilustração!!!
fig = plt.figure()
ax = Axes3D(fig)

verts1 = [[0, 0, 1],[1, 0, 0],[0, 1, 0]]
verts2 = [[0, 0, 0],[1, 0, 0],[0, 1, 1]]
verts3 = [[0, 0, 0],[1, 0, 1],[0, 1, 0]]

#ax.add_collection3d(Poly3DCollection([verts1]))
#ax.add_collection3d(Poly3DCollection([verts2]))
ax.add_collection3d(Poly3DCollection([verts3]))
plt.show()
#-------------------------------------------------------------------------------------------------------------

##primeira derivada em x
#dN1x = sp.diff(N1, x)#.subs({r: r1, s: s1})
#dN2x = sp.diff(N2, x)#.subs({r: r2, s: s2})
#dN3x = sp.diff(N3, x)#.subs({r: r3, s: s3})
#dN4r = sp.diff(N4, r)#.subs({r: r4, s: s4})
##convertendo para função lambda nuympy
#ndN1r = sp.utilities.lambdify([r, s], dN1r, "numpy")
#ndN2r = sp.utilities.lambdify([r, s], dN2r, "numpy")
#ndN3r = sp.utilities.lambdify([r, s], dN3r, "numpy")
#ndN4r = sp.utilities.lambdify([r, s], dN4r, "numpy")
#
##primeira derivada em s
#dN1y = sp.diff(N1, y)#.subs({r: r1, s: s1})
#dN2y = sp.diff(N2, y)#.subs({r: r2, s: s2})
#dN3y = sp.diff(N3, y)#.subs({r: r3, s: s3})
#dN4s = sp.diff(N4, s)#.subs({r: r4, s: s4})
##convertendo para função lambda nuympy
#ndN1s = sp.utilities.lambdify([r, s], dN1s, "numpy")
#ndN2s = sp.utilities.lambdify([r, s], dN2s, "numpy")
#ndN3s = sp.utilities.lambdify([r, s], dN3s, "numpy")
#ndN4s = sp.utilities.lambdify([r, s], dN4s, "numpy")
#
##gerando a matriz dNdx analítica
#x1 = sp.Symbol('x1')
#y1 = sp.Symbol('y1')
#x2 = sp.Symbol('x2')
#y2 = sp.Symbol('y2')
#x3 = sp.Symbol('x3')
#y3 = sp.Symbol('y3')
#x4 = sp.Symbol('x4')
#y4 = sp.Symbol('y4')
    
##Matriz dos nós de um elemento
#Xe = sp.Matrix([[x1, y1],[x2, y2], [x3, y3], [x4, y4]])
##Matriz das derivadas das funções de interpolação do elemento padrão no sistema r s
#dNds = sp.Matrix([[dN1r, dN1s], [dN2r, dN2s], [dN3r, dN3s], [dN4r, dN4s]])
#
##Jacobiano analítico
#J = Xe.T * dNds
#JI = J.inv()
#
##derivadas das funções de interpolação do elemento no sistema local x y
#dNdx = dNds * JI
#
#B = sp.Matrix([[dN1x, 0, dN2x, 0, dN3x, 0],
#              [0, dN1y, 0, dN2y, 0, dN3y],
#              [dN1y, dN1x, dN2y, dN2x, dN3y, dN3x]])
#
##tensores constitutivos
#E, nu = sp.symbols('E, nu')
#
##tensor constitutivo para o estado plano de tensões
##D_t = E/(1 - nu**2) * sp.Matrix([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu**2)/( 2*(1 + nu) )]])
#D_t = sp.Matrix([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu**2)/( 2*(1 + nu) )]])
#
##tensor constitutivo para o estado plano de deformação
#D_d = E/( (1 + nu)*(1 - 2*nu) )*sp.Matrix([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2*nu)/2]])
#
##integração da matriz de rigidez com espessura constante no elemento
#t = sp.Symbol('t')
#
#BtDB_t = B.T * D_t * B
#BtDB_d = B.T * D_d * B
#
#ke_t = t * A_t * BtDB_t
#ke_d = t * A_t * BtDB_d






#### iniciando do código numérico ---------------------------------------------------------------------------------------------------------------------
#def dNdx(Xe, pg):
#    '''
#    Função para a determinação da matriz das derivadas das funções de interpolação já no sistema x y e do jacobiano
#    
#    Parâmetros
#    ----------
#    
#    Xe: array numpy com as coordenadas de cada nó dos elementos dispostas no sentido horário, com o primeiro nó o correspondente ao segundo quadrante
#    
#    >>>
#        2 ----- 1
#        |       |
#        |       |
#        3-------4
#        
#    >>> Xe = np.array([ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ])
#    
#    pg: coordenadas do ponto de gauss utilizado
#    
#    >>> pg = np.array([ [xpg, ypg] ])
#    
#    retorna a matriz B para cada ponto de gauss
#    '''
#    r = pg[0]
#    s = pg[1]
#    x1 = Xe[0,0]
#    y1 = Xe[0,1]
#    x2 = Xe[1,0]
#    y2 = Xe[1,1]
#    x3 = Xe[2,0]
#    y3 = Xe[2,1]
#    x4 = Xe[3,0]
#    y4 = Xe[3,1]
#    
#    J = np.array([ [x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4), x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4)],
#                    [y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4), y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)]])
#    
#    dNdx = np.array([ [ (r/4 + 1/4)*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) + (s/4 + 1/4)*(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4)) - (x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))/((-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))),   (r/4 + 1/4)*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) - (s/4 + 1/4)*(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))],
#                       [(-r/4 + 1/4)*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) + (-s/4 - 1/4)*(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4)) - (x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))/((-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))), (-r/4 + 1/4)*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) - (-s/4 - 1/4)*(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))],
#                       [  (r/4 - 1/4)*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) + (s/4 - 1/4)*(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4)) - (x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))/((-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))),   (r/4 - 1/4)*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) - (s/4 - 1/4)*(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))],
#                       [(-r/4 - 1/4)*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) + (-s/4 + 1/4)*(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4)) - (x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))/((-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))), (-r/4 - 1/4)*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) - (-s/4 + 1/4)*(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))]])
#    B1x = dNdx[0,0]
#    B1y = dNdx[0,1]
#    B2x = dNdx[1,0]
#    B2y = dNdx[1,1]
#    B3x = dNdx[2,0]
#    B3y = dNdx[2,1]
#    B4x = dNdx[3,0]
#    B4y = dNdx[3,1]
#    B = np.array([[B1x,   0, B2x,   0, B3x,   0, B4x,   0],
#                  [  0, B1y,   0, B2y,   0, B3y,   0, B4y],
#                  [B1y, B1x, B2y, B2x, B3y, B3x, B4y, B4x]])
#    return B, J
#
#def ke(Xe, E, nu, t):
#    '''
#    Função para a geração das matrizes de rigidez dos elementos função das coordenadas dos elementos no sistema global, o módulo de elasticidade
#    do material (E), o corficiente de poisson do material (nu) e da espessura (t), considerando 4 pontos de gauss para a integração
#    
#    Parâmetros
#    ----------
#    
#    Xe: array numpy com as coordenadas de cada nó dos elementos dispostas no sentido antihorário, com o primeiro nó o correspondente ao primeiro quadrante.
#        
#    >>> Xe = np.array([ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ])
#    '''
#    #matriz constitutiva do material
#    D = E/(1 - nu**2) * np.array([[1, nu, 0],
#                                  [nu, 1, 0],
#                                  [0, 0, (1 - nu**2)/(2 + 2*nu)]])
#    #número de graus de liberdade por elemento
#    GLe = 8
#    #coordenadas e pesos dos pontos de Gauss
#    PG = np.array([[0.5773502691896258, 0.5773502691896258],
#                   [-0.5773502691896258, 0.5773502691896258],
#                   [-0.5773502691896258, -0.5773502691896258],
#                   [0.5773502691896258, -0.5773502691896258]])
#    wPG = np.array([[1., 1.],
#                    [1., 1.],
#                    [1., 1.],
#                    [1., 1.]])
#    Be = []
#    Ke = np.zeros((GLe, GLe))
#    for p in range(PG.shape[0]):
#        B, J = dNdx(Xe, PG[p])
#        Be.append(B)
#        Ke += np.matmul( np.matmul(np.transpose(B), D), B) * wPG[p, 0] * wPG[p, 1] * np.linalg.det(J) * t
#    return Ke, Be
#
##coordenadas dos nós da estrutura
#NOS = np.array([ [30., 0.],
#                 [60., 0.],
#                 [90., 0.],
#                 [120., 0.],
#                 [30., 20.],
#                 [60., 20.],
#                 [90., 20.],
#                 [120., 20.],
#                 [0., 0.],
#                 [0., 20.],])
#
##incidência dos elementos !!! DEVE SEGUIR A ORDEM DAS FUNÇÕES DE INTERPOLAÇÃO DEFINIDA NA FUNÇÃO dNdx !!!
#IE = np.array([ [9, 4, 0, 8],
#                [4, 5, 1, 0],
#                [5, 6, 2, 1],
#                [6, 7, 3, 2],])
#
##malha de elementos
#Xe = []
#for e in IE:
#    Xe.append( np.array([ NOS[e[0]], NOS[e[1]], NOS[e[2]], NOS[e[3]] ]) )
#    
##propriedades mecânicas do material da estrutura e espessura
#E = 20000. #kN/cm2
#nu = 0.3
#t = 10. #cm
##resistência a compressão e a tração para aplicação do critério de Christensen, aço A-36
#Sc = 25. #kN/cm2
#St = 5. #kN/cm2
##coesão e o ângulo de atrito para os critérios de Mohr-Coulomb e Drucker-Prager (http://www.pcc.usp.br/files/text/publications/BT_00231.pdf)
#phi = 51. * np.pi/180.
#coesao = 0.00073 #kN/cm2
#
##determinação da matriz de rigidez dos elementos
#Ke1, Be1 = ke(Xe[0], E, nu, t)
#Ke2, Be2 = ke(Xe[1], E, nu, t)
#Ke3, Be3 = ke(Xe[2], E, nu, t)
#Ke4, Be4 = ke(Xe[3], E, nu, t)
#
##indexação dos graus de liberdade
#ID1 = np.repeat(IE[0]*2, 2) + np.tile(np.array([0, 1]), 4)
#ID2 = np.repeat(IE[1]*2, 2) + np.tile(np.array([0, 1]), 4)
#ID3 = np.repeat(IE[2]*2, 2) + np.tile(np.array([0, 1]), 4)
#ID4 = np.repeat(IE[3]*2, 2) + np.tile(np.array([0, 1]), 4)
#
##graus de liberdade da estrutura
#GL = NOS.shape[0]*2
#DOF = GL - 4
#
##montagem da matriz de rigidez da estrutura
#K = np.zeros((GL, GL))
#for i in range(8):
#    for j in range(8):
#        K[ ID1[i], ID1[j] ] += Ke1[i, j]
#        K[ ID2[i], ID2[j] ] += Ke2[i, j]
#        K[ ID3[i], ID3[j] ] += Ke3[i, j]
#        K[ ID4[i], ID4[j] ] += Ke4[i, j]
#
##separação das matrizes de rigidez
#Ku = K[:DOF, :DOF]
#Kr = K[GL-DOF:, :DOF]
#
##vetor de forças nodais
#F = np.zeros(GL)
#F[7] = -150. #kN
#F[15] = -150. #kN
#
#Fu = F[:DOF]
#Fr = F[GL-DOF:]
#
#Uu = np.linalg.solve(Ku, Fu)
#Rr = np.matmul(Kr, Uu) - Fr
#
#U = np.zeros(GL)
#U[:DOF] = Uu
#
#Uxy = U.reshape(NOS.shape)
#
###visualização dos deslocamentos ---------------------------------------------------------------------------------------------------------------------
##fig = go.Figure(data = go.Contour(z=Uxy[:,0], x=NOS[:,0], y=NOS[:,1], colorscale='Jet', contours=dict(
##            showlabels = True, # show labels on contours
##            labelfont = dict(size = 12, color = 'white') ) ) )
##fig.update_layout(title="Deslocamentos em X", autosize=True, width=1200, height=400)
##fig.write_html('deslocamentos.html')
#
###visualização dos vetores de deslocamentos dos nós
##fig = plty.figure_factory.create_quiver(NOS[:,0], NOS[:,1], Uxy[:,0], Uxy[:,1])
##fig.write_html('deslocamentosVetor.html')
#
###geração do arquivo vtu
##pontos = NOS
##celulas = {'quad': IE}
##meshio.write_points_cells(
##        "teste.vtu",
##        pontos,
##        celulas,
##        # Optionally provide extra data on points, cells, etc.
##        point_data = {"U": Uxy},
##        # cell_data=cell_data,
##        # field_data=field_data
##        )
#
##-----------------------------------------------------------------------------------------------------------------------------------------------------
#
##determinação dos deslocamentos por elemento
#Ue = []
#Ue.append( U[ID1] )
#Ue.append( U[ID2] )
#Ue.append( U[ID3] )
#Ue.append( U[ID4] )
#
#
##determinação das deformações por ponto de Gauss -----------------------------------------------------------------------------------------------------
#epsilon1 = []
#epsilon2 = []
#epsilon3 = []
#epsilon4 = []
#for b in range(4): #range na quantidade de pontos de Gauss
#    epsilon1.append( np.matmul(Be1[b], Ue[0]) )
#    epsilon2.append( np.matmul(Be2[b], Ue[1]) )
#    epsilon3.append( np.matmul(Be3[b], Ue[2]) )
#    epsilon4.append( np.matmul(Be4[b], Ue[3]) )
#
##determinação das tensões por ponto de Gauss ---------------------------------------------------------------------------------------------------------
##matriz constitutiva do material
#D = E/(1 - nu**2) * np.array([[1, nu, 0],
#                              [nu, 1, 0],
#                              [0, 0, (1 - nu**2)/(2 + 2*nu)]])
#
#sigma1 = []
#sigma2 = []
#sigma3 = []
#sigma4 = []
#for b in range(4): #range na quantidade de pontos de Gauss
#    sigma1.append( np.matmul(D, epsilon1[b]) )
#    sigma2.append( np.matmul(D, epsilon2[b]) )
#    sigma3.append( np.matmul(D, epsilon3[b]) )
#    sigma4.append( np.matmul(D, epsilon4[b]) )
#
##cálculo das tensões principais nos pontos de Gauss---------------------------------------------------------------------------------------------------
##tensão principal máxima, tensão principal mínima, ângulo das tensões principais, tensão máxima de cisalhamento, tensão equivalente de von Mises
#sigmaPP1 = []
#sigmaPP2 = []
#sigmaPP3 = []
#sigmaPP4 = []
#
#def principaisPG(sigmas):
#    '''
#    Função para a determinação da tensão principal 1 (sigmaMAX), tensão principal 2 (sigmaMIN), 
#    ângulo das tensões principais, tensão máxima de cisalhamento, tensão equivalente de von Mises, de Christensen para materiais frágeis com
#    sigmaC <= 0.5 sigmaT (que deve ser menor que 1), de Morh-Coulomb de Drucker-Prager para a tensão fora do plano igual a zero
#    
#    sigmas é um array de uma dimensão contendo sigma_x, sigma_y e tau_xy
#    
#    retorna um array de uma dimensão com as quantidades acima
#    '''
#    sigma_x = sigmas[0]
#    sigma_y = sigmas[1]
#    tay_xy = sigmas[2]
#    
#    sigmaMAX = (sigma_x + sigma_y)/2 + np.sqrt( ((sigma_x - sigma_y)/2)**2 + tay_xy**2 )
#    sigmaMIN = (sigma_x + sigma_y)/2 - np.sqrt( ((sigma_x - sigma_y)/2)**2 + tay_xy**2 )
#    theta = 1./2. * np.arctan( 2*tay_xy/(sigma_x - sigma_y) )
#    tauMAX = (sigmaMAX - sigmaMIN)/2
#    sigmaEQvM = np.sqrt( sigmaMAX**2 - sigmaMAX*sigmaMIN + sigmaMIN**2 )
#    sigmaEQc = (1/St - 1/Sc)*(sigmaMAX + sigmaMIN) + 1/(St*Sc)*(sigmaMAX**2 - sigmaMAX*sigmaMIN + sigmaMIN**2)
#    sigmaEQmc = 2*( (sigmaMAX + sigmaMIN)/2.*np.sin(phi) + coesao*np.cos(phi) )/(sigmaMAX - sigmaMIN)
#    A = 2*1.4142135623730951*np.sin(phi)/(3 - np.sin(phi))
#    B = 3.*coesao*np.cos(phi)/np.sin(phi)
#    sigmaEQdp = ( (sigmaMAX - sigmaMIN)**2 + sigmaMAX**2 + sigmaMIN**2 )/( A**2*(sigmaMAX + sigmaMIN + B)**2 )
#    
#    return np.array([ sigmaMAX, sigmaMIN, theta, tauMAX, sigmaEQvM, sigmaEQc, sigmaEQmc, sigmaEQdp ])
#
#for p in range(4):
#    sigmaPP1.append( principaisPG(sigma1[p]) )
#    sigmaPP2.append( principaisPG(sigma2[p]) )
#    sigmaPP3.append( principaisPG(sigma3[p]) )
#    sigmaPP4.append( principaisPG(sigma4[p]) )
#
##cálculo das tensões nos nós, interpolando com as funções de interpolação dos elementos





