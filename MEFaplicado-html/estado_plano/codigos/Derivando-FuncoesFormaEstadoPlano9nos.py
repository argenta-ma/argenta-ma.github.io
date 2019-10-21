#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Funções de forma ok!

@author: markinho
"""

import sympy as sp
import numpy as np
import dNdx9nos
#import matplotlib.pyplot as plt
#import plotly.graph_objects as go
import meshio

#elemento padrão
l = sp.Symbol('l')

r1 = 1
s1 = 1

r2 = -1
s2 = 1

r3 = -1
s3 = -1

r4 = 1
s4 = -1

r5 = 0
s5 = 1

r6 = -1
s6 = 0

r7 = 0
s7 = -1

r8 = 1
s8 = 0

r9 = 0
s9 = 0

u1 = sp.Symbol('u1')
u2 = sp.Symbol('u2')
u3 = sp.Symbol('u3')
u4 = sp.Symbol('u4')
u5 = sp.Symbol('u5')
u6 = sp.Symbol('u6')
u7 = sp.Symbol('u7')
u8 = sp.Symbol('u8')
u9 = sp.Symbol('u9')

#polinomio completo do segundo grau completo
#c0 + c1 x1 + c2 x2 + c3 x1**2 + c4 x1 x2 + c5 x2**2

Mat_Coef = sp.Matrix([[1, r1, s1, r1*s1, r1**2, r1**2*s1, r1**2*s1**2, r1*s1**2, s1**2],  #no1
                      [1, r2, s2, r2*s2, r2**2, r2**2*s2, r2**2*s2**2, r2*s2**2, s2**2],  #no2
                      [1, r3, s3, r3*s3, r3**2, r3**2*s3, r3**2*s3**2, r3*s3**2, s3**2],  #no3
                      [1, r4, s4, r4*s4, r4**2, r4**2*s4, r4**2*s4**2, r4*s4**2, s4**2],  #no4
                      [1, r5, s5, r5*s5, r5**2, r5**2*s5, r5**2*s5**2, r5*s5**2, s5**2],  #no5
                      [1, r6, s6, r6*s6, r6**2, r6**2*s6, r6**2*s6**2, r6*s6**2, s6**2],  #no6
                      [1, r7, s7, r7*s7, r7**2, r7**2*s7, r7**2*s7**2, r7*s7**2, s7**2],  #no7
                      [1, r8, s8, r8*s8, r8**2, r8**2*s8, r8**2*s8**2, r8*s8**2, s8**2],  #no8
                      [1, r9, s9, r9*s9, r9**2, r9**2*s9, r9**2*s9**2, r9*s9**2, s9**2]]) #no9

U = sp.Matrix([u1, u2, u3, u4, u5, u6, u7, u8, u9])

Coefs = Mat_Coef.inv() * U

A = Coefs[0]
B = Coefs[1]
C = Coefs[2]
D = Coefs[3]
E = Coefs[4]
F = Coefs[5]
G = Coefs[6]
H = Coefs[7]
I = Coefs[8]

r = sp.Symbol('r')
s = sp.Symbol('s')

Ns = sp.expand(A + B*r + C*s + D*r*s + E*r**2 + F*r**2*s + G*r**2*s**2 + H*r*s**2 + I*s**2)

N1 = sp.Add(*[argi for argi in Ns.args if argi.has(u1)]).subs(u1, 1)
N2 = sp.Add(*[argi for argi in Ns.args if argi.has(u2)]).subs(u2, 1)
N3 = sp.Add(*[argi for argi in Ns.args if argi.has(u3)]).subs(u3, 1)
N4 = sp.Add(*[argi for argi in Ns.args if argi.has(u4)]).subs(u4, 1)
N5 = sp.Add(*[argi for argi in Ns.args if argi.has(u5)]).subs(u5, 1)
N6 = sp.Add(*[argi for argi in Ns.args if argi.has(u6)]).subs(u6, 1)
N7 = sp.Add(*[argi for argi in Ns.args if argi.has(u7)]).subs(u7, 1)
N8 = sp.Add(*[argi for argi in Ns.args if argi.has(u8)]).subs(u8, 1)
N9 = sp.Add(*[argi for argi in Ns.args if argi.has(u9)]).subs(u9, 1)

N = sp.Matrix([N1, N2, N3, N4, N5, N6, N7, N8, N9])

##grafico das funcoes de forma no plotly -------------------------------------------------------------------
#nN1 = sp.utilities.lambdify([r, s], N1, "numpy")
#nN2 = sp.utilities.lambdify([r, s], N2, "numpy")
#nN3 = sp.utilities.lambdify([r, s], N3, "numpy")
#nN4 = sp.utilities.lambdify([r, s], N4, "numpy")
#nN5 = sp.utilities.lambdify([r, s], N5, "numpy")
#nN6 = sp.utilities.lambdify([r, s], N6, "numpy")
#nN7 = sp.utilities.lambdify([r, s], N7, "numpy")
#nN8 = sp.utilities.lambdify([r, s], N8, "numpy")
#nN9 = sp.utilities.lambdify([r, s], N9, "numpy")
#
#rl = np.linspace(-1., 1., 100)
#sl = np.linspace(-1., 1., 100)
#
#rm, sm = np.meshgrid(rl, sl)
#
#dados1 = plty.graph_objs.Surface(z=list(nN1(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados2 = plty.graph_objs.Surface(z=list(nN2(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados3 = plty.graph_objs.Surface(z=list(nN3(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados4 = plty.graph_objs.Surface(z=list(nN4(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados5 = plty.graph_objs.Surface(z=list(nN5(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados6 = plty.graph_objs.Surface(z=list(nN6(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados7 = plty.graph_objs.Surface(z=list(nN7(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados8 = plty.graph_objs.Surface(z=list(nN8(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados9 = plty.graph_objs.Surface(z=list(nN9(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#
#fig = plty.subplots.make_subplots(rows=3, cols=3,
#    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
#           [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
#           [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]])
#
#fig.add_trace(dados1, row=1, col=1)
#fig.add_trace(dados2, row=1, col=2)
#fig.add_trace(dados3, row=1, col=3)
#fig.add_trace(dados4, row=2, col=1)
#fig.add_trace(dados5, row=2, col=2)
#fig.add_trace(dados6, row=2, col=3)
#fig.add_trace(dados7, row=3, col=1)
#fig.add_trace(dados8, row=3, col=2)
#fig.add_trace(dados9, row=3, col=3)
#
#fig.update_layout(title="Funções de forma do elemento quadrático" , autosize=True)#, width=900, height=700)
#
#fig.write_html('funcoesForma9nos.html')
##-------------------------------------------------------------------------------------------------------------

##primeira derivada em r
#dN1r = sp.diff(N1, r)#.subs({r: r1, s: s1})
#dN2r = sp.diff(N2, r)#.subs({r: r2, s: s2})
#dN3r = sp.diff(N3, r)#.subs({r: r3, s: s3})
#dN4r = sp.diff(N4, r)#.subs({r: r4, s: s4})
#dN5r = sp.diff(N5, r)#.subs({r: r5, s: s5})
#dN6r = sp.diff(N6, r)#.subs({r: r6, s: s6})
#dN7r = sp.diff(N7, r)#.subs({r: r7, s: s7})
#dN8r = sp.diff(N8, r)#.subs({r: r8, s: s8})
#dN9r = sp.diff(N9, r)#.subs({r: r9, s: s9})
##convertendo para função lambda nuympy
#ndN1r = sp.utilities.lambdify([r, s], dN1r, "numpy")
#ndN2r = sp.utilities.lambdify([r, s], dN2r, "numpy")
#ndN3r = sp.utilities.lambdify([r, s], dN3r, "numpy")
#ndN4r = sp.utilities.lambdify([r, s], dN4r, "numpy")
#ndN5r = sp.utilities.lambdify([r, s], dN5r, "numpy")
#ndN6r = sp.utilities.lambdify([r, s], dN6r, "numpy")
#ndN7r = sp.utilities.lambdify([r, s], dN7r, "numpy")
#ndN8r = sp.utilities.lambdify([r, s], dN8r, "numpy")
#ndN9r = sp.utilities.lambdify([r, s], dN9r, "numpy")
#
##primeira derivada em s
#dN1s = sp.diff(N1, s)#.subs({r: r1, s: s1})
#dN2s = sp.diff(N2, s)#.subs({r: r2, s: s2})
#dN3s = sp.diff(N3, s)#.subs({r: r3, s: s3})
#dN4s = sp.diff(N4, s)#.subs({r: r4, s: s4})
#dN5s = sp.diff(N5, s)#.subs({r: r5, s: s5})
#dN6s = sp.diff(N6, s)#.subs({r: r6, s: s6})
#dN7s = sp.diff(N7, s)#.subs({r: r7, s: s7})
#dN8s = sp.diff(N8, s)#.subs({r: r8, s: s8})
#dN9s = sp.diff(N9, s)#.subs({r: r9, s: s9})
##convertendo para função lambda nuympy
#ndN1s = sp.utilities.lambdify([r, s], dN1s, "numpy")
#ndN2s = sp.utilities.lambdify([r, s], dN2s, "numpy")
#ndN3s = sp.utilities.lambdify([r, s], dN3s, "numpy")
#ndN4s = sp.utilities.lambdify([r, s], dN4s, "numpy")
#ndN5s = sp.utilities.lambdify([r, s], dN5s, "numpy")
#ndN6s = sp.utilities.lambdify([r, s], dN6s, "numpy")
#ndN7s = sp.utilities.lambdify([r, s], dN7s, "numpy")
#ndN8s = sp.utilities.lambdify([r, s], dN8s, "numpy")
#ndN9s = sp.utilities.lambdify([r, s], dN9s, "numpy")
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
#x5 = sp.Symbol('x5')
#y5 = sp.Symbol('y5')
#x6 = sp.Symbol('x6')
#y6 = sp.Symbol('y6')
#x7 = sp.Symbol('x7')
#y7 = sp.Symbol('y7')
#x8 = sp.Symbol('x8')
#y8 = sp.Symbol('y8')
#x9 = sp.Symbol('x9')
#y9 = sp.Symbol('y9')
#
##Matriz dos nós de um elemento
#Xe = sp.Matrix([[x1, y1],[x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6], [x7, y7], [x8, y8], [x9, y9]])
##Matriz das derivadas das funções de interpolação do elemento padrão no sistema r s
#dNds = sp.Matrix([[dN1r, dN1s], [dN2r, dN2s], [dN3r, dN3s], [dN4r, dN4s], [dN5r, dN5s], [dN6r, dN6s], [dN7r, dN7s], [dN8r, dN8s], [dN9r, dN9s]])
#
##Jacobiano analítico
#J = Xe.T * dNds
#JI = J.inv()
#
##derivadas das funções de interpolação do elemento no sistema local x y
#dNdx = dNds * JI

### iniciando do código numérico ---------------------------------------------------------------------------------------------------------------------
def ke(Xe, E, nu, t):
    '''
    Função para a geração das matrizes de rigidez dos elementos função das coordenadas dos elementos no sistema global, o módulo de elasticidade
    do material (E), o corficiente de poisson do material (nu) e da espessura (t), considerando 4 pontos de gauss para a integração
    
    Parâmetros
    ----------
    
    Xe: array numpy com as coordenadas de cada nó dos elementos dispostas no sentido antihorário, com o primeiro nó o correspondente ao primeiro quadrante.
        
    >>> Xe = np.array([ [x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6], [x7, y7], [x8, y8], [x9, y9] ])
    '''
    #matriz constitutiva do material
    D = E/(1 - nu**2) * np.array([[1, nu, 0],
                                  [nu, 1, 0],
                                  [0, 0, (1 - nu**2)/(2 + 2*nu)]])
    #número de graus de liberdade por elemento
    GLe = 18
    #coordenadas dos pontos de gauss - 3 cada direção
    PG = np.array([[ 0.774596669241483, 0.774596669241483],
                   [ 0.000000000000000, 0.774596669241483],
                   [-0.774596669241483, 0.774596669241483],
                   [ 0.774596669241483, 0.000000000000000],
                   [ 0.000000000000000, 0.000000000000000],
                   [-0.774596669241483, 0.000000000000000],
                   [ 0.774596669241483,-0.774596669241483],
                   [ 0.000000000000000,-0.774596669241483],
                   [-0.774596669241483,-0.774596669241483]])
                 
    #pesos de cada ponto de gauss
    wPG = np.array([[0.555555555555556,0.555555555555556],
                    [0.888888888888889,0.555555555555556],
                    [0.555555555555556,0.555555555555556],
                    [0.555555555555556,0.888888888888889],
                    [0.888888888888889,0.888888888888889],
                    [0.555555555555556,0.888888888888889],
                    [0.555555555555556,0.555555555555556],
                    [0.888888888888889,0.555555555555556],
                    [0.555555555555556,0.555555555555556]])
    Ke = np.zeros((GLe, GLe))
    for p in range(PG.shape[0]):
        B, J = dNdx9nos.dNdx(Xe, PG[p])
        Ke += np.matmul( np.matmul(np.transpose(B), D), B) * wPG[p, 0] * wPG[p, 1] * np.linalg.det(J) * t
    return Ke

#coordenadas dos nós da estrutura
NOS = np.zeros((27,2))
NOS[:,0] = np.tile(np.arange(0, 135, 15, dtype=float), 3)
NOS[:,1] = np.concatenate((np.zeros(9), np.ones(9)*10., np.ones(9)*20.), axis=0)

#incidência dos elementos !!! DEVE SEGUIR A ORDEM DAS FUNÇÕES DE INTERPOLAÇÃO DEFINIDA NA FUNÇÃO dNdx !!!
IE = np.array([ [20, 18, 0, 2, 19, 9, 1, 11, 10],
                [22, 20, 2, 4, 21, 11, 3, 13, 12],
                [24, 22, 4, 6, 23, 13, 5, 15, 14],
                [26, 24, 6, 8, 25, 15, 7, 17, 16]])

#malha de elementos
Xe = []
for e in IE:
    Xe.append( np.array([ NOS[e[0]], NOS[e[1]], NOS[e[2]], NOS[e[3]], NOS[e[4]], NOS[e[5]], NOS[e[6]], NOS[e[7]], NOS[e[8]] ]) )
    
#propriedades mecânicas do material da estrutura e espessura
E = 20000. #kN/cm2
nu = 0.3
t = 10. #cm

#determinação da matriz de rigidez dos elementos
Ke1 = ke(Xe[0], E, nu, t)
Ke2 = ke(Xe[1], E, nu, t)
Ke3 = ke(Xe[2], E, nu, t)
Ke4 = ke(Xe[3], E, nu, t)

#indexação dos graus de liberdade
ID1 = np.repeat(IE[0]*2, 2) + np.tile(np.array([0, 1]), 9)
ID2 = np.repeat(IE[1]*2, 2) + np.tile(np.array([0, 1]), 9)
ID3 = np.repeat(IE[2]*2, 2) + np.tile(np.array([0, 1]), 9)
ID4 = np.repeat(IE[3]*2, 2) + np.tile(np.array([0, 1]), 9)

#graus de liberdade da estrutura
GL = NOS.shape[0]*2 #dois graus de liberdade por nó da estrutura
GLe = 9*2 #dois graus de liberdade por nó do elemento

#montagem da matriz de rigidez da estrutura
K = np.zeros((GL, GL))
for i in range(GLe):
    for j in range(GLe):
        K[ ID1[i], ID1[j] ] += Ke1[i, j]
        K[ ID2[i], ID2[j] ] += Ke2[i, j]
        K[ ID3[i], ID3[j] ] += Ke3[i, j]
        K[ ID4[i], ID4[j] ] += Ke4[i, j]

#nós livre, restringidos e respectivos graus de liberdade
NOSr = np.array([0, 9, 18])
NOSl = np.delete(np.arange(0, NOS.shape[0], dtype=int), NOSr, axis=0)
GLr = np.repeat(NOSr*2, 2) + np.tile(np.array([0, 1]), NOSr.shape[0])
GLl = np.repeat(NOSl*2, 2) + np.tile(np.array([0, 1]), NOSl.shape[0])

#separação das matrizes de rigidez
Ku = np.delete(np.delete(K, GLr, 0), GLr, 1)
Kr = np.delete(np.delete(K, GLl, 0), GLr, 1)

#vetor de forças nodais
F = np.zeros(GL)
F[17] = -10000. #kN
F[35] = -10000. #kN
F[53] = -10000. #kN

Fu = np.delete(F, GLr, 0)
Fr = np.delete(F, GLl, 0)

Uu = np.linalg.solve(Ku, Fu)
Rr = np.matmul(Kr, Uu) - Fr

U = np.zeros(GL)
U[GLl] = Uu

Uxy = U.reshape(NOS.shape)

##visualização dos deslocamentos
#fig = go.Figure(data = go.Contour(z=Uxy[:,0], x=NOS[:,0], y=NOS[:,1], colorscale='Jet', contours=dict(
#            start=-170,
#            end=170,
#            size=10,
#            showlabels = True, # show labels on contours
#            labelfont = dict(size = 12, color = 'white') ) ) )
#fig.update_layout(title="Deslocamentos em X", autosize=True, width=1200, height=400)
#fig.write_html('deslocamentos9.html')

#geração do arquivo vtu
pontos = NOS
celulas = {'quad9': IE}
meshio.write_points_cells(
        "teste9.vtu",
        pontos,
        celulas,
        # Optionally provide extra data on points, cells, etc.
        point_data = {"U": Uxy},
        # cell_data=cell_data,
        # field_data=field_data
        )


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!! AQUI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



















###############!!!!!!!!!!!!!!!!!!!!! AQUI!!!!!!!!!!!!!!!!!



##para treliça

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
