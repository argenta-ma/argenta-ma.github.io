#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Funções de forma ok!

@author: markinho
"""

import sympy as sp
import numpy as np
import dNdx16nos
#import matplotlib.pyplot as plt
import plotly.graph_objects as go

#elemento padrão
l = sp.Symbol('l')

r1 = 1.
s1 = 1.

r2 = -1.
s2 = 1.

r3 = -1.
s3 = -1.

r4 = 1.
s4 = -1.

r5 = 1./3.
s5 = 1.

r6 = -1./3.
s6 = 1.

r7 = -1.
s7 = 1./3.

r8 = -1.
s8 = -1./3.

r9 = -1./3.
s9 = -1.

r10 = 1./3.
s10 = -1.

r11 = 1.
s11 = -1./3.

r12 = 1.
s12 = 1./3.

r13 = 1./3.
s13 = 1./3.

r14 = -1./3.
s14 = 1./3.

r15 = -1./3.
s15 = -1./3.

r16 = 1./3.
s16 = -1./3.

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

#polinomio completo do segundo grau completo
#c0 + c1 x1 + c2 x2 + c3 x1**2 + c4 x1 x2 + c5 x2**2

Mat_Coef = sp.Matrix([[1, r1, s1, r1*s1, r1**2, r1**2*s1, r1**2*s1**2, r1*s1**2, s1**2, r1**3, r1**3*s1, r1**3*s1**2, r1**3*s1**3, r1**2*s1**3, r1*s1**3, s1**3],  #no1
                      [1, r2, s2, r2*s2, r2**2, r2**2*s2, r2**2*s2**2, r2*s2**2, s2**2, r2**3, r2**3*s2, r2**3*s2**2, r2**3*s2**3, r2**2*s2**3, r2*s2**3, s2**3],  #no2
                      [1, r3, s3, r3*s3, r3**2, r3**2*s3, r3**2*s3**2, r3*s3**2, s3**2, r3**3, r3**3*s3, r3**3*s3**2, r3**3*s3**3, r3**2*s3**3, r3*s3**3, s3**3],  #no3
                      [1, r4, s4, r4*s4, r4**2, r4**2*s4, r4**2*s4**2, r4*s4**2, s4**2, r4**3, r4**3*s4, r4**3*s4**2, r4**3*s4**3, r4**2*s4**3, r4*s4**3, s4**3],  #no4
                      [1, r5, s5, r5*s5, r5**2, r5**2*s5, r5**2*s5**2, r5*s5**2, s5**2, r5**3, r5**3*s5, r5**3*s5**2, r5**3*s5**3, r5**2*s5**3, r5*s5**3, s5**3],  #no5
                      [1, r6, s6, r6*s6, r6**2, r6**2*s6, r6**2*s6**2, r6*s6**2, s6**2, r6**3, r6**3*s6, r6**3*s6**2, r6**3*s6**3, r6**2*s6**3, r6*s6**3, s6**3],  #no6
                      [1, r7, s7, r7*s7, r7**2, r7**2*s7, r7**2*s7**2, r7*s7**2, s7**2, r7**3, r7**3*s7, r7**3*s7**2, r7**3*s7**3, r7**2*s7**3, r7*s7**3, s7**3],  #no7
                      [1, r8, s8, r8*s8, r8**2, r8**2*s8, r8**2*s8**2, r8*s8**2, s8**2, r8**3, r8**3*s8, r8**3*s8**2, r8**3*s8**3, r8**2*s8**3, r8*s8**3, s8**3],  #no8
                      [1, r9, s9, r9*s9, r9**2, r9**2*s9, r9**2*s9**2, r9*s9**2, s9**2, r9**3, r9**3*s9, r9**3*s9**2, r9**3*s9**3, r9**2*s9**3, r9*s9**3, s9**3],  #no9
                      [1, r10, s10, r10*s10, r10**2, r10**2*s10, r10**2*s10**2, r10*s10**2, s10**2, r10**3, r10**3*s10, r10**3*s10**2, r10**3*s10**3, r10**2*s10**3, r10*s10**3, s10**3],  #no10
                      [1, r11, s11, r11*s11, r11**2, r11**2*s11, r11**2*s11**2, r11*s11**2, s11**2, r11**3, r11**3*s11, r11**3*s11**2, r11**3*s11**3, r11**2*s11**3, r11*s11**3, s11**3],  #no11
                      [1, r12, s12, r12*s12, r12**2, r12**2*s12, r12**2*s12**2, r12*s12**2, s12**2, r12**3, r12**3*s12, r12**3*s12**2, r12**3*s12**3, r12**2*s12**3, r12*s12**3, s12**3],  #no12
                      [1, r13, s13, r13*s13, r13**2, r13**2*s13, r13**2*s13**2, r13*s13**2, s13**2, r13**3, r13**3*s13, r13**3*s13**2, r13**3*s13**3, r13**2*s13**3, r13*s13**3, s13**3],  #no13
                      [1, r14, s14, r14*s14, r14**2, r14**2*s14, r14**2*s14**2, r14*s14**2, s14**2, r14**3, r14**3*s14, r14**3*s14**2, r14**3*s14**3, r14**2*s14**3, r14*s14**3, s14**3],  #no14
                      [1, r15, s15, r15*s15, r15**2, r15**2*s15, r15**2*s15**2, r15*s15**2, s15**2, r15**3, r15**3*s15, r15**3*s15**2, r15**3*s15**3, r15**2*s15**3, r15*s15**3, s15**3],  #no15
                      [1, r16, s16, r16*s16, r16**2, r16**2*s16, r16**2*s16**2, r16*s16**2, s16**2, r16**3, r16**3*s16, r16**3*s16**2, r16**3*s16**3, r16**2*s16**3, r16*s16**3, s16**3]]) #no16

U = sp.Matrix([u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16])

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
J = Coefs[9]
K = Coefs[10]
L = Coefs[11]
M = Coefs[12]
N = Coefs[13]
O = Coefs[14]
P = Coefs[15]

r = sp.Symbol('r')
s = sp.Symbol('s')

Ns = sp.expand(A + B*r + C*s + D*r*s + E*r**2 + F*r**2*s + G*r**2*s**2 + H*r*s**2 + I*s**2 + J*r**3 + K*r**3*s + L*r**3*s**2 + M*r**3*s**3 + N*r**2*s**3 + O*r*s**3 + P*s**3)

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

N = sp.Matrix([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16])

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
#nN10 = sp.utilities.lambdify([r, s], N10, "numpy")
#nN11 = sp.utilities.lambdify([r, s], N11, "numpy")
#nN12 = sp.utilities.lambdify([r, s], N12, "numpy")
#nN13 = sp.utilities.lambdify([r, s], N13, "numpy")
#nN14 = sp.utilities.lambdify([r, s], N14, "numpy")
#nN15 = sp.utilities.lambdify([r, s], N15, "numpy")
#nN16 = sp.utilities.lambdify([r, s], N16, "numpy")
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
#dados10 = plty.graph_objs.Surface(z=list(nN10(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados11 = plty.graph_objs.Surface(z=list(nN11(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados12 = plty.graph_objs.Surface(z=list(nN12(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados13 = plty.graph_objs.Surface(z=list(nN13(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados14 = plty.graph_objs.Surface(z=list(nN14(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados15 = plty.graph_objs.Surface(z=list(nN15(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados16 = plty.graph_objs.Surface(z=list(nN16(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#
#fig = plty.subplots.make_subplots(rows=4, cols=4,
#    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
#           [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
#           [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
#           [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]])
#
#fig.add_trace(dados1, row=1, col=1)
#fig.add_trace(dados2, row=1, col=2)
#fig.add_trace(dados3, row=1, col=3)
#fig.add_trace(dados4, row=1, col=4)
#fig.add_trace(dados5, row=2, col=1)
#fig.add_trace(dados6, row=2, col=2)
#fig.add_trace(dados7, row=2, col=3)
#fig.add_trace(dados8, row=2, col=4)
#fig.add_trace(dados9, row=3, col=1)
#fig.add_trace(dados10, row=3, col=2)
#fig.add_trace(dados11, row=3, col=3)
#fig.add_trace(dados12, row=3, col=4)
#fig.add_trace(dados13, row=4, col=1)
#fig.add_trace(dados14, row=4, col=2)
#fig.add_trace(dados15, row=4, col=3)
#fig.add_trace(dados16, row=4, col=4)
#
#fig.update_layout(title="Funções de forma do elemento quadrático" , autosize=True)#, width=900, height=700)
#
#fig.write_html('funcoesForma16nos.html')
##-------------------------------------------------------------------------------------------------------------

##primeira derivada em r
#dNr = []
#for Ni in N:
#    dNr.append(sp.diff(Ni, r))
###convertendo para função lambda nuympy
##ndNr = []
##for dNi in dNr:
##    ndNr.append(sp.utilities.lambdify([r, s], dNi, "numpy"))
#
##primeira derivada em s
#dNs = []
#for Ni in N:
#    dNs.append(sp.diff(Ni, s))
###convertendo para função lambda nuympy
##ndNs = []
##for dNi in dNs:
##    ndNs.append(sp.utilities.lambdify([r, s], dNi, "numpy"))
#
##gerando a matriz dNdx analítica
#x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, x13, y13, x14, y14, x15, y15, x16, y16 = sp.symbols('x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, x13, y13, x14, y14, x15, y15, x16, y16')
#
##Matriz dos nós de um elemento
#Xe = sp.Matrix([[x1, y1], [x2, y2], [x3, y3], [x4, y4], 
#                [x5, y5], [x6, y6], [x7, y7], [x8, y8], 
#                [x9, y9], [x10, y10], [x11, y11], [x12, y12],
#                [x13, y13], [x14, y14], [x15, y15], [x16, y16]])
##Matriz das derivadas das funções de interpolação do elemento padrão no sistema r s
#dNds = sp.Matrix([[dNr[0], dNs[0]], [dNr[1], dNs[1]], [dNr[2], dNs[2]], [dNr[3], dNs[3]], 
#                  [dNr[4], dNs[4]], [dNr[5], dNs[5]], [dNr[6], dNs[6]], [dNr[7], dNs[7]], 
#                  [dNr[8], dNs[8]], [dNr[9], dNs[9]], [dNr[10], dNs[10]], [dNr[11], dNs[11]], 
#                  [dNr[12], dNs[12]], [dNr[13], dNs[13]], [dNr[14], dNs[14]], [dNr[15], dNs[15]]])
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
        
    >>> Xe = np.array([ [x1, y1], [x2, y2], [x3, y3], [x4, y4], 
                        [x5, y5], [x6, y6], [x7, y7], [x8, y8], 
                        [x9, y9], [x10, y10], [x11, y11], [x12, y12],
                        [x13, y13], [x14, y14], [x15, y15], [x16, y16] ])
    '''
    #matriz constitutiva do material
    D = E/(1 - nu**2) * np.array([[1, nu, 0],
                                  [nu, 1, 0],
                                  [0, 0, (1 - nu**2)/(2 + 2*nu)]])
    #número de graus de liberdade por elemento
    GLe = 32
    #coordenadas dos pontos de gauss - 4 cada direção
    PG = np.array([[0.861136311594053,0.861136311594053],
                     [0.339981043584856,0.861136311594053],
                     [-0.339981043584856,0.861136311594053],
                     [-0.861136311594053,0.861136311594053],
                     [0.861136311594053,0.339981043584856],
                     [0.339981043584856,0.339981043584856],
                     [-0.339981043584856,0.339981043584856],
                     [-0.861136311594053,0.339981043584856],
                     [0.861136311594053,-0.339981043584856],
                     [0.339981043584856,-0.339981043584856],
                     [-0.339981043584856,-0.339981043584856],
                     [-0.861136311594053,-0.339981043584856],
                     [0.861136311594053,-0.861136311594053],
                     [0.339981043584856,-0.861136311594053],
                     [-0.339981043584856,-0.861136311594053],
                     [-0.861136311594053,-0.861136311594053]])
    #pesos de cada ponto de gauss
    wPG = np.array([[0.347854845137454,0.347854845137454],
                      [0.652145154862546,0.347854845137454],
                      [0.652145154862546,0.347854845137454],
                      [0.347854845137454,0.347854845137454],
                      [0.347854845137454,0.652145154862546],
                      [0.652145154862546,0.652145154862546],
                      [0.652145154862546,0.652145154862546],
                      [0.347854845137454,0.652145154862546],
                      [0.347854845137454,0.652145154862546],
                      [0.652145154862546,0.652145154862546],
                      [0.652145154862546,0.652145154862546],
                      [0.347854845137454,0.652145154862546],
                      [0.347854845137454,0.347854845137454],
                      [0.652145154862546,0.347854845137454],
                      [0.652145154862546,0.347854845137454],
                      [0.347854845137454,0.347854845137454]])
    Ke = np.zeros((GLe, GLe))
    for p in range(PG.shape[0]):
        B, J = dNdx16nos.dNdx(Xe, PG[p])
        Ke += np.matmul( np.matmul(np.transpose(B), D), B) * wPG[p, 0] * wPG[p, 1] * np.linalg.det(J) * t
    return Ke

#coordenadas dos nós da estrutura
NOS = np.zeros((52,2))
NOS[:,0] = np.tile(np.arange(0, 130, 10, dtype=float), 4)
NOS[:,1] = np.concatenate((np.zeros(13), np.ones(13)*20./3, np.ones(13)*20./3*2, np.ones(13)*20.), axis=0)

#incidência dos elementos !!! DEVE SEGUIR A ORDEM DAS FUNÇÕES DE INTERPOLAÇÃO DEFINIDA NA FUNÇÃO dNdx !!!
IE = np.array([ [42, 39, 0, 3, 41, 40, 26, 13, 1, 2, 16, 29, 28, 27, 14, 15],
                [45, 42, 3, 6, 44, 43, 29, 16, 4, 5, 19, 32, 31, 30, 17, 18],
                [48, 45, 6, 9, 47, 46, 32, 19, 7, 8, 22, 35, 34, 33, 20, 21],
                [51, 48, 9, 12, 50, 49, 35, 22, 10, 11, 25, 38, 37, 36, 23, 24] ])

#malha de elementos
Xe = []
for e in IE:
    Xe.append( np.array([ NOS[e[0]], NOS[e[1]], NOS[e[2]], NOS[e[3]], 
                          NOS[e[4]], NOS[e[5]], NOS[e[6]], NOS[e[7]], 
                          NOS[e[8]], NOS[e[9]], NOS[e[10]], NOS[e[11]], 
                          NOS[e[12]], NOS[e[13]], NOS[e[14]], NOS[e[15]] ]) )
    
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
ID1 = np.repeat(IE[0]*2, 2) + np.tile(np.array([0, 1]), 16)
ID2 = np.repeat(IE[1]*2, 2) + np.tile(np.array([0, 1]), 16)
ID3 = np.repeat(IE[2]*2, 2) + np.tile(np.array([0, 1]), 16)
ID4 = np.repeat(IE[3]*2, 2) + np.tile(np.array([0, 1]), 16)

#graus de liberdade da estrutura
GL = NOS.shape[0]*2 #dois graus de liberdade por nó da estrutura
GLe = 32 #dois graus de liberdade por nó do elemento

#montagem da matriz de rigidez da estrutura
K = np.zeros((GL, GL))
for i in range(GLe):
    for j in range(GLe):
        K[ ID1[i], ID1[j] ] += Ke1[i, j]
        K[ ID2[i], ID2[j] ] += Ke2[i, j]
        K[ ID3[i], ID3[j] ] += Ke3[i, j]
        K[ ID4[i], ID4[j] ] += Ke4[i, j]

#nós livre, restringidos e respectivos graus de liberdade
NOSr = np.array([0, 13, 26, 39])
NOSl = np.delete(np.arange(0, NOS.shape[0], dtype=int), NOSr, axis=0)
GLr = np.repeat(NOSr*2, 2) + np.tile(np.array([0, 1]), NOSr.shape[0])
GLl = np.repeat(NOSl*2, 2) + np.tile(np.array([0, 1]), NOSl.shape[0])

#separação das matrizes de rigidez
Ku = np.delete(np.delete(K, GLr, 0), GLr, 1)
Kr = np.delete(np.delete(K, GLl, 0), GLr, 1)

#vetor de forças nodais
F = np.zeros(GL)
F[25] = -30000./4 #kN
F[51] = -30000./4 #kN
F[77] = -30000./4 #kN
F[103] = -30000./4 #kN

Fu = np.delete(F, GLr, 0)
Fr = np.delete(F, GLl, 0)

Uu = np.linalg.solve(Ku, Fu)
Rr = np.matmul(Kr, Uu) - Fr

U = np.zeros(GL)
U[GLl] = Uu

##visualização dos deslocamentos
#Uxy = U.reshape(NOS.shape)
#fig = go.Figure(data = go.Contour(z=Uxy[:,0], x=NOS[:,0], y=NOS[:,1], colorscale='Jet', contours=dict(
#            start=-170,
#            end=170,
#            size=10,
#            showlabels = True, # show labels on contours
#            labelfont = dict(size = 12, color = 'white') ) ) )
#fig.update_layout(title="Deslocamentos em X", autosize=True, width=1200, height=400)
#fig.write_html('deslocamentos16.html')





#!!!!!!!!!!!!!!!!!!!!!!!!!!!!! AQUI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



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