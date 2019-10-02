#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Vetor de cargas equivalentes a distrubuída da placa OK!
Carga de vento OK!!!

Resultados OK!

@author: markinho
"""

import sympy as sp
import numpy as np
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt


def rigidez_portico(E, A, I, scL):
    '''
    Matriz de rigidez do elemento de pórtico já com a rotação incorporada
    
    Elemento de pórtico de 3 nos
    
    E: módulo de elasticidade em kN/cm2
    I: inércia da seção transversal em torno do eixo que sai do plano em cm4
    A: área da seção transversal em cm2
    scL: array numpy com os senos, cossenos e comprimentos das barras em cm
    np.zeros(E.shape[0]): para extender os zeros
    
    '''   
    s = scL[0]
    c = scL[1]
    L = scL[2]
    return np.array([ [7*A*E*c**2/(3*L) + 5092*E*I*s**2/(35*L**3),   7*A*E*c*s/(3*L) - 5092*E*I*c*s/(35*L**3), -1138*E*I*s/(35*L**2),  -8*A*E*c**2/(3*L) - 512*E*I*s**2/(5*L**3),    -8*A*E*c*s/(3*L) + 512*E*I*c*s/(5*L**3), -384*E*I*s/(7*L**2),   A*E*c**2/(3*L) - 1508*E*I*s**2/(35*L**3),     A*E*c*s/(3*L) + 1508*E*I*c*s/(35*L**3),  -242*E*I*s/(35*L**2)],
                        [  7*A*E*c*s/(3*L) - 5092*E*I*c*s/(35*L**3), 7*A*E*s**2/(3*L) + 5092*E*I*c**2/(35*L**3),  1138*E*I*c/(35*L**2),    -8*A*E*c*s/(3*L) + 512*E*I*c*s/(5*L**3),  -8*A*E*s**2/(3*L) - 512*E*I*c**2/(5*L**3),  384*E*I*c/(7*L**2),     A*E*c*s/(3*L) + 1508*E*I*c*s/(35*L**3),   A*E*s**2/(3*L) - 1508*E*I*c**2/(35*L**3),   242*E*I*c/(35*L**2)],
                        [                     -1138*E*I*s/(35*L**2),                       1138*E*I*c/(35*L**2),        332*E*I/(35*L),                         128*E*I*s/(5*L**2),                        -128*E*I*c/(5*L**2),        64*E*I/(7*L),                        242*E*I*s/(35*L**2),                       -242*E*I*c/(35*L**2),         38*E*I/(35*L)],
                        [ -8*A*E*c**2/(3*L) - 512*E*I*s**2/(5*L**3),    -8*A*E*c*s/(3*L) + 512*E*I*c*s/(5*L**3),    128*E*I*s/(5*L**2), 16*A*E*c**2/(3*L) + 1024*E*I*s**2/(5*L**3),   16*A*E*c*s/(3*L) - 1024*E*I*c*s/(5*L**3),                0,  -8*A*E*c**2/(3*L) - 512*E*I*s**2/(5*L**3),    -8*A*E*c*s/(3*L) + 512*E*I*c*s/(5*L**3),   -128*E*I*s/(5*L**2)],
                        [   -8*A*E*c*s/(3*L) + 512*E*I*c*s/(5*L**3),  -8*A*E*s**2/(3*L) - 512*E*I*c**2/(5*L**3),   -128*E*I*c/(5*L**2),   16*A*E*c*s/(3*L) - 1024*E*I*c*s/(5*L**3), 16*A*E*s**2/(3*L) + 1024*E*I*c**2/(5*L**3),                0,    -8*A*E*c*s/(3*L) + 512*E*I*c*s/(5*L**3),  -8*A*E*s**2/(3*L) - 512*E*I*c**2/(5*L**3),    128*E*I*c/(5*L**2)],
                        [                       -384*E*I*s/(7*L**2),                         384*E*I*c/(7*L**2),          64*E*I/(7*L),                                       0,                                       0,       256*E*I/(7*L),                         384*E*I*s/(7*L**2),                        -384*E*I*c/(7*L**2),          64*E*I/(7*L)],
                        [  A*E*c**2/(3*L) - 1508*E*I*s**2/(35*L**3),     A*E*c*s/(3*L) + 1508*E*I*c*s/(35*L**3),   242*E*I*s/(35*L**2),  -8*A*E*c**2/(3*L) - 512*E*I*s**2/(5*L**3),    -8*A*E*c*s/(3*L) + 512*E*I*c*s/(5*L**3),  384*E*I*s/(7*L**2), 7*A*E*c**2/(3*L) + 5092*E*I*s**2/(35*L**3),   7*A*E*c*s/(3*L) - 5092*E*I*c*s/(35*L**3),  1138*E*I*s/(35*L**2)],
                        [    A*E*c*s/(3*L) + 1508*E*I*c*s/(35*L**3),   A*E*s**2/(3*L) - 1508*E*I*c**2/(35*L**3),  -242*E*I*c/(35*L**2),    -8*A*E*c*s/(3*L) + 512*E*I*c*s/(5*L**3),  -8*A*E*s**2/(3*L) - 512*E*I*c**2/(5*L**3), -384*E*I*c/(7*L**2),   7*A*E*c*s/(3*L) - 5092*E*I*c*s/(35*L**3), 7*A*E*s**2/(3*L) + 5092*E*I*c**2/(35*L**3), -1138*E*I*c/(35*L**2)],
                        [                      -242*E*I*s/(35*L**2),                        242*E*I*c/(35*L**2),         38*E*I/(35*L),                        -128*E*I*s/(5*L**2),                         128*E*I*c/(5*L**2),        64*E*I/(7*L),                       1138*E*I*s/(35*L**2),                      -1138*E*I*c/(35*L**2),        332*E*I/(35*L)]])

def angulos_comprimentos(nos, elementos):
    '''
    Função para calcular os senos e os cossenos de cada barra e o seu comprimento
    
    no1: coordenadas do nó 1 em array([x, y])
    no2: coordenadas do nó 2 em array([x, y])
    
    retorna array com elementos na primeira dimensão e [sen, cos, comprimento] na segunda
    
    '''    
    sen_cos_comp_comp = np.zeros( (elementos.shape[0], 3) )
    
    no1 = nos[elementos[:,0]] #nós iniciais
    no2 = nos[elementos[:,2]] #nós finais
    sen_cos_comp_comp[:,2] = np.sqrt( (no2[:,0] - no1[:,0])**2 + (no2[:,1] - no1[:,1])**2) #comprimento
    sen_cos_comp_comp[:,0] = (no2[:,1] - no1[:,1])/( sen_cos_comp_comp[:,2] ) #seno
    sen_cos_comp_comp[:,1] = (no2[:,0] - no1[:,0])/( sen_cos_comp_comp[:,2] ) #cosseno
    
    return sen_cos_comp_comp

GL = np.arange(0, 21).reshape(7,3)
#IE = np.array([[9, 7, 0],[0, 1, 2],[2, 3, 4],[4, 5, 6], [10, 8, 6]])
IE = np.array([ [9, 0, 1], [1, 2, 3], [3, 4, 5], [5, 6, 7], [10, 8, 7] ])
#nos = np.array([ [-470., 470.], [-410., 470.], [-350., 470.], [-150., 470.], [50., 470.], [260., 470.], [470., 470.], 
#                [-470., 235.], [470., 235.], [-470., 0.], [470, 0] ])
nos = np.array([ [0, 235], [0, 470], [60, 470], [120, 470], [320, 470], [520, 470], [730, 470], [940, 470], [940, 235], [0,0], [940, 0] ])

scL = angulos_comprimentos(nos, IE)

d = 20. #cm
t_w = 1.25 #cm
b_f = 40. #cm
t_f = 1.25 #cm
h = d - 2 * t_f
I_z = b_f*d**3/12 - (b_f-2*t_w)*h**3/12 #cm4
Ar = d*b_f - h*(b_f-2*t_w) #cm2
#matriz de rigidez dos elementos
Ke = []
for e in range(IE.shape[0]):
    Ke.append(rigidez_portico(20000., Ar, I_z, scL[e])) #kN/cm2, cm2 e cm4

ID = []
for e in IE:
    ID.append( [3*e[0], 3*e[0]+1, 3*e[0]+2, 3*e[1], 3*e[1]+1, 3*e[1]+2, 3*e[2], 3*e[2]+1, 3*e[2]+2] )

K = np.zeros((nos.shape[0]*3, nos.shape[0]*3))

for e in range(IE.shape[0]):
    for i in range(9):
        for j in range(9):
            K[ ID[e][i], ID[e][j] ] += Ke[e][i, j]

dof = nos.shape[0]*3 - 6

Ku = K[:dof, :dof]
Kr = K[dof:, :dof]

##determinação das forças nodais equivalentes !!!!ERRO AQUI!!! ---------------------------------------------------------------------
#para viga
r = sp.Symbol('r')
s = sp.Symbol('s')
l = sp.Symbol('l')
x1 = -l/2
x2 = 0
x3 = l/2
u1 = sp.Symbol('u1')
u2 = sp.Symbol('u2')
u3 = sp.Symbol('u3')
u4 = sp.Symbol('u4')
u5 = sp.Symbol('u5')
u6 = sp.Symbol('u6')

Mat_Coef = sp.Matrix([[1, -l/2, l**2/4, -l**3/8, l**4/16, -l**5/32],
                      [0, 1, -l, 3*l**2/4, -l**3/2, 5*l**4/16],
                      [1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [1, l/2, l**2/4, l**3/8, l**4/16, l**5/32],
                      [0, 1, l, 3*l**2/4, l**3/2, 5*l**4/16]])

U = sp.Matrix([u1, u2, u3, u4, u5, u6])

Coefs = Mat_Coef.inv() * U

A = Coefs[0]
B = Coefs[1]
C = Coefs[2]
D = Coefs[3]
E = Coefs[4]
F = Coefs[5]

Ns = sp.expand(A + B*r + C*r**2 + D*r**3 + E*r**4 + F*r**5)

N1 = sp.Add(*[argi for argi in Ns.args if argi.has(u1)]).subs(u1, 1)
N2 = sp.Add(*[argi for argi in Ns.args if argi.has(u2)]).subs(u2, 1)
N3 = sp.Add(*[argi for argi in Ns.args if argi.has(u3)]).subs(u3, 1)
N4 = sp.Add(*[argi for argi in Ns.args if argi.has(u4)]).subs(u4, 1)
N5 = sp.Add(*[argi for argi in Ns.args if argi.has(u5)]).subs(u5, 1)
N6 = sp.Add(*[argi for argi in Ns.args if argi.has(u6)]).subs(u6, 1)
Nn = sp.Matrix([N1, N2, N3, N4, N5, N6])

#Determinação da carga distribuída na viga superior - PLACA
Lvs = scL[2,2] #cm
g = 300./400 * 9.81/1000 #sp.Symbol('g') #em kN
#g = 0.01 #kN/cm
Nnn = Nn.subs({l: Lvs})
Feg = -g * sp.integrate( Nnn, (r, -Lvs/2, Lvs/2) )

##determinação da força não-linear estimativa do vento analítica
##com a origem no centro do elemento
#xA = -235
#xB = 235
#lv = xB - xA
#vi = 0.0046587 * (r + sp.Rational(lv, 2) )**sp.Rational(1, 5)
#Nvi = sp.expand(sp.Matrix([N1.subs({l: lv}), N2.subs({l: lv}), N3.subs({l: lv}), N4.subs({l: lv}), N5.subs({l: lv}), N6.subs({l: lv})]) * vi)
#Fevi = sp.integrate(Nvi, (r, xA, xB)).evalf()

#resultado de acima
Fevi = -np.array([ 1.15548506063797, 43.1624305176797, 3.43185982081697, 26.0157115449028, 1.65863999243194, -53.9826014556733])
#Fevi = np.zeros(6)

#Determinação da carga distribuída na viga superior !!!!ERRRO AQUI!!!----------------------------------------------------------------
Lvs1 = scL[0,2] #cm
Lvs2 = scL[1,2] #cm
Lvs3 = scL[2,2] #cm
Lvs4 = scL[3,2] #cm
q = 0.02 #kN/cm
#q = 0.01 #kN/cm
Nnn = Nn.subs({l: Lvs})
Feq2 = -q * sp.integrate( Nn.subs({l: Lvs2}), (r, -Lvs2/2, Lvs2/2) )
##Feq2 = np.zeros(6)
Feq3 = -q * sp.integrate( Nn.subs({l: Lvs3}), (r, -Lvs3/2, Lvs3/2) )
Feq4 = -q * sp.integrate( Nn.subs({l: Lvs4}), (r, -Lvs4/2, Lvs4/2) )
#Fevi = -q * sp.integrate( Nn.subs({l: Lvs1}), (r, -Lvs1/2, Lvs1/2) )
##Feq4 = np.zeros(6)

##cargas distribuídas constantes
#def cdcP(s, c, L, gx, gy):
#    '''
#    Cálculo das forças nodais equivalentes a uma carga distribuída constante ao elemento de pórtico.
#    '''
#    return np.array([  L*c*(c*gx + gy*s)/6 - 7*L*s*(c*gy - gx*s)/30,
#                          7*L*c*(c*gy - gx*s)/30 + L*s*(c*gx + gy*s)/6,
#                                                 L**2*(c*gy - gx*s)/60,
#                        2*L*c*(c*gx + gy*s)/3 - 8*L*s*(c*gy - gx*s)/15,
#                        8*L*c*(c*gy - gx*s)/15 + 2*L*s*(c*gx + gy*s)/3,
#                                                                     0,
#                          L*c*(c*gx + gy*s)/6 - 7*L*s*(c*gy - gx*s)/30,
#                          7*L*c*(c*gy - gx*s)/30 + L*s*(c*gx + gy*s)/6,
#                                                -L**2*(c*gy - gx*s)/60])
#
#Feq1 = cdcP(scL[0,0], scL[0,1], scL[0,2], 0.01, 0.)
Feq1 = np.array([0, Fevi[0], Fevi[1], 0, Fevi[2], Fevi[3], 0, Fevi[4], Fevi[5]], dtype=float)
#Feg = cdcP(scL[2,0], scL[2,1], scL[2,2], 0., -0.01)
Feg = np.array([0, Feg[0], Feg[1], 0, Feg[2], Feg[3], 0, Feg[4], Feg[5]], dtype=float)
#Feq2 = cdcP(scL[1,0], scL[1,1], scL[1,2], 0., -0.01)
Feq2 = np.array([0, Feq2[0], Feq2[1], 0, Feq2[2], Feq2[3], 0, Feq2[4], Feq2[5]], dtype=float) 
#Feq3 = cdcP(scL[2,0], scL[2,1], scL[2,2], 0., -0.01)
Feq3 = np.array([0, Feq3[0], Feq3[1], 0, Feq3[2], Feq3[3], 0, Feq3[4], Feq3[5]], dtype=float) 
#Feq4 = cdcP(scL[3,0], scL[3,1], scL[3,2], 0., -0.01)
Feq4 = np.array([0, Feq4[0], Feq4[1], 0, Feq4[2], Feq4[3], 0, Feq4[4], Feq4[5]], dtype=float)
    
#Determinação das demais cargas como pórtico (1 já com rotação) !!!ERRO AQUI!!!---------------------------------------------------------------------------------------
Fe = []
RFv = np.array([[0, -1, 0, 0,  0, 0, 0,  0, 0], 
                [1,  0, 0, 0,  0, 0, 0,  0, 0], 
                [0,  0, 1, 0,  0, 0, 0,  0, 0],
                [0 , 0 ,0 ,0, -1, 0, 0,  0, 0],
                [0,  0, 0, 1,  0, 0, 0,  0, 0],
                [0,  0, 0, 0,  0, 1, 0,  0, 0],
                [0,  0, 0, 0,  0, 0, 0, -1, 0],
                [0,  0, 0, 0,  0, 0, 1,  0, 0],
                [0,  0, 0, 0,  0, 0, 0,  0, 1]])
Fe.append(np.matmul( RFv, Feq1 ))
#Fe.append(Feq1)
Fe.append(Feq2)
Fe.append(Feq3 + Feg)
Fe.append(Feq4)
Fe.append(np.zeros(9))

#Determinação do vetor de cargas nodais equivalentes para cálculo dos deslocamentos
Ft = np.zeros(nos.shape[0]*3)
for e in range(IE.shape[0]):
    for i in range(9):
        Ft[ ID[e][i] ] += Fe[e][i]
FU = Ft[:dof]
FR = Ft[dof:]

#determinação dos deslocamentos
Un = np.linalg.solve(Ku, FU)
R = np.matmul(Kr, Un) - FR

U = np.zeros(nos.shape[0]*3)
U[:dof] = Un

#reescrevendo os deslocamentos no sistema local do elemento
u = []

for e in range(IE.shape[0]):
    ugs = np.zeros(9)
    for i in range(9):
        ugs[i] = U[ ID[e][i] ]
    u.append(ugs)

R13 = np.array([[ 0, 1, 0,  0, 0, 0,  0, 0, 0], 
                [-1, 0, 0,  0, 0, 0,  0, 0, 0], 
                [ 0, 0, 1,  0, 0, 0,  0, 0, 0],
                [ 0 ,0 ,0 , 0, 1, 0,  0, 0, 0],
                [ 0, 0, 0, -1, 0, 0,  0, 0, 0],
                [ 0, 0, 0,  0, 0, 1,  0, 0, 0],
                [ 0, 0, 0,  0, 0, 0,  0, 1, 0],
                [ 0, 0, 0,  0, 0, 0, -1, 0, 0],
                [ 0, 0, 0,  0, 0, 0,  0, 0, 1]])

u[0] = np.dot(R13, u[0])
u[-1] = np.dot(R13, u[-1])

#calculo das deformações, tensões, momento, corte e normal em cada elemento no eixo local ------------------------------------------------------------
def esP(U, L, E, A, h, I, pontos=100):
    x = np.linspace(-L/2, L/2, pontos)
    deslocamentos = U[0]*(-x/L + 2*x**2/L**2) + U[1]*(4*x**2/L**2 - 10*x**3/L**3 - 8*x**4/L**4 + 24*x**5/L**5) + U[2]*(x**2/(2*L) - x**3/L**2 - 2*x**4/L**3 + 4*x**5/L**4) + U[3]*(1 - 4*x**2/L**2) + U[4]*(1 - 8*x**2/L**2 + 16*x**4/L**4) + U[5]*(x - 8*x**3/L**2 + 16*x**5/L**4) + U[6]*(x/L + 2*x**2/L**2) + U[7]*(4*x**2/L**2 + 10*x**3/L**3 - 8*x**4/L**4 - 24*x**5/L**5) + U[8]*(-x**2/(2*L) - x**3/L**2 + 2*x**4/L**3 + 4*x**5/L**4)
    rotacoes = U[0]*(-1/L + 4*x/L**2) + U[1]*(8/L**2 - 60*x/L**3 - 96*x**2/L**4 + 480*x**3/L**5) + U[2]*(1/L - 6*x/L**2 - 24*x**2/L**3 + 80*x**3/L**4) + U[4]*(-16/L**2 + 192*x**2/L**4) + U[5]*(-48*x/L**2 + 320*x**3/L**4) + U[6]*(1/L + 4*x/L**2) + U[7]*(8/L**2 + 60*x/L**3 - 96*x**2/L**4 - 480*x**3/L**5) + U[8]*(-1/L - 6*x/L**2 + 24*x**2/L**3 + 80*x**3/L**4) - U[3]*8*x/L**2
    #deformacoes = U[0]*(-1/L + 4*x/L**2) + U[1]*(8/L**2 - 60*x/L**3 - 96*x**2/L**4 + 480*x**3/L**5) + U[2]*(1/L - 6*x/L**2 - 24*x**2/L**3 + 80*x**3/L**4) + U[4]*(-16/L**2 + 192*x**2/L**4) + U[5]*(-48*x/L**2 + 320*x**3/L**4) + U[6]*(1/L + 4*x/L**2) + U[7]*(8/L**2 + 60*x/L**3 - 96*x**2/L**4 - 480*x**3/L**5) + U[8]*(-1/L - 6*x/L**2 + 24*x**2/L**3 + 80*x**3/L**4) - U[3]*8*x/L**2
    #tensoes = E * deformacoes
    momento = - (E * I) * ( U[1]*(8/L**2 - 60*x/L**3 - 96*x**2/L**4 + 480*x**3/L**5) + U[2]*(1/L - 6*x/L**2 - 24*x**2/L**3 + 80*x**3/L**4) + U[4]*(-16/L**2 + 192*x**2/L**4) + U[5]*(-48*x/L**2 + 320*x**3/L**4) + U[7]*(8/L**2 + 60*x/L**3 - 96*x**2/L**4 - 480*x**3/L**5) + U[8]*(-1/L - 6*x/L**2 + 24*x**2/L**3 + 80*x**3/L**4) )
    cortante = (E * I) * ( U[1]*(-60/L**3 - 192*x/L**4 + 1440*x**2/L**5) + U[2]*(-6/L**2 - 48*x/L**3 + 240*x**2/L**4) + U[5]*(-48/L**2 + 960*x**2/L**4) + U[7]*(60/L**3 - 192*x/L**4 - 1440*x**2/L**5) + U[8]*(-6/L**2 + 48*x/L**3 + 240*x**2/L**4) + U[4]*384*x/L**4 )
    normal = (E * A) * ( U[0]*(-1/L + 4*x/L**2) + U[6]*(1/L + 4*x/L**2) - 8*U[3]*x/L**2 )
    
    #aborgadem reversa
    tensoes = normal/A + momento/I * h/2
    deformacoes = tensoes/E
    
    return deslocamentos, rotacoes, deformacoes, tensoes, momento, cortante, normal, x
    
E = 20000. #kN/cm2
deslocamentos1, rotacoes1, deformacoes1, tensoes1, momentos1, corte1, normal1, varElem1 = esP(u[0], scL[0, 2], E, Ar, d, I_z)
deslocamentos2, rotacoes2, deformacoes2, tensoes2, momentos2, corte2, normal2, varElem2 = esP(u[1], scL[1, 2], E, Ar, d, I_z)
deslocamentos3, rotacoes3, deformacoes3, tensoes3, momentos3, corte3, normal3, varElem3 = esP(u[2], scL[2, 2], E, Ar, d, I_z)
deslocamentos4, rotacoes4, deformacoes4, tensoes4, momentos4, corte4, normal4, varElem4 = esP(u[3], scL[3, 2], E, Ar, d, I_z)
deslocamentos5, rotacoes5, deformacoes5, tensoes5, momentos5, corte5, normal5, varElem5 = esP(u[4], scL[4, 2], E, Ar, d, I_z)



#matriz das derivadas das funções de interpolação do pórtico
Bv = -s * sp.diff( sp.diff(Nn, r), r)
Bv2 = sp.diff( sp.diff(Nn, r), r)
Bp = sp.Matrix([[-1/l + 4*r/l**2, 0., 0., -8*r/l**2, 0., 0., 1/l + 4*r/l**2, 0., 0.],
                [0.,  Bv[0], Bv[1], 0., Bv[2], Bv[3], 0., Bv[4], Bv[5] ]])
Bp2 = sp.Matrix([0.,  Bv2[0], Bv2[1], 0., Bv2[2], Bv2[3], 0., Bv2[4], Bv2[5] ])
    
#deformações nos elementos
epsilon = []

for e in range(IE.shape[0]):
    epsilon.append( Bp.subs({l: scL[e,2]}) * u[e][:, np.newaxis] )

#tensões nos elementos
E = 20000. #kN/cm2

sigma = []
for e in range(IE.shape[0]):
    sigma.append( E*epsilon[e] )

#esforços normais
Ap = 143.75 #cm2
N = []
for e in range(IE.shape[0]):
    N.append( Ap * sigma[e][0] )

#momentos fletores nas barras
M = []
for e in range(IE.shape[0]):
    M.append( 2 * t_w * sp.integrate( s * sigma[e][1], (s, -h/2, h/2 ) ) + 2 * b_f * sp.integrate( s * sigma[e][1], (s, h/2, d/2 ) ) )

#esforço cortante 
V = []
for e in range(IE.shape[0]):
    V.append( sp.diff(M[e], r) )

#grafico dos deslocamentos, normais, momento e cortante ---------------------------------------------------------------------------------------------
#funcoes de forma de treliça e viga
Nt = sp.Matrix([-r/l + 2*r**2/l**2, 1 - 4*r**2/l**2, r/l + 2*r**2/l**2])
Np = sp.Matrix([Nn[0], Nn[1], Nn[2], Nn[3], Nn[4], Nn[5]])

Ymin = np.min(nos[:,1])
Ymax = np.max(nos[:,1])
Xmin = np.min(nos[:,0])
Xmax = np.max(nos[:,0])

Y = np.linspace(-235, 235, 100)
X1 = np.linspace(-60, 60, 100)
X2 = np.linspace(-200, 200, 100)
X3 = np.linspace(-210, 210, 100)


##esforço normal
#escala_n = 20
#plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
#plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
#plt.plot(-np.ones(100)*N[0]*escala_n - 470, Y)
#plt.plot(np.linspace(-470, 470, 100), np.ones(100)*N[1].subs({r:0})*escala_n + 235)
#plt.plot(-np.ones(100)*N[2].subs({r:0})*escala_n + 470, Y)
#plt.show()

#esforço cortante
V1f = sp.utilities.lambdify([r], V[0], "numpy")
V2f = sp.utilities.lambdify([r], V[1], "numpy")
V3f = sp.utilities.lambdify([r], V[2], "numpy")
V4f = sp.utilities.lambdify([r], V[3], "numpy")
V5f = sp.utilities.lambdify([r], V[4], "numpy")

escala_v = 20
plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
plt.plot(V1f(Y)*escala_v - 470, Y)
plt.plot(X1 - 470 + 60, -V2f(X1)*escala_v + 235)
plt.plot(X2 - 350 + 200, -V3f(X2)*escala_v + 235)
plt.plot(X3 + 50 + 210, -V4f(X3)*escala_v + 235)
plt.plot(V5f(Y)*escala_v + 470, Y)
plt.show()

###momento fletor
M1f = sp.utilities.lambdify([r], M[0], "numpy")
M2f = sp.utilities.lambdify([r], M[1], "numpy")
M3f = sp.utilities.lambdify([r], M[2], "numpy")
M4f = sp.utilities.lambdify([r], M[3], "numpy")
M5f = sp.utilities.lambdify([r], M[4], "numpy")

escala_v = 0.1
plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
plt.plot(-M1f(Y)*escala_v -470, Y)
plt.plot(X1 - 470 + 60, M2f(X1)*escala_v + 235)
plt.plot(X2 - 350 + 200, M3f(X2)*escala_v + 235)
plt.plot(X3 + 50 + 210, M4f(X3)*escala_v + 235)
plt.plot(-M5f(Y)*escala_v +470, Y)
plt.show()


##com as funções de forma ----------------------------------------------------------------------------------
#escala_v = 20.
#plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
#plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
#plt.plot(-normal1*escala_v - 470, varElem1)
#plt.plot(varElem2 + 60 - 470, normal2*escala_v + 235)
#plt.plot(varElem3 + 320 - 470, normal3*escala_v + 235)
#plt.plot(varElem4 + 730 - 470, normal4*escala_v + 235)
#plt.plot(-normal5*escala_v + 470, varElem5)
#plt.show()
#
#escala_v = 20.
#plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
#plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
#plt.plot(-corte1*escala_v - 470, varElem1)
#plt.plot(varElem2 + 60 - 470, corte2*escala_v + 235)
#plt.plot(varElem3 + 320 - 470, corte3*escala_v + 235)
#plt.plot(varElem4 + 730 - 470, corte4*escala_v + 235)
#plt.plot(-corte5*escala_v + 470, varElem5)
#plt.show()
#
#escala_v = 0.1
#plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
#plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
#plt.plot(-momentos1*escala_v - 470, varElem1)
#plt.plot(varElem2 + 60 - 470, momentos2*escala_v + 235)
#plt.plot(varElem3 + 320 - 470, momentos3*escala_v + 235)
#plt.plot(varElem4 + 730 - 470, momentos4*escala_v + 235)
#plt.plot(-momentos5*escala_v + 470, varElem5)
#plt.show()