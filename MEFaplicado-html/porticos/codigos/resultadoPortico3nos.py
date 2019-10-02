#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Vetor de cargas equivalentes a distrubuída da placa OK!

Carga de vento OK!!!


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
    no2 = nos[elementos[:,1]] #nós finais
    sen_cos_comp_comp[:,2] = np.sqrt( (no2[:,0] - no1[:,0])**2 + (no2[:,1] - no1[:,1])**2) #comprimento
    sen_cos_comp_comp[:,0] = (no2[:,1] - no1[:,1])/( sen_cos_comp_comp[:,2] ) #seno
    sen_cos_comp_comp[:,1] = (no2[:,0] - no1[:,0])/( sen_cos_comp_comp[:,2] ) #cosseno
    
    return sen_cos_comp_comp

GL = np.arange(0, 21).reshape(7,3)
#IE = np.array([[5, 3, 0],[0, 1, 2],[6, 4, 2]]) #com os 3 nós do elemento
IE = np.array([[5, 0],[0, 2],[6, 2]]) #só com os nós extremos
nos = np.array([ [-470., 470.], [0., 470.], [470., 470.], [-470., 235.], [470., 235.], [-470., 0.], [470, 0] ])

scL = angulos_comprimentos(nos, IE)

d = 20. #cm
t_w = 1.25 #cm
b_f = 40. #cm
t_f = 1.25 #cm
h = d - 2 * t_f
I_z = b_f*d**3/12 - (b_f-2*t_w)*h**3/12 #cm4
Ar = d*b_f - h*(b_f-2*t_w) #cm2
#matriz de rigidez dos elementos
Ke1 = rigidez_portico(20000., Ar, I_z, scL[0]) #kN/cm2, cm2 e cm4
Ke2 = rigidez_portico(20000., Ar, I_z, scL[1])
Ke3 = rigidez_portico(20000., Ar, I_z, scL[2])

ID1 = np.array([15, 16, 17, 9, 10, 11, 0, 1, 2])
ID2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
ID3 = np.array([18, 19, 20, 12, 13, 14, 6, 7, 8])

K = np.zeros((21, 21))

for i in range(9):
    for j in range(9):
        K[ ID1[i], ID1[j] ] += Ke1[i, j]
        K[ ID2[i], ID2[j] ] += Ke2[i, j]
        K[ ID3[i], ID3[j] ] += Ke3[i, j]

Ku = K[:15, :15]
Kr = K[15:, :15]

##determinação das forças nodais equivalentes ---------------------------------------------------------------------
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

#determinação da força com descontinuidade numérico
g = 300./400 * 9.81/1000 #sp.Symbol('g') #em kN
de = 420. #sp.Symbol('de') #espeçamento da carga a borda
dp = 400. #sp.Symbol('dp') #largura da carga
A = scL[1,2] - de - dp - scL[0,2]
B = scL[1,2]/2 - de
Nnn = Nn.subs({l: scL[1,2]})
Feg = - g * sp.integrate( Nnn, (r, A, B) )

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

#Determinação da carga distribuída na viga superior, ja como pórtico----------------------------------------------------------------
Lvs = scL[1,2] #cm
q = 0.02 #kN/cm
Nnn = Nn.subs({l: Lvs})
Feq = -q * sp.integrate( Nnn, (r, -Lvs/2, Lvs/2) )
#Feq = np.zeros(6)

#Determinação das demais cargas como pórtico (1 já com rotação)---------------------------------------------------------------------------------------
Fe3 = np.zeros(9)
Feg = np.array([0, Feg[0], Feg[1], 0, Feg[2], Feg[3], 0, Feg[4], Feg[5]], dtype=float)
Feq = np.array([0, Feq[0], Feq[1], 0, Feq[2], Feq[3], 0, Feq[4], Feq[5]], dtype=float)
Fe2 = Feg + Feq
RFv = np.array([[0, -1, 0, 0,  0, 0, 0,  0, 0], 
                [1,  0, 0, 0,  0, 0, 0,  0, 0], 
                [0,  0, 1, 0,  0, 0, 0,  0, 0],
                [0 , 0 ,0 ,0, -1, 0, 0,  0, 0],
                [0,  0, 0, 1,  0, 0, 0,  0, 0],
                [0,  0, 0, 0,  0, 1, 0,  0, 0],
                [0,  0, 0, 0,  0, 0, 0, -1, 0],
                [0,  0, 0, 0,  0, 0, 1,  0, 0],
                [0,  0, 0, 0,  0, 0, 0,  0, 1]])
Fevi = np.array([0, Fevi[0], Fevi[1], 0, Fevi[2], Fevi[3], 0, Fevi[4], Fevi[5]], dtype=float)
Fe1 = np.matmul( RFv,  Fevi)

#Determinação do vetor de cargas nodais equivalentes para cálculo dos deslocamentos
Ft = np.zeros(21)
for i in range(9):
    Ft[ ID1[i] ] += Fe1[i]
    Ft[ ID2[i] ] += Fe2[i]
    Ft[ ID3[i] ] += Fe3[i]
FU = Ft[:15]
FR = Ft[15:]

#determinação dos deslocamentos
Un = np.linalg.solve(Ku, FU)
R = np.matmul(Kr, Un) - FR

U = np.zeros(21)
U[:15] = Un

#reescrevendo os deslocamentos no sistema local do elemento
ug1 = np.zeros(9)
ug2 = np.zeros(9)
ug3 = np.zeros(9)

for i in range(9):
    ug1[i] = U[ ID1[i] ]
    ug2[i] = U[ ID2[i] ]
    ug3[i] = U[ ID3[i] ]

R13 = np.array([[ 0, 1, 0,  0, 0, 0,  0, 0, 0], 
                [-1, 0, 0,  0, 0, 0,  0, 0, 0], 
                [ 0, 0, 1,  0, 0, 0,  0, 0, 0],
                [ 0 ,0 ,0 , 0, 1, 0,  0, 0, 0],
                [ 0, 0, 0, -1, 0, 0,  0, 0, 0],
                [ 0, 0, 0,  0, 0, 1,  0, 0, 0],
                [ 0, 0, 0,  0, 0, 0,  0, 1, 0],
                [ 0, 0, 0,  0, 0, 0, -1, 0, 0],
                [ 0, 0, 0,  0, 0, 0,  0, 0, 1]])

u1 = np.dot(R13, ug1)
u2 = ug2
u3 = np.dot(R13, ug3)

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
deslocamentos1, rotacoes1, deformacoes1, tensoes1, momentos1, corte1, normal1, varElem1 = esP(u1, scL[0, 2], E, Ar, d, I_z)
deslocamentos2, rotacoes2, deformacoes2, tensoes2, momentos2, corte2, normal2, varElem2 = esP(u2, scL[1, 2], E, Ar, d, I_z)
deslocamentos3, rotacoes3, deformacoes3, tensoes3, momentos3, corte3, normal3, varElem3 = esP(u3, scL[2, 2], E, Ar, d, I_z)



#matriz das derivadas das funções de interpolação do pórtico
Bv = -s * sp.diff( sp.diff(Nn, r), r)
Bp = sp.Matrix([[-1/l + 4*r/l**2, 0., 0., -8*r/l**2, 0., 0., 1/l + 4*r/l**2, 0., 0.],
                [0.,  Bv[0], Bv[1], 0., Bv[2], Bv[3], 0., Bv[4], Bv[5] ]])

#deformações nos elementos
epsilon_1 = Bp.subs({l: scL[0,2]}) * u1[:, np.newaxis]
epsilonA_1 = epsilon_1[0]
epsilonF_1 = epsilon_1[1]

epsilon_2 = Bp.subs({l: scL[1,2]}) * u2[:, np.newaxis]
epsilonA_2 = epsilon_2[0]
epsilonF_2 = epsilon_2[1]

epsilon_3 = Bp.subs({l: scL[2,2]}) * u3[:, np.newaxis]
epsilonA_3 = epsilon_3[0]
epsilonF_3 = epsilon_3[1]

#tensões nos elementos
E = 20000 #kN/cm2

sigmaA_1 = E*epsilonA_1
sigmaF_1 = E*epsilonF_1
sigmaA_2 = E*epsilonA_2
sigmaF_2 = E*epsilonF_2
sigmaA_3 = E*epsilonA_3
sigmaF_3 = E*epsilonF_3

#esforços normais
Ap = 143.75 #cm2
N_1 = Ap * sigmaA_1
N_2 = Ap * sigmaA_2
N_3 = Ap * sigmaA_3

#momentos fletores nas barras
M1 = 2 * t_w * sp.integrate( s * sigmaF_1, (s, -h/2, h/2 ) ) + 2 * b_f * sp.integrate( s * sigmaF_1, (s, h/2, h/2 + t_f ) )
M2 = 2 * t_w * sp.integrate( s * sigmaF_2, (s, -h/2, h/2 ) ) + 2 * b_f * sp.integrate( s * sigmaF_2, (s, h/2, h/2 + t_f ) )
M3 = 2 * t_w * sp.integrate( s * sigmaF_3, (s, -h/2, h/2 ) ) + 2 * b_f * sp.integrate( s * sigmaF_3, (s, h/2, h/2 + t_f ) )

#esforço cortante 
V1 = sp.diff(M1, r)
V2 = sp.diff(M2, r)
V3 = sp.diff(M3, r)

###na mão, elemento 1
##momento1 = - (E * I_z) * ( u1[1]*(8/l**2 - 60*r/l**3 - 96*r**2/l**4 + 480*r**3/l**5) + u1[2]*(1/l - 6*r/l**2 - 24*r**2/l**3 + 80*r**3/l**4) + u1[4]*(-16/l**2 + 192*r**2/l**4) + u1[5]*(-48*r/l**2 + 320*r**3/l**4) + u1[7]*(8/l**2 + 60*r/l**3 - 96*r**2/l**4 - 480*r**3/l**5) + u1[8]*(-1/l - 6*r/l**2 + 24*r**2/l**3 + 80*r**3/l**4) )
##cortante1 = (E * I_z) * ( u1[1]*(-60/l**3 - 192*r/l**4 + 1440*r**2/l**5) + u1[2]*(-6/l**2 - 48*r/l**3 + 240*r**2/l**4) + u1[5]*(-48/l**2 + 960*r**2/l**4) + u1[7]*(60/l**3 - 192*r/l**4 - 1440*r**2/l**5) + u1[8]*(-6/l**2 + 48*r/l**3 + 240*r**2/l**4) + u1[4]*384*r/l**4 )
##normal1 = (E * A) * ( u1[0]*(-1/l + 4*r/l**2) + u1[6]*(1/l + 4*r/l**2) - 8*u1[3]*r/l**2 )
##
###na mão, elemento 2
##momento2 = - (E * I_z) * ( u2[1]*(8/l**2 - 60*r/l**3 - 96*r**2/l**4 + 480*r**3/l**5) + u2[2]*(1/l - 6*r/l**2 - 24*r**2/l**3 + 80*r**3/l**4) + u2[4]*(-16/l**2 + 192*r**2/l**4) + u2[5]*(-48*r/l**2 + 320*r**3/l**4) + u2[7]*(8/l**2 + 60*r/l**3 - 96*r**2/l**4 - 480*r**3/l**5) + u2[8]*(-1/l - 6*r/l**2 + 24*r**2/l**3 + 80*r**3/l**4) )
##cortante2 = (E * I_z) * ( u2[1]*(-60/l**3 - 192*r/l**4 + 1440*r**2/l**5) + u2[2]*(-6/l**2 - 48*r/l**3 + 240*r**2/l**4) + u2[5]*(-48/l**2 + 960*r**2/l**4) + u2[7]*(60/l**3 - 192*r/l**4 - 1440*r**2/l**5) + u2[8]*(-6/l**2 + 48*r/l**3 + 240*r**2/l**4) + u2[4]*384*r/l**4 )
##normal2 = (E * A) * ( u2[0]*(-1/l + 4*r/l**2) + u2[6]*(1/l + 4*r/l**2) - 8*u2[3]*r/l**2 )
##
###na mão, elemento 3
##momento3 = - (E * I_z) * ( u3[1]*(8/l**2 - 60*r/l**3 - 96*r**2/l**4 + 480*r**3/l**5) + u3[2]*(1/l - 6*r/l**2 - 24*r**2/l**3 + 80*r**3/l**4) + u3[4]*(-16/l**2 + 192*r**2/l**4) + u3[5]*(-48*r/l**2 + 320*r**3/l**4) + u3[7]*(8/l**2 + 60*r/l**3 - 96*r**2/l**4 - 480*r**3/l**5) + u3[8]*(-1/l - 6*r/l**2 + 24*r**2/l**3 + 80*r**3/l**4) )
##cortante3 = (E * I_z) * ( u3[1]*(-60/l**3 - 192*r/l**4 + 1440*r**2/l**5) + u3[2]*(-6/l**2 - 48*r/l**3 + 240*r**2/l**4) + u3[5]*(-48/l**2 + 960*r**2/l**4) + u3[7]*(60/l**3 - 192*r/l**4 - 1440*r**2/l**5) + u3[8]*(-6/l**2 + 48*r/l**3 + 240*r**2/l**4) + u3[4]*384*r/l**4 )
##normal3 = (E * A) * ( u3[0]*(-1/l + 4*r/l**2) + u3[6]*(1/l + 4*r/l**2) - 8*u3[3]*r/l**2 )
#
#
##grafico dos deslocamentos, normais, momento e cortante ---------------------------------------------------------------------------------------------
##funcoes de forma de treliça e viga
#Nt = sp.Matrix([-r/l + 2*r**2/l**2, 1 - 4*r**2/l**2, r/l + 2*r**2/l**2])
#Np = sp.Matrix([Nn[0], Nn[1], Nn[2], Nn[3], Nn[4], Nn[5]])
#
#u1t = np.array([u1[0], u1[3], u1[6]])
#u1p = np.array([u1[1], u1[2], u1[4], u1[5], u1[7], u1[8]])
#u2t = np.array([u2[0], u2[3], u2[6]])
#u2p = np.array([u2[1], u2[2], u2[4], u2[5], u2[7], u2[8]])
#u3t = np.array([u3[0], u3[3], u3[6]])
#u3p = np.array([u3[1], u3[2], u3[4], u3[5], u3[7], u3[8]])
#
#u1Nt = Nt.T*u1t[:, np.newaxis]
#u1Np = Np.T*u1p[:, np.newaxis]
#u2Nt = Nt.T*u2t[:, np.newaxis]
#u2Np = Np.T*u2p[:, np.newaxis]
#u3Nt = Nt.T*u3t[:, np.newaxis]
#u3Np = Np.T*u3p[:, np.newaxis]
#
#
##convertendo para função python
#u1Nt = sp.utilities.lambdify([r, l], u1Nt[0], "numpy")
#u1Np = sp.utilities.lambdify([r, l], u1Np[0], "numpy")
#u2Nt = sp.utilities.lambdify([r, l], u2Nt[0], "numpy")
#u2Np = sp.utilities.lambdify([r, l], u2Np[0], "numpy")
#u3Nt = sp.utilities.lambdify([r, l], u3Nt[0], "numpy")
#u3Np = sp.utilities.lambdify([r, l], u3Np[0], "numpy")

Y = np.linspace(-scL[0,2]/2, scL[0,2]/2, 100)
X = np.linspace(-scL[1,2]/2, scL[1,2]/2, 100)

####gráfico dos deslocamentos !!!!!!!!!!!!!!!!!!! ERRADO!!!!!!!!!!!!!!!!!
##escala = 100
##plt.plot([0, 0, scL[1,2], scL[1,2]], [0, scL[0,2], scL[0,2], 0], color="gray") #elementos
##plt.scatter([0, 0, scL[1,2], scL[1,2]], [0, scL[0,2], scL[0,2], 0], s=15, color="gray") #nós
##plt.plot(-u1Np(Y, scL[0,2])*escala, u1Nt(Y, scL[0,2])*escala + Y, '--', color='blue')
##plt.plot(u2Nt(X, scL[1,2])*escala + X - u1Np(Y, scL[0,2])[-1]*escala/2, -u2Np(X, scL[1,2])*escala + scL[0,2], '--', color='blue')
##plt.plot(u3Np(Y, scL[0,2])*escala + scL[1,2], u3Nt(Y, scL[0,2]) + Y, '--', color='blue')
##plt.yticks(np.arange(0, 520, step=20))
##plt.show()
#
###esforço normal
##escala_n = 7
##plt.plot([-scL[1,2]/2, -scL[1,2]/2, scL[1,2]/2, scL[1,2]/2], [-235, 235, 235, -235], color="gray") #elementos
##plt.scatter([-scL[1,2]/2, -scL[1,2]/2, scL[1,2]/2, scL[1,2]/2], [-235, 235, 235, -235], s=15, color="gray") #nós
##plt.plot(np.ones(100)*N_1*escala_n, Y)
##plt.plot(X, np.ones(100)*N_2*escala_n + scL[0,2])
##plt.plot(np.ones(100)*N_3*escala_n + scL[1,2], Y)
##plt.show()
#
#esforço cortante
V1f = sp.utilities.lambdify([r], V1, "numpy")
V2f = sp.utilities.lambdify([r], V2, "numpy")
V3f = sp.utilities.lambdify([r], V3, "numpy")
escala_v = 20
plt.plot([-scL[1,2]/2, -scL[1,2]/2, scL[1,2]/2, scL[1,2]/2], [-235, 235, 235, -235], color="gray") #elementos
plt.scatter([-scL[1,2]/2, -scL[1,2]/2, scL[1,2]/2, scL[1,2]/2], [-235, 235, 235, -235], s=15, color="gray") #nós
plt.plot(V1f(Y)*escala_v  - scL[1,2]/2, Y)
plt.plot(X, -V2f(X)*escala_v + scL[0,2]/2)
plt.plot(V3f(Y)*escala_v + scL[1,2]/2, Y)
plt.show()

###momento fletor
#M1f = sp.utilities.lambdify([r], M1, "numpy")
#M2f = sp.utilities.lambdify([r], M2, "numpy")
#M3f = sp.utilities.lambdify([r], M3, "numpy")
#escala_m = 0.1
#plt.plot([-scL[1,2]/2, -scL[1,2]/2, scL[1,2]/2, scL[1,2]/2], [-235, 235, 235, -235], color="gray") #elementos
#plt.scatter([-scL[1,2]/2, -scL[1,2]/2, scL[1,2]/2, scL[1,2]/2], [-235, 235, 235, -235], s=15, color="gray") #nós
#plt.plot(-M1f(Y)*escala_m - scL[1,2]/2, Y)
#plt.plot(X, M2f(X)*escala_m + scL[0,2]/2)
#plt.plot(-M3f(Y)*escala_m + scL[1,2]/2, Y)
#plt.show()

###com as funções de forma ----------------------------------------------------------------------------------
#escala_v = 20.
#plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
#plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
#plt.plot(-normal1*escala_v - 470, varElem1)
#plt.plot(varElem2, normal2*escala_v + 235)
#plt.plot(-normal3*escala_v + 470, varElem3)
#plt.show()
#
escala_v = 20.
plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
plt.plot(-corte1*escala_v - 470, varElem1)
plt.plot(varElem2, corte2*escala_v + 235)
plt.plot(-corte3*escala_v + 470, varElem3)
plt.show()
#
escala_v = 0.1
plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
plt.plot(-momentos1*escala_v - 470, varElem1)
plt.plot(varElem2, momentos2*escala_v + 235)
plt.plot(-momentos3*escala_v + 470, varElem3)
plt.show()