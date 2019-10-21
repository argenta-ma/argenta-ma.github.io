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

def rigidez_portico(E, A, I_z, scL):
    '''
    Função para a deterrminação das matrizes de rigidez do elemento de pórtico
    '''
    s = scL[0]
    c = scL[1]
    L = scL[2]
    return np.array([[A * E * c**2/L + 12 * E * I_z * s**2/L**3, A * E * s * c/L - 12 * E * I_z * s * c/L**3,  - 6 * E * I_z * s/L**2,  - A * E * c**2/L - 12 * E * I_z * s**2/L**3,  - A * E * s * c/L + 12 * E * I_z * s * c/L**3,  - 6 * E * I_z * s/L**2 ],
                    [A * E * s * c/L - 12 * E * I_z * s * c/L**3, A * E * s**2/L + 12 * E * I_z * c**2/L**3, 6 * E * I_z * c/L**2,  - A * E * s * c/L + 12 * E * I_z * s * c/L**3,  - A * E * s**2/L - 12 * E * I_z * c**2/L**3, 6 * E * I_z * c/L**2 ],
                    [ - 6 * E * I_z * s/L**2, 6 * E * I_z * c/L**2, 4 * E * I_z/L  , 6 * E * I_z * s/L**2,  - 6 * E * I_z * c/L**2, 2 * E * I_z/L ],
                    [ - A * E * c**2/L - 12 * E * I_z * s**2/L**3,  - A * E * s * c/L + 12 * E * I_z * s * c/L**3, 6 * E * I_z * s/L**2, A * E * c**2/L + 12 * E * I_z * s**2/L**3, A * E * s * c/L - 12 * E * I_z * s * c/L**3, 6 * E * I_z * s/L**2],
                    [ - A * E * s * c/L + 12 * E * I_z * s * c/L**3,  - A * E * s**2/L - 12 * E * I_z * c**2/L**3,  - 6 * E * I_z * c/L**2, A * E * s * c/L - 12 * E * I_z * s * c/L**3, A * E * s**2/L + 12 * E * I_z * c**2/L**3,  - 6 * E * I_z * c/L**2],
                    [ - 6 * E * I_z * s/L**2, 6 * E * I_z * c/L**2, 2 * E * I_z/L, 6 * E * I_z * s/L**2,  - 6 * E * I_z * c/L**2, 4 * E * I_z/L ]])

def angulos_comprimentos(nos, elementos):
    '''
    Função para calcular os senos e os cossenos de cada barra e o seu comprimento
    
    no1: coordenadas do nó 1 em array([x, y])
    no2: coordenadas do nó 2 em array([x, y])
    
    retorna array com elementos na primeira dimensão e [sen, cos, comprimento] na segunda
    
    '''    
    sen_cos_comp_comp = np.zeros( (elementos.shape[0], 3) )
    
    no1 = nos[ elementos[:,0] ] #nós iniciais
    no2 = nos[ elementos[:,1] ] #nós finais
    sen_cos_comp_comp[:,2] = np.sqrt( (no2[:,0] - no1[:,0])**2 + (no2[:,1] - no1[:,1])**2) #comprimento
    sen_cos_comp_comp[:,0] = (no2[:,1] - no1[:,1])/( sen_cos_comp_comp[:,2] ) #seno
    sen_cos_comp_comp[:,1] = (no2[:,0] - no1[:,0])/( sen_cos_comp_comp[:,2] ) #cosseno
            
    
    return sen_cos_comp_comp

GL = np.array([[6, 7, 8], [0, 1, 2], [3, 4, 5], [9, 10, 11]])
nos = np.array([ [-470, 0], [-470, 470], [470, 470], [470, 0] ], dtype=float)
IE = np.array([ [0, 1], [1, 2], [3, 2] ], dtype=int)

#determinação dos ângulos e comprimentos
scL = angulos_comprimentos(nos, IE)

d = 20. #cm
t_w = 1.25 #cm
b_f = 40. #cm
t_f = 1.25 #cm
h = d - 2 * t_f
I_z = b_f*d**3/12 - (b_f-2*t_w)*h**3/12 #cm4
Ar = d*b_f - h*(b_f-2*t_w)
#matriz de rigidez dos elementos
Ke1 = rigidez_portico(20000, Ar, I_z, scL[0]) #kN/cm2, cm2 e cm4
Ke2 = rigidez_portico(20000, Ar, I_z, scL[1])
Ke3 = rigidez_portico(20000, Ar, I_z, scL[2])

#montagem do coletor
C = np.zeros((IE.shape[0], GL.size), dtype=int) + (-1) #-1 para diferenciar o grau de liberdade 0 de um valor vazio no coletor
for i in range(2): #2 nos por elemento
    for j in range(3): #3 graus de liberdade por elemento
        for b in range(3): #3 elementos na estrutura
            C[b, GL[IE[b,i], j] ] = (i+1)**2 + j - 1 #somar 1 no i pois python inicia em 0


#detrminação de Ku
Ku = np.zeros((6,6)) #6 graus de liberdade livres
for i in range(6):
    for j in range(6):
        if C[0, i] != -1 and C[0, j] != -1:
            Ku[i, j] += Ke1[ C[0, i], C[0, j] ]
        if C[1, i] != -1 and C[1, j] != -1:
            Ku[i, j] += Ke2[ C[1, i], C[1, j] ]
        if C[2, i] != -1 and C[2, j] != -1:
            Ku[i, j] += Ke3[ C[2, i], C[2, j] ]

#detrminação de Kr
Kr = np.zeros((6,6)) #6 graus de liberdade livres e 6 graus de liberdade restringidos
for i in range(6):
    for j in range(6):
        if C[0, i+6] != -1 and C[0, j] != -1:
            Kr[i, j] += Ke1[ C[0, i+6], C[0, j] ]
        if C[1, i+6] != -1 and C[1, j] != -1:
            Kr[i, j] += Ke2[ C[1, i+6], C[1, j] ]
        if C[2, i+6] != -1 and C[2, j] != -1:
            Kr[i, j] += Ke3[ C[2, i+6], C[2, j] ]

#determinação das forças nodais equivalentes
#para viga
r = sp.Symbol('r')
s = sp.Symbol('s')
l = sp.Symbol('l')
x1 = -l/2
x2 = l/2
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

Acte = Coefs[0]
Bcte = Coefs[1]
Ccte = Coefs[2]
Dcte = Coefs[3]

Ns = sp.expand(Acte + Bcte*r + Ccte*r**2 + Dcte*r**3)

N1 = sp.Add(*[argi for argi in Ns.args if argi.has(u1)]).subs(u1, 1)
N2 = sp.Add(*[argi for argi in Ns.args if argi.has(u2)]).subs(u2, 1)
N3 = sp.Add(*[argi for argi in Ns.args if argi.has(u3)]).subs(u3, 1)
N4 = sp.Add(*[argi for argi in Ns.args if argi.has(u4)]).subs(u4, 1)
Nn = sp.Matrix([N1, N2, N3, N4])

##determinação da força com descontinuidade analítica
#g = sp.Symbol('g')
#r1 = sp.Symbol('r1') #espeçamento da carga a borda
#r2 = sp.Symbol('r2') #largura da carga
#Feg = - g * sp.integrate( Nn, (x, r1, r2) )

#determinação da força com descontinuidade numérico
g = 300./400 * 9.81/1000 #sp.Symbol('g') #em kN
de = 420. #sp.Symbol('de') #espeçamento da carga a borda
dp = 400. #sp.Symbol('dp') #largura da carga
A = scL[1,2] - de - dp - scL[0,2]
B = scL[1,2]/2 - de
Nnn = Nn.subs({l: scL[1,2]})
Feg = - g * sp.integrate( Nnn, (r, A, B) )


#Determinação da carga distribuída na viga superior----------------------------------------------------------------
Lvs = 940 #cm
q = 0.02 #kN/cm
Feq = -q * sp.integrate( Nnn, (r, -Lvs/2, Lvs/2) )
#Feq = np.zeros(6)

##teste com viga em balanço usando ke2
#Kvb = Ke2[:3, :3]
#Fvb = np.array([0, Feg[0], Feg[1]], dtype=float)
#Uvb = np.linalg.solve(Kvb, Fvb)
#xA = -235
#xB = 235
#Lv = xB - xA
##funções de forma com origem no nó inicial -----------------------------------------------------------------
#x1i = 0
#x2i = l
#Mat_Coefi = sp.Matrix([[1, x1i, x1i**2, x1i**3],
#                      [0, 1, 2*x1i, 3*x1i**2],
#                      [1, x2i, x2i**2, x2i**3],
#                      [0, 1, 2*x2i, 3*x2i**2]])
#
#Coefsi = Mat_Coefi.inv() * U
#
#Ai = Coefsi[0]
#Bi = Coefsi[1]
#Ci = Coefsi[2]
#Di = Coefsi[3]
#
#Nsi = sp.expand(Ai + Bi*r + Ci*r**2 + Di*r**3)
#
#N1i = sp.Add(*[argi for argi in Nsi.args if argi.has(u1)]).subs(u1, 1)
#N2i = sp.Add(*[argi for argi in Nsi.args if argi.has(u2)]).subs(u2, 1)
#N3i = sp.Add(*[argi for argi in Nsi.args if argi.has(u3)]).subs(u3, 1)
#N4i = sp.Add(*[argi for argi in Nsi.args if argi.has(u4)]).subs(u4, 1)
#Nni = sp.Matrix([N1i, N2i, N3i, N4i])
##------------------------------------------------------------------------------
#xA = -235
#xB = 235
#Lv = xB - xA
##determinação da força não-linear analítica com as funções de forma no início do elemento
#xA = 0.
#xB = 300.
#Lv = xB - xA
#
#vi = 0.0046587 * x**0.2
#Nvi = sp.expand(Nni * vi)
#Fevi = np.array( sp.integrate(Nvi, (r, xA, xB)).subs({l: Lv}) , dtype=float).flatten()

##com a origem no centro do elemento
#xA = -235
#xB = 235
#lv = xB - xA
#vi = 0.0046587 * (r + sp.Rational(Lv, 2) )**sp.Rational(1, 5)
#Nvi = sp.expand(sp.Matrix([N1.subs({l: lv}), N2.subs({l: lv}), N3.subs({l: lv}), N4.subs({l: lv})]) * vi)
#Fevi = sp.integrate(Nvi, (r, xA, xB)).evalf()

#resultado de acima
Fevi = -np.array([ 2.78838610441379,  238.280267104451,   3.4575987694731, -262.108293814896])
#Fevi = np.zeros(6)

#TESTANDO:

##Viga analítica com a origem da extremidade do elemento para comparação: em balanço com carga do vento vi com comprimento de 300 cm
#Ev = 20000.
#Av = 10.*40.
#Iv = 10.*40.**3/12.
##resultante equivalente Rvi e centróide xvi
#xA = 0
#xB = 470
#vi = 0.0046587 * r**(0.2) #reescrevendo para considerar a origem na extremidade do elemento!
#Rvi = sp.integrate(vi, (r, xA, xB))
#xvi = sp.integrate(vi*r, (r, xA, xB))/Rvi
##reações de apoio
#RA = Rvi
#MRA = Rvi*xvi
#
##força resultante da carga de vento na seção e centroíde
#Rvix = sp.integrate(vi, (r, xA, x))
#xvix = sp.integrate(vi*r, (r, xA, x))/Rvix
##momento na seção do vão
#Ms = sp.expand(RA*r - MRA - Rvix*(r - xvix))
##rotações da viga
#dMsdx = sp.integrate(Ms, r)
##deflexões
#w = sp.integrate(dMsdx, r)/(Ev*Iv) !!!!!!!!!!!!!!!!!!!!!!!!!ERRADO!!
#dWdx = sp.diff(w, r)
#
#wEX = w.subs({r: Lv}).evalf()
#dWdxEX = dWdx.subs({r: Lv}).evalf()
#
##matriz de rigidez dos elementos 2 nós
#Kevb = rigidez_portico(Ev, Av, Iv, [0, 1, Lv]) #kN/cm2, cm2 e cm4
#Kuvb = Kevb[3:, 3:]
#FeviP = np.array([0, Fevi[0], Fevi[1], 0, Fevi[2], Fevi[3]], dtype=float)
#Fuvb = -FeviP[3:]
#Uvb = np.linalg.solve(Kuvb, Fuvb)
#
##comparativo
#print('w', wEX, 'Uv', Uvb[1])
#print('dWdx', dWdxEX, 'Rv', Uvb[2])

#Determinação das demais cargas como pórtico (1 já com rotação)---------------------------------------------------------------------------------------
Fe3 = np.zeros(6)
Feq = np.array([0, Feq[0], Feq[1], 0, Feq[2], Feq[3]], dtype=float)
Feg = np.array([0, Feg[0], Feg[1], 0, Feg[2], Feg[3]], dtype=float)
Fe2 = Feq + Feg
RFv = np.array([[0, -1, 0, 0,  0, 0], 
                [1,  0, 0, 0,  0, 0], 
                [0,  0, 1, 0,  0, 0],
                [0 , 0 ,0 ,0, -1, 0],
                [0,  0, 0, 1,  0, 0],
                [0,  0, 0, 0,  0, 1]])
Fevi = np.array([0, Fevi[0], Fevi[1], 0, Fevi[2], Fevi[3]], dtype=float)
Fe1 = np.matmul( RFv, Fevi )

#Determinação do vetor de cargas nodais equivalentes para cálculo dos deslocamentos
FU = np.array([Fe1[3] + Fe2[0], Fe1[4] + Fe2[1], Fe1[5] + Fe2[2], Fe2[3], Fe2[4], Fe2[5]], dtype=float)
FR = np.array([Fe1[0], Fe1[1], Fe1[2], 0, 0, 0], dtype=float)

#determinação dos deslocamentos
Un = np.linalg.solve(Ku, FU)
R = np.dot(Kr, Un) - FR

U = np.zeros(12)
U[:6] = Un

#reescrevendo os deslocamentos no sistema local do elemento
ug1 = np.zeros(6)
ug2 = np.zeros(6)
ug3 = np.zeros(6)

for i in range(12):
    if C[0, i] >= 0:
        ug1[ C[0, i] ] = U[i]
    if C[1, i] >= 0:
        ug2[ C[1, i] ] = U[i]
    if C[2, i] >= 0:
        ug3[ C[2, i] ] = U[i]

R13 = np.array([[ 0, 1, 0,  0, 0, 0], 
                [-1, 0, 0,  0, 0, 0], 
                [ 0, 0, 1,  0, 0, 0],
                [ 0 ,0 ,0 , 0, 1, 0],
                [ 0, 0, 0, -1, 0, 0],
                [ 0, 0, 0,  0, 0, 1],])

u1 = np.dot(R13, ug1)
u2 = ug2
u3 = np.dot(R13, ug3)

#matriz das derivadas das funções de interpolação do pórtico
Bv = -s * sp.diff( sp.diff(Nn, r), r)
Bp = sp.Matrix([[ -1/l, 0, 0, 1/l, 0, 0 ], [ 0,  Bv[0], Bv[1], 0, Bv[2], Bv[3] ] ])

#Nnp = sp.Matrix([ r - 1/l, Nn[0], Nn[1], r + 1/l, Nn[2], Nn[3] ])
#Bv1 = sp.diff(Nn, r)
#Bp1 = sp.Matrix([ -1/l,  Bv1[0], Bv1[1], 1/l, Bv1[2], Bv1[3] ])
#Bv2 = sp.diff(Bv1, r)
#Bp2 = sp.Matrix([ -1/l,  Bv2[0], Bv2[1], 1/l, Bv2[2], Bv2[3] ])
#Bv3 = sp.diff(Bv2, r)
#Usym = sp.MatrixSymbol('U', 6, 1)
#UMsym = sp.Matrix(Usym)
#UMsymV = UMsym[[1,2,4,5],:]
#
#deslocamentosS = Nnp.T * UMsym
#deformacoesS = Bp1.T * UMsym
##tensoesS = deformacoesS * E
#rotacoesS = Bv1.T * UMsymV
#momentoS = Bv2.T * UMsymV
#cortanteS = Bv3.T * UMsymV

##calculo das deformações, tensões, momento, corte e normal em cada elemento no eixo local ------------------------------------------------------------
#def esP(U, l, E, A, h, I, pontos=100):
#    r = np.linspace(-l/2, l/2, pontos)
#    U = U[:, np.newaxis]
#    deslocamentos = (r - 1/l)*U[0, 0] + (r + 1/l)*U[3, 0] + (1/2 - 3*r/(2*l) + 2*r**3/l**3)*U[1, 0] + (1/2 + 3*r/(2*l) - 2*r**3/l**3)*U[4, 0] + (-l/8 - r/4 + r**2/(2*l) + r**3/l**2)*U[5, 0] + (l/8 - r/4 - r**2/(2*l) + r**3/l**2)*U[2, 0]
#    rotacoes = (-3/(2*l) + 6*r**2/l**3)*U[1, 0] + (3/(2*l) - 6*r**2/l**3)*U[4, 0] + (-1/4 - r/l + 3*r**2/l**2)*U[2, 0] + (-1/4 + r/l + 3*r**2/l**2)*U[5, 0]
#    momento = (E * I) * ( (-1/l + 6*r/l**2)*U[2, 0] + (1/l + 6*r/l**2)*U[5, 0] + 12*r*U[1, 0]/l**3 - 12*r*U[4, 0]/l**3 )
#    cortante = (E * I) * ( 6*U[2, 0]/l**2 + 6*U[5, 0]/l**2 + 12*U[1, 0]/l**3 - 12*U[4, 0]/l**3 )*np.ones(pontos)
#    normal = (E * A) * ( U[0,0]*(- 1/l) + U[3, 0]*(1/l) )*np.ones(pontos)
#    
#    #aborgadem reversa
#    tensoes = normal/A + momento/I * h/2
#    deformacoes = tensoes/E
#    
#    return deslocamentos, rotacoes, deformacoes, tensoes, momento, cortante, normal, r
#    
#E = 20000. #kN/cm2
#deslocamentos1, rotacoes1, deformacoes1, tensoes1, momentos1, corte1, normal1, varElem1 = esP(u1, scL[0, 2], E, Ar, d, I_z)
#deslocamentos2, rotacoes2, deformacoes2, tensoes2, momentos2, corte2, normal2, varElem2 = esP(u2, scL[1, 2], E, Ar, d, I_z)
#deslocamentos3, rotacoes3, deformacoes3, tensoes3, momentos3, corte3, normal3, varElem3 = esP(u3, scL[2, 2], E, Ar, d, I_z)


#deformações nos elementos
epsilon_1 = Bp.subs({l: 470}) * u1[:, np.newaxis]
epsilonA_1 = epsilon_1[0]
epsilonF_1 = epsilon_1[1]

epsilon_2 = Bp.subs({l: 940}) * u2[:, np.newaxis]
epsilonA_2 = epsilon_2[0]
epsilonF_2 = epsilon_2[1]

epsilon_3 = Bp.subs({l: 470}) * u3[:, np.newaxis]
epsilonA_3 = epsilon_3[0]
epsilonF_3 = epsilon_3[1]

#tensões nos elementos
E = 20000. #kN/cm2

sigmaA_1 = E*epsilonA_1
sigmaF_1 = E*epsilonF_1
sigmaA_2 = E*epsilonA_2
sigmaF_2 = E*epsilonF_2
sigmaA_3 = E*epsilonA_3
sigmaF_3 = E*epsilonF_3

#tensões axiais
Ap = 143.75 #cm2
N_1 = Ap * sigmaA_1
N_2 = Ap * sigmaA_2
N_3 = Ap * sigmaA_3

#momentos fletores nas barras
M1 = 2 * t_w * sp.integrate( s * sigmaF_1, (s, -h/2, h/2 ) ) + 2 * b_f * sp.integrate( s * sigmaF_1, (s, h/2, h/2 + t_f ) )
M2 = 2 * t_w * sp.integrate( s * sigmaF_2, (s, -h/2, h/2 ) ) + 2 * b_f * sp.integrate( s * sigmaF_2, (s, h/2, h/2 + t_f ) )
M3 = 2 * t_w * sp.integrate( s * sigmaF_3, (s, -h/2, h/2 ) ) + 2 * b_f * sp.integrate( s * sigmaF_3, (s, h/2, h/2 + t_f ) )

#esforço cortante ---------------------------------------------------------------------------------------------------
V1 = sp.diff(M1, r)
V2 = sp.diff(M2, r)
V3 = sp.diff(M3, r)

#grafico dos deslocamentos, normais, momento e cortante
#funcoes de forma de treliça e viga
Nt = sp.Matrix([1/2 - r/l, r/l])
Np = Nn

u1t = np.array([u1[0], u1[3]])
u1p = np.array([u1[1], u1[2], u1[4], u1[5]])
u2t = np.array([u2[0], u2[3]])
u2p = np.array([u2[1], u2[2], u2[4], u2[5]])
u3t = np.array([u3[0], u3[3]])
u3p = np.array([u3[1], u3[2], u3[4], u3[5]])

u1Nt = Nt.T*u1t[:, np.newaxis]
u1Np = Np.T*u1p[:, np.newaxis]
u2Nt = Nt.T*u2t[:, np.newaxis]
u2Np = Np.T*u2p[:, np.newaxis]
u3Nt = Nt.T*u3t[:, np.newaxis]
u3Np = Np.T*u3p[:, np.newaxis]


#convertendo para função python
u1Nt = sp.utilities.lambdify([r, l], u1Nt[0], "numpy")
u1Np = sp.utilities.lambdify([r, l], u1Np[0], "numpy")
u2Nt = sp.utilities.lambdify([r, l], u2Nt[0], "numpy")
u2Np = sp.utilities.lambdify([r, l], u2Np[0], "numpy")
u3Nt = sp.utilities.lambdify([r, l], u3Nt[0], "numpy")
u3Np = sp.utilities.lambdify([r, l], u3Np[0], "numpy")

Y = np.linspace(-235, 235, 100)
X = np.linspace(-470, 470, 100)

##gráfico dos deslocamentos !!!!!!!!!!!!!!!!!!! MUITO MAL FEITO!!!!!!!!!!!!!!!!!
#escala = 1000
#plt.plot([0, 0, 920, 920], [0, 470, 470, 0], color="gray") #elementos
#plt.scatter([0, 0, 920, 920], [0, 470, 470, 0], s=15, color="gray") #nós
#plt.plot(-u1Np(Y, 470)*escala, u1Nt(Y, 470)*escala + Y + 235, '--', color='blue')
#plt.plot(u2Nt(X, 920)*escala + X - u1Np(Y, 470)[-1]*escala/2 + 470, u2Np(X, 920)*escala + 470, '--', color='blue')
#plt.plot(-u3Np(Y, 470)*escala + 920, u3Nt(Y, 470) + Y + 235, '--', color='blue')
#plt.yticks(np.arange(0, 520, step=20))
#plt.show()

#esforço normal
escala_n = 7
plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
plt.plot(np.ones(100)*N_1*escala_n - 470, Y)
plt.plot(X, np.ones(100)*N_2*escala_n + 235)
plt.plot(np.ones(100)*N_3*escala_n + 470, Y)
plt.show()

#esforço cortante
escala_v = 30
plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
plt.plot(np.ones(100)*V1*escala_v - 470, Y)
plt.plot(X, np.ones(100)*V2*escala_v + 235)
plt.plot(np.ones(100)*V3*escala_v + 470, Y)
plt.show()

#momento fletor
M1f = sp.utilities.lambdify([r], M1, "numpy")
M2f = sp.utilities.lambdify([r], M2, "numpy")
M3f = sp.utilities.lambdify([r], M3, "numpy")
escala_m = 0.1
plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
plt.plot(-M1f(Y)*escala_m - 470, Y)
plt.plot(X, M2f(X)*escala_m + 235)
plt.plot(-M3f(Y)*escala_m + 470, Y)
plt.show()

###com as funções de forma ----------------------------------------------------------------------------------
#escala_v = 20.
#plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
#plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
#plt.plot(-normal1*escala_v - 470, varElem1)
#plt.plot(varElem2, normal2*escala_v + 235)
#plt.plot(-normal3*escala_v + 470, varElem3)
#plt.show()
#
#escala_v = 20.
#plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
#plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
#plt.plot(-corte1*escala_v - 470, varElem1)
#plt.plot(varElem2, corte2*escala_v + 235)
#plt.plot(-corte3*escala_v + 470, varElem3)
#plt.show()
#
#escala_v = 0.1
#plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
#plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
#plt.plot(-momentos1*escala_v - 470, varElem1)
#plt.plot(varElem2, momentos2*escala_v + 235)
#plt.plot(-momentos3*escala_v + 470, varElem3)
#plt.show()