#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Que M?!?!?!?
Não é porcentagem!!! ERRADO!!!

Nada feito ainda!!!

@author: markinho
"""

import sympy as sp
import numpy as np
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
import meshio

#### DADOS DE ENTRADA ------------------------------------------------------------------------
nos = np.array([ [0, 0], [0, 5.], [1., 5.], [4., 5.], [8., 5.], [8., 0.] ], dtype=float)
IE = np.array([ [0, 1], [1, 2], [2, 3], [3, 4], [5, 4] ], dtype=int)

#nos com restrição
nosR = np.array([0, 5])
#graus de liberdade restringidos em cada nó: 1 = restringido; 0 = livre
nosR_GL = np.array([[1, 1, 1],
                    [1, 1, 1]])

#secao retangular
b = 0.1 #m
h = 0.4 #m

#Rigidezes dos elementos
El = np.ones(IE.shape[0])*200000. #kPa
#Áreas das seções transversais dos elementos
Ar = np.ones(IE.shape[0])*b*h #m2
#Inércias das seções transversais dos elementos
Iz = np.ones(IE.shape[0])*b*h**3/12 #m4

#coeficientes de semi-rigidez (0 - engaste, 1 - livre)
alphas = np.zeros((IE.shape[0], 2)) #nó inicial e nó final
alphas[1, 0] = 0.
alphas[3, 1] = 0.

#carregamentos distribuidos constantes nos elementos na direção x, y global!!
forcas = np.zeros((IE.shape[0], 2))
forcas[0, 0] = 3. #kN/m
forcas[2, 1] = -5. #kN/m

def rigidez_portico_semi(E, A, I_z, scL, alpha):
    '''
    Função para a deterrminação das matrizes de rigidez do elemento de pórtico
    '''
    s = scL[0]
    c = scL[1]
    L = scL[2]
    ai = alpha[0]
    aj = alpha[1]
    return np.array([[ A*E*c**2/L + 12*E*I_z*s**2*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)),    A*E*c*s/L - 12*E*I_z*c*s*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)), -6*E*I_z*s*(2*aj + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1)), -A*E*c**2/L - 12*E*I_z*s**2*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)),   -A*E*c*s/L + 12*E*I_z*c*s*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)), -6*E*I_z*s*(2*ai + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1))],
                    [   A*E*c*s/L - 12*E*I_z*c*s*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)),  A*E*s**2/L + 12*E*I_z*c**2*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)),  6*E*I_z*c*(2*aj + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1)),   -A*E*c*s/L + 12*E*I_z*c*s*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)), -A*E*s**2/L - 12*E*I_z*c**2*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)),  6*E*I_z*c*(2*ai + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1))],
                    [                     -6*E*I_z*s*(2*aj + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1)),                       6*E*I_z*c*(2*aj + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1)),       4*E*I_z*(3*aj + 1)/(L*(12*ai*aj + 4*ai + 4*aj + 1)),                       6*E*I_z*s*(2*aj + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1)),                      -6*E*I_z*c*(2*aj + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1)),                  2*E*I_z/(L*(12*ai*aj + 4*ai + 4*aj + 1))],
                    [-A*E*c**2/L - 12*E*I_z*s**2*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)),   -A*E*c*s/L + 12*E*I_z*c*s*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)),  6*E*I_z*s*(2*aj + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1)),  A*E*c**2/L + 12*E*I_z*s**2*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)),    A*E*c*s/L - 12*E*I_z*c*s*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)),  6*E*I_z*s*(2*ai + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1))],
                    [  -A*E*c*s/L + 12*E*I_z*c*s*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)), -A*E*s**2/L - 12*E*I_z*c**2*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)), -6*E*I_z*c*(2*aj + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1)),    A*E*c*s/L - 12*E*I_z*c*s*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)),  A*E*s**2/L + 12*E*I_z*c**2*(ai + aj + 1)/(L**3*(12*ai*aj + 4*ai + 4*aj + 1)), -6*E*I_z*c*(2*ai + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1))],
                    [                     -6*E*I_z*s*(2*ai + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1)),                       6*E*I_z*c*(2*ai + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1)),                  2*E*I_z/(L*(12*ai*aj + 4*ai + 4*aj + 1)),                       6*E*I_z*s*(2*ai + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1)),                      -6*E*I_z*c*(2*ai + 1)/(L**2*(12*ai*aj + 4*ai + 4*aj + 1)),       4*E*I_z*(3*ai + 1)/(L*(12*ai*aj + 4*ai + 4*aj + 1))]])

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

#determinação dos ângulos e comprimentos
scL = angulos_comprimentos(nos, IE)


#matriz de rigidez dos elementos
Ke = []
for e in range(IE.shape[0]):
    Ke.append(rigidez_portico_semi(El[e], Ar[e], Iz[e], scL[e], alphas[e]))
#    Ke.append(rigidez_portico(El[e], Ar[e], Iz[e], scL[e]))
Ke = np.array(Ke)

#graus de liberdade da estrutura totais, GL
GL = nos.shape[0]*3 #3 número de graus de liberdade por nó

#indexação do elemento
ID = []
for e in range(IE.shape[0]):
    ID.append( np.repeat(IE[e]*3, 3) + np.tile(np.array([0, 1, 2]), 2) )
ID = np.array(ID)

#montagem da matriz de rigidez da estrutura
K = np.zeros((GL, GL))
for i in range(6):
    for j in range(6):
        for e in range(IE.shape[0]):
            K[ ID[e, i], ID[e, j] ] += Ke[e, i, j]


#graus de liberdade restringidos
GL_R = []
for n in nosR:
    GL_R.append( np.repeat(n*3, 3) + np.array([0, 1, 2]) )
GL_R = np.extract( nosR_GL.flatten(), np.array(GL_R).flatten() )

#graus de liberdade livres
DOF = np.delete(np.arange(0, nos.shape[0]*3), GL_R)

#separação das matrizes de rigidez
Ku = np.delete(np.delete(K, GL_R, axis=0), GL_R, axis=1)
Kr = np.delete(np.delete(K, DOF, axis=0), GL_R, axis=1)

#cargas distribuídas constantes no sistema global
def cdg(scL, forcas, alpha):
    '''
    Cálculo das forças nodais equivalentes a uma carga distribuída constante ao elemento de pórtico.
    '''
    s = scL[0]
    c = scL[1]
    L = scL[2]
    ai = alpha[0]
    aj = alpha[1]
    gx = forcas[0]
    gy = forcas[1]
    M = np.array([[1,0,0,1,0,0],
                  [0,2*(1+2*ai)/L**2/(1+4*aj+4*ai+12*ai*aj),(1+2*aj)/L/(1+4*aj+4*ai+12*ai*aj),0,-2*(1+2*ai)/L**2/(1+4*aj+4*ai+12*ai*aj),(1-2*aj+4*ai)/L/(1+4*aj+4*ai+12*ai*aj)],
                  [0,(1+2*aj)/L/(1+4*aj+4*ai+12*ai*aj),4/3*(1+3*aj)/(1+4*aj+4*ai+12*ai*aj)-1/3/(1+4*aj+4*ai+12*ai*aj),0,-(1+2*aj)/L/(1+4*aj+4*ai+12*ai*aj),-2/3*(1+3*aj)/(1+4*aj+4*ai+12*ai*aj)+2/3/(1+4*aj+4*ai+12*ai*aj)],
                  [1,0,0,1,0,0],
                  [0,-2*(1+2*ai)/L**2/(1+4*aj+4*ai+12*ai*aj),-(1+2*aj)/L/(1+4*aj+4*ai+12*ai*aj),0,2*(1+2*ai)/L**2/(1+4*aj+4*ai+12*ai*aj),-(1-2*aj+4*ai)/L/(1+4*aj+4*ai+12*ai*aj)],
                  [0,(1-2*aj+4*ai)/L/(1+4*aj+4*ai+12*ai*aj),-2/3*(1+3*aj)/(1+4*aj+4*ai+12*ai*aj)+2/3/(1+4*aj+4*ai+12*ai*aj),0,-(1-2*aj+4*ai)/L/(1+4*aj+4*ai+12*ai*aj),-1/3/(1+4*aj+4*ai+12*ai*aj)+4/3*(1+3*ai)/(1+4*aj+4*ai+12*ai*aj)]])
    print(M)
    fe = np.array([0.5*L*c*(c*gx + gy*s) - L*s*(c*gy - gx*s)*0.5,
                     L*c*(c*gy - gx*s)/2 + 0.5*L*s*(c*gx + gy*s),
                     L**2*(c*gy - gx*s)*0.08333333333333333,
                     0.5*L*c*(c*gx + gy*s) - L*s*(c*gy - gx*s)*0.5,
                     L*c*(c*gy - gx*s)/2 + 0.5*L*s*(c*gx + gy*s),
                     -L**2*(c*gy - gx*s)*0.08333333333333333])
    return np.matmul(M, fe)





#cálculo do vetor de cargas nodais equivalentes às distribuídas
Feq = np.zeros(GL)
for e in range(IE.shape[0]):
    F = cdg(scL[e], forcas[e], alphas[e])
    for i in range(6):
        Feq[ ID[e][i] ] += F[i]

FU = np.delete(Feq, GL_R, axis=0)
FR = np.delete(Feq, DOF, axis=0)

#determinação dos deslocamentos
Un = np.linalg.solve(Ku, FU)
R = np.dot(Kr, Un) - FR

U = np.zeros(GL)
U[DOF] = Un

Uxyr = U.reshape((nos.shape[0],3))

#reescrevendo os deslocamentos no sistema local do elemento
u = np.zeros((IE.shape[0], 6))

for e in range(IE.shape[0]):
    s = scL[e, 0]
    c = scL[e, 1]
    rotacao = np.array([[ c, s, 0,  0, 0, 0], 
                        [-s, c, 0,  0, 0, 0], 
                        [ 0, 0, 1,  0, 0, 0],
                        [ 0 ,0 ,0 , c, s, 0],
                        [ 0, 0, 0, -s, c, 0],
                        [ 0, 0, 0,  0, 0, 1]])
    u[e] = np.matmul( U[ ID[e] ], rotacao)

#calculo das deformações, tensões, momento, corte e normal em cada elemento no eixo local ------------------------------------------------------------
def esP(ue, l, E, A, h, I, pontos=100):
    r = np.linspace(-l/2, l/2, pontos)
    ue = ue[:, np.newaxis]
    deslocamentos = (r - 1/l)*ue[0, 0] + (r + 1/l)*ue[3, 0] + (1/2 - 3*r/(2*l) + 2*r**3/l**3)*ue[1, 0] + (1/2 + 3*r/(2*l) - 2*r**3/l**3)*ue[4, 0] + (-l/8 - r/4 + r**2/(2*l) + r**3/l**2)*ue[5, 0] + (l/8 - r/4 - r**2/(2*l) + r**3/l**2)*ue[2, 0]
    rotacoes = (-3/(2*l) + 6*r**2/l**3)*ue[1, 0] + (3/(2*l) - 6*r**2/l**3)*ue[4, 0] + (-1/4 - r/l + 3*r**2/l**2)*ue[2, 0] + (-1/4 + r/l + 3*r**2/l**2)*ue[5, 0]
    momento = (E * I) * ( (-1/l + 6*r/l**2)*ue[2, 0] + (1/l + 6*r/l**2)*ue[5, 0] + 12*r*ue[1, 0]/l**3 - 12*r*ue[4, 0]/l**3 )
    cortante = (E * I) * ( 6*ue[2, 0]/l**2 + 6*ue[5, 0]/l**2 + 12*ue[1, 0]/l**3 - 12*ue[4, 0]/l**3 )*np.ones(pontos)
    normal = (E * A) * ( ue[0,0]*(- 1/l) + ue[3, 0]*(1/l) )*np.ones(pontos)
    
    #aborgadem reversa
    tensoes = normal/A + momento/I * h/2
    deformacoes = tensoes/E
    
    return deslocamentos, rotacoes, deformacoes, tensoes, momento, cortante, normal, r

resultados = []
for e in range(IE.shape[0]):
    resultados.append( esP(u[e], scL[e, 2], El[e], Ar[e], h, Iz[e]) )
resultados = np.array(resultados) #elementos, quantidade de resultados, número de pontos para o gráfico

##grafico dos resultados ----------------------------------------------------------------------------------------------------------------------------
#Os resultados para cada elemento são: deslocamentos - 0, rotacoes - 1, deformacoes - 2, tensoes - 3, momento - 4, cortante - 5, normal - 6, r - 7
#plt.plot(resultados[0, 0], resultados[0, 7]+scL[0, 2]/2)
#plt.plot(resultados[1, 0]+0.5+resultados[0, 0, -1], resultados[1, 7]+5)
#plt.plot(resultados[2, 0]+3.5, resultados[2, 7]+5)
#plt.plot(resultados[3, 0]+6.5, resultados[3, 7]+5)
#plt.plot(resultados[4, 0]+8, resultados[4, 7]+scL[4, 2]/2)



##geração do arquivo vtu ???
#pontos = nos
#celulas = {'line': IE}
#meshio.write_points_cells(
#        "resultadosPoericoSemi.vtu",
#        pontos,
#        celulas,
#        # Optionally provide extra data on points, cells, etc.
#        point_data = {"U": Uxyr},
#        # cell_data=cell_data,
#        # field_data=field_data
#        )





















##funcoes de forma de treliça e viga
#Nt = sp.Matrix([1/2 - r/l, r/l])
#Np = Nn
#
#u1t = np.array([u1[0], u1[3]])
#u1p = np.array([u1[1], u1[2], u1[4], u1[5]])
#u2t = np.array([u2[0], u2[3]])
#u2p = np.array([u2[1], u2[2], u2[4], u2[5]])
#u3t = np.array([u3[0], u3[3]])
#u3p = np.array([u3[1], u3[2], u3[4], u3[5]])
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
#
#Y = np.linspace(-235, 235, 100)
#X = np.linspace(-470, 470, 100)
#
###gráfico dos deslocamentos !!!!!!!!!!!!!!!!!!! MUITO MAL FEITO!!!!!!!!!!!!!!!!!
##escala = 1000
##plt.plot([0, 0, 920, 920], [0, 470, 470, 0], color="gray") #elementos
##plt.scatter([0, 0, 920, 920], [0, 470, 470, 0], s=15, color="gray") #nós
##plt.plot(-u1Np(Y, 470)*escala, u1Nt(Y, 470)*escala + Y + 235, '--', color='blue')
##plt.plot(u2Nt(X, 920)*escala + X - u1Np(Y, 470)[-1]*escala/2 + 470, u2Np(X, 920)*escala + 470, '--', color='blue')
##plt.plot(-u3Np(Y, 470)*escala + 920, u3Nt(Y, 470) + Y + 235, '--', color='blue')
##plt.yticks(np.arange(0, 520, step=20))
##plt.show()
#
##esforço normal
#escala_n = 7
#plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
#plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
#plt.plot(np.ones(100)*N_1*escala_n - 470, Y)
#plt.plot(X, np.ones(100)*N_2*escala_n + 235)
#plt.plot(np.ones(100)*N_3*escala_n + 470, Y)
#plt.show()
#
##esforço cortante
#escala_v = 30
#plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
#plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
#plt.plot(np.ones(100)*V1*escala_v - 470, Y)
#plt.plot(X, np.ones(100)*V2*escala_v + 235)
#plt.plot(np.ones(100)*V3*escala_v + 470, Y)
#plt.show()
#
##momento fletor
#M1f = sp.utilities.lambdify([r], M1, "numpy")
#M2f = sp.utilities.lambdify([r], M2, "numpy")
#M3f = sp.utilities.lambdify([r], M3, "numpy")
#escala_m = 0.1
#plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
#plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
#plt.plot(-M1f(Y)*escala_m - 470, Y)
#plt.plot(X, M2f(X)*escala_m + 235)
#plt.plot(-M3f(Y)*escala_m + 470, Y)
#plt.show()
#
####com as funções de forma ----------------------------------------------------------------------------------
##escala_v = 20.
##plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
##plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
##plt.plot(-normal1*escala_v - 470, varElem1)
##plt.plot(varElem2, normal2*escala_v + 235)
##plt.plot(-normal3*escala_v + 470, varElem3)
##plt.show()
##
##escala_v = 20.
##plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
##plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
##plt.plot(-corte1*escala_v - 470, varElem1)
##plt.plot(varElem2, corte2*escala_v + 235)
##plt.plot(-corte3*escala_v + 470, varElem3)
##plt.show()
##
##escala_v = 0.1
##plt.plot([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], color="gray") #elementos
##plt.scatter([-scL[0,2], -scL[0,2], scL[0,2], scL[0,2]], [-235, 235, 235, -235], s=15, color="gray") #nós
##plt.plot(-momentos1*escala_v - 470, varElem1)
##plt.plot(varElem2, momentos2*escala_v + 235)
##plt.plot(-momentos3*escala_v + 470, varElem3)
##plt.show()