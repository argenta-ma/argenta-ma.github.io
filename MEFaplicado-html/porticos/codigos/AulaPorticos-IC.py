#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 08:28:59 2021

@author: argenta
"""

import numpy as np

def rigidez_portico(E, A, I_z, scL):
    '''
    Função para a deterrminação das matrizes de rigidez do elemento de pórtico
    '''
    s = scL[:,0]
    c = scL[:,1]
    L = scL[:,2]
    return np.array([[A * E * c**2/L + 12 * E * I_z * s**2/L**3, A * E * s * c/L - 12 * E * I_z * s * c/L**3,  - 6 * E * I_z * s/L**2,  - A * E * c**2/L - 12 * E * I_z * s**2/L**3,  - A * E * s * c/L + 12 * E * I_z * s * c/L**3,  - 6 * E * I_z * s/L**2 ],
                    [A * E * s * c/L - 12 * E * I_z * s * c/L**3, A * E * s**2/L + 12 * E * I_z * c**2/L**3, 6 * E * I_z * c/L**2,  - A * E * s * c/L + 12 * E * I_z * s * c/L**3,  - A * E * s**2/L - 12 * E * I_z * c**2/L**3, 6 * E * I_z * c/L**2 ],
                    [ - 6 * E * I_z * s/L**2, 6 * E * I_z * c/L**2, 4 * E * I_z/L  , 6 * E * I_z * s/L**2,  - 6 * E * I_z * c/L**2, 2 * E * I_z/L ],
                    [ - A * E * c**2/L - 12 * E * I_z * s**2/L**3,  - A * E * s * c/L + 12 * E * I_z * s * c/L**3, 6 * E * I_z * s/L**2, A * E * c**2/L + 12 * E * I_z * s**2/L**3, A * E * s * c/L - 12 * E * I_z * s * c/L**3, 6 * E * I_z * s/L**2],
                    [ - A * E * s * c/L + 12 * E * I_z * s * c/L**3,  - A * E * s**2/L - 12 * E * I_z * c**2/L**3,  - 6 * E * I_z * c/L**2, A * E * s * c/L - 12 * E * I_z * s * c/L**3, A * E * s**2/L + 12 * E * I_z * c**2/L**3,  - 6 * E * I_z * c/L**2],
                    [ - 6 * E * I_z * s/L**2, 6 * E * I_z * c/L**2, 2 * E * I_z/L, 6 * E * I_z * s/L**2,  - 6 * E * I_z * c/L**2, 4 * E * I_z/L ]])

def angulosComprimentos(cNos, iElem):
    '''
    Função para calcular os senos e cossenos e comprimentos de elementos
    '''
    scc = np.zeros( (iElem.shape[0], 3) )
    noI = cNos[ iElem[:,0] ]
    noF = cNos[ iElem[:,1] ]
    scc[:,2] = np.sqrt( (noF[:,0] - noI[:,0])**2 + (noF[:,1] - noI[:,1])**2 )
    scc[:,0] = (noF[:,1] - noI[:,1])/scc[:,2]
    scc[:,1] = (noF[:,0] - noI[:,0])/scc[:,2]
    return scc

#matriz dos graus de liberdade da estrutura
GL = np.array([[ 6,  7,  8], #no1
               [ 0,  1,  2], #no2
               [ 3,  4,  5], #no3
               [ 9, 10, 11]]) #no4
GLsLivres = 6

#matriz de incidencia dos elementos
IE = np.array([[0, 1], #elem1
               [1, 2], #elem2
               [3, 2]]) #elem3

#coordenadas dos nós da estrutura
NOS = np.array([[-47., 0.], #coord no1
                [-47., 47.], #coord no2
                [47., 47.], #coord no3
                [47., 0.]]) #coord no4

#propriedades dos materiais
Ec = 12000. #KN/cm2 MPA - KN/cm2 *0.1
largura = 0.4 #cm
altura = 0.4 #cm
espessura = 0.2 #cm
Ab = largura*altura - (largura - espessura)*(altura - espessura) #cm2
Iz = largura*altura**3/12 - (largura - espessura)*(altura - espessura)**3/12 #cm4

#indexação dos elementos
ID = np.zeros( (6, IE.shape[0]), dtype=int )
for b in range(IE.shape[0]):
    ID[:,b] = np.array([GL[ IE[b,0], 0 ], GL[ IE[b,0], 1 ], GL[ IE[b,0], 2 ], GL[ IE[b,1], 0 ], GL[ IE[b,1], 1 ], GL[ IE[b,1], 2 ]], dtype=int)

#criação das matrizes dos elementos
scL = angulosComprimentos(NOS, IE)
kes = rigidez_portico(Ec, Ab, Iz, scL)
K = np.zeros( (GL.max() + 1, GL.max() + 1) )

for i in range(kes.shape[0]):
    for j in range(kes.shape[1]):
        for e in range(IE.shape[0]):
            K[ ID[i,e], ID[j,e] ] += kes[i, j, e]

Ku = K[:GLsLivres,:GLsLivres]

#vetor de forças nodais da estrutura
F = np.zeros((GL.max() + 1)) #em kN
F[0] = 0.1 #kN
F[1] = 0.3 #kN
F[3] = -0.1 #kN
F[4] = 0.3 #kN

Fu = F[:GLsLivres]

#calcular os deslocamentos
U = np.linalg.solve(Ku, Fu)

#deslocamentos completos de todos os graus de liberdade
Uc = np.zeros(GL.max() + 1)
Uc[:GLsLivres] = U

#deslocamentos dos elementos
uee = np.zeros((6, IE.shape[0]))
for b in range(IE.shape[0]):
    uee[:, b] = Uc[ ID[:, b] ]

#rotação dos graus de liberdade dos elementos na estrutura para os elementos isolados
ue = np.zeros((6, IE.shape[0]))
for b in range(IE.shape[0]):
    R = np.array([[ scL[b,1], scL[b,0], 0,        0,        0, 0],
                  [-scL[b,0], scL[b,1], 0,        0,        0, 0],
                  [        0,        0, 1,        0,        0, 0],
                  [        0,        0, 0, scL[b,1], scL[b,0], 0],
                  [        0,        0, 0,-scL[b,0], scL[b,1], 0],
                  [        0,        0, 0,        0,        0, 1]])
    ue[:,b] = np.matmul(R, uee[:, b])

#esforços nas barras
def esforcos(u, l, Ab, Ec):
    '''
    Cálculo dos esforços nas barras
    '''
    u0 = u[0]
    u1 = u[1]
    u2 = u[2]
    u3 = u[3]
    u4 = u[4]
    u5 = u[5]
    
    N = Ab*Ec*(-u0/l + u3/l)
    r = -0.5*l
    M1 = 0.166666666666667*altura**3*espessura*(Ec*l**2*u2 - Ec*l**2*u5 - 6*Ec*l*r*u2 - 6*Ec*l*r*u5 - 12*Ec*r*u1 + 12*Ec*r*u4)/l**3 + 2*largura*(-0.0416666666666667*altura**3*(Ec*l**2*u2 - Ec*l**2*u5 - 6*Ec*l*r*u2 - 6*Ec*l*r*u5 - 12*Ec*r*u1 + 12*Ec*r*u4)/l**3 + (0.5*altura + espessura)**3*(Ec*l**2*u2 - Ec*l**2*u5 - 6*Ec*l*r*u2 - 6*Ec*l*r*u5 - 12*Ec*r*u1 + 12*Ec*r*u4)/(3*l**3))
    r = 0.5*l
    M2 = 0.166666666666667*altura**3*espessura*(Ec*l**2*u2 - Ec*l**2*u5 - 6*Ec*l*r*u2 - 6*Ec*l*r*u5 - 12*Ec*r*u1 + 12*Ec*r*u4)/l**3 + 2*largura*(-0.0416666666666667*altura**3*(Ec*l**2*u2 - Ec*l**2*u5 - 6*Ec*l*r*u2 - 6*Ec*l*r*u5 - 12*Ec*r*u1 + 12*Ec*r*u4)/l**3 + (0.5*altura + espessura)**3*(Ec*l**2*u2 - Ec*l**2*u5 - 6*Ec*l*r*u2 - 6*Ec*l*r*u5 - 12*Ec*r*u1 + 12*Ec*r*u4)/(3*l**3))
    M = np.max([abs(M1), abs(M2)])
    Q = 0.166666666666667*altura**3*espessura*(-6*Ec*l*u2 - 6*Ec*l*u5 - 12*Ec*u1 + 12*Ec*u4)/l**3 + 2*largura*(-0.0416666666666667*altura**3*(-6*Ec*l*u2 - 6*Ec*l*u5 - 12*Ec*u1 + 12*Ec*u4)/l**3 + (0.5*altura + espessura)**3*(-6*Ec*l*u2 - 6*Ec*l*u5 - 12*Ec*u1 + 12*Ec*u4)/(3*l**3))
    return N, M, Q

esforcos_e = np.zeros((3, IE.shape[0]))
for b in range(IE.shape[0]):
    esforcos_e[:,b] = esforcos(ue[:,b], scL[b,2], Ab, Ec)