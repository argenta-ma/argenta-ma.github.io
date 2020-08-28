#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Pórticos espaciais.

Nada feito ainda!!

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
nos = np.array([ [0, 0, 2], [0, 3, 2], [4, 3, 2], [4, 0, 2], 
                [0, 0, 0], [0, 3, 0], [4, 3, 0], [4, 0, 0] ], dtype=float)
#o nó adicional é o nó utilizado para a orientação do eixo local y dos elementos
IE = np.array([ [0, 1, 2], [1, 2, 0], [3, 2, 1], [1, 5, 0], [2, 6, 3], [4, 5, 7], [5, 6, 4], [7, 6, 4] ], dtype=int)

#nos com restrição
nosR = np.array([0, 3, 4, 7])
#graus de liberdade restringidos em cada nó: 1 = restringido; 0 = livre
nosR_GL = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]])

#secao retangular
b = 0.1 #m
h = 0.4 #m

#Rigidezes dos materiais dos elementos
El = np.ones(IE.shape[0])*200000. #kPa
nu = 0.3
Gl = El*nu/(2*(1+nu**2)) #kPa
#Áreas das seções transversais dos elementos
Ar = np.ones(IE.shape[0])*b*h #m2
#Inércias das seções transversais dos elementos
Iz = np.ones(IE.shape[0])*b*h**3/12 #m4
Iy = np.ones(IE.shape[0])*h*b**3/12 #m4
Ix = np.ones(IE.shape[0])*h*b**3*(1./3. - 0.21+b/h*(1 - b**4/(12*h**4))) #m4 inercia torção, constante de torção de Saint Venant para retângulos

##carregamentos distribuidos constantes nos elementos na direção x, y e z global!!
#distribuidas = np.zeros((IE.shape[0], 3))
#distribuidas[0, 0] = 3. #kN/m
#distribuidas[2, 1] = -5. #kN/m

#carregamentos concentrados nos nós
concentradas = np.zeros((nos.shape[0], 3))
concentradas[1, 0] = 10. #kN
concentradas[2, 2] = -10. #kN
concentradas[5, 1] = -10. #kN


def rigidez_portico(E, G, A, Ii, Ij, Ik, cosL):
    '''
    Função para a deterrminação das matrizes de rigidez do elemento de pórtico
    '''
    lx = cosL[0]
    ly = cosL[1]
    lz = cosL[2]
    mx = cosL[3]
    my = cosL[4]
    mz = cosL[5]
    nx = cosL[6]
    ny = cosL[7]
    nz = cosL[8]
    l = cosL[9]
    return np.array([[ A*E*lx**2/l + 12*E*Ij*lz**2/l**3 + 12*E*Ik*ly**2/l**3,  A*E*lx*mx/l + 12*E*Ij*lz*mz/l**3 + 12*E*Ik*ly*my/l**3,  A*E*lx*nx/l + 12*E*Ij*lz*nz/l**3 + 12*E*Ik*ly*ny/l**3,          6*E*Ij*ly*lz/l**2 + 6*E*Ik*ly*lz/l**2,          6*E*Ij*lz*my/l**2 + 6*E*Ik*ly*mz/l**2,          6*E*Ij*lz*ny/l**2 + 6*E*Ik*ly*nz/l**2, -A*E*lx**2/l - 12*E*Ij*lz**2/l**3 - 12*E*Ik*ly**2/l**3, -A*E*lx*mx/l - 12*E*Ij*lz*mz/l**3 - 12*E*Ik*ly*my/l**3, -A*E*lx*nx/l - 12*E*Ij*lz*nz/l**3 - 12*E*Ik*ly*ny/l**3,          6*E*Ij*ly*lz/l**2 + 6*E*Ik*ly*lz/l**2,          6*E*Ij*lz*my/l**2 + 6*E*Ik*ly*mz/l**2,          6*E*Ij*lz*ny/l**2 + 6*E*Ik*ly*nz/l**2],
                    [ A*E*lx*mx/l + 12*E*Ij*lz*mz/l**3 + 12*E*Ik*ly*my/l**3,  A*E*mx**2/l + 12*E*Ij*mz**2/l**3 + 12*E*Ik*my**2/l**3,  A*E*mx*nx/l + 12*E*Ij*mz*nz/l**3 + 12*E*Ik*my*ny/l**3,          6*E*Ij*ly*mz/l**2 + 6*E*Ik*lz*my/l**2,          6*E*Ij*my*mz/l**2 + 6*E*Ik*my*mz/l**2,          6*E*Ij*mz*ny/l**2 + 6*E*Ik*my*nz/l**2, -A*E*lx*mx/l - 12*E*Ij*lz*mz/l**3 - 12*E*Ik*ly*my/l**3, -A*E*mx**2/l - 12*E*Ij*mz**2/l**3 - 12*E*Ik*my**2/l**3, -A*E*mx*nx/l - 12*E*Ij*mz*nz/l**3 - 12*E*Ik*my*ny/l**3,          6*E*Ij*ly*mz/l**2 + 6*E*Ik*lz*my/l**2,          6*E*Ij*my*mz/l**2 + 6*E*Ik*my*mz/l**2,          6*E*Ij*mz*ny/l**2 + 6*E*Ik*my*nz/l**2],
                    [ A*E*lx*nx/l + 12*E*Ij*lz*nz/l**3 + 12*E*Ik*ly*ny/l**3,  A*E*mx*nx/l + 12*E*Ij*mz*nz/l**3 + 12*E*Ik*my*ny/l**3,  A*E*nx**2/l + 12*E*Ij*nz**2/l**3 + 12*E*Ik*ny**2/l**3,          6*E*Ij*ly*nz/l**2 + 6*E*Ik*lz*ny/l**2,          6*E*Ij*my*nz/l**2 + 6*E*Ik*mz*ny/l**2,          6*E*Ij*ny*nz/l**2 + 6*E*Ik*ny*nz/l**2, -A*E*lx*nx/l - 12*E*Ij*lz*nz/l**3 - 12*E*Ik*ly*ny/l**3, -A*E*mx*nx/l - 12*E*Ij*mz*nz/l**3 - 12*E*Ik*my*ny/l**3, -A*E*nx**2/l - 12*E*Ij*nz**2/l**3 - 12*E*Ik*ny**2/l**3,          6*E*Ij*ly*nz/l**2 + 6*E*Ik*lz*ny/l**2,          6*E*Ij*my*nz/l**2 + 6*E*Ik*mz*ny/l**2,          6*E*Ij*ny*nz/l**2 + 6*E*Ik*ny*nz/l**2],
                    [                 6*E*Ij*ly*lz/l**2 + 6*E*Ik*ly*lz/l**2,                  6*E*Ij*ly*mz/l**2 + 6*E*Ik*lz*my/l**2,                  6*E*Ij*ly*nz/l**2 + 6*E*Ik*lz*ny/l**2, 4*E*Ij*ly**2/l + 4*E*Ik*lz**2/l + G*Ii*lx**2/l, 4*E*Ij*ly*my/l + 4*E*Ik*lz*mz/l + G*Ii*lx*mx/l, 4*E*Ij*ly*ny/l + 4*E*Ik*lz*nz/l + G*Ii*lx*nx/l,                 -6*E*Ij*ly*lz/l**2 - 6*E*Ik*ly*lz/l**2,                 -6*E*Ij*ly*mz/l**2 - 6*E*Ik*lz*my/l**2,                 -6*E*Ij*ly*nz/l**2 - 6*E*Ik*lz*ny/l**2, 2*E*Ij*ly**2/l + 2*E*Ik*lz**2/l - G*Ii*lx**2/l, 2*E*Ij*ly*my/l + 2*E*Ik*lz*mz/l - G*Ii*lx*mx/l, 2*E*Ij*ly*ny/l + 2*E*Ik*lz*nz/l - G*Ii*lx*nx/l],
                    [                 6*E*Ij*lz*my/l**2 + 6*E*Ik*ly*mz/l**2,                  6*E*Ij*my*mz/l**2 + 6*E*Ik*my*mz/l**2,                  6*E*Ij*my*nz/l**2 + 6*E*Ik*mz*ny/l**2, 4*E*Ij*ly*my/l + 4*E*Ik*lz*mz/l + G*Ii*lx*mx/l, 4*E*Ij*my**2/l + 4*E*Ik*mz**2/l + G*Ii*mx**2/l, 4*E*Ij*my*ny/l + 4*E*Ik*mz*nz/l + G*Ii*mx*nx/l,                 -6*E*Ij*lz*my/l**2 - 6*E*Ik*ly*mz/l**2,                 -6*E*Ij*my*mz/l**2 - 6*E*Ik*my*mz/l**2,                 -6*E*Ij*my*nz/l**2 - 6*E*Ik*mz*ny/l**2, 2*E*Ij*ly*my/l + 2*E*Ik*lz*mz/l - G*Ii*lx*mx/l, 2*E*Ij*my**2/l + 2*E*Ik*mz**2/l - G*Ii*mx**2/l, 2*E*Ij*my*ny/l + 2*E*Ik*mz*nz/l - G*Ii*mx*nx/l],
                    [                 6*E*Ij*lz*ny/l**2 + 6*E*Ik*ly*nz/l**2,                  6*E*Ij*mz*ny/l**2 + 6*E*Ik*my*nz/l**2,                  6*E*Ij*ny*nz/l**2 + 6*E*Ik*ny*nz/l**2, 4*E*Ij*ly*ny/l + 4*E*Ik*lz*nz/l + G*Ii*lx*nx/l, 4*E*Ij*my*ny/l + 4*E*Ik*mz*nz/l + G*Ii*mx*nx/l, 4*E*Ij*ny**2/l + 4*E*Ik*nz**2/l + G*Ii*nx**2/l,                 -6*E*Ij*lz*ny/l**2 - 6*E*Ik*ly*nz/l**2,                 -6*E*Ij*mz*ny/l**2 - 6*E*Ik*my*nz/l**2,                 -6*E*Ij*ny*nz/l**2 - 6*E*Ik*ny*nz/l**2, 2*E*Ij*ly*ny/l + 2*E*Ik*lz*nz/l - G*Ii*lx*nx/l, 2*E*Ij*my*ny/l + 2*E*Ik*mz*nz/l - G*Ii*mx*nx/l, 2*E*Ij*ny**2/l + 2*E*Ik*nz**2/l - G*Ii*nx**2/l],
                    [-A*E*lx**2/l - 12*E*Ij*lz**2/l**3 - 12*E*Ik*ly**2/l**3, -A*E*lx*mx/l - 12*E*Ij*lz*mz/l**3 - 12*E*Ik*ly*my/l**3, -A*E*lx*nx/l - 12*E*Ij*lz*nz/l**3 - 12*E*Ik*ly*ny/l**3,         -6*E*Ij*ly*lz/l**2 - 6*E*Ik*ly*lz/l**2,         -6*E*Ij*lz*my/l**2 - 6*E*Ik*ly*mz/l**2,         -6*E*Ij*lz*ny/l**2 - 6*E*Ik*ly*nz/l**2,  A*E*lx**2/l + 12*E*Ij*lz**2/l**3 + 12*E*Ik*ly**2/l**3,  A*E*lx*mx/l + 12*E*Ij*lz*mz/l**3 + 12*E*Ik*ly*my/l**3,  A*E*lx*nx/l + 12*E*Ij*lz*nz/l**3 + 12*E*Ik*ly*ny/l**3,         -6*E*Ij*ly*lz/l**2 - 6*E*Ik*ly*lz/l**2,         -6*E*Ij*lz*my/l**2 - 6*E*Ik*ly*mz/l**2,         -6*E*Ij*lz*ny/l**2 - 6*E*Ik*ly*nz/l**2],
                    [-A*E*lx*mx/l - 12*E*Ij*lz*mz/l**3 - 12*E*Ik*ly*my/l**3, -A*E*mx**2/l - 12*E*Ij*mz**2/l**3 - 12*E*Ik*my**2/l**3, -A*E*mx*nx/l - 12*E*Ij*mz*nz/l**3 - 12*E*Ik*my*ny/l**3,         -6*E*Ij*ly*mz/l**2 - 6*E*Ik*lz*my/l**2,         -6*E*Ij*my*mz/l**2 - 6*E*Ik*my*mz/l**2,         -6*E*Ij*mz*ny/l**2 - 6*E*Ik*my*nz/l**2,  A*E*lx*mx/l + 12*E*Ij*lz*mz/l**3 + 12*E*Ik*ly*my/l**3,  A*E*mx**2/l + 12*E*Ij*mz**2/l**3 + 12*E*Ik*my**2/l**3,  A*E*mx*nx/l + 12*E*Ij*mz*nz/l**3 + 12*E*Ik*my*ny/l**3,         -6*E*Ij*ly*mz/l**2 - 6*E*Ik*lz*my/l**2,         -6*E*Ij*my*mz/l**2 - 6*E*Ik*my*mz/l**2,         -6*E*Ij*mz*ny/l**2 - 6*E*Ik*my*nz/l**2],
                    [-A*E*lx*nx/l - 12*E*Ij*lz*nz/l**3 - 12*E*Ik*ly*ny/l**3, -A*E*mx*nx/l - 12*E*Ij*mz*nz/l**3 - 12*E*Ik*my*ny/l**3, -A*E*nx**2/l - 12*E*Ij*nz**2/l**3 - 12*E*Ik*ny**2/l**3,         -6*E*Ij*ly*nz/l**2 - 6*E*Ik*lz*ny/l**2,         -6*E*Ij*my*nz/l**2 - 6*E*Ik*mz*ny/l**2,         -6*E*Ij*ny*nz/l**2 - 6*E*Ik*ny*nz/l**2,  A*E*lx*nx/l + 12*E*Ij*lz*nz/l**3 + 12*E*Ik*ly*ny/l**3,  A*E*mx*nx/l + 12*E*Ij*mz*nz/l**3 + 12*E*Ik*my*ny/l**3,  A*E*nx**2/l + 12*E*Ij*nz**2/l**3 + 12*E*Ik*ny**2/l**3,         -6*E*Ij*ly*nz/l**2 - 6*E*Ik*lz*ny/l**2,         -6*E*Ij*my*nz/l**2 - 6*E*Ik*mz*ny/l**2,         -6*E*Ij*ny*nz/l**2 - 6*E*Ik*ny*nz/l**2],
                    [                 6*E*Ij*ly*lz/l**2 + 6*E*Ik*ly*lz/l**2,                  6*E*Ij*ly*mz/l**2 + 6*E*Ik*lz*my/l**2,                  6*E*Ij*ly*nz/l**2 + 6*E*Ik*lz*ny/l**2, 2*E*Ij*ly**2/l + 2*E*Ik*lz**2/l - G*Ii*lx**2/l, 2*E*Ij*ly*my/l + 2*E*Ik*lz*mz/l - G*Ii*lx*mx/l, 2*E*Ij*ly*ny/l + 2*E*Ik*lz*nz/l - G*Ii*lx*nx/l,                 -6*E*Ij*ly*lz/l**2 - 6*E*Ik*ly*lz/l**2,                 -6*E*Ij*ly*mz/l**2 - 6*E*Ik*lz*my/l**2,                 -6*E*Ij*ly*nz/l**2 - 6*E*Ik*lz*ny/l**2, 4*E*Ij*ly**2/l + 4*E*Ik*lz**2/l + G*Ii*lx**2/l, 4*E*Ij*ly*my/l + 4*E*Ik*lz*mz/l + G*Ii*lx*mx/l, 4*E*Ij*ly*ny/l + 4*E*Ik*lz*nz/l + G*Ii*lx*nx/l],
                    [                 6*E*Ij*lz*my/l**2 + 6*E*Ik*ly*mz/l**2,                  6*E*Ij*my*mz/l**2 + 6*E*Ik*my*mz/l**2,                  6*E*Ij*my*nz/l**2 + 6*E*Ik*mz*ny/l**2, 2*E*Ij*ly*my/l + 2*E*Ik*lz*mz/l - G*Ii*lx*mx/l, 2*E*Ij*my**2/l + 2*E*Ik*mz**2/l - G*Ii*mx**2/l, 2*E*Ij*my*ny/l + 2*E*Ik*mz*nz/l - G*Ii*mx*nx/l,                 -6*E*Ij*lz*my/l**2 - 6*E*Ik*ly*mz/l**2,                 -6*E*Ij*my*mz/l**2 - 6*E*Ik*my*mz/l**2,                 -6*E*Ij*my*nz/l**2 - 6*E*Ik*mz*ny/l**2, 4*E*Ij*ly*my/l + 4*E*Ik*lz*mz/l + G*Ii*lx*mx/l, 4*E*Ij*my**2/l + 4*E*Ik*mz**2/l + G*Ii*mx**2/l, 4*E*Ij*my*ny/l + 4*E*Ik*mz*nz/l + G*Ii*mx*nx/l],
                    [                 6*E*Ij*lz*ny/l**2 + 6*E*Ik*ly*nz/l**2,                  6*E*Ij*mz*ny/l**2 + 6*E*Ik*my*nz/l**2,                  6*E*Ij*ny*nz/l**2 + 6*E*Ik*ny*nz/l**2, 2*E*Ij*ly*ny/l + 2*E*Ik*lz*nz/l - G*Ii*lx*nx/l, 2*E*Ij*my*ny/l + 2*E*Ik*mz*nz/l - G*Ii*mx*nx/l, 2*E*Ij*ny**2/l + 2*E*Ik*nz**2/l - G*Ii*nx**2/l,                 -6*E*Ij*lz*ny/l**2 - 6*E*Ik*ly*nz/l**2,                 -6*E*Ij*mz*ny/l**2 - 6*E*Ik*my*nz/l**2,                 -6*E*Ij*ny*nz/l**2 - 6*E*Ik*ny*nz/l**2, 4*E*Ij*ly*ny/l + 4*E*Ik*lz*nz/l + G*Ii*lx*nx/l, 4*E*Ij*my*ny/l + 4*E*Ik*mz*nz/l + G*Ii*mx*nx/l, 4*E*Ij*ny**2/l + 4*E*Ik*nz**2/l + G*Ii*nx**2/l]])

def cossenos_diretores(nos, elementos):
    '''
    Função para calcular os senos e os cossenos de cada barra e o seu comprimento
    
    no1: coordenadas do nó 1 em array([x, y, z])
    no2: coordenadas do nó 2 em array([x, y, z])
    
    elementos no formato [i, f, p], no inicial i, nó final f, nó posicional p (direção dos eixos da seção transversal)
    
    retorna array com elementos na primeira dimensão e [lx, ly, lz, mx, my, mz, nx, ny, nz, comprimento] na segunda
    
    '''    
    resultados = np.zeros( (elementos.shape[0], 10) )
    
    no1 = nos[ elementos[:,0] ] #nós iniciais
    no2 = nos[ elementos[:,1] ] #nós finais
    nop = nos[ elementos[:,2] ] #nós posicionais
    
    ix = no2[:,0] - no1[:,0]
    iy = no2[:,1] - no1[:,1]
    iz = no2[:,2] - no1[:,2]
    
    L = np.sqrt(ix**2 + iy**2 + iz**2) #comprimento do elemento
    lx = ix/L #lx para todos os elementos
    mx = iy/L #mx para todos os elementos
    nx = iz/L #nx para todos os elementos
    
    lr = nop[:,0] - no1[:,0]
    mr = nop[:,1] - no1[:,1]
    nr = nop[:,2] - no1[:,2]
    
    kx = mx*nr - nx*mr
    ky = nx*lr - lx*nr
    kz = lx*mr - mx*lr
    
    Lz = np.sqrt(kx**2 + ky**2 + kz**2)
    
    lz = kx/Lz#lz para todos os elementos
    mz = ky/Lz#mz para todos os elementos
    nz = kz/Lz#nz para todos os elementos
    
    jx = mx*nz - nx*mz
    jy = nx*lz - lx*nz
    jz = lx*mz - mx*lz
    
    Ly = np.sqrt(jx**2 + jy**2 + jz**2)
    
    ly = jx/Ly#ly para todos os elementos
    my = jy/Ly#my para todos os elementos
    ny = jz/Ly#ny para todos os elementos
    
    resultados[:,0] = lx
    resultados[:,1] = ly
    resultados[:,2] = lz
    resultados[:,3] = mx
    resultados[:,4] = my
    resultados[:,5] = mz
    resultados[:,6] = nx
    resultados[:,7] = ny
    resultados[:,8] = nz
    resultados[:,9] = L
    
    return resultados

#determinação dos cossenos diretoras e comprimentos
cosdirL = cossenos_diretores(nos, IE)

#matriz de rigidez dos elementos
Ke = []
for e in range(IE.shape[0]):
    Ke.append(rigidez_portico(El[e], Gl[e], Ar[e], Ix[e], Iy[e], Iz[e], cosdirL[e]))
Ke = np.array(Ke)

##graus de liberdade da estrutura totais, GL
#GL = nos.shape[0]*3 #3 número de graus de liberdade por nó
#
##indexação do elemento
#ID = []
#for e in range(IE.shape[0]):
#    ID.append( np.repeat(IE[e]*3, 3) + np.tile(np.array([0, 1, 2]), 2) )
#ID = np.array(ID)
#
##montagem da matriz de rigidez da estrutura
#K = np.zeros((GL, GL))
#for i in range(6):
#    for j in range(6):
#        for e in range(IE.shape[0]):
#            K[ ID[e, i], ID[e, j] ] += Ke[e, i, j]
#
#
##graus de liberdade restringidos
#GL_R = []
#for n in nosR:
#    GL_R.append( np.repeat(n*3, 3) + np.array([0, 1, 2]) )
#GL_R = np.extract( nosR_GL.flatten(), np.array(GL_R).flatten() )
#
##graus de liberdade livres
#DOF = np.delete(np.arange(0, nos.shape[0]*3), GL_R)
#
##separação das matrizes de rigidez
#Ku = np.delete(np.delete(K, GL_R, axis=0), GL_R, axis=1)
#Kr = np.delete(np.delete(K, DOF, axis=0), GL_R, axis=1)
#
##cargas distribuídas constantes no sistema global
#def cdg(scL, forcas, alpha):
#    '''
#    Cálculo das forças nodais equivalentes a uma carga distribuída constante ao elemento de pórtico.
#    '''
#    s = scL[0]
#    c = scL[1]
#    L = scL[2]
#    ai = alpha[0]
#    aj = alpha[1]
#    gx = forcas[0]
#    gy = forcas[1]
#    M = np.array([[1,0,0,1,0,0],
#                  [0,2*(1+2*ai)/L**2/(1+4*aj+4*ai+12*ai*aj),(1+2*aj)/L/(1+4*aj+4*ai+12*ai*aj),0,-2*(1+2*ai)/L**2/(1+4*aj+4*ai+12*ai*aj),(1-2*aj+4*ai)/L/(1+4*aj+4*ai+12*ai*aj)],
#                  [0,(1+2*aj)/L/(1+4*aj+4*ai+12*ai*aj),4/3*(1+3*aj)/(1+4*aj+4*ai+12*ai*aj)-1/3/(1+4*aj+4*ai+12*ai*aj),0,-(1+2*aj)/L/(1+4*aj+4*ai+12*ai*aj),-2/3*(1+3*aj)/(1+4*aj+4*ai+12*ai*aj)+2/3/(1+4*aj+4*ai+12*ai*aj)],
#                  [1,0,0,1,0,0],
#                  [0,-2*(1+2*ai)/L**2/(1+4*aj+4*ai+12*ai*aj),-(1+2*aj)/L/(1+4*aj+4*ai+12*ai*aj),0,2*(1+2*ai)/L**2/(1+4*aj+4*ai+12*ai*aj),-(1-2*aj+4*ai)/L/(1+4*aj+4*ai+12*ai*aj)],
#                  [0,(1-2*aj+4*ai)/L/(1+4*aj+4*ai+12*ai*aj),-2/3*(1+3*aj)/(1+4*aj+4*ai+12*ai*aj)+2/3/(1+4*aj+4*ai+12*ai*aj),0,-(1-2*aj+4*ai)/L/(1+4*aj+4*ai+12*ai*aj),-1/3/(1+4*aj+4*ai+12*ai*aj)+4/3*(1+3*ai)/(1+4*aj+4*ai+12*ai*aj)]])
#    print(M)
#    fe = np.array([0.5*L*c*(c*gx + gy*s) - L*s*(c*gy - gx*s)*0.5,
#                     L*c*(c*gy - gx*s)/2 + 0.5*L*s*(c*gx + gy*s),
#                     L**2*(c*gy - gx*s)*0.08333333333333333,
#                     0.5*L*c*(c*gx + gy*s) - L*s*(c*gy - gx*s)*0.5,
#                     L*c*(c*gy - gx*s)/2 + 0.5*L*s*(c*gx + gy*s),
#                     -L**2*(c*gy - gx*s)*0.08333333333333333])
#    return np.matmul(M, fe)
#
#
#
#
#
##cálculo do vetor de cargas nodais equivalentes às distribuídas
#Feq = np.zeros(GL)
#for e in range(IE.shape[0]):
#    F = cdg(scL[e], forcas[e], alphas[e])
#    for i in range(6):
#        Feq[ ID[e][i] ] += F[i]
#
#FU = np.delete(Feq, GL_R, axis=0)
#FR = np.delete(Feq, DOF, axis=0)
#
##determinação dos deslocamentos
#Un = np.linalg.solve(Ku, FU)
#R = np.dot(Kr, Un) - FR
#
#U = np.zeros(GL)
#U[DOF] = Un
#
#Uxyr = U.reshape((nos.shape[0],3))
#
##reescrevendo os deslocamentos no sistema local do elemento
#u = np.zeros((IE.shape[0], 6))
#
#for e in range(IE.shape[0]):
#    s = scL[e, 0]
#    c = scL[e, 1]
#    rotacao = np.array([[ c, s, 0,  0, 0, 0], 
#                        [-s, c, 0,  0, 0, 0], 
#                        [ 0, 0, 1,  0, 0, 0],
#                        [ 0 ,0 ,0 , c, s, 0],
#                        [ 0, 0, 0, -s, c, 0],
#                        [ 0, 0, 0,  0, 0, 1]])
#    u[e] = np.matmul( U[ ID[e] ], rotacao)
#
##calculo das deformações, tensões, momento, corte e normal em cada elemento no eixo local ------------------------------------------------------------
#def esP(ue, l, E, A, h, I, pontos=100):
#    r = np.linspace(-l/2, l/2, pontos)
#    ue = ue[:, np.newaxis]
#    deslocamentos = (r - 1/l)*ue[0, 0] + (r + 1/l)*ue[3, 0] + (1/2 - 3*r/(2*l) + 2*r**3/l**3)*ue[1, 0] + (1/2 + 3*r/(2*l) - 2*r**3/l**3)*ue[4, 0] + (-l/8 - r/4 + r**2/(2*l) + r**3/l**2)*ue[5, 0] + (l/8 - r/4 - r**2/(2*l) + r**3/l**2)*ue[2, 0]
#    rotacoes = (-3/(2*l) + 6*r**2/l**3)*ue[1, 0] + (3/(2*l) - 6*r**2/l**3)*ue[4, 0] + (-1/4 - r/l + 3*r**2/l**2)*ue[2, 0] + (-1/4 + r/l + 3*r**2/l**2)*ue[5, 0]
#    momento = (E * I) * ( (-1/l + 6*r/l**2)*ue[2, 0] + (1/l + 6*r/l**2)*ue[5, 0] + 12*r*ue[1, 0]/l**3 - 12*r*ue[4, 0]/l**3 )
#    cortante = (E * I) * ( 6*ue[2, 0]/l**2 + 6*ue[5, 0]/l**2 + 12*ue[1, 0]/l**3 - 12*ue[4, 0]/l**3 )*np.ones(pontos)
#    normal = (E * A) * ( ue[0,0]*(- 1/l) + ue[3, 0]*(1/l) )*np.ones(pontos)
#    
#    #aborgadem reversa
#    tensoes = normal/A + momento/I * h/2
#    deformacoes = tensoes/E
#    
#    return deslocamentos, rotacoes, deformacoes, tensoes, momento, cortante, normal, r
#
#resultados = []
#for e in range(IE.shape[0]):
#    resultados.append( esP(u[e], scL[e, 2], El[e], Ar[e], h, Iz[e]) )
#resultados = np.array(resultados) #elementos, quantidade de resultados, número de pontos para o gráfico

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





