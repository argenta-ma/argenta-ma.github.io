#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Valores ok!
Inclusive tensões máximas, mínimas e vonMises!

@author: markinho
"""

import numpy as np
import meshio

### iniciando do código numérico ---------------------------------------------------------------------------------------------------------------------
def dNdx(Xe, pg):
    '''
    Função para a determinação da matriz das derivadas das funções de interpolação já no sistema x y e do jacobiano
    
    Parâmetros
    ----------
    
    Xe: array numpy com as coordenadas de cada nó dos elementos dispostas no sentido horário, com o primeiro nó o correspondente ao segundo quadrante
    
    >>>
        2 ----- 1
        |       |
        |       |
        3-------4
        
    >>> Xe = np.array([ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ])
    
    pg: coordenadas do ponto de gauss utilizado
    
    >>> pg = np.array([ [xpg, ypg] ])
    
    retorna a matriz B para cada ponto de gauss
    '''
    r = pg[0]
    s = pg[1]
    x1 = Xe[0,0]
    y1 = Xe[0,1]
    x2 = Xe[1,0]
    y2 = Xe[1,1]
    x3 = Xe[2,0]
    y3 = Xe[2,1]
    x4 = Xe[3,0]
    y4 = Xe[3,1]
    
    J = np.array([ [x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4), x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4)],
                    [y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4), y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)]])
    
    dNdx = np.array([ [ (r/4 + 1/4)*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) + (s/4 + 1/4)*(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4)) - (x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))/((-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))),   (r/4 + 1/4)*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) - (s/4 + 1/4)*(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))],
                       [(-r/4 + 1/4)*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) + (-s/4 - 1/4)*(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4)) - (x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))/((-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))), (-r/4 + 1/4)*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) - (-s/4 - 1/4)*(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))],
                       [  (r/4 - 1/4)*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) + (s/4 - 1/4)*(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4)) - (x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))/((-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))),   (r/4 - 1/4)*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) - (s/4 - 1/4)*(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))],
                       [(-r/4 - 1/4)*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) + (-s/4 + 1/4)*(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4)) - (x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))/((-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))), (-r/4 - 1/4)*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) - (-s/4 + 1/4)*(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))]])
    B1x = dNdx[0,0]
    B1y = dNdx[0,1]
    B2x = dNdx[1,0]
    B2y = dNdx[1,1]
    B3x = dNdx[2,0]
    B3y = dNdx[2,1]
    B4x = dNdx[3,0]
    B4y = dNdx[3,1]
    B = np.array([[B1x,   0, B2x,   0, B3x,   0, B4x,   0],
                  [  0, B1y,   0, B2y,   0, B3y,   0, B4y],
                  [B1y, B1x, B2y, B2x, B3y, B3x, B4y, B4x]])
    return B, J

def ke(Xe, E, nu, t):
    '''
    Função para a geração das matrizes de rigidez dos elementos função das coordenadas dos elementos no sistema global, o módulo de elasticidade
    do material (E), o corficiente de poisson do material (nu) e da espessura (t), considerando 4 pontos de gauss para a integração
    
    Parâmetros
    ----------
    
    Xe: array numpy com as coordenadas de cada nó dos elementos dispostas no sentido antihorário, com o primeiro nó o correspondente ao primeiro quadrante.
        
    >>> Xe = np.array([ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ])
    '''
    #matriz constitutiva do material
    D = E/(1 - nu**2) * np.array([[1, nu, 0],
                                  [nu, 1, 0],
                                  [0, 0, (1 - nu**2)/(2 + 2*nu)]])
    #número de graus de liberdade por elemento
    GLe = 8
    #coordenadas e pesos dos pontos de Gauss
    PG = np.array([[0.5773502691896258, 0.5773502691896258],
                   [-0.5773502691896258, 0.5773502691896258],
                   [-0.5773502691896258, -0.5773502691896258],
                   [0.5773502691896258, -0.5773502691896258]])
    wPG = np.array([[1., 1.],
                    [1., 1.],
                    [1., 1.],
                    [1., 1.]])
    Be = []
    Ke = np.zeros((GLe, GLe))
    for p in range(PG.shape[0]):
        B, J = dNdx(Xe, PG[p])
        Be.append(B)
        Ke += np.matmul( np.matmul(np.transpose(B), D), B) * wPG[p, 0] * wPG[p, 1] * np.linalg.det(J) * t
    return Ke, Be

#coordenadas dos nós da estrutura
NOS = np.array([ [0., 0.],
                 [150., 0.],
                 [0., 170.],
                 [150., 170.],
                 [0., 340.],
                 [150., 340.],
                 [0., 510.],
                 [150., 510.],
                 [0., 680.],
                 [150., 680.],
                 [0., 850.],
                 [150., 850.]])

#incidência dos elementos !!! DEVE SEGUIR A ORDEM DAS FUNÇÕES DE INTERPOLAÇÃO DEFINIDA NA FUNÇÃO dNdx !!!
IE = np.array([[ 3,  2,  0,  1],
               [ 5,  4,  2,  3],
               [ 7,  6,  4,  5],
               [ 9,  8,  6,  7],
               [11, 10,  8,  9]])

#malha de elementos
Xe = []
for e in IE:
    Xe.append( np.array([ NOS[e[0]], NOS[e[1]], NOS[e[2]], NOS[e[3]] ]) )
    
#propriedades mecânicas do material da estrutura e espessura
E = 3313.00468 #kN/cm2
nu = 0.2
t = 20. #cm
#resistência a compressão e a tração para aplicação do critério de Christensen, concreto C35
Sc = 35. #kN/cm2
St = 3.5 #kN/cm2
#coesão e o ângulo de atrito para os critérios de Mohr-Coulomb e Drucker-Prager (http://www.pcc.usp.br/files/text/publications/BT_00231.pdf)
phi = 51. * np.pi/180.
coesao = 0.00073 #kN/cm2

#determinação da matriz de rigidez dos elementos
Ke1, Be1 = ke(Xe[0], E, nu, t)
Ke2, Be2 = ke(Xe[1], E, nu, t)
Ke3, Be3 = ke(Xe[2], E, nu, t)
Ke4, Be4 = ke(Xe[3], E, nu, t)
Ke5, Be5 = ke(Xe[4], E, nu, t)

#indexação dos graus de liberdade
ID1 = np.repeat(IE[0]*2, 2) + np.tile(np.array([0, 1]), 4)
ID2 = np.repeat(IE[1]*2, 2) + np.tile(np.array([0, 1]), 4)
ID3 = np.repeat(IE[2]*2, 2) + np.tile(np.array([0, 1]), 4)
ID4 = np.repeat(IE[3]*2, 2) + np.tile(np.array([0, 1]), 4)
ID5 = np.repeat(IE[4]*2, 2) + np.tile(np.array([0, 1]), 4)

#graus de liberdade da estrutura
GL = NOS.shape[0]*2
DOF = GL - 4

#montagem da matriz de rigidez da estrutura
K = np.zeros((GL, GL))
for i in range(8):
    for j in range(8):
        K[ ID1[i], ID1[j] ] += Ke1[i, j]
        K[ ID2[i], ID2[j] ] += Ke2[i, j]
        K[ ID3[i], ID3[j] ] += Ke3[i, j]
        K[ ID4[i], ID4[j] ] += Ke4[i, j]
        K[ ID5[i], ID5[j] ] += Ke5[i, j]

#separação das matrizes de rigidez
Ku = K[:DOF, :DOF]
Kr = K[DOF:, :DOF]

#vetor de forças nodais de todos os elementos (ver Derivando-FuncoesFormaEstadoPlano4nos.py)
fe = np.array([ 0.    , -3.1875,  0.    , -3.1875,  0.    , -3.1875,  0.    , -3.1875])
F = np.zeros(GL)
for i in range(8):
    F[ ID1[i] ] += fe[i]
    F[ ID2[i] ] += fe[i]
    F[ ID3[i] ] += fe[i]
    F[ ID4[i] ] += fe[i]
    F[ ID5[i] ] += fe[i]

Fu = F[:DOF]
Fr = F[DOF:]

Uu = np.linalg.solve(Ku, Fu)
Rr = np.matmul(Kr, Uu) - Fr

U = np.zeros(GL)
U[:DOF] = Uu

Uxy = U.reshape(NOS.shape)

##visualização dos deslocamentos ---------------------------------------------------------------------------------------------------------------------
#fig = go.Figure(data = go.Contour(z=Uxy[:,0], x=NOS[:,0], y=NOS[:,1], colorscale='Jet', contours=dict(
#            showlabels = True, # show labels on contours
#            labelfont = dict(size = 12, color = 'white') ) ) )
#fig.update_layout(title="Deslocamentos em X", autosize=True, width=1200, height=400)
#fig.write_html('deslocamentos.html')

##visualização dos vetores de deslocamentos dos nós
#fig = plty.figure_factory.create_quiver(NOS[:,0], NOS[:,1], Uxy[:,0], Uxy[:,1])
#fig.write_html('deslocamentosVetor.html')

##geração do arquivo vtu
#pontos = NOS
#celulas = {'quad': IE}
#meshio.write_points_cells(
#        "teste.vtu",
#        pontos,
#        celulas,
#        # Optionally provide extra data on points, cells, etc.
#        point_data = {"U": Uxy},
#        # cell_data=cell_data,
#        # field_data=field_data
#        )

#-----------------------------------------------------------------------------------------------------------------------------------------------------

#determinação dos deslocamentos por elemento
Ue = []
Ue.append( U[ID1] )
Ue.append( U[ID2] )
Ue.append( U[ID3] )
Ue.append( U[ID4] )
Ue.append( U[ID5] )


#determinação das deformações por ponto de Gauss -----------------------------------------------------------------------------------------------------
epsilon1 = []
epsilon2 = []
epsilon3 = []
epsilon4 = []
epsilon5 = []
for p in range(4): #range nos pontos de Gauss
    epsilon1.append( np.matmul(Be1[p], Ue[0]) )
    epsilon2.append( np.matmul(Be2[p], Ue[1]) )
    epsilon3.append( np.matmul(Be3[p], Ue[2]) )
    epsilon4.append( np.matmul(Be4[p], Ue[3]) )
    epsilon5.append( np.matmul(Be5[p], Ue[3]) )

#determinação das tensões por ponto de Gauss ---------------------------------------------------------------------------------------------------------
#matriz constitutiva do material
D = E/(1 - nu**2) * np.array([[1, nu, 0],
                              [nu, 1, 0],
                              [0, 0, (1 - nu**2)/(2 + 2*nu)]])

sigma1 = []
sigma2 = []
sigma3 = []
sigma4 = []
sigma5 = []
for p in range(4): #range na quantidade de pontos de Gauss
    sigma1.append( np.matmul(D, epsilon1[p]) )
    sigma2.append( np.matmul(D, epsilon2[p]) )
    sigma3.append( np.matmul(D, epsilon3[p]) )
    sigma4.append( np.matmul(D, epsilon4[p]) )
    sigma5.append( np.matmul(D, epsilon5[p]) )

##para mudar para latex
#sigma11 = np.zeros((3,4))
#sigma21 = np.zeros((3,4))
#sigma31 = np.zeros((3,4))
#sigma41 = np.zeros((3,4))
#sigma51 = np.zeros((3,4))
#for b in range(4):
#    sigma11[:,b] = sigma1[b]
#    sigma21[:,b] = sigma2[b]
#    sigma31[:,b] = sigma3[b]
#    sigma41[:,b] = sigma4[b]
#    sigma51[:,b] = sigma5[b]

#cálculo das tensões principais nos pontos de Gauss---------------------------------------------------------------------------------------------------
#tensão principal máxima, tensão principal mínima, ângulo das tensões principais, tensão máxima de cisalhamento, tensão equivalente de von Mises
sigmaPP1 = []
sigmaPP2 = []
sigmaPP3 = []
sigmaPP4 = []
sigmaPP5 = []

def principaisPG(sigmas):
    '''
    Função para a determinação da tensão principal 1 (sigmaMAX), tensão principal 2 (sigmaMIN), 
    ângulo das tensões principais, tensão máxima de cisalhamento, tensão equivalente de von Mises, de Christensen para materiais frágeis com
    sigmaC <= 0.5 sigmaT (que deve ser menor que 1), de Morh-Coulomb de Drucker-Prager para a tensão fora do plano igual a zero
    
    sigmas é um array de uma dimensão contendo sigma_x, sigma_y e tau_xy
    
    retorna um array de uma dimensão com as quantidades acima
    '''
    sigma_x = sigmas[0]
    sigma_y = sigmas[1]
    tay_xy = sigmas[2]
    
    sigmaMAX = (sigma_x + sigma_y)/2 + np.sqrt( ((sigma_x - sigma_y)/2)**2 + tay_xy**2 )
    sigmaMIN = (sigma_x + sigma_y)/2 - np.sqrt( ((sigma_x - sigma_y)/2)**2 + tay_xy**2 )
    theta = 1./2. * np.arctan( 2*tay_xy/(sigma_x - sigma_y) )
    tauMAX = (sigmaMAX - sigmaMIN)/2
    sigmaEQvM = np.sqrt( sigmaMAX**2 - sigmaMAX*sigmaMIN + sigmaMIN**2 )
    sigmaEQc = (1/St - 1/Sc)*(sigmaMAX + sigmaMIN) + 1/(2*St*Sc)*( (sigmaMAX - sigmaMIN)**2 + sigmaMAX**2 + sigmaMIN**2)
    sigmaEQmc = ( sigmaMAX * (1 + np.sin(phi)) - sigmaMIN * (1 - np.sin(phi)) )/(2*coesao*np.cos(phi))
    A = 2*1.4142135623730951*np.sin(phi)/(3 - np.sin(phi))
    B = 3.*coesao*np.cos(phi)/np.sin(phi)
    sigmaEQdp = ( (sigmaMAX - sigmaMIN)**2 + sigmaMAX**2 + sigmaMIN**2 )/( A**2*(sigmaMAX + sigmaMIN + B)**2 )
    
    return np.array([ sigmaMAX, sigmaMIN, theta, tauMAX, sigmaEQvM, sigmaEQc, sigmaEQmc, sigmaEQdp ])

for p in range(4):
    sigmaPP1.append( principaisPG(sigma1[p]) )
    sigmaPP2.append( principaisPG(sigma2[p]) )
    sigmaPP3.append( principaisPG(sigma3[p]) )
    sigmaPP4.append( principaisPG(sigma4[p]) )
    sigmaPP5.append( principaisPG(sigma5[p]) )


#para gerar as tensões principais para o latex
#ppais = np.array(sigmaPP4)
#ppaisT = ppais.T
#res = ppaisT[:3,:4]

#para gerar as tensões equivalentes para o latex
ppais = np.array(sigmaPP4)
ppaisT = ppais.T
res = ppaisT[4:,:]



##geração do arquivo vtu
#pontos = NOS
#celulas = {'quad9': IE}
#meshio.write_points_cells(
#        "teste9.vtu",
#        pontos,
#        celulas,
#        # Optionally provide extra data on points, cells, etc.
#        point_data = {"U": Uxy},
#        # cell_data=cell_data,
#        # field_data=field_data
#        )

#cálculo das tensões nos nós, interpolando com as funções de interpolação dos elementos