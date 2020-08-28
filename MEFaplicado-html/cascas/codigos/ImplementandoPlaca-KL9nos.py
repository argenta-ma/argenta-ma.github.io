#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

OK! Verificar os resultados!!!


O elemento padrão:

    2 -- 5 -- 1
    |         |
    6    9    8
    |         |
    3 -- 7 -- 4

Nós restringidos em x = 0
Nós com carga em x = lx

@author: markinho
"""

import numpy as np
import scipy.sparse as sps
#import meshio

def dNdx(Xe, pg):
    '''
    Função para a determinação da matriz das derivadas das funções de interpolação já no sistema x y e do jacobiano
    
    Parâmetros
    ----------
    
    Xe: array numpy com as coordenadas de cada nó dos elementos dispostas no sentido horário, com o primeiro nó o correspondente ao segundo quadrante
    
    >>>
        2 -- 5 -- 1
        |         |
        6    9    8
        |         |
        3 -- 7 -- 4
        
    >>> Xe = np.array([ [x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6], [x7, y7], [x8, y8], [x9, y9] ])
    
    pg: coordenadas do ponto de gauss utilizado
    
    >>> pg = np.array([ [xpg, ypg] ])
    
    retorna a matriz B para cada ponto de gauss
    '''
    r = pg[0]
    s = pg[1]

    #das funções de forma do elemento de placa de 9 nós com 3 GL por 2 sendo os as duas rotações derivadas da deflexão
    dNdr = np.array([[-15*r**4*s**2/8 - 15*r**4*s/8 - r**3*s**2 - r**3*s + 15*r**2*s**2/8 + 15*r**2*s/8 - 3*r*s**5/4 - r*s**4/2 + 5*r*s**3/4 + 3*r*s**2/2 + r*s/2 - 3*s**5/8 - s**4/4 + 5*s**3/8 + s**2/4 - s/4,r*s**5/4 + r*s**4/4 - r*s**3/4 - r*s**2/4 + s**5/8 + s**4/8 - s**3/8 - s**2/8,-5*r**4*s**2/8 - 5*r**4*s/8 - r**3*s**2/2 - r**3*s/2 + 3*r**2*s**2/8 + 3*r**2*s/8 + r*s**2/4 + r*s/4,15*r**4*s**2/8 + 15*r**4*s/8 - r**3*s**2 - r**3*s - 15*r**2*s**2/8 - 15*r**2*s/8 - 3*r*s**5/4 - r*s**4/2 + 5*r*s**3/4 + 3*r*s**2/2 + r*s/2 + 3*s**5/8 + s**4/4 - 5*s**3/8 - s**2/4 + s/4,r*s**5/4 + r*s**4/4 - r*s**3/4 - r*s**2/4 - s**5/8 - s**4/8 + s**3/8 + s**2/8,-5*r**4*s**2/8 - 5*r**4*s/8 + r**3*s**2/2 + r**3*s/2 + 3*r**2*s**2/8 + 3*r**2*s/8 - r*s**2/4 - r*s/4,15*r**4*s**2/8 - 15*r**4*s/8 - r**3*s**2 + r**3*s - 15*r**2*s**2/8 + 15*r**2*s/8 + 3*r*s**5/4 - r*s**4/2 - 5*r*s**3/4 + 3*r*s**2/2 - r*s/2 - 3*s**5/8 + s**4/4 + 5*s**3/8 - s**2/4 - s/4,r*s**5/4 - r*s**4/4 - r*s**3/4 + r*s**2/4 - s**5/8 + s**4/8 + s**3/8 - s**2/8,-5*r**4*s**2/8 + 5*r**4*s/8 + r**3*s**2/2 - r**3*s/2 + 3*r**2*s**2/8 - 3*r**2*s/8 - r*s**2/4 + r*s/4,-15*r**4*s**2/8 + 15*r**4*s/8 - r**3*s**2 + r**3*s + 15*r**2*s**2/8 - 15*r**2*s/8 + 3*r*s**5/4 - r*s**4/2 - 5*r*s**3/4 + 3*r*s**2/2 - r*s/2 + 3*s**5/8 - s**4/4 - 5*s**3/8 + s**2/4 + s/4,r*s**5/4 - r*s**4/4 - r*s**3/4 + r*s**2/4 + s**5/8 - s**4/8 - s**3/8 + s**2/8,-5*r**4*s**2/8 + 5*r**4*s/8 - r**3*s**2/2 + r**3*s/2 + 3*r**2*s**2/8 - 3*r**2*s/8 + r*s**2/4 - r*s/4,2*r**3*s**2 + 2*r**3*s + 3*r*s**5/2 + r*s**4 - 5*r*s**3/2 - 3*r*s**2 - r*s,-r*s**5/2 - r*s**4/2 + r*s**3/2 + r*s**2/2,-5*r**4*s**2/2 - 5*r**4*s/2 + 3*r**2*s**2 + 3*r**2*s - s**2/2 - s/2,-15*r**4*s**2/4 + 15*r**4/4 + 2*r**3*s**2 - 2*r**3 + 15*r**2*s**2/4 - 15*r**2/4 + r*s**4 - 3*r*s**2 + 2*r - s**4/2 + s**2/2,r*s**5 - 2*r*s**3 + r*s - s**5/2 + s**3 - s/2,5*r**4*s**2/4 - 5*r**4/4 - r**3*s**2 + r**3 - 3*r**2*s**2/4 + 3*r**2/4 + r*s**2/2 - r/2,2*r**3*s**2 - 2*r**3*s - 3*r*s**5/2 + r*s**4 + 5*r*s**3/2 - 3*r*s**2 + r*s,-r*s**5/2 + r*s**4/2 + r*s**3/2 - r*s**2/2,-5*r**4*s**2/2 + 5*r**4*s/2 + 3*r**2*s**2 - 3*r**2*s - s**2/2 + s/2,15*r**4*s**2/4 - 15*r**4/4 + 2*r**3*s**2 - 2*r**3 - 15*r**2*s**2/4 + 15*r**2/4 + r*s**4 - 3*r*s**2 + 2*r + s**4/2 - s**2/2,r*s**5 - 2*r*s**3 + r*s + s**5/2 - s**3 + s/2,5*r**4*s**2/4 - 5*r**4/4 + r**3*s**2 - r**3 - 3*r**2*s**2/4 + 3*r**2/4 - r*s**2/2 + r/2,-4*r**3*s**2 + 4*r**3 - 2*r*s**4 + 6*r*s**2 - 4*r,-2*r*s**5 + 4*r*s**3 - 2*r*s,5*r**4*s**2 - 5*r**4 - 6*r**2*s**2 + 6*r**2 + s**2 - 1],
                     [-3*r**5*s/4 - 3*r**5/8 - r**4*s/2 - r**4/4 + 5*r**3*s/4 + 5*r**3/8 - 15*r**2*s**4/8 - r**2*s**3 + 15*r**2*s**2/8 + 3*r**2*s/2 + r**2/4 - 15*r*s**4/8 - r*s**3 + 15*r*s**2/8 + r*s/2 - r/4,5*r**2*s**4/8 + r**2*s**3/2 - 3*r**2*s**2/8 - r**2*s/4 + 5*r*s**4/8 + r*s**3/2 - 3*r*s**2/8 - r*s/4,-r**5*s/4 - r**5/8 - r**4*s/4 - r**4/8 + r**3*s/4 + r**3/8 + r**2*s/4 + r**2/8,3*r**5*s/4 + 3*r**5/8 - r**4*s/2 - r**4/4 - 5*r**3*s/4 - 5*r**3/8 - 15*r**2*s**4/8 - r**2*s**3 + 15*r**2*s**2/8 + 3*r**2*s/2 + r**2/4 + 15*r*s**4/8 + r*s**3 - 15*r*s**2/8 - r*s/2 + r/4,5*r**2*s**4/8 + r**2*s**3/2 - 3*r**2*s**2/8 - r**2*s/4 - 5*r*s**4/8 - r*s**3/2 + 3*r*s**2/8 + r*s/4,-r**5*s/4 - r**5/8 + r**4*s/4 + r**4/8 + r**3*s/4 + r**3/8 - r**2*s/4 - r**2/8,3*r**5*s/4 - 3*r**5/8 - r**4*s/2 + r**4/4 - 5*r**3*s/4 + 5*r**3/8 + 15*r**2*s**4/8 - r**2*s**3 - 15*r**2*s**2/8 + 3*r**2*s/2 - r**2/4 - 15*r*s**4/8 + r*s**3 + 15*r*s**2/8 - r*s/2 - r/4,5*r**2*s**4/8 - r**2*s**3/2 - 3*r**2*s**2/8 + r**2*s/4 - 5*r*s**4/8 + r*s**3/2 + 3*r*s**2/8 - r*s/4,-r**5*s/4 + r**5/8 + r**4*s/4 - r**4/8 + r**3*s/4 - r**3/8 - r**2*s/4 + r**2/8,-3*r**5*s/4 + 3*r**5/8 - r**4*s/2 + r**4/4 + 5*r**3*s/4 - 5*r**3/8 + 15*r**2*s**4/8 - r**2*s**3 - 15*r**2*s**2/8 + 3*r**2*s/2 - r**2/4 + 15*r*s**4/8 - r*s**3 - 15*r*s**2/8 + r*s/2 + r/4,5*r**2*s**4/8 - r**2*s**3/2 - 3*r**2*s**2/8 + r**2*s/4 + 5*r*s**4/8 - r*s**3/2 - 3*r*s**2/8 + r*s/4,-r**5*s/4 + r**5/8 - r**4*s/4 + r**4/8 + r**3*s/4 - r**3/8 + r**2*s/4 - r**2/8,r**4*s + r**4/2 + 15*r**2*s**4/4 + 2*r**2*s**3 - 15*r**2*s**2/4 - 3*r**2*s - r**2/2 - 15*s**4/4 - 2*s**3 + 15*s**2/4 + 2*s,-5*r**2*s**4/4 - r**2*s**3 + 3*r**2*s**2/4 + r**2*s/2 + 5*s**4/4 + s**3 - 3*s**2/4 - s/2,-r**5*s - r**5/2 + 2*r**3*s + r**3 - r*s - r/2,-3*r**5*s/2 + r**4*s + 5*r**3*s/2 + 2*r**2*s**3 - 3*r**2*s - 2*r*s**3 + r*s,5*r**2*s**4/2 - 3*r**2*s**2 + r**2/2 - 5*r*s**4/2 + 3*r*s**2 - r/2,r**5*s/2 - r**4*s/2 - r**3*s/2 + r**2*s/2,r**4*s - r**4/2 - 15*r**2*s**4/4 + 2*r**2*s**3 + 15*r**2*s**2/4 - 3*r**2*s + r**2/2 + 15*s**4/4 - 2*s**3 - 15*s**2/4 + 2*s,-5*r**2*s**4/4 + r**2*s**3 + 3*r**2*s**2/4 - r**2*s/2 + 5*s**4/4 - s**3 - 3*s**2/4 + s/2,-r**5*s + r**5/2 + 2*r**3*s - r**3 - r*s + r/2,3*r**5*s/2 + r**4*s - 5*r**3*s/2 + 2*r**2*s**3 - 3*r**2*s + 2*r*s**3 - r*s,5*r**2*s**4/2 - 3*r**2*s**2 + r**2/2 + 5*r*s**4/2 - 3*r*s**2 + r/2,r**5*s/2 + r**4*s/2 - r**3*s/2 - r**2*s/2,-2*r**4*s - 4*r**2*s**3 + 6*r**2*s + 4*s**3 - 4*s,-5*r**2*s**4 + 6*r**2*s**2 - r**2 + 5*s**4 - 6*s**2 + 1,2*r**5*s - 4*r**3*s + 2*r*s]])
    d2Ndr2 = np.array([[s*(-15*r**3*s/2 - 15*r**3/2 - 3*r**2*s - 3*r**2 + 15*r*s/4 + 15*r/4 - 3*s**4/4 - s**3/2 + 5*s**2/4 + 3*s/2 + 1/2),s**2*(s**3 + s**2 - s - 1)/4,s*(-10*r**3*s - 10*r**3 - 6*r**2*s - 6*r**2 + 3*r*s + 3*r + s + 1)/4,s*(15*r**3*s/2 + 15*r**3/2 - 3*r**2*s - 3*r**2 - 15*r*s/4 - 15*r/4 - 3*s**4/4 - s**3/2 + 5*s**2/4 + 3*s/2 + 1/2),s**2*(s**3 + s**2 - s - 1)/4,s*(-10*r**3*s - 10*r**3 + 6*r**2*s + 6*r**2 + 3*r*s + 3*r - s - 1)/4,s*(15*r**3*s/2 - 15*r**3/2 - 3*r**2*s + 3*r**2 - 15*r*s/4 + 15*r/4 + 3*s**4/4 - s**3/2 - 5*s**2/4 + 3*s/2 - 1/2),s**2*(s**3 - s**2 - s + 1)/4,s*(-10*r**3*s + 10*r**3 + 6*r**2*s - 6*r**2 + 3*r*s - 3*r - s + 1)/4,s*(-15*r**3*s/2 + 15*r**3/2 - 3*r**2*s + 3*r**2 + 15*r*s/4 - 15*r/4 + 3*s**4/4 - s**3/2 - 5*s**2/4 + 3*s/2 - 1/2),s**2*(s**3 - s**2 - s + 1)/4,s*(-10*r**3*s + 10*r**3 - 6*r**2*s + 6*r**2 + 3*r*s - 3*r + s - 1)/4,s*(6*r**2*s + 6*r**2 + 3*s**4/2 + s**3 - 5*s**2/2 - 3*s - 1),s**2*(-s**3 - s**2 + s + 1)/2,2*r*s*(-5*r**2*s - 5*r**2 + 3*s + 3),-15*r**3*s**2 + 15*r**3 + 6*r**2*s**2 - 6*r**2 + 15*r*s**2/2 - 15*r/2 + s**4 - 3*s**2 + 2,s*(s**4 - 2*s**2 + 1),5*r**3*s**2 - 5*r**3 - 3*r**2*s**2 + 3*r**2 - 3*r*s**2/2 + 3*r/2 + s**2/2 - 1/2,s*(6*r**2*s - 6*r**2 - 3*s**4/2 + s**3 + 5*s**2/2 - 3*s + 1),s**2*(-s**3 + s**2 + s - 1)/2,2*r*s*(-5*r**2*s + 5*r**2 + 3*s - 3),15*r**3*s**2 - 15*r**3 + 6*r**2*s**2 - 6*r**2 - 15*r*s**2/2 + 15*r/2 + s**4 - 3*s**2 + 2,s*(s**4 - 2*s**2 + 1),5*r**3*s**2 - 5*r**3 + 3*r**2*s**2 - 3*r**2 - 3*r*s**2/2 + 3*r/2 - s**2/2 + 1/2,2*(-6*r**2*s**2 + 6*r**2 - s**4 + 3*s**2 - 2),2*s*(-s**4 + 2*s**2 - 1),4*r*(5*r**2*s**2 - 5*r**2 - 3*s**2 + 3)],
                       [r*(-3*r**4/4 - r**3/2 + 5*r**2/4 - 15*r*s**3/2 - 3*r*s**2 + 15*r*s/4 + 3*r/2 - 15*s**3/2 - 3*s**2 + 15*s/4 + 1/2),r*(10*r*s**3 + 6*r*s**2 - 3*r*s - r + 10*s**3 + 6*s**2 - 3*s - 1)/4,r**2*(-r**3 - r**2 + r + 1)/4,r*(3*r**4/4 - r**3/2 - 5*r**2/4 - 15*r*s**3/2 - 3*r*s**2 + 15*r*s/4 + 3*r/2 + 15*s**3/2 + 3*s**2 - 15*s/4 - 1/2),r*(10*r*s**3 + 6*r*s**2 - 3*r*s - r - 10*s**3 - 6*s**2 + 3*s + 1)/4,r**2*(-r**3 + r**2 + r - 1)/4,r*(3*r**4/4 - r**3/2 - 5*r**2/4 + 15*r*s**3/2 - 3*r*s**2 - 15*r*s/4 + 3*r/2 - 15*s**3/2 + 3*s**2 + 15*s/4 - 1/2),r*(10*r*s**3 - 6*r*s**2 - 3*r*s + r - 10*s**3 + 6*s**2 + 3*s - 1)/4,r**2*(-r**3 + r**2 + r - 1)/4,r*(-3*r**4/4 - r**3/2 + 5*r**2/4 + 15*r*s**3/2 - 3*r*s**2 - 15*r*s/4 + 3*r/2 + 15*s**3/2 - 3*s**2 - 15*s/4 + 1/2),r*(10*r*s**3 - 6*r*s**2 - 3*r*s + r + 10*s**3 - 6*s**2 - 3*s + 1)/4,r**2*(-r**3 - r**2 + r + 1)/4,r**4 + 15*r**2*s**3 + 6*r**2*s**2 - 15*r**2*s/2 - 3*r**2 - 15*s**3 - 6*s**2 + 15*s/2 + 2,-5*r**2*s**3 - 3*r**2*s**2 + 3*r**2*s/2 + r**2/2 + 5*s**3 + 3*s**2 - 3*s/2 - 1/2,r*(-r**4 + 2*r**2 - 1),r*(-3*r**4/2 + r**3 + 5*r**2/2 + 6*r*s**2 - 3*r - 6*s**2 + 1),2*r*s*(5*r*s**2 - 3*r - 5*s**2 + 3),r**2*(r**3 - r**2 - r + 1)/2,r**4 - 15*r**2*s**3 + 6*r**2*s**2 + 15*r**2*s/2 - 3*r**2 + 15*s**3 - 6*s**2 - 15*s/2 + 2,-5*r**2*s**3 + 3*r**2*s**2 + 3*r**2*s/2 - r**2/2 + 5*s**3 - 3*s**2 - 3*s/2 + 1/2,r*(-r**4 + 2*r**2 - 1),r*(3*r**4/2 + r**3 - 5*r**2/2 + 6*r*s**2 - 3*r + 6*s**2 - 1),2*r*s*(5*r*s**2 - 3*r + 5*s**2 - 3),r**2*(r**3 + r**2 - r - 1)/2,2*(-r**4 - 6*r**2*s**2 + 3*r**2 + 6*s**2 - 2),4*s*(-5*r**2*s**2 + 3*r**2 + 5*s**2 - 3),2*r*(r**4 - 2*r**2 + 1)],
                       [-15*r**4*s/4 - 15*r**4/8 - 2*r**3*s - r**3 + 15*r**2*s/4 + 15*r**2/8 - 15*r*s**4/4 - 2*r*s**3 + 15*r*s**2/4 + 3*r*s + r/2 - 15*s**4/8 - s**3 + 15*s**2/8 + s/2 - 1/4,s*(5*r*s**3/4 + r*s**2 - 3*r*s/4 - r/2 + 5*s**3/8 + s**2/2 - 3*s/8 - 1/4),r*(-5*r**3*s/4 - 5*r**3/8 - r**2*s - r**2/2 + 3*r*s/4 + 3*r/8 + s/2 + 1/4),15*r**4*s/4 + 15*r**4/8 - 2*r**3*s - r**3 - 15*r**2*s/4 - 15*r**2/8 - 15*r*s**4/4 - 2*r*s**3 + 15*r*s**2/4 + 3*r*s + r/2 + 15*s**4/8 + s**3 - 15*s**2/8 - s/2 + 1/4,s*(5*r*s**3/4 + r*s**2 - 3*r*s/4 - r/2 - 5*s**3/8 - s**2/2 + 3*s/8 + 1/4),r*(-5*r**3*s/4 - 5*r**3/8 + r**2*s + r**2/2 + 3*r*s/4 + 3*r/8 - s/2 - 1/4),15*r**4*s/4 - 15*r**4/8 - 2*r**3*s + r**3 - 15*r**2*s/4 + 15*r**2/8 + 15*r*s**4/4 - 2*r*s**3 - 15*r*s**2/4 + 3*r*s - r/2 - 15*s**4/8 + s**3 + 15*s**2/8 - s/2 - 1/4,s*(5*r*s**3/4 - r*s**2 - 3*r*s/4 + r/2 - 5*s**3/8 + s**2/2 + 3*s/8 - 1/4),r*(-5*r**3*s/4 + 5*r**3/8 + r**2*s - r**2/2 + 3*r*s/4 - 3*r/8 - s/2 + 1/4),-15*r**4*s/4 + 15*r**4/8 - 2*r**3*s + r**3 + 15*r**2*s/4 - 15*r**2/8 + 15*r*s**4/4 - 2*r*s**3 - 15*r*s**2/4 + 3*r*s - r/2 + 15*s**4/8 - s**3 - 15*s**2/8 + s/2 + 1/4,s*(5*r*s**3/4 - r*s**2 - 3*r*s/4 + r/2 + 5*s**3/8 - s**2/2 - 3*s/8 + 1/4),r*(-5*r**3*s/4 + 5*r**3/8 - r**2*s + r**2/2 + 3*r*s/4 - 3*r/8 + s/2 - 1/4),r*(4*r**2*s + 2*r**2 + 15*s**4/2 + 4*s**3 - 15*s**2/2 - 6*s - 1),r*s*(-5*s**3/2 - 2*s**2 + 3*s/2 + 1),-5*r**4*s - 5*r**4/2 + 6*r**2*s + 3*r**2 - s - 1/2,s*(-15*r**4/2 + 4*r**3 + 15*r**2/2 + 4*r*s**2 - 6*r - 2*s**2 + 1),5*r*s**4 - 6*r*s**2 + r - 5*s**4/2 + 3*s**2 - 1/2,r*s*(5*r**3/2 - 2*r**2 - 3*r/2 + 1),r*(4*r**2*s - 2*r**2 - 15*s**4/2 + 4*s**3 + 15*s**2/2 - 6*s + 1),r*s*(-5*s**3/2 + 2*s**2 + 3*s/2 - 1),-5*r**4*s + 5*r**4/2 + 6*r**2*s - 3*r**2 - s + 1/2,s*(15*r**4/2 + 4*r**3 - 15*r**2/2 + 4*r*s**2 - 6*r + 2*s**2 - 1),5*r*s**4 - 6*r*s**2 + r + 5*s**4/2 - 3*s**2 + 1/2,r*s*(5*r**3/2 + 2*r**2 - 3*r/2 - 1),4*r*s*(-2*r**2 - 2*s**2 + 3), 2*r*(-5*s**4 + 6*s**2 - 1),2*s*(5*r**4 - 6*r**2 + 1)]])
    
    #das funções de forma do elemento do estado plano de 9 nós
    dN_w = np.array([[r*s**2/2 + r*s/2 + s**2/4 + s/4, r**2*s/2 + r**2/4 + r*s/2 + r/4],
                        [r*s**2/2 + r*s/2 - s**2/4 - s/4, r**2*s/2 + r**2/4 - r*s/2 - r/4],
                        [r*s**2/2 - r*s/2 - s**2/4 + s/4, r**2*s/2 - r**2/4 - r*s/2 + r/4],
                        [r*s**2/2 - r*s/2 + s**2/4 - s/4, r**2*s/2 - r**2/4 + r*s/2 - r/4],
                        [                  -r*s**2 - r*s,      -r**2*s - r**2/2 + s + 1/2],
                        [     -r*s**2 + r + s**2/2 - 1/2,                   -r**2*s + r*s],
                        [                  -r*s**2 + r*s,      -r**2*s + r**2/2 + s - 1/2],
                        [     -r*s**2 + r - s**2/2 + 1/2,                   -r**2*s - r*s],
                        [                 2*r*s**2 - 2*r,                  2*r**2*s - 2*s]])
    dN2_w = np.array([[ s*(s + 1)/2,  r*(r + 1)/2],
                        [ s*(s + 1)/2,  r*(r - 1)/2],
                        [ s*(s - 1)/2,  r*(r - 1)/2],
                        [ s*(s - 1)/2,  r*(r + 1)/2],
                        [  -s*(s + 1),    -r**2 + 1],
                        [   -s**2 + 1,   r*(-r + 1)],
                        [  s*(-s + 1),    -r**2 + 1],
                        [   -s**2 + 1,   -r*(r + 1)],
                        [2*(s**2 - 1), 2*(r**2 - 1)]])
    dNrs_w = np.array([[r*s + r/2 + s/2 + 1/4],
                        [r*s + r/2 - s/2 - 1/4],
                        [r*s - r/2 - s/2 + 1/4],
                        [r*s - r/2 + s/2 - 1/4],
                        [         -r*(2*s + 1)],
                        [         s*(-2*r + 1)],
                        [         r*(-2*s + 1)],
                        [         -s*(2*r + 1)],
                        [                4*r*s]])
    
    #Jacobiano calculado utilizando as funções de forma do estado plano
    J = np.matmul(Xe.T, dN_w)
    dJ = np.matmul(Xe.T, dN2_w)
    dJrs = np.matmul(Xe.T, dNrs_w)
    
    J23 = np.array([ [dJ[0,0], dJ[1,0]],
                     [dJ[0,1], dJ[1,1]],
                     [dJrs[0,0], dJrs[1,0]] ])
    
    #jacobiano expandido
    Jex = np.array([ [     J[0,0]**2,     J[1,0]**2,               2*J[0,0]*J[1,0] ],
                     [     J[0,1]**2,     J[1,1]**2,               2*J[0,1]*J[1,1] ],
                     [ J[0,0]*J[0,1], J[1,0]*J[1,1], J[1,0]*J[0,1] + J[0,0]*J[1,1] ]])
    
    JI = np.linalg.inv(J)
    JexI = np.linalg.inv(Jex)
    
    ##derivadas das funções de interpolação do elemento no sistema local x y para placas
    dNdxl = np.matmul(dNdr.T, JI)
    B = np.matmul(JexI, ( d2Ndr2 - np.matmul(J23, dNdxl.T) ) )

    return B, J

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
    D = 1/12 * E * t**3/(1 - nu**2) * np.array([ [1, nu, 0], 
                                                  [nu, 1, 0], 
                                                  [0,  0, (1 - nu)/2] ])
    #número de graus de liberdade por elemento
    GLe = 27
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
        B, J = dNdx(Xe, PG[p])
        Ke += np.matmul( np.matmul(np.transpose(B), D), B) * wPG[p, 0] * wPG[p, 1] * np.linalg.det(J)
    return Ke

#tamanho da placa
lx = 8. #m
ly = 4. #m

#propriedades mecânicas do material da estrutura e espessura
E = 200e6 #kN/m2
nu = 0.3
t = 0.1 #m

#discretização em elementos por direção
#dx = 4 #elementos
#dy = 2 #elementos
dx = 128 #elementos
dy = 64 #elementos

print('Iniciando a geração da malha...')
#geração das coordenadas dos nós
xv, yv = np.meshgrid(np.linspace(0., lx, dx*2 + 1, endpoint=True), np.linspace(0., ly, dy*2 + 1, endpoint=True))
NOS = np.zeros((xv.shape[0]*xv.shape[1], 2))
NOS[:,0] = xv.flatten()
NOS[:,1] = yv.flatten()

#geração da incidência dos elementos
indicesNOS = np.arange(0, xv.shape[0]*xv.shape[1]).reshape(xv.shape)

infx = np.arange(0, dx*2, 2)
supx = np.arange(0, dx*2, 2) + 3

infy = np.arange(0, dy*2, 2)
supy = np.arange(0, dy*2, 2) + 3

IE = []

for i in range(dx):
    for j in range(dy):
        IE.append(indicesNOS[infy[j]:supy[j], infx[i]:supx[i]].flatten())

del indicesNOS, infx, supx, infy, supy, xv, yv

IE = np.array(IE)
IE = IE[:, [8, 6, 0, 2, 7, 3, 1, 5, 4]] #reordenando os nós de cada elementos para bater com as funções de forma!!

#indexação dos graus de liberdade
ID = []

for e in IE:
    ID.append( np.repeat(e*3, 3) + np.tile(np.array([0, 1, 2]), 9) )#3 graus de liberdade no elemento IE[i]*3, 3 e 9 nós no np.tile

ID = np.array(ID)

print('Geração da malha completa!')
print("Iniciando a determinação das matrizes de rigidez do elemento...")

#determinação da matriz de rigidez dos elementos, todas são iguais!!!
Ke = []
for e in IE:
    Ke.append( ke(NOS[e], E, nu, t) )
Ke = np.array(Ke)

del dNdx, ke

print('Determinação das matrizes de rigidez dos elementos completa!')
print("Iniciando a determinação das matrizes da estrutura...")

#seleção dos nós restringidos em x = 0 e dos nós com carga em x = lx e determinação dos graus de liberdade restringidos e com carga (só no transversal de deslcamento)
nR = []
nF = []
for n in np.arange(0, NOS.shape[0]):
    if NOS[n, 0] == 0.:
        nR.append(n)
    elif NOS[n, 0] == lx:
        nF.append(n)
nR = np.repeat(np.array(nR)*3, 3) + np.tile(np.array([0, 1, 2]), np.array(nR).shape[0]) #graus de liberdade restringidos
nF = np.array(nF)*3 #graus de liberdade com carga concentrada
nL = np.delete(np.arange(0, ID.max()+1), nR) #graus de liberdade livres

#quantidade de graus de liberdade da estrutura GL e de graus de liberdade livres DOF
GL = NOS.shape[0]*3
DOF = GL - nR.shape[0]*3

#montagem da matriz de rigidez da estrutura
K = np.zeros((GL, GL))
for e in range(ID.shape[0]):
    for i in range(ID.shape[1]):
        for j in range(ID.shape[1]):
            K[ ID[e, i], ID[e, j] ] += Ke[e, i, j]

del Ke

#separação das matrizes de rigidez
Ku = np.delete(np.delete(K, nR, 0), nR, 1)
Kr = np.delete(np.delete(K, nL, 0), nR, 1)

#vetor de forças nodais
F = np.zeros(GL)
F[nF] = -100./nF.shape[0] #kN
print('Valor da carga no nó: ', -100./nF.shape[0])

Fu = np.delete(F, nR, 0)
Fr = np.delete(F, nL, 0)

print('Determinação das matrizes de rigidez da estrutura completa!')
print("Iniciando o cálculo dos deslocamentos...")

#del F, K

#cálculo dos deslocamentos -------------------------------------------------------------------------------------
Uu = np.linalg.solve(Ku, Fu)
Rr = np.matmul(Kr, Uu) - Fr

U = np.zeros(GL)
U[nL] = Uu

Uzxy = U.reshape(NOS.shape[0], 3)

print('Cálculo dos deslocamentos completo!')
print(Uzxy[-1])

##geração do arquivo vtu
#pontos = np.zeros((NOS.shape[0], 3))
#pontos[:, [0, 1]] = NOS
#celulas = {'quad9': IE}
#meshio.write_points_cells(
#        "placa9.vtu",
#        pontos,
#        celulas,
#        # Optionally provide extra data on points, cells, etc.
#        point_data = {"U": Uzxy},
#        # cell_data=cell_data,
#        # field_data=field_data
#        )




