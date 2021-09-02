#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 08:30:06 2021

@author: argenta
"""

import sympy as sp

x, y, z, l, E, A, Ao, t, nu = sp.symbols('x, y, z, l, E, A, Ao, t, nu')
t_w, b, t_f, b_f, h = sp.symbols('t_w, b, t_f, b_f, h') #h é a distância entre faces internas de mesas

#derivadas das funções de forma (de /home/markinho/Dropbox/DocumentaçõesMK/MEFaplicado-mkdocs/docs/porticos_espacial/Codigos/Derivando-FuncoesFormaPorticoEspacial2nos.py):
#trelica e torcao
Bt = sp.Matrix([[-1/l],[ 1/l]])
#viga direção de y
Bvy = -y*sp.Matrix([[      12*x/l**3],
                    [-1/l + 6*x/l**2],
                    [     -12*x/l**3],
                    [ 1/l + 6*x/l**2]])
#viga direção de z
Bvz = -z*sp.Matrix([[      12*x/l**3],
                    [-1/l + 6*x/l**2],
                    [     -12*x/l**3],
                    [ 1/l + 6*x/l**2]])

# matriz das derivadas das funções de interpolação do pórtico
# Bv = -s * sp.diff( sp.diff(Nn, r), r)
# Bp = sp.Matrix([[ -1/l, 0, 0, 1/l, 0, 0 ], [ 0,  Bv[0], Bv[1], 0, Bv[2], Bv[3] ] ])

##vetor de deformações genérico
ug0, ug1, ug2, ug3, ug4, ug5, ug6, ug7, ug8, ug9, ug10, ug11 = sp.symbols('ug0, ug1, ug2, ug3, ug4, ug5, ug6, ug7, ug8, ug9, ug10, ug11')

UgN = sp.Matrix([ug0, ug6])
UgT = sp.Matrix([ug3, ug9])

UgMy = sp.Matrix([ug2, ug5, ug8, ug11])
UgMz = sp.Matrix([ug1, ug4, ug7, ug10])


#deformacoes
epsilonN = (Bt.T*UgN)[0]
epsilonT = (Bt.T*UgT)[0]
epsilonMy = (Bvy.T*UgMy)[0]
epsilonMz = (Bvz.T*UgMz)[0]

#tensoes
sigmaN = epsilonN*E
sigmaT = epsilonT*E/(2*(1 + nu))
sigmaMy = epsilonMy*E
sigmaMz = epsilonMz*E

#integrações
normais = sigmaN*A
torsao = 2*sigmaT*t*Ao
momentoy = 2 * t_w * sp.integrate( y * sigmaMy, (y, -h/2, h/2 ) ) + 2 * b_f * sp.integrate( y * sigmaMy, (y, h/2, h/2 + t_f ) )
momentoz = 2 * t_f * sp.integrate( z * sigmaMz, (z, -b_f/2, b_f/2 ) ) + 2 * h * sp.integrate( z * sigmaMz, (z, b_f/2 - t_w, b_f/2 ) )
cortey = sp.diff(momentoy, x)
cortez = sp.diff(momentoz, x)