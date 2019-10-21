#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

ValueError: Matrix det == 0; not invertible.

NÃO FUNCIONA!!!! IMPLEMENTAR O MITC9!!!

@author: markinho
"""

import sympy as sp
import numpy as np
#import dNdx9nosKL
#import matplotlib.pyplot as plt
#import plotly.graph_objects as go
#import meshio

#elemento padrão
r1, s1, r2, s2, r3, s3, r4, s4, r5, s5, r6, s6, r7, s7, r8, s8, r9, s9 = sp.symbols('r1, s1, r2, s2, r3, s3, r4, s4, r5, s5, r6, s6, r7, s7, r8, s8, r9, s9')
##coordenadas dos nós
#r1 = 1
#s1 = 1
#
#r2 = -1
#s2 = 1
#
#r3 = -1
#s3 = -1
#
#r4 = 1
#s4 = -1
#
#r5 = 0
#s5 = 1
#
#r6 = -1
#s6 = 0
#
#r7 = 0
#s7 = -1
#
#r8 = 1
#s8 = 0
#
#r9 = 0
#s9 = 0

#u1, u2, u3, u4, u5, u6, u7, u8, u9 = sp.symbols('u1, u2, u3, u4, u5, u6, u7, u8, u9')
w1, w2, w3, w4, w5, w6, w7, w8, w9 = sp.symbols('w1, w2, w3, w4, w5, w6, w7, w8, w9')
rr1, rr2, rr3, rr4, rr5, rr6, rr7, rr8, rr9 = sp.symbols('rr1, rr2, rr3, rr4, rr5, rr6, rr7, rr8, rr9')
rs1, rs2, rs3, rs4, rs5, rs6, rs7, rs8, rs9 = sp.symbols('rs1, rs2, rs3, rs4, rs5, rs6, rs7, rs8, rs9')

##estado plano de tensões
#Mat_CoefEP = sp.Matrix([[1, r1, s1, r1*s1, r1**2, r1**2*s1, r1**2*s1**2, r1*s1**2, s1**2],  #no1
#                        [1, r2, s2, r2*s2, r2**2, r2**2*s2, r2**2*s2**2, r2*s2**2, s2**2],  #no2
#                        [1, r3, s3, r3*s3, r3**2, r3**2*s3, r3**2*s3**2, r3*s3**2, s3**2],  #no3
#                        [1, r4, s4, r4*s4, r4**2, r4**2*s4, r4**2*s4**2, r4*s4**2, s4**2],  #no4
#                        [1, r5, s5, r5*s5, r5**2, r5**2*s5, r5**2*s5**2, r5*s5**2, s5**2],  #no5
#                        [1, r6, s6, r6*s6, r6**2, r6**2*s6, r6**2*s6**2, r6*s6**2, s6**2],  #no6
#                        [1, r7, s7, r7*s7, r7**2, r7**2*s7, r7**2*s7**2, r7*s7**2, s7**2],  #no7
#                        [1, r8, s8, r8*s8, r8**2, r8**2*s8, r8**2*s8**2, r8*s8**2, s8**2],  #no8
#                        [1, r9, s9, r9*s9, r9**2, r9**2*s9, r9**2*s9**2, r9*s9**2, s9**2]]) #no9

#placa plana
Mat_CoefP = sp.Matrix([[1, r1, s1, r1*s1, r1**2, r1**2*s1, r1**2*s1**2, r1*s1**2, s1**2,   r1**3,   r1**3*s1,   r1**3*s1**2,   r1**3*s1**3,   r1**2*s1**3,   r1*s1**3,   s1**3,    r1**4,   r1**4*s1,   r1**4*s1**2,   r1**4*s1**3,   r1**4*s1**4,   r1**3*s1**4,   r1**2*s1**4,    r1*s1**4,    s1**4,   r1**5,   s1**5],  #no1
                       [0,  0,  1,    r1,     0,    r1**2,  r1**2*2*s1,  r1*2*s1,  2*s1,       0,      r1**3,    r1**3*2*s1, r1**3*3*s1**2, r1**2*3*s1**2, r1*3*s1**2, 3*s1**2,        0,      r1**4,    r1**4*2*s1, r1**4*3*s1**2, r1**4*4*s1**3, r1**3*4*s1**3, r1**2*4*s1**3,  r1*4*s1**3,  4*s1**3,       0, 5*s1**4],  #no1Rs
                       [0,  1,  0,    s1,  2*r1,  2*r1*s1,  2*r1*s1**2,    s1**2,     0, 3*r1**2, 3*r1**2*s1, 3*r1**2*s1**2, 3*r1**2*s1**3,    2*r1*s1**3,      s1**3,       0,  4*r1**3, 4*r1**3*s1, 4*r1**3*s1**2, 4*r1**3*s1**3, 4*r1**3*s1**4, 3*r1**2*s1**4,    2*r1*s1**4,       s1**4,        0, 5*r1**4,       0],  #no1Rr                       
                       [1, r2, s2, r2*s2, r2**2, r2**2*s2, r2**2*s2**2, r2*s2**2, s2**2,   r2**3,   r2**3*s2,   r2**3*s2**2,   r2**3*s2**3,   r2**2*s2**3,   r2*s2**3,   s2**3,    r2**4,   r2**4*s2,   r2**4*s2**2,   r2**4*s2**3,   r2**4*s2**4,   r2**3*s2**4,   r2**2*s2**4,    r2*s2**4,    s2**4,   r2**5,   s2**5],  #no2                       
                       [0,  0,  1,    r2,     0,    r2**2,  r2**2*2*s2,  r2*2*s2,  2*s2,       0,      r2**3,    r2**3*2*s2, r2**3*3*s2**2, r2**2*3*s2**2, r2*3*s2**2, 3*s2**2,        0,      r2**4,    r2**4*2*s2, r2**4*3*s2**2, r2**4*4*s2**3, r2**3*4*s2**3, r2**2*4*s2**3,  r2*4*s2**3,  4*s2**3,       0, 5*s2**4],  #no2Rs                       
                       [0,  1,  0,    s2,  2*r2,  2*r2*s2,  2*r2*s2**2,    s2**2,     0, 3*r2**2, 3*r2**2*s2, 3*r2**2*s2**2, 3*r2**2*s2**3,    2*r2*s2**3,      s2**3,       0,  4*r2**3, 4*r2**3*s2, 4*r2**3*s2**2, 4*r2**3*s2**3, 4*r2**3*s2**4, 3*r2**2*s2**4,    2*r2*s2**4,       s2**4,        0, 5*r2**4,       0],  #no2Rr
                       [1, r3, s3, r3*s3, r3**2, r3**2*s3, r3**2*s3**2, r3*s3**2, s3**2,   r3**3,   r3**3*s3,   r3**3*s3**2,   r3**3*s3**3,   r3**2*s3**3,   r3*s3**3,   s3**3,    r3**4,   r3**4*s3,   r3**4*s3**2,   r3**4*s3**3,   r3**4*s3**4,   r3**3*s3**4,   r3**2*s3**4,    r3*s3**4,    s3**4,   r3**5,   s3**5],  #no3
                       [0,  0,  1,    r3,     0,    r3**2,  r3**2*2*s3,  r3*2*s3,  2*s3,       0,      r3**3,    r3**3*2*s3, r3**3*3*s3**2, r3**2*3*s3**2, r3*3*s3**2, 3*s3**2,        0,      r3**4,    r3**4*2*s3, r3**4*3*s3**2, r3**4*4*s3**3, r3**3*4*s3**3, r3**2*4*s3**3,  r3*4*s3**3,  4*s3**3,       0, 5*s3**4],  #no3Rs                       
                       [0,  1,  0,    s3,  2*r3,  2*r3*s3,  2*r3*s3**2,    s3**2,     0, 3*r3**2, 3*r3**2*s3, 3*r3**2*s3**2, 3*r3**2*s3**3,    2*r3*s3**3,      s3**3,       0,  4*r3**3, 4*r3**3*s3, 4*r3**3*s3**2, 4*r3**3*s3**3, 4*r3**3*s3**4, 3*r3**2*s3**4,    2*r3*s3**4,       s3**4,        0, 5*r3**4,       0],  #no3Rr                       
                       [1, r4, s4, r4*s4, r4**2, r4**2*s4, r4**2*s4**2, r4*s4**2, s4**2,   r4**3,   r4**3*s4,   r4**3*s4**2,   r4**3*s4**3,   r4**2*s4**3,   r4*s4**3,   s4**3,    r4**4,   r4**4*s4,   r4**4*s4**2,   r4**4*s4**3,   r4**4*s4**4,   r4**3*s4**4,   r4**2*s4**4,    r4*s4**4,    s4**4,   r4**5,   s4**5],  #no4
                       [0,  0,  1,    r4,     0,    r4**2,  r4**2*2*s4,  r4*2*s4,  2*s4,       0,      r4**3,    r4**3*2*s4, r4**3*3*s4**2, r4**2*3*s4**2, r4*3*s4**2, 3*s4**2,        0,      r4**4,    r4**4*2*s4, r4**4*3*s4**2, r4**4*4*s4**3, r4**3*4*s4**3, r4**2*4*s4**3,  r4*4*s4**3,  4*s4**3,       0, 5*s4**4],  #no4Rs                       
                       [0,  1,  0,    s4,  2*r4,  2*r4*s4,  2*r4*s4**2,    s4**2,     0, 3*r4**2, 3*r4**2*s4, 3*r4**2*s4**2, 3*r4**2*s4**3,    2*r4*s4**3,      s4**3,       0,  4*r4**3, 4*r4**3*s4, 4*r4**3*s4**2, 4*r4**3*s4**3, 4*r4**3*s4**4, 3*r4**2*s4**4,    2*r4*s4**4,       s4**4,        0, 5*r4**4,       0],  #no4Rr                       
                       [1, r5, s5, r5*s5, r5**2, r5**2*s5, r5**2*s5**2, r5*s5**2, s5**2,   r5**3,   r5**3*s5,   r5**3*s5**2,   r5**3*s5**3,   r5**2*s5**3,   r5*s5**3,   s5**3,    r5**4,   r5**4*s5,   r5**4*s5**2,   r5**4*s5**3,   r5**4*s5**4,   r5**3*s5**4,   r5**2*s5**4,    r5*s5**4,    s5**4,   r5**5,   s5**5],  #no5
                       [0,  0,  1,    r5,     0,    r5**2,  r5**2*2*s5,  r5*2*s5,  2*s5,       0,      r5**3,    r5**3*2*s5, r5**3*3*s5**2, r5**2*3*s5**2, r5*3*s5**2, 3*s5**2,        0,      r5**4,    r5**4*2*s5, r5**4*3*s5**2, r5**4*4*s5**3, r5**3*4*s5**3, r5**2*4*s5**3,  r5*4*s5**3,  4*s5**3,       0, 5*s5**4],  #no5Rs                       
                       [0,  1,  0,    s5,  2*r5,  2*r5*s5,  2*r5*s5**2,    s5**2,     0, 3*r5**2, 3*r5**2*s5, 3*r5**2*s5**2, 3*r5**2*s5**3,    2*r5*s5**3,      s5**3,       0,  4*r5**3, 4*r5**3*s5, 4*r5**3*s5**2, 4*r5**3*s5**3, 4*r5**3*s5**4, 3*r5**2*s5**4,    2*r5*s5**4,       s5**4,        0, 5*r5**4,       0],  #no5Rr                       
                       [1, r6, s6, r6*s6, r6**2, r6**2*s6, r6**2*s6**2, r6*s6**2, s6**2,   r6**3,   r6**3*s6,   r6**3*s6**2,   r6**3*s6**3,   r6**2*s6**3,   r6*s6**3,   s6**3,    r6**4,   r6**4*s6,   r6**4*s6**2,   r6**4*s6**3,   r6**4*s6**4,   r6**3*s6**4,   r6**2*s6**4,    r6*s6**4,    s6**4,   r6**5,   s6**5],  #no6
                       [0,  0,  1,    r6,     0,    r6**2,  r6**2*2*s6,  r6*2*s6,  2*s6,       0,      r6**3,    r6**3*2*s6, r6**3*3*s6**2, r6**2*3*s6**2, r6*3*s6**2, 3*s6**2,        0,      r6**4,    r6**4*2*s6, r6**4*3*s6**2, r6**4*4*s6**3, r6**3*4*s6**3, r6**2*4*s6**3,  r6*4*s6**3,  4*s6**3,       0, 5*s6**4],  #no6Rs                       
                       [0,  1,  0,    s6,  2*r6,  2*r6*s6,  2*r6*s6**2,    s6**2,     0, 3*r6**2, 3*r6**2*s6, 3*r6**2*s6**2, 3*r6**2*s6**3,    2*r6*s6**3,      s6**3,       0,  4*r6**3, 4*r6**3*s6, 4*r6**3*s6**2, 4*r6**3*s6**3, 4*r6**3*s6**4, 3*r6**2*s6**4,    2*r6*s6**4,       s6**4,        0, 5*r6**4,       0],  #no6Rr                       
                       [1, r7, s7, r7*s7, r7**2, r7**2*s7, r7**2*s7**2, r7*s7**2, s7**2,   r7**3,   r7**3*s7,   r7**3*s7**2,   r7**3*s7**3,   r7**2*s7**3,   r7*s7**3,   s7**3,    r7**4,   r7**4*s7,   r7**4*s7**2,   r7**4*s7**3,   r7**4*s7**4,   r7**3*s7**4,   r7**2*s7**4,    r7*s7**4,    s7**4,   r7**5,   s7**5],  #no7
                       [0,  0,  1,    r7,     0,    r7**2,  r7**2*2*s7,  r7*2*s7,  2*s7,       0,      r7**3,    r7**3*2*s7, r7**3*3*s7**2, r7**2*3*s7**2, r7*3*s7**2, 3*s7**2,        0,      r7**4,    r7**4*2*s7, r7**4*3*s7**2, r7**4*4*s7**3, r7**3*4*s7**3, r7**2*4*s7**3,  r7*4*s7**3,  4*s7**3,       0, 5*s7**4],  #no7Rs                       
                       [0,  1,  0,    s7,  2*r7,  2*r7*s7,  2*r7*s7**2,    s7**2,     0, 3*r7**2, 3*r7**2*s7, 3*r7**2*s7**2, 3*r7**2*s7**3,    2*r7*s7**3,      s7**3,       0,  4*r7**3, 4*r7**3*s7, 4*r7**3*s7**2, 4*r7**3*s7**3, 4*r7**3*s7**4, 3*r7**2*s7**4,    2*r7*s7**4,       s7**4,        0, 5*r7**4,       0],  #no7Rr                       
                       [1, r8, s8, r8*s8, r8**2, r8**2*s8, r8**2*s8**2, r8*s8**2, s8**2,   r8**3,   r8**3*s8,   r8**3*s8**2,   r8**3*s8**3,   r8**2*s8**3,   r8*s8**3,   s8**3,    r8**4,   r8**4*s8,   r8**4*s8**2,   r8**4*s8**3,   r8**4*s8**4,   r8**3*s8**4,   r8**2*s8**4,    r8*s8**4,    s8**4,   r8**5,   s8**5],  #no8
                       [0,  0,  1,    r8,     0,    r8**2,  r8**2*2*s8,  r8*2*s8,  2*s8,       0,      r8**3,    r8**3*2*s8, r8**3*3*s8**2, r8**2*3*s8**2, r8*3*s8**2, 3*s8**2,        0,      r8**4,    r8**4*2*s8, r8**4*3*s8**2, r8**4*4*s8**3, r8**3*4*s8**3, r8**2*4*s8**3,  r8*4*s8**3,  4*s8**3,       0, 5*s8**4],  #no8Rs                       
                       [0,  1,  0,    s8,  2*r8,  2*r8*s8,  2*r8*s8**2,    s8**2,     0, 3*r8**2, 3*r8**2*s8, 3*r8**2*s8**2, 3*r8**2*s8**3,    2*r8*s8**3,      s8**3,       0,  4*r8**3, 4*r8**3*s8, 4*r8**3*s8**2, 4*r8**3*s8**3, 4*r8**3*s8**4, 3*r8**2*s8**4,    2*r8*s8**4,       s8**4,        0, 5*r8**4,       0],  #no8Rr                       
                       [1, r9, s9, r9*s9, r9**2, r9**2*s9, r9**2*s9**2, r9*s9**2, s9**2,   r9**3,   r9**3*s9,   r9**3*s9**2,   r9**3*s9**3,   r9**2*s9**3,   r9*s9**3,   s9**3,    r9**4,   r9**4*s9,   r9**4*s9**2,   r9**4*s9**3,   r9**4*s9**4,   r9**3*s9**4,   r9**2*s9**4,    r9*s9**4,    s9**4,   r9**5,   s9**5],  #no9
                       [0,  0,  1,    r9,     0,    r9**2,  r9**2*2*s9,  r9*2*s9,  2*s9,       0,      r9**3,    r9**3*2*s9, r9**3*3*s9**2, r9**2*3*s9**2, r9*3*s9**2, 3*s9**2,        0,      r9**4,    r9**4*2*s9, r9**4*3*s9**2, r9**4*4*s9**3, r9**3*4*s9**3, r9**2*4*s9**3,  r9*4*s9**3,  4*s9**3,       0, 5*s9**4],  #no9Rs
                       [0,  1,  0,    s9,  2*r9,  2*r9*s9,  2*r9*s9**2,    s9**2,     0, 3*r9**2, 3*r9**2*s9, 3*r9**2*s9**2, 3*r9**2*s9**3,    2*r9*s9**3,      s9**3,       0,  4*r9**3, 4*r9**3*s9, 4*r9**3*s9**2, 4*r9**3*s9**3, 4*r9**3*s9**4, 3*r9**2*s9**4,    2*r9*s9**4,       s9**4,        0, 5*r9**4,       0]]) #no9Rr
                       

#teste = np.array(Mat_CoefP, dtype=float)

#U = sp.Matrix([u1, u2, u3, u4, u5, u6, u7, u8, u9])
WR = sp.Matrix([w1, rr1, rs1, w2, rr2, rs2, w3, rr3, rs3, w4, rr4, rs4, w5, rr5, rs5, w6, rr6, rs6, w7, rr7, rs7, w8, rr8, rs8, w9, rr9, rs9])

#CoefsEP = Mat_CoefEP.inv() * U
CoefsP = Mat_CoefP.inv() * WR

#Aep, Bep, Cep, Dep, Eep, Fep, Gep, Hep, Iep = CoefsEP[0], CoefsEP[1], CoefsEP[2], CoefsEP[3], CoefsEP[4], CoefsEP[5], CoefsEP[6], CoefsEP[7], CoefsEP[8]
Ap, Bp, Cp, Dp, Ep, Fp, Gp, Hp, Ip, Jp, Kp, Lp, Mp, Np, Op, Pp, Qp, Rp, Sp, Tp, Up, Vp, Xp, Wp, Yp, Zp, AAp = CoefsP[0], CoefsP[1], CoefsP[2], CoefsP[3], CoefsP[4], CoefsP[5], CoefsP[6], CoefsP[7], CoefsP[8], CoefsP[9], CoefsP[10], CoefsP[11], CoefsP[12], CoefsP[13], CoefsP[14], CoefsP[15], CoefsP[16], CoefsP[17], CoefsP[18], CoefsP[19], CoefsP[20],CoefsP[21], CoefsP[22], CoefsP[23], CoefsP[24], CoefsP[25], CoefsP[26]

r = sp.Symbol('r')
s = sp.Symbol('s')

#estado plano
#NsEP = sp.expand(Aep + Bep*r + Cep*s + Dep*r*s + Eep*r**2 + Fep*r**2*s + Gep*r**2*s**2 + Hep*r*s**2 + Iep*s**2)

#N1EP = sp.Add(*[argi for argi in NsEP.args if argi.has(u1)]).subs(u1, 1)
#N2EP = sp.Add(*[argi for argi in NsEP.args if argi.has(u2)]).subs(u2, 1)
#N3EP = sp.Add(*[argi for argi in NsEP.args if argi.has(u3)]).subs(u3, 1)
#N4EP = sp.Add(*[argi for argi in NsEP.args if argi.has(u4)]).subs(u4, 1)
#N5EP = sp.Add(*[argi for argi in NsEP.args if argi.has(u5)]).subs(u5, 1)
#N6EP = sp.Add(*[argi for argi in NsEP.args if argi.has(u6)]).subs(u6, 1)
#N7EP = sp.Add(*[argi for argi in NsEP.args if argi.has(u7)]).subs(u7, 1)
#N8EP = sp.Add(*[argi for argi in NsEP.args if argi.has(u8)]).subs(u8, 1)
#N9EP = sp.Add(*[argi for argi in NsEP.args if argi.has(u9)]).subs(u9, 1)
#
#NEP = sp.Matrix([N1EP, N2EP, N3EP, N4EP, N5EP, N6EP, N7EP, N8EP, N9EP])

#placa
NsP = sp.expand(Ap + Bp*r1 + Cp*s1 + Dp*r1*s1 + Ep*r1**2 + Fp*r1**2*s1 + Gp*r1**2*s1**2 + Hp*r1*s1**2 + Ip*s1**2 + Jp*r1**3 + Kp*r1**3*s1 + Lp*r1**3*s1**2 + Mp*r1**3*s1**3 + Np*r1**2*s1**3 + Op*r1*s1**3 + Pp*s1**3 +  Qp*r1**4 + Rp*r1**4*s1 + Sp*r1**4*s1**2 + Tp*r1**4*s1**3 + Up*r1**4*s1**4 + Vp*r1**3*s1**4 + Xp*r1**2*s1**4 +  Wp*r1*s1**4 +  Yp*s1**4 + Zp*r1**5 + AAp*s1**5)

N1p = sp.Add(*[argi for argi in NsP.args if argi.has( w1)]).subs( w1, 1)
N2p = sp.Add(*[argi for argi in NsP.args if argi.has(rr1)]).subs(rr1, 1)
N3p = sp.Add(*[argi for argi in NsP.args if argi.has(rs1)]).subs(rs1, 1)
N4p = sp.Add(*[argi for argi in NsP.args if argi.has( w2)]).subs( w2, 1)
N5p = sp.Add(*[argi for argi in NsP.args if argi.has(rr2)]).subs(rr2, 1)
N6p = sp.Add(*[argi for argi in NsP.args if argi.has(rs2)]).subs(rs2, 1)
N7p = sp.Add(*[argi for argi in NsP.args if argi.has( w3)]).subs( w3, 1)
N8p = sp.Add(*[argi for argi in NsP.args if argi.has(rr3)]).subs(rr3, 1)
N9p = sp.Add(*[argi for argi in NsP.args if argi.has(rs3)]).subs(rs3, 1)
N10p = sp.Add(*[argi for argi in NsP.args if argi.has( w4)]).subs( w4, 1)
N11p = sp.Add(*[argi for argi in NsP.args if argi.has(rr4)]).subs(rr4, 1)
N12p = sp.Add(*[argi for argi in NsP.args if argi.has(rs4)]).subs(rs4, 1)
N13p = sp.Add(*[argi for argi in NsP.args if argi.has( w5)]).subs( w5, 1)
N14p = sp.Add(*[argi for argi in NsP.args if argi.has(rr5)]).subs(rr5, 1)
N15p = sp.Add(*[argi for argi in NsP.args if argi.has(rs5)]).subs(rs5, 1)
N16p = sp.Add(*[argi for argi in NsP.args if argi.has( w6)]).subs( w6, 1)
N17p = sp.Add(*[argi for argi in NsP.args if argi.has(rr6)]).subs(rr6, 1)
N18p = sp.Add(*[argi for argi in NsP.args if argi.has(rs6)]).subs(rs6, 1)
N19p = sp.Add(*[argi for argi in NsP.args if argi.has( w7)]).subs( w7, 1)
N20p = sp.Add(*[argi for argi in NsP.args if argi.has(rr7)]).subs(rr7, 1)
N21p = sp.Add(*[argi for argi in NsP.args if argi.has(rs7)]).subs(rs7, 1)
N22p = sp.Add(*[argi for argi in NsP.args if argi.has( w8)]).subs( w8, 1)
N23p = sp.Add(*[argi for argi in NsP.args if argi.has(rr8)]).subs(rr8, 1)
N24p = sp.Add(*[argi for argi in NsP.args if argi.has(rs8)]).subs(rs8, 1)
N25p = sp.Add(*[argi for argi in NsP.args if argi.has( w9)]).subs( w9, 1)
N26p = sp.Add(*[argi for argi in NsP.args if argi.has(rr9)]).subs(rr9, 1)
N27p = sp.Add(*[argi for argi in NsP.args if argi.has(rs9)]).subs(rs9, 1)

NP = sp.Matrix([N1p, N2p, N3p, N4p, N5p, N6p, N7p, N8p, N9p, N10p, N11p, N12p, N13p, N14p, N15p, N16p, N17p, N18p, N19p, N20p, N21p, N22p, N23p, N24p, N25p, N26p, N27p])

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
#
#fig = plty.subplots.make_subplots(rows=3, cols=3,
#    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
#           [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
#           [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]])
#
#fig.add_trace(dados1, row=1, col=1)
#fig.add_trace(dados2, row=1, col=2)
#fig.add_trace(dados3, row=1, col=3)
#fig.add_trace(dados4, row=2, col=1)
#fig.add_trace(dados5, row=2, col=2)
#fig.add_trace(dados6, row=2, col=3)
#fig.add_trace(dados7, row=3, col=1)
#fig.add_trace(dados8, row=3, col=2)
#fig.add_trace(dados9, row=3, col=3)
#
#fig.update_layout(title="Funções de forma do elemento quadrático" , autosize=True)#, width=900, height=700)
#
#fig.write_html('funcoesForma9nos.html')
##-------------------------------------------------------------------------------------------------------------

##primeira derivada em r
#dN1r = sp.diff(N1, r)#.subs({r: r1, s: s1})
#dN2r = sp.diff(N2, r)#.subs({r: r2, s: s2})
#dN3r = sp.diff(N3, r)#.subs({r: r3, s: s3})
#dN4r = sp.diff(N4, r)#.subs({r: r4, s: s4})
#dN5r = sp.diff(N5, r)#.subs({r: r5, s: s5})
#dN6r = sp.diff(N6, r)#.subs({r: r6, s: s6})
#dN7r = sp.diff(N7, r)#.subs({r: r7, s: s7})
#dN8r = sp.diff(N8, r)#.subs({r: r8, s: s8})
#dN9r = sp.diff(N9, r)#.subs({r: r9, s: s9})
##convertendo para função lambda nuympy
#ndN1r = sp.utilities.lambdify([r, s], dN1r, "numpy")
#ndN2r = sp.utilities.lambdify([r, s], dN2r, "numpy")
#ndN3r = sp.utilities.lambdify([r, s], dN3r, "numpy")
#ndN4r = sp.utilities.lambdify([r, s], dN4r, "numpy")
#ndN5r = sp.utilities.lambdify([r, s], dN5r, "numpy")
#ndN6r = sp.utilities.lambdify([r, s], dN6r, "numpy")
#ndN7r = sp.utilities.lambdify([r, s], dN7r, "numpy")
#ndN8r = sp.utilities.lambdify([r, s], dN8r, "numpy")
#ndN9r = sp.utilities.lambdify([r, s], dN9r, "numpy")
#
##primeira derivada em s
#dN1s = sp.diff(N1, s)#.subs({r: r1, s: s1})
#dN2s = sp.diff(N2, s)#.subs({r: r2, s: s2})
#dN3s = sp.diff(N3, s)#.subs({r: r3, s: s3})
#dN4s = sp.diff(N4, s)#.subs({r: r4, s: s4})
#dN5s = sp.diff(N5, s)#.subs({r: r5, s: s5})
#dN6s = sp.diff(N6, s)#.subs({r: r6, s: s6})
#dN7s = sp.diff(N7, s)#.subs({r: r7, s: s7})
#dN8s = sp.diff(N8, s)#.subs({r: r8, s: s8})
#dN9s = sp.diff(N9, s)#.subs({r: r9, s: s9})
##convertendo para função lambda nuympy
#ndN1s = sp.utilities.lambdify([r, s], dN1s, "numpy")
#ndN2s = sp.utilities.lambdify([r, s], dN2s, "numpy")
#ndN3s = sp.utilities.lambdify([r, s], dN3s, "numpy")
#ndN4s = sp.utilities.lambdify([r, s], dN4s, "numpy")
#ndN5s = sp.utilities.lambdify([r, s], dN5s, "numpy")
#ndN6s = sp.utilities.lambdify([r, s], dN6s, "numpy")
#ndN7s = sp.utilities.lambdify([r, s], dN7s, "numpy")
#ndN8s = sp.utilities.lambdify([r, s], dN8s, "numpy")
#ndN9s = sp.utilities.lambdify([r, s], dN9s, "numpy")
#
##gerando a matriz dNdx analítica
#x1 = sp.Symbol('x1')
#y1 = sp.Symbol('y1')
#x2 = sp.Symbol('x2')
#y2 = sp.Symbol('y2')
#x3 = sp.Symbol('x3')
#y3 = sp.Symbol('y3')
#x4 = sp.Symbol('x4')
#y4 = sp.Symbol('y4')
#x5 = sp.Symbol('x5')
#y5 = sp.Symbol('y5')
#x6 = sp.Symbol('x6')
#y6 = sp.Symbol('y6')
#x7 = sp.Symbol('x7')
#y7 = sp.Symbol('y7')
#x8 = sp.Symbol('x8')
#y8 = sp.Symbol('y8')
#x9 = sp.Symbol('x9')
#y9 = sp.Symbol('y9')
#
##Matriz dos nós de um elemento
#Xe = sp.Matrix([[x1, y1],[x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6], [x7, y7], [x8, y8], [x9, y9]])
##Matriz das derivadas das funções de interpolação do elemento padrão no sistema r s
#dNds = sp.Matrix([[dN1r, dN1s], [dN2r, dN2s], [dN3r, dN3s], [dN4r, dN4s], [dN5r, dN5s], [dN6r, dN6s], [dN7r, dN7s], [dN8r, dN8s], [dN9r, dN9s]])
#
##Jacobiano analítico
#J = Xe.T * dNds
#JI = J.inv()
#
##derivadas das funções de interpolação do elemento no sistema local x y
#dNdx = dNds * JI

#### iniciando do código numérico ---------------------------------------------------------------------------------------------------------------------
#def ke(Xe, E, nu, t):
#    '''
#    Função para a geração das matrizes de rigidez dos elementos função das coordenadas dos elementos no sistema global, o módulo de elasticidade
#    do material (E), o corficiente de poisson do material (nu) e da espessura (t), considerando 4 pontos de gauss para a integração
#    
#    Parâmetros
#    ----------
#    
#    Xe: array numpy com as coordenadas de cada nó dos elementos dispostas no sentido antihorário, com o primeiro nó o correspondente ao primeiro quadrante.
#        
#    >>> Xe = np.array([ [x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6], [x7, y7], [x8, y8], [x9, y9] ])
#    '''
#    #matriz constitutiva do material
#    D = E/(1 - nu**2) * np.array([[1, nu, 0],
#                                  [nu, 1, 0],
#                                  [0, 0, (1 - nu**2)/(2 + 2*nu)]])
#    #número de graus de liberdade por elemento
#    GLe = 18
#    #coordenadas dos pontos de gauss - 3 cada direção
#    PG = np.array([[ 0.774596669241483, 0.774596669241483],
#                   [ 0.000000000000000, 0.774596669241483],
#                   [-0.774596669241483, 0.774596669241483],
#                   [ 0.774596669241483, 0.000000000000000],
#                   [ 0.000000000000000, 0.000000000000000],
#                   [-0.774596669241483, 0.000000000000000],
#                   [ 0.774596669241483,-0.774596669241483],
#                   [ 0.000000000000000,-0.774596669241483],
#                   [-0.774596669241483,-0.774596669241483]])
#                 
#    #pesos de cada ponto de gauss
#    wPG = np.array([[0.555555555555556,0.555555555555556],
#                    [0.888888888888889,0.555555555555556],
#                    [0.555555555555556,0.555555555555556],
#                    [0.555555555555556,0.888888888888889],
#                    [0.888888888888889,0.888888888888889],
#                    [0.555555555555556,0.888888888888889],
#                    [0.555555555555556,0.555555555555556],
#                    [0.888888888888889,0.555555555555556],
#                    [0.555555555555556,0.555555555555556]])
#    Ke = np.zeros((GLe, GLe))
#    for p in range(PG.shape[0]):
#        B, J = dNdx9nos.dNdx(Xe, PG[p])
#        Ke += np.matmul( np.matmul(np.transpose(B), D), B) * wPG[p, 0] * wPG[p, 1] * np.linalg.det(J) * t
#    return Ke
#
##coordenadas dos nós da estrutura
#NOS = np.zeros((27,2))
#NOS[:,0] = np.tile(np.arange(0, 135, 15, dtype=float), 3)
#NOS[:,1] = np.concatenate((np.zeros(9), np.ones(9)*10., np.ones(9)*20.), axis=0)
#
##incidência dos elementos !!! DEVE SEGUIR A ORDEM DAS FUNÇÕES DE INTERPOLAÇÃO DEFINIDA NA FUNÇÃO dNdx !!!
#IE = np.array([ [20, 18, 0, 2, 19, 9, 1, 11, 10],
#                [22, 20, 2, 4, 21, 11, 3, 13, 12],
#                [24, 22, 4, 6, 23, 13, 5, 15, 14],
#                [26, 24, 6, 8, 25, 15, 7, 17, 16]])
#
##malha de elementos
#Xe = []
#for e in IE:
#    Xe.append( np.array([ NOS[e[0]], NOS[e[1]], NOS[e[2]], NOS[e[3]], NOS[e[4]], NOS[e[5]], NOS[e[6]], NOS[e[7]], NOS[e[8]] ]) )
#    
##propriedades mecânicas do material da estrutura e espessura
#E = 20000. #kN/cm2
#nu = 0.3
#t = 10. #cm
#
##determinação da matriz de rigidez dos elementos
#Ke1 = ke(Xe[0], E, nu, t)
#Ke2 = ke(Xe[1], E, nu, t)
#Ke3 = ke(Xe[2], E, nu, t)
#Ke4 = ke(Xe[3], E, nu, t)
#
##indexação dos graus de liberdade
#ID1 = np.repeat(IE[0]*2, 2) + np.tile(np.array([0, 1]), 9)
#ID2 = np.repeat(IE[1]*2, 2) + np.tile(np.array([0, 1]), 9)
#ID3 = np.repeat(IE[2]*2, 2) + np.tile(np.array([0, 1]), 9)
#ID4 = np.repeat(IE[3]*2, 2) + np.tile(np.array([0, 1]), 9)
#
##graus de liberdade da estrutura
#GL = NOS.shape[0]*2 #dois graus de liberdade por nó da estrutura
#GLe = 9*2 #dois graus de liberdade por nó do elemento
#
##montagem da matriz de rigidez da estrutura
#K = np.zeros((GL, GL))
#for i in range(GLe):
#    for j in range(GLe):
#        K[ ID1[i], ID1[j] ] += Ke1[i, j]
#        K[ ID2[i], ID2[j] ] += Ke2[i, j]
#        K[ ID3[i], ID3[j] ] += Ke3[i, j]
#        K[ ID4[i], ID4[j] ] += Ke4[i, j]
#
##nós livre, restringidos e respectivos graus de liberdade
#NOSr = np.array([0, 9, 18])
#NOSl = np.delete(np.arange(0, NOS.shape[0], dtype=int), NOSr, axis=0)
#GLr = np.repeat(NOSr*2, 2) + np.tile(np.array([0, 1]), NOSr.shape[0])
#GLl = np.repeat(NOSl*2, 2) + np.tile(np.array([0, 1]), NOSl.shape[0])
#
##separação das matrizes de rigidez
#Ku = np.delete(np.delete(K, GLr, 0), GLr, 1)
#Kr = np.delete(np.delete(K, GLl, 0), GLr, 1)
#
##vetor de forças nodais
#F = np.zeros(GL)
#F[17] = -10000. #kN
#F[35] = -10000. #kN
#F[53] = -10000. #kN
#
#Fu = np.delete(F, GLr, 0)
#Fr = np.delete(F, GLl, 0)
#
#Uu = np.linalg.solve(Ku, Fu)
#Rr = np.matmul(Kr, Uu) - Fr
#
#U = np.zeros(GL)
#U[GLl] = Uu
#
#Uxy = U.reshape(NOS.shape)
#
###visualização dos deslocamentos
##fig = go.Figure(data = go.Contour(z=Uxy[:,0], x=NOS[:,0], y=NOS[:,1], colorscale='Jet', contours=dict(
##            start=-170,
##            end=170,
##            size=10,
##            showlabels = True, # show labels on contours
##            labelfont = dict(size = 12, color = 'white') ) ) )
##fig.update_layout(title="Deslocamentos em X", autosize=True, width=1200, height=400)
##fig.write_html('deslocamentos9.html')
#
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


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!! AQUI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



















###############!!!!!!!!!!!!!!!!!!!!! AQUI!!!!!!!!!!!!!!!!!



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
