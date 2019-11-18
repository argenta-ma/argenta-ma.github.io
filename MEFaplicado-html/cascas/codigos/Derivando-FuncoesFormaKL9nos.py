#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Na plotagem: PORQUE N2, N3, N6, N11, N15 e N23 NEGATIVOS???



O elemento padrão:

    2 -- 5 -- 1
    |         |
    6    9    8
    |         |
    3 -- 7 -- 4

@author: markinho
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
#import meshio

#elemento padrão
r1 = 1
s1 = 1

r2 = -1
s2 = 1

r3 = -1
s3 = -1

r4 = 1
s4 = -1

r5 = 0
s5 = 1

r6 = -1
s6 = 0

r7 = 0
s7 = -1

r8 = 1
s8 = 0

r9 = 0
s9 = 0

u1 = sp.Symbol('u1')
u2 = sp.Symbol('u2')
u3 = sp.Symbol('u3')
u4 = sp.Symbol('u4')
u5 = sp.Symbol('u5')
u6 = sp.Symbol('u6')
u7 = sp.Symbol('u7')
u8 = sp.Symbol('u8')
u9 = sp.Symbol('u9')
u10 = sp.Symbol('u10')
u11 = sp.Symbol('u11')
u12 = sp.Symbol('u12')
u13 = sp.Symbol('u13')
u14 = sp.Symbol('u14')
u15 = sp.Symbol('u15')
u16 = sp.Symbol('u16')
u17 = sp.Symbol('u17')
u18 = sp.Symbol('u18')
u19 = sp.Symbol('u19')
u20 = sp.Symbol('u20')
u21 = sp.Symbol('u21')
u22 = sp.Symbol('u22')
u23 = sp.Symbol('u23')
u24 = sp.Symbol('u24')
u25 = sp.Symbol('u25')
u26 = sp.Symbol('u26')
u27 = sp.Symbol('u27')


###polinomio maluco incompleto para montagem da matriz dos coeficientes
#x = sp.Symbol('x')
#y = sp.Symbol('y')
#pmi = sp.Matrix([1, x, x*y, y, x**2, x**2*y, x**2*y**2, x*y**2, y**2, x**3, x**3*y, x**3*y**2, x**2*y**3, x*y**3, y**3, x**4, x**4*y, x**4*y**2, x**2*y**4, x*y**4, y**4, x**5, x**5*y, x**5*y**2, x**2*y**5, x*y**5, y**5])
#pmi = pmi.T
#dpmidx = sp.diff(pmi, x)
#dpmidy = sp.diff(pmi, y)
#
#r1,s1, r2, s2, r3, s3, r4, s4, r5, s5, r6, s6, r7, s7, r8, s8, r9, s9 = sp.symbols('r1,s1, r2, s2, r3, s3, r4, s4, r5, s5, r6, s6, r7, s7, r8, s8, r9, s9')
#
#pmiN1 = pmi.subs({x: r1, y: s1})
#dpmidxN1 = sp.diff(pmiN1, r1)
#dpmidyN1= sp.diff(pmiN1, s1)
#
#pmiN2 = pmi.subs({x: r2, y: s2})
#dpmidxN2 = sp.diff(pmiN2, r2)
#dpmidyN2= sp.diff(pmiN2, s2)
#
#pmiN3 = pmi.subs({x: r3, y: s3})
#dpmidxN3 = sp.diff(pmiN3, r3)
#dpmidyN3= sp.diff(pmiN3, s3)
#
#pmiN4 = pmi.subs({x: r4, y: s4})
#dpmidxN4 = sp.diff(pmiN4, r4)
#dpmidyN4= sp.diff(pmiN4, s4)
#
#pmiN5 = pmi.subs({x: r5, y: s5})
#dpmidxN5 = sp.diff(pmiN5, r5)
#dpmidyN5= sp.diff(pmiN5, s5)
#
#pmiN6 = pmi.subs({x: r6, y: s6})
#dpmidxN6 = sp.diff(pmiN6, r6)
#dpmidyN6= sp.diff(pmiN6, s6)
#
#pmiN7 = pmi.subs({x: r7, y: s7})
#dpmidxN7 = sp.diff(pmiN7, r7)
#dpmidyN7= sp.diff(pmiN7, s7)
#
#pmiN8 = pmi.subs({x: r8, y: s8})
#dpmidxN8 = sp.diff(pmiN8, r8)
#dpmidyN8= sp.diff(pmiN8, s8)
#
#pmiN9 = pmi.subs({x: r9, y: s9})
#dpmidxN9 = sp.diff(pmiN9, r9)
#dpmidyN9= sp.diff(pmiN9, s9)
#
#Mat_Coef = sp.Matrix([pmiN1,     #no1
#                      dpmidxN1,  #no2
#                      dpmidyN1,  #no3
#                      pmiN2,     #no4
#                      dpmidxN2,  #no5
#                      dpmidyN2,  #no6
#                      pmiN3,     #no7
#                      dpmidxN3,  #no8
#                      dpmidyN3,  #no9
#                      pmiN4,     #no10
#                      dpmidxN4,  #no11
#                      dpmidyN4,  #no12
#                      pmiN5,     #no13
#                      dpmidxN5,  #no14
#                      dpmidyN5,  #no15
#                      pmiN6,     #no16
#                      dpmidxN6,  #no17
#                      dpmidyN6,  #no18
#                      pmiN7,     #no19
#                      dpmidxN7,  #no20
#                      dpmidyN7,  #no21
#                      pmiN8,     #no22
#                      dpmidxN8,  #no23
#                      dpmidyN8,  #no24
#                      pmiN9,     #no25
#                      dpmidxN9,  #no26
#                      dpmidyN9])  #no27

Mat_Coef = sp.Matrix([[1, r1, r1*s1, s1, r1**2, r1**2*s1, r1**2*s1**2, r1*s1**2, s1**2,   r1**3,   r1**3*s1,   r1**3*s1**2,   r1**2*s1**3,   r1*s1**3,   s1**3,   r1**4,   r1**4*s1,   r1**4*s1**2,   r1**2*s1**4,   r1*s1**4,   s1**4,   r1**5,   r1**5*s1,   r1**5*s1**2,   r1**2*s1**5,   r1*s1**5,   s1**5],
                        [0,  1,    s1,  0,  2*r1,  2*r1*s1,  2*r1*s1**2,    s1**2,     0, 3*r1**2, 3*r1**2*s1, 3*r1**2*s1**2,    2*r1*s1**3,      s1**3,       0, 4*r1**3, 4*r1**3*s1, 4*r1**3*s1**2,    2*r1*s1**4,      s1**4,       0, 5*r1**4, 5*r1**4*s1, 5*r1**4*s1**2,    2*r1*s1**5,      s1**5,       0],
                        [0,  0,    r1,  1,     0,    r1**2,  2*r1**2*s1,  2*r1*s1,  2*s1,       0,      r1**3,    2*r1**3*s1, 3*r1**2*s1**2, 3*r1*s1**2, 3*s1**2,       0,      r1**4,    2*r1**4*s1, 4*r1**2*s1**3, 4*r1*s1**3, 4*s1**3,       0,      r1**5,    2*r1**5*s1, 5*r1**2*s1**4, 5*r1*s1**4, 5*s1**4],
                        [1, r2, r2*s2, s2, r2**2, r2**2*s2, r2**2*s2**2, r2*s2**2, s2**2,   r2**3,   r2**3*s2,   r2**3*s2**2,   r2**2*s2**3,   r2*s2**3,   s2**3,   r2**4,   r2**4*s2,   r2**4*s2**2,   r2**2*s2**4,   r2*s2**4,   s2**4,   r2**5,   r2**5*s2,   r2**5*s2**2,   r2**2*s2**5,   r2*s2**5,   s2**5],
                        [0,  1,    s2,  0,  2*r2,  2*r2*s2,  2*r2*s2**2,    s2**2,     0, 3*r2**2, 3*r2**2*s2, 3*r2**2*s2**2,    2*r2*s2**3,      s2**3,       0, 4*r2**3, 4*r2**3*s2, 4*r2**3*s2**2,    2*r2*s2**4,      s2**4,       0, 5*r2**4, 5*r2**4*s2, 5*r2**4*s2**2,    2*r2*s2**5,      s2**5,       0],
                        [0,  0,    r2,  1,     0,    r2**2,  2*r2**2*s2,  2*r2*s2,  2*s2,       0,      r2**3,    2*r2**3*s2, 3*r2**2*s2**2, 3*r2*s2**2, 3*s2**2,       0,      r2**4,    2*r2**4*s2, 4*r2**2*s2**3, 4*r2*s2**3, 4*s2**3,       0,      r2**5,    2*r2**5*s2, 5*r2**2*s2**4, 5*r2*s2**4, 5*s2**4],
                        [1, r3, r3*s3, s3, r3**2, r3**2*s3, r3**2*s3**2, r3*s3**2, s3**2,   r3**3,   r3**3*s3,   r3**3*s3**2,   r3**2*s3**3,   r3*s3**3,   s3**3,   r3**4,   r3**4*s3,   r3**4*s3**2,   r3**2*s3**4,   r3*s3**4,   s3**4,   r3**5,   r3**5*s3,   r3**5*s3**2,   r3**2*s3**5,   r3*s3**5,   s3**5],
                        [0,  1,    s3,  0,  2*r3,  2*r3*s3,  2*r3*s3**2,    s3**2,     0, 3*r3**2, 3*r3**2*s3, 3*r3**2*s3**2,    2*r3*s3**3,      s3**3,       0, 4*r3**3, 4*r3**3*s3, 4*r3**3*s3**2,    2*r3*s3**4,      s3**4,       0, 5*r3**4, 5*r3**4*s3, 5*r3**4*s3**2,    2*r3*s3**5,      s3**5,       0],
                        [0,  0,    r3,  1,     0,    r3**2,  2*r3**2*s3,  2*r3*s3,  2*s3,       0,      r3**3,    2*r3**3*s3, 3*r3**2*s3**2, 3*r3*s3**2, 3*s3**2,       0,      r3**4,    2*r3**4*s3, 4*r3**2*s3**3, 4*r3*s3**3, 4*s3**3,       0,      r3**5,    2*r3**5*s3, 5*r3**2*s3**4, 5*r3*s3**4, 5*s3**4],
                        [1, r4, r4*s4, s4, r4**2, r4**2*s4, r4**2*s4**2, r4*s4**2, s4**2,   r4**3,   r4**3*s4,   r4**3*s4**2,   r4**2*s4**3,   r4*s4**3,   s4**3,   r4**4,   r4**4*s4,   r4**4*s4**2,   r4**2*s4**4,   r4*s4**4,   s4**4,   r4**5,   r4**5*s4,   r4**5*s4**2,   r4**2*s4**5,   r4*s4**5,   s4**5],
                        [0,  1,    s4,  0,  2*r4,  2*r4*s4,  2*r4*s4**2,    s4**2,     0, 3*r4**2, 3*r4**2*s4, 3*r4**2*s4**2,    2*r4*s4**3,      s4**3,       0, 4*r4**3, 4*r4**3*s4, 4*r4**3*s4**2,    2*r4*s4**4,      s4**4,       0, 5*r4**4, 5*r4**4*s4, 5*r4**4*s4**2,    2*r4*s4**5,      s4**5,       0],
                        [0,  0,    r4,  1,     0,    r4**2,  2*r4**2*s4,  2*r4*s4,  2*s4,       0,      r4**3,    2*r4**3*s4, 3*r4**2*s4**2, 3*r4*s4**2, 3*s4**2,       0,      r4**4,    2*r4**4*s4, 4*r4**2*s4**3, 4*r4*s4**3, 4*s4**3,       0,      r4**5,    2*r4**5*s4, 5*r4**2*s4**4, 5*r4*s4**4, 5*s4**4],
                        [1, r5, r5*s5, s5, r5**2, r5**2*s5, r5**2*s5**2, r5*s5**2, s5**2,   r5**3,   r5**3*s5,   r5**3*s5**2,   r5**2*s5**3,   r5*s5**3,   s5**3,   r5**4,   r5**4*s5,   r5**4*s5**2,   r5**2*s5**4,   r5*s5**4,   s5**4,   r5**5,   r5**5*s5,   r5**5*s5**2,   r5**2*s5**5,   r5*s5**5,   s5**5],
                        [0,  1,    s5,  0,  2*r5,  2*r5*s5,  2*r5*s5**2,    s5**2,     0, 3*r5**2, 3*r5**2*s5, 3*r5**2*s5**2,    2*r5*s5**3,      s5**3,       0, 4*r5**3, 4*r5**3*s5, 4*r5**3*s5**2,    2*r5*s5**4,      s5**4,       0, 5*r5**4, 5*r5**4*s5, 5*r5**4*s5**2,    2*r5*s5**5,      s5**5,       0],
                        [0,  0,    r5,  1,     0,    r5**2,  2*r5**2*s5,  2*r5*s5,  2*s5,       0,      r5**3,    2*r5**3*s5, 3*r5**2*s5**2, 3*r5*s5**2, 3*s5**2,       0,      r5**4,    2*r5**4*s5, 4*r5**2*s5**3, 4*r5*s5**3, 4*s5**3,       0,      r5**5,    2*r5**5*s5, 5*r5**2*s5**4, 5*r5*s5**4, 5*s5**4],
                        [1, r6, r6*s6, s6, r6**2, r6**2*s6, r6**2*s6**2, r6*s6**2, s6**2,   r6**3,   r6**3*s6,   r6**3*s6**2,   r6**2*s6**3,   r6*s6**3,   s6**3,   r6**4,   r6**4*s6,   r6**4*s6**2,   r6**2*s6**4,   r6*s6**4,   s6**4,   r6**5,   r6**5*s6,   r6**5*s6**2,   r6**2*s6**5,   r6*s6**5,   s6**5],
                        [0,  1,    s6,  0,  2*r6,  2*r6*s6,  2*r6*s6**2,    s6**2,     0, 3*r6**2, 3*r6**2*s6, 3*r6**2*s6**2,    2*r6*s6**3,      s6**3,       0, 4*r6**3, 4*r6**3*s6, 4*r6**3*s6**2,    2*r6*s6**4,      s6**4,       0, 5*r6**4, 5*r6**4*s6, 5*r6**4*s6**2,    2*r6*s6**5,      s6**5,       0],
                        [0,  0,    r6,  1,     0,    r6**2,  2*r6**2*s6,  2*r6*s6,  2*s6,       0,      r6**3,    2*r6**3*s6, 3*r6**2*s6**2, 3*r6*s6**2, 3*s6**2,       0,      r6**4,    2*r6**4*s6, 4*r6**2*s6**3, 4*r6*s6**3, 4*s6**3,       0,      r6**5,    2*r6**5*s6, 5*r6**2*s6**4, 5*r6*s6**4, 5*s6**4],
                        [1, r7, r7*s7, s7, r7**2, r7**2*s7, r7**2*s7**2, r7*s7**2, s7**2,   r7**3,   r7**3*s7,   r7**3*s7**2,   r7**2*s7**3,   r7*s7**3,   s7**3,   r7**4,   r7**4*s7,   r7**4*s7**2,   r7**2*s7**4,   r7*s7**4,   s7**4,   r7**5,   r7**5*s7,   r7**5*s7**2,   r7**2*s7**5,   r7*s7**5,   s7**5],
                        [0,  1,    s7,  0,  2*r7,  2*r7*s7,  2*r7*s7**2,    s7**2,     0, 3*r7**2, 3*r7**2*s7, 3*r7**2*s7**2,    2*r7*s7**3,      s7**3,       0, 4*r7**3, 4*r7**3*s7, 4*r7**3*s7**2,    2*r7*s7**4,      s7**4,       0, 5*r7**4, 5*r7**4*s7, 5*r7**4*s7**2,    2*r7*s7**5,      s7**5,       0],
                        [0,  0,    r7,  1,     0,    r7**2,  2*r7**2*s7,  2*r7*s7,  2*s7,       0,      r7**3,    2*r7**3*s7, 3*r7**2*s7**2, 3*r7*s7**2, 3*s7**2,       0,      r7**4,    2*r7**4*s7, 4*r7**2*s7**3, 4*r7*s7**3, 4*s7**3,       0,      r7**5,    2*r7**5*s7, 5*r7**2*s7**4, 5*r7*s7**4, 5*s7**4],
                        [1, r8, r8*s8, s8, r8**2, r8**2*s8, r8**2*s8**2, r8*s8**2, s8**2,   r8**3,   r8**3*s8,   r8**3*s8**2,   r8**2*s8**3,   r8*s8**3,   s8**3,   r8**4,   r8**4*s8,   r8**4*s8**2,   r8**2*s8**4,   r8*s8**4,   s8**4,   r8**5,   r8**5*s8,   r8**5*s8**2,   r8**2*s8**5,   r8*s8**5,   s8**5],
                        [0,  1,    s8,  0,  2*r8,  2*r8*s8,  2*r8*s8**2,    s8**2,     0, 3*r8**2, 3*r8**2*s8, 3*r8**2*s8**2,    2*r8*s8**3,      s8**3,       0, 4*r8**3, 4*r8**3*s8, 4*r8**3*s8**2,    2*r8*s8**4,      s8**4,       0, 5*r8**4, 5*r8**4*s8, 5*r8**4*s8**2,    2*r8*s8**5,      s8**5,       0],
                        [0,  0,    r8,  1,     0,    r8**2,  2*r8**2*s8,  2*r8*s8,  2*s8,       0,      r8**3,    2*r8**3*s8, 3*r8**2*s8**2, 3*r8*s8**2, 3*s8**2,       0,      r8**4,    2*r8**4*s8, 4*r8**2*s8**3, 4*r8*s8**3, 4*s8**3,       0,      r8**5,    2*r8**5*s8, 5*r8**2*s8**4, 5*r8*s8**4, 5*s8**4],
                        [1, r9, r9*s9, s9, r9**2, r9**2*s9, r9**2*s9**2, r9*s9**2, s9**2,   r9**3,   r9**3*s9,   r9**3*s9**2,   r9**2*s9**3,   r9*s9**3,   s9**3,   r9**4,   r9**4*s9,   r9**4*s9**2,   r9**2*s9**4,   r9*s9**4,   s9**4,   r9**5,   r9**5*s9,   r9**5*s9**2,   r9**2*s9**5,   r9*s9**5,   s9**5],
                        [0,  1,    s9,  0,  2*r9,  2*r9*s9,  2*r9*s9**2,    s9**2,     0, 3*r9**2, 3*r9**2*s9, 3*r9**2*s9**2,    2*r9*s9**3,      s9**3,       0, 4*r9**3, 4*r9**3*s9, 4*r9**3*s9**2,    2*r9*s9**4,      s9**4,       0, 5*r9**4, 5*r9**4*s9, 5*r9**4*s9**2,    2*r9*s9**5,      s9**5,       0],
                        [0,  0,    r9,  1,     0,    r9**2,  2*r9**2*s9,  2*r9*s9,  2*s9,       0,      r9**3,    2*r9**3*s9, 3*r9**2*s9**2, 3*r9*s9**2, 3*s9**2,       0,      r9**4,    2*r9**4*s9, 4*r9**2*s9**3, 4*r9*s9**3, 4*s9**3,       0,      r9**5,    2*r9**5*s9, 5*r9**2*s9**4, 5*r9*s9**4, 5*s9**4]])

ue = sp.Matrix([u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18, u19, u20, u21, u22, u23, u24, u25, u26, u27])

Coefs = Mat_Coef.inv() * ue

A = Coefs[0]
B = Coefs[1]
C = Coefs[2]
D = Coefs[3]
E = Coefs[4]
F = Coefs[5]
G = Coefs[6]
H = Coefs[7]
I = Coefs[8]
J = Coefs[9]
K = Coefs[10]
L = Coefs[11]
M = Coefs[12]
N = Coefs[13]
O = Coefs[14]
P = Coefs[15]
Q = Coefs[16]
R = Coefs[17]
S = Coefs[18]
T = Coefs[19]
U = Coefs[20]
W = Coefs[21]
X = Coefs[22]
Y = Coefs[23]
Z = Coefs[24]
A1 = Coefs[25]
A2 = Coefs[26]

r = sp.Symbol('r')
s = sp.Symbol('s')

Ns = sp.expand(A + B*r + C*r*s + D*s + E*r**2 + F*r**2*s + G*r**2*s**2 + H*r*s**2 + I*s**2 + J*r**3 + K*r**3*s + L*r**3*s**2 + M*r**2*s**3 + N*r*s**3 + O*s**3 + P*r**4 + Q*r**4*s + R*r**4*s**2 + S*r**2*s**4 + T*r*s**4 + U*s**4 + W*r**5 + X*r**5*s + Y*r**5*s**2 + Z*r**2*s**5 + A1*r*s**5 + A2*s**5)

N1 = sp.Add(*[argi for argi in Ns.args if argi.has(u1)]).subs(u1, 1)
N2 = sp.Add(*[argi for argi in Ns.args if argi.has(u2)]).subs(u2, 1)
N3 = sp.Add(*[argi for argi in Ns.args if argi.has(u3)]).subs(u3, 1)
N4 = sp.Add(*[argi for argi in Ns.args if argi.has(u4)]).subs(u4, 1)
N5 = sp.Add(*[argi for argi in Ns.args if argi.has(u5)]).subs(u5, 1)
N6 = sp.Add(*[argi for argi in Ns.args if argi.has(u6)]).subs(u6, 1)
N7 = sp.Add(*[argi for argi in Ns.args if argi.has(u7)]).subs(u7, 1)
N8 = sp.Add(*[argi for argi in Ns.args if argi.has(u8)]).subs(u8, 1)
N9 = sp.Add(*[argi for argi in Ns.args if argi.has(u9)]).subs(u9, 1)
N10 = sp.Add(*[argi for argi in Ns.args if argi.has(u10)]).subs(u10, 1)
N11 = sp.Add(*[argi for argi in Ns.args if argi.has(u11)]).subs(u11, 1)
N12 = sp.Add(*[argi for argi in Ns.args if argi.has(u12)]).subs(u12, 1)
N13 = sp.Add(*[argi for argi in Ns.args if argi.has(u13)]).subs(u13, 1)
N14 = sp.Add(*[argi for argi in Ns.args if argi.has(u14)]).subs(u14, 1)
N15 = sp.Add(*[argi for argi in Ns.args if argi.has(u15)]).subs(u15, 1)
N16 = sp.Add(*[argi for argi in Ns.args if argi.has(u16)]).subs(u16, 1)
N17 = sp.Add(*[argi for argi in Ns.args if argi.has(u17)]).subs(u17, 1)
N18 = sp.Add(*[argi for argi in Ns.args if argi.has(u18)]).subs(u18, 1)
N19 = sp.Add(*[argi for argi in Ns.args if argi.has(u19)]).subs(u19, 1)
N20 = sp.Add(*[argi for argi in Ns.args if argi.has(u20)]).subs(u20, 1)
N21 = sp.Add(*[argi for argi in Ns.args if argi.has(u21)]).subs(u21, 1)
N22 = sp.Add(*[argi for argi in Ns.args if argi.has(u22)]).subs(u22, 1)
N23 = sp.Add(*[argi for argi in Ns.args if argi.has(u23)]).subs(u23, 1)
N24 = sp.Add(*[argi for argi in Ns.args if argi.has(u24)]).subs(u24, 1)
N25 = sp.Add(*[argi for argi in Ns.args if argi.has(u25)]).subs(u25, 1)
N26 = sp.Add(*[argi for argi in Ns.args if argi.has(u26)]).subs(u26, 1)
N27 = sp.Add(*[argi for argi in Ns.args if argi.has(u27)]).subs(u27, 1)


N = sp.Matrix([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17, N18, N19, N20, N21, N22, N23, N24, N25, N26, N27])
##Na plotagem: PORQUE N2, N3, N6, N11, N15 e N23 NEGATIVOS??? corrigindo abaixo
Nc = sp.Matrix([N1, -N2, -N3, N4, N5, -N6, N7, N8, N9, N10, -N11, N12, N13, N14, -N15, N16, N17, N18, N19, N20, N21, N22, -N23, N24, N25, N26, N27]).T

##plotagem com o matplotlib -------------------------------------------------------------------------------
#nN1 = sp.utilities.lambdify([r, s], N1, "numpy")
#nN2 = sp.utilities.lambdify([r, s], N2, "numpy")
#nN3 = sp.utilities.lambdify([r, s], N3, "numpy")
#nN4 = sp.utilities.lambdify([r, s], N4, "numpy")
#nN5 = sp.utilities.lambdify([r, s], N5, "numpy")
#nN6 = sp.utilities.lambdify([r, s], N6, "numpy")
#nN7 = sp.utilities.lambdify([r, s], N7, "numpy")
#nN8 = sp.utilities.lambdify([r, s], N8, "numpy")
#nN9 = sp.utilities.lambdify([r, s], N9, "numpy")
#nN10 = sp.utilities.lambdify([r, s], N10, "numpy")
#nN11 = sp.utilities.lambdify([r, s], N11, "numpy")
#nN12 = sp.utilities.lambdify([r, s], N12, "numpy")
#nN13 = sp.utilities.lambdify([r, s], N13, "numpy")
#nN14 = sp.utilities.lambdify([r, s], N14, "numpy")
#nN15 = sp.utilities.lambdify([r, s], N15, "numpy")
#nN16 = sp.utilities.lambdify([r, s], N16, "numpy")
#nN17 = sp.utilities.lambdify([r, s], N17, "numpy")
#nN18 = sp.utilities.lambdify([r, s], N18, "numpy")
#nN19 = sp.utilities.lambdify([r, s], N19, "numpy")
#nN20 = sp.utilities.lambdify([r, s], N20, "numpy")
#nN21 = sp.utilities.lambdify([r, s], N21, "numpy")
#nN22 = sp.utilities.lambdify([r, s], N22, "numpy")
#nN23 = sp.utilities.lambdify([r, s], N23, "numpy")
#nN24 = sp.utilities.lambdify([r, s], N24, "numpy")
#nN25 = sp.utilities.lambdify([r, s], N25, "numpy")
#nN26 = sp.utilities.lambdify([r, s], N26, "numpy")
#nN27 = sp.utilities.lambdify([r, s], N27, "numpy")
#
#rl = np.linspace(-1., 1., 100)
#sl = np.linspace(-1., 1., 100)
#
#rm, sm = np.meshgrid(rl, sl)
#
##para o nó 1, 2, 3 e 4
#fig = plt.figure()
##ax = Axes3D(fig)
#
#ax = fig.add_subplot(4, 3, 1, projection='3d')
#surf = ax.plot_surface(rm, sm, nN1(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 2, projection='3d')
#surf = ax.plot_surface(rm, sm, -nN2(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 3, projection='3d')
#surf = ax.plot_surface(rm, sm, -nN3(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(4, 3, 4, projection='3d')
#surf = ax.plot_surface(rm, sm, nN4(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 5, projection='3d')
#surf = ax.plot_surface(rm, sm, nN5(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 6, projection='3d')
#surf = ax.plot_surface(rm, sm, -nN6(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(4, 3, 7, projection='3d')
#surf = ax.plot_surface(rm, sm, nN7(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 8, projection='3d')
#surf = ax.plot_surface(rm, sm, nN8(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 9, projection='3d')
#surf = ax.plot_surface(rm, sm, nN9(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(4, 3, 10, projection='3d')
#surf = ax.plot_surface(rm, sm, nN10(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 11, projection='3d')
#surf = ax.plot_surface(rm, sm, -nN11(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(4, 3, 12, projection='3d')
#surf = ax.plot_surface(rm, sm, nN12(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#plt.show()
#
##para o nó 5, 6, 7, 8 e 9
#fig = plt.figure()
##ax = Axes3D(fig)
#
#ax = fig.add_subplot(5, 3, 1, projection='3d')
#surf = ax.plot_surface(rm, sm, nN13(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 2, projection='3d')
#surf = ax.plot_surface(rm, sm, nN14(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 3, projection='3d')
#surf = ax.plot_surface(rm, sm, -nN15(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(5, 3, 4, projection='3d')
#surf = ax.plot_surface(rm, sm, nN16(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 5, projection='3d')
#surf = ax.plot_surface(rm, sm, nN17(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 6, projection='3d')
#surf = ax.plot_surface(rm, sm, nN18(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(5, 3, 7, projection='3d')
#surf = ax.plot_surface(rm, sm, nN19(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 8, projection='3d')
#surf = ax.plot_surface(rm, sm, nN20(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 9, projection='3d')
#surf = ax.plot_surface(rm, sm, nN21(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(5, 3, 10, projection='3d')
#surf = ax.plot_surface(rm, sm, nN22(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 11, projection='3d')
#surf = ax.plot_surface(rm, sm, -nN23(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 12, projection='3d')
#surf = ax.plot_surface(rm, sm, nN24(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(5, 3, 13, projection='3d')
#surf = ax.plot_surface(rm, sm, nN25(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 14, projection='3d')
#surf = ax.plot_surface(rm, sm, nN26(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#ax = fig.add_subplot(5, 3, 15, projection='3d')
#surf = ax.plot_surface(rm, sm, nN27(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#plt.show()
#
##-------------------------------------------------------------------------------------------------------------

#resolvendo o equilíbrio minimizando o funcional de energia potencial total

z = sp.Symbol('z')
t = sp.Symbol('t')
Ee = sp.Symbol('Ee')
nu = sp.Symbol('nu')

I3 = sp.Identity(3)

w = (Nc * ue)[0]

epsilon = sp.Matrix([ [sp.diff(w, r, r)],
                      [sp.diff(w, s, s)],
                      [2*sp.diff(w, r, s)]])

epsilonN = sp.Matrix([ sp.diff(Nc, r, r),
                      sp.diff(Nc, s, s),
                      2*sp.diff(Nc, r, s)])

De = sp.Matrix([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu)/2]])
Ep = 1/12 * Ee * t**3/(1 - nu**2)

integrando = epsilonN.T * D * epsilonN

PI = sp.integrate( sp.integrate(integrando, r), s)
PI = ue.T * PI * ue
ku1 = sp.diff(sp.expand(PI[0,0]), u1)


k1_1 = sp.Add(*[argi for argi in ku1.args if argi.has(u1)]).subs(u1, 1)
k1_2 = sp.Add(*[argi for argi in ku1.args if argi.has(u2)]).subs(u2, 1)
k1_3 = sp.Add(*[argi for argi in ku1.args if argi.has(u3)]).subs(u3, 1)
k1_4 = sp.Add(*[argi for argi in ku1.args if argi.has(u4)]).subs(u4, 1)
k1_5 = sp.Add(*[argi for argi in ku1.args if argi.has(u5)]).subs(u5, 1)
k1_6 = sp.Add(*[argi for argi in ku1.args if argi.has(u6)]).subs(u6, 1)
k1_7 = sp.Add(*[argi for argi in ku1.args if argi.has(u7)]).subs(u7, 1)
k1_8 = sp.Add(*[argi for argi in ku1.args if argi.has(u8)]).subs(u8, 1)
k1_9 = sp.Add(*[argi for argi in ku1.args if argi.has(u9)]).subs(u9, 1)
k1_10 = sp.Add(*[argi for argi in ku1.args if argi.has(u10)]).subs(u10, 1)
k1_11 = sp.Add(*[argi for argi in ku1.args if argi.has(u11)]).subs(u11, 1)
k1_12 = sp.Add(*[argi for argi in ku1.args if argi.has(u12)]).subs(u12, 1)
k1_13 = sp.Add(*[argi for argi in ku1.args if argi.has(u13)]).subs(u13, 1)
k1_14 = sp.Add(*[argi for argi in ku1.args if argi.has(u14)]).subs(u14, 1)
k1_15 = sp.Add(*[argi for argi in ku1.args if argi.has(u15)]).subs(u15, 1)
k1_16 = sp.Add(*[argi for argi in ku1.args if argi.has(u16)]).subs(u16, 1)
k1_17 = sp.Add(*[argi for argi in ku1.args if argi.has(u17)]).subs(u17, 1)
k1_18 = sp.Add(*[argi for argi in ku1.args if argi.has(u18)]).subs(u18, 1)
k1_19 = sp.Add(*[argi for argi in ku1.args if argi.has(u19)]).subs(u19, 1)
k1_20 = sp.Add(*[argi for argi in ku1.args if argi.has(u20)]).subs(u20, 1)
k1_21 = sp.Add(*[argi for argi in ku1.args if argi.has(u21)]).subs(u21, 1)
k1_22 = sp.Add(*[argi for argi in ku1.args if argi.has(u22)]).subs(u22, 1)
k1_23 = sp.Add(*[argi for argi in ku1.args if argi.has(u23)]).subs(u23, 1)
k1_24 = sp.Add(*[argi for argi in ku1.args if argi.has(u24)]).subs(u24, 1)
k1_25 = sp.Add(*[argi for argi in ku1.args if argi.has(u25)]).subs(u25, 1)
k1_26 = sp.Add(*[argi for argi in ku1.args if argi.has(u26)]).subs(u26, 1)
k1_27 = sp.Add(*[argi for argi in ku1.args if argi.has(u27)]).subs(u27, 1)

























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
#
#
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!! AQUI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
################!!!!!!!!!!!!!!!!!!!!! AQUI!!!!!!!!!!!!!!!!!
#
#
#
###para treliça
#
###cálculo da matriz de rigidez
##Bp = sp.Matrix([ddN1, ddN2, ddN3, ddN4])
##Bmulp = Bp * Bp.T
##
##E = sp.Symbol('E')
##I = sp.Symbol('I')
##
##Kep = E*I*sp.integrate( Bmulp, (x, x1, x3) )
##
##A = sp.Symbol('A')
##
##Bt = sp.Matrix([dL1, dL2])
##Bmult = Bt * Bt.T
##
##Ket = E*A*sp.integrate( Bmult, (x, x1, x3) )
##
##Ke = sp.zeros(6, 6)
##
##Ke[1:3,1:3] = Kep[0:2,0:2]
##Ke[1:3,4:] = Kep[0:2,2:4]
##
##Ke[4:,1:3] = Kep[2:4,0:2]
##Ke[4:,4:] = Kep[2:4,2:4]
##
##Ke[0,0] = Ket[0,0]
##Ke[0,3] = Ket[0,1]
##
##Ke[3,0] = Ket[1,0]
##Ke[3,3] = Ket[1,1]
##
###matriz de rotação
##c = sp.Symbol('c')
##s = sp.Symbol('s')
##
##rot = sp.Matrix([ [c, -s, 0], [s, c, 0], [0, 0, 1] ])
##
##R = sp.zeros(6,6)
##R[:3,:3] = rot[:,:]
##R[3:,3:] = rot[:,:]
##
##Keg = R * Ke
##KeG = Keg * R.T
##
##
###deslocamentos no elemento
##N_EL = sp.Matrix([[L1],
##                  [N1],
##                  [N2],
##                  [L2],
##                  [N3],
##                  [N4]])
##
###para cálculo das deformações e tensões
##dN_ES = sp.Matrix([[dL1],
##                  [dN1],
##                  [dN2],
##                  [dL2],
##                  [dN3],
##                  [dN4]])
###para o cálculo do momento
##dN_M = sp.Matrix([[ddN1],
##                  [ddN2],
##                  [ddN3],
##                  [ddN4]])
###para o cálculo do cortante
##dN_C = sp.Matrix([[dddN1],
##                  [dddN2],
##                  [dddN3],
##                  [dddN4]])
###para cálculo da normal
##dN_N = sp.Matrix([[dL1],
##                  [dL2]])
##
###vetor de deformações genérico
##ug0, ug1, ug2, ug3, ug4, ug5 = sp.symbols('ug0 ug1 ug2 ug3 ug4 ug5')
##Ug = sp.Matrix([ug0, ug1, ug2, ug3, ug4, ug5])
##UgM = sp.Matrix([ug1, ug2, ug4, ug5])
##UgN = sp.Matrix([ug0, ug3])
##
##deslocamentos = N_EL.transpose() * Ug
##deformacoes = dN_ES.transpose() * Ug
##tensoes = E * deformacoes
##momento = dN_M.transpose() * UgM
##cortante = dN_C.transpose() * UgM
##normal = dN_N.transpose() * UgN
##
###cargas distribuída constante
##gx = sp.Symbol('gx')
##gy = sp.Symbol('gy')
##
##g_axial = c*gx + s*gy
##g_transv = -s*gx + c*gy
##
##n1 = g_axial*sp.integrate(L1, (x, x1, x3))
##n2 = g_axial*sp.integrate(L2, (x, x1, x3))
##
##f1 = g_transv*sp.integrate(N1, (x, x1, x3))
##f2 = g_transv*sp.integrate(N2, (x, x1, x3))
##f4 = g_transv*sp.integrate(N3, (x, x1, x3))
##f5 = g_transv*sp.integrate(N4, (x, x1, x3))
##
##F = sp.Matrix([n1, f1, f2, n2, f4, f5])
##Fg = R * F
##
####verificando com viga simplesmente apoiada com 1 elemento
###Kvs = Ke[2:7,2:7]
###
###F.row_del(0)
###F.row_del(0)
###F.row_del(-1)
###F.row_del(-1)
###
###U = Kvs.inv()*F
##
###verificando com viga em balanço com carga na extremidade
##F = np.zeros(6)
##F[4] = -10.
##
##Kvb = Ke[3:,3:]
##Fvb = F[3:, np.newaxis]
##
##U = Kvb.inv() * Fvb
##U1 = U[1].subs(L, 4).subs(E, 200. * 1e6).subs(I, 1000. * 0.01**4)
##U2 = U[2].subs(L, 4).subs(E, 200. * 1e6).subs(I, 1000. * 0.01**4)
