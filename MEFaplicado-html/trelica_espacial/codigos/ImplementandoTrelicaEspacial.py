#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:19:51 2020

Treliça espacial


@author: markinho
"""

import sympy as sp

#cossenos diretores
cx, cy, cz = sp.symbols('cx cy cz')
lx, ly, lz = sp.symbols('lx ly lz')
mx, my, mz = sp.symbols('mx my mz')
nx, ny, nz = sp.symbols('nx ny nz')
u1, u2, u3, u4, u5, u6 = sp.symbols('u1 u2 u3 u4 u5 u6')
E, A, L = sp.symbols('E A L')

C = sp.Matrix([[cx], [cy], [cz], [-cx], [-cy], [-cz]])

kg = E*A/L * C * C.T

R3 = sp.Matrix([[lx, mx, nx, 0, 0, 0],
                [ly, my, ny, 0, 0, 0],
                [lz, mz, nz, 0, 0, 0],
                [0, 0, 0, lx, mx, nx],
                [0, 0, 0, ly, my, ny],
                [0, 0, 0, lz, mz, nz]])

D3 = sp.Matrix([[lx, mx, nx, 0, 0, 0],
                [0, 0, 0, lx, mx, nx]])

ug = sp.Matrix([[u1], [u2], [u3], [u4], [u5], [u6]])

ul = R3 * ug
ulD = D3 * ug