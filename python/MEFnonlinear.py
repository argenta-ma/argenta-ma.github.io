# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:37:30 2016

@author: markinho
"""
import numpy as np
import matplotlib.pyplot as plt

M = 1000 #kNm
Iz = 0.000533 #m4
E = 200000000 #kN/m2
l = 10 #m

def funcL(x):
    return M*x**2/(2*E*Iz)

def funcNL(x):
    return E*Iz/M*(1-np.sqrt(1-(M*x/(E*Iz))**2))

x = np.linspace(0, l, 100)
yL = funcL(x)
yNL = funcNL(x)

plt.plot(x, yL, 'r', linewidth=2)
plt.plot(x, yNL, 'b', linewidth=2)
plt.xlim(xmin=0, xmax=10.2)
#plt.axis('equal')
plt.show()
