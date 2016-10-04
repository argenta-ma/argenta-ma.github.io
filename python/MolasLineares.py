# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:22:56 2016

@author: markinho
"""

import numpy as np

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

def k1(u1, u2):
    return 300*u1**2 + 400*u1*u2 - 200*u2**2 + 150*u1 - 100*u2

def k2(u1, u2):
    return 200*u2**2 - 400*u1*u2 + 200*u2**2 - 100*u1 + 100*u2 - 100

u = np.linspace(-1, 1, 100)
u1 = u
u2 = u
u1, u2 = np.meshgrid(u1, u2)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(u1, u2, k1(u1, u2), rstride=8, cstride=8, alpha=0.3, color='r')
ax.plot_surface(u1, u2, k2(u1, u2), rstride=8, cstride=8, alpha=0.3, color='b')

plt.show()