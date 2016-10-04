# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:59:54 2016

@author: markinho
"""

import numpy as np
import matplotlib.pyplot as plt

def sigma_mat(E, m, epsilon):
    return E*np.arctan(m*epsilon)

epsilon = np.linspace(0., 0.05, 100)
m = 40
E = 100

plt.grid()
plt.plot(epsilon, sigma_mat(E, m, epsilon))
plt.show()