# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:20:11 2017

@author: markinho
"""

import numpy as np

def integracao(f, a, b, n=100):
    """
    Integra f de a a b,
    usando a regra dos trapézios com n intervalos.
    """
    x = np.linspace(a, b, n+1)    # Coordenadas dos intervalos
    h = x[1] - x[0]            # Espaçamento entre intervalos
    I = h*(sum(f(x)) - 0.5*(f(a) + f(b)))
    return I

# Definindo nosso integrando
def minha_funcao(x):
    return np.exp(-x**2)

menos_infinito = -20  # Aproximação de menos infinito
I = integracao(minha_funcao, menos_infinito, 1, n=1000)
print 'Valor da integral:', I

from sympy import *
t, v0, g = symbols('t v0 g')
y = v0*t - Rational(1,2)*g*t**2
dydt = diff(y, t)                     # 1a derivada
dydt
print 'aceleração:', diff(y, t, t)  # 2a derivada
>>> y2 = integrate(dydt, t)
>>> y2

y = v0*t - Rational(1,2)*g*t**2
roots = solve(y, t)    # resolve y=0 para t
roots
x, y = symbols('x y')
f = -sin(x)*sin(y) + cos(x)*cos(y)
simplify(f)
expand(sin(x+y), trig=True)  # necessita de uma dica trigonométrica