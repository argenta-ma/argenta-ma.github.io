#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:54:46 2020

Funções de forma para a viga de Timoshenko usando linked interpolation.

Onate, volume 2, 2.8.3, página 66 do PDF.

Usando 3 pontos para obter uma interpolação quarquica e depois linkando com a condição
de que gamma_xz deve desaparecer para vigas esbeltas (de Euler-Bernouilli) para obter
uma viga de Timshenko de 2 nós

      s
     ^
     |
r -- r -- r   -> r

r0   r1   r2


Final: 

    s
   ^
   |
r --- r   -> r

r0   r2    

Graus de liberdade no final:
    
      ^ s
      |
   ,      ,
 1(r ---3(r   --> r
   ^      ^  
   |0     |2

  no0    no1

Translação: 0, 2
Rotação: 1, 3

Fazendo para comparar com o Onate!

Não funcionando....
Conforme refina a malha, os valores aumentam...!!!???

@author: markinho
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


#para viga
L = sp.Symbol('L')

#elemento padrão vai de -1 até 1 em r, para comparar com o Onate
r0 = -1
r1 = 0 #objetivo elimitar esse pelo link
r2 = 1

#somente para os graus de liberdade de deslocamentos
u0 = sp.Symbol('u0')
u1 = sp.Symbol('u1')
u2 = sp.Symbol('u2')

theta0 = sp.Symbol('theta0')
theta1 = sp.Symbol('theta1')
theta2 = sp.Symbol('theta2')

###!!!! objetivo é eliminar 1 e 3 deirando apenas 3 nós, usando a condição de
###!!!! gamma_rz = 0, que é a condição de vigas esbeltas

Mat_Coef = sp.Matrix([[1, r0, r0**2],
                      [1, r1, r1**2],
                      [1, r2, r2**2]])

U = sp.Matrix([u0, u1, u2])

Coefs = Mat_Coef.inv() * U

A = Coefs[0]
B = Coefs[1]
C = Coefs[2]

r = sp.Symbol('r')

Ns = sp.expand(A + B*r + C*r**2)

N0 = sp.Add(*[argi for argi in Ns.args if argi.has(u0)]).subs(u0, 1)
N1 = sp.Add(*[argi for argi in Ns.args if argi.has(u1)]).subs(u1, 1)
N2 = sp.Add(*[argi for argi in Ns.args if argi.has(u2)]).subs(u2, 1)

# #geração dos gráficos --------------------------------------------------------------
# #convertendo para função python
# nN0 = sp.utilities.lambdify([r, L], N0, "numpy")
# nN1 = sp.utilities.lambdify([r, L], N1, "numpy")
# nN2 = sp.utilities.lambdify([r, L], N2, "numpy")

# L = 1.
# r = np.linspace(-L/2., L/2, 100)

# plt.plot(r, nN0(r, L), label="N0")
# plt.plot(r, nN1(r, L), label="N1")
# plt.plot(r, nN2(r, L), label="N2")
# plt.title('Deslocamentos')
# plt.legend(loc=2)
# plt.show()
# #-------------------------------------------------------------------------------

#montando o w e o theta ----------------------------------------------------------------------------------
w = Ns
wOnate = sp.expand(sp.Rational(1, 2)*(r - 1)*r*u0 + (1 - r**2)*u1 + sp.Rational(1, 2)*(r + 1)*r*u2)

theta = eval(str(Ns).replace('u', 'theta'))
thetaOnate = sp.expand(sp.Rational(1, 2)*(r - 1)*r*theta0 + (1 - r**2)*theta1 + sp.Rational(1, 2)*(r + 1)*r*theta2)
#!!!ATE AQUI, OK!!!

#Fazendo no início de -1 a 1, então precisa multiplicar por 2/L a parcela da derivada para considerar em X, mesmo sendo em r
gamma_xz = sp.expand(2/L*sp.diff(w, r) - theta)

#obtendo apenas os termos independentes
gamma_xz_cte = (gamma_xz + sp.O(r**1)).removeO()
#dos termos independentes
theta1c = gamma_xz_cte + theta1 #já sai um valor para theta1: the average slope equals the rotation at the mid-node, which is a physical condition for slender beams!!! Onate pag. 67.

#obtendo somente os lineares
gamma_xz_linear = sp.collect(gamma_xz, r, evaluate=False)[r]

#obtendo somente os termos quadráticos
gamma_xz_quad = sp.collect(gamma_xz, r**2, evaluate=False)[r**2]

#isolar das equações acima, w1, e theta1 para resolver Ax = B
incognitas = sp.Matrix([theta1, u1])

incognitas = sp.solve([gamma_xz_linear, gamma_xz_quad], [theta1, u1])

#substituindo novamente em w, theta e gamma_xy para obter as interpolações dos deslocamentos verticais, rotações e da deformação de cisalhamento
wLinked = sp.expand(w.subs({u1:incognitas[u1]}))
wLinkedOnate = sp.expand(sp.Rational(1, 2)*(1 - r)*u0 + sp.Rational(1, 2)*(1 + r)*u2 + L*sp.Rational(1, 8)*(1 - r**2)*(theta0 - theta2))

thetaLinked = sp.expand(theta.subs({theta1:incognitas[theta1]}))
thetaLinkedOnate = sp.expand(sp.Rational(1, 2)*(1 - r)*theta0 + sp.Rational(1, 2)*(1 + r)*theta2)

# ###!!! ???? Como gamma_xz vem de dw/dx + theta, essas funções já representam essa soma, no entanto para derivar elementos finitos, deve seguir a derivação
# ### clássica do elemento de Timoshenko
# ### gama_xz3 não existe aqui! Somente foi utilizado para achar w3 e theta3 ?????? Travou para simplesmente apoiada com 1 elemento!
# # gamma_xz3 = sp.expand(gamma_xz.subs({theta1:incognitas[0], u1:incognitas[1]}))

## TODOS, DEPENDEM DE TODOS OS GRAUS DE LIBERDADE u0, theta0, u2, theta2, u4, theta4!!!!!!!

# #obtendo as funções de interpolação para cada um dos três nós para w3, theta3 e gamma_xz3
# wNu0 = sp.Add(*[argi for argi in wLinked.args if argi.has(u0)]).subs(u0, 1)
# wNu2 = sp.Add(*[argi for argi in wLinked.args if argi.has(u2)]).subs(u2, 1)
# wNtheta0 = sp.Add(*[argi for argi in wLinked.args if argi.has(theta0)]).subs(theta0, 1)
# wNtheta2 = sp.Add(*[argi for argi in wLinked.args if argi.has(theta2)]).subs(theta2, 1) 

# thetaNu0 = sp.Add(*[argi for argi in thetaLinked.args if argi.has(u0)]).subs(u0, 1) # É IGUAL A ZERO!!! Ou seja, thetaLinked não é função de u0!!!
# thetaNu2 = sp.Add(*[argi for argi in thetaLinked.args if argi.has(u2)]).subs(u2, 1) # É IGUAL A ZERO!!! Ou seja, thetaLinked não é função de u2!!!
# thetaNtheta0 = sp.Add(*[argi for argi in thetaLinked.args if argi.has(theta0)]).subs(theta0, 1)
# thetaNtheta2 = sp.Add(*[argi for argi in thetaLinked.args if argi.has(theta2)]).subs(theta2, 1)

# # # Não existe aqui!!
# # gamma_xzNu0 = sp.Add(*[argi for argi in gamma_xz3.args if argi.has(u0)]).subs(u0, 1)
# # gamma_xzNu2 = sp.Add(*[argi for argi in gamma_xz3.args if argi.has(u2)]).subs(u2, 1)
# # gamma_xzNtheta0 = sp.Add(*[argi for argi in gamma_xz3.args if argi.has(theta0)]).subs(theta0, 1)
# # gamma_xzNtheta2 = sp.Add(*[argi for argi in gamma_xz3.args if argi.has(theta2)]).subs(theta2, 1)


# ### !!!! AS FUNÇÕES PARA THETA E GAMMA SÃO AS MESMAS, em outas palavras, o campo de interpolação para o cisalhamento é o mesmo das rotações!

# # #geração dos gráficos -------------------------------------------------------------- Resultados interessantes!!!!
# # #convertendo para função python
# # wN0 = sp.utilities.lambdify([r, L], wNu0, "numpy")
# # wN2 = sp.utilities.lambdify([r, L], wNu2, "numpy")
# # wthetaN0 = sp.utilities.lambdify([r, L], wNtheta0, "numpy")
# # wthetaN2 = sp.utilities.lambdify([r, L], wNtheta2, "numpy")

# # thetawN0 = sp.utilities.lambdify([r, L], thetaNu0, "numpy")
# # thetawN2 = sp.utilities.lambdify([r, L], thetaNu2, "numpy")
# # thetathetaN0 = sp.utilities.lambdify([r, L], thetaNtheta0, "numpy")
# # thetathetaN2 = sp.utilities.lambdify([r, L], thetaNtheta2, "numpy")

# # # # Não existe aqui!!
# # # gamma_xz_wN0 = sp.utilities.lambdify([r, L], gamma_xzNu0, "numpy")
# # # gamma_xz_wN2 = sp.utilities.lambdify([r, L], gamma_xzNu2, "numpy")
# # # gamma_xz_thetaN0 = sp.utilities.lambdify([r, L], gamma_xzNtheta0, "numpy")
# # # gamma_xz_thetaN2 = sp.utilities.lambdify([r, L], gamma_xzNtheta2, "numpy")

# # L = 1.
# # r = np.linspace(-L/2., L/2, 100)

# # #w
# # plt.plot(r, wN0(r, L), label="wN0")
# # plt.plot(r, wN2(r, L), label="wN2")

# # # plt.plot(r, thetawN0(r, L), label="wthetaN0")
# # # plt.plot(r, thetawN2(r, L), label="wthetaN2")

# # # # theta
# # # plt.plot(r, wthetaN0(r, L), label="thetawN0")
# # # plt.plot(r, wthetaN2(r, L), label="thetawN2") #dá zero pois está no meio!!!

# # # plt.plot(r, thetathetaN0(r, L), label="thetaN0")
# # # plt.plot(r, thetathetaN2(r, L), label="thetaN2")

# # # # gamma ## Não existe aqui!!
# # # plt.plot(r, gamma_xz_wN0(r, L), label="gamma_xz_wN0")
# # # plt.plot(r, gamma_xz_wN2(r, L), label="gamma_xz_wN2")

# # # plt.plot(r, gamma_xz_thetaN0(r, L), label="gamma_xz_thetaN0")
# # # plt.plot(r, gamma_xz_thetaN2(r, L), label="gamma_xz_thetaN2")

# # plt.title('Deslocamentos')
# # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# # plt.show()

# # #-------------------------------------------------------------------------------------

# # Derivação do elemento de Timoshenko e suas matrizes de rigidez
# dtheta_dr = sp.diff(thetaLinked, r)
# #Em elementos finitos, gamma_xz3 existe aqui!!! ???? Travou para simplesmente apoiada com 1 elemento!
# gamma_xzLinked = sp.expand(2/L*sp.diff(wLinked, r) - thetaLinked)
# #fazendo a interpolação do gamma_xz direto da calculada

# ###### Derivação das matrizes de rigidez

# #extraindo as derivadas das funções de interpolação para theta
# #nó 1
# tB0 = sp.Add(*[argi for argi in dtheta_dr.args if argi.has(u0)]).subs(u0, 1)
# tB1 = sp.Add(*[argi for argi in dtheta_dr.args if argi.has(theta0)]).subs(theta0, 1)
# #nó 2
# tB2 = sp.Add(*[argi for argi in dtheta_dr.args if argi.has(u2)]).subs(u2, 1)
# tB3 = sp.Add(*[argi for argi in dtheta_dr.args if argi.has(theta2)]).subs(theta2, 1)

# #extraindo as derivadas das funções de interpolação para gamma_xz
# #nó 1
# gB0 = sp.Add(*[argi for argi in gamma_xzLinked.args if argi.has(u0)]).subs(u0, 1)
# gB1 = sp.Add(*[argi for argi in gamma_xzLinked.args if argi.has(theta0)]).subs(theta0, 1)
# #nó 2
# gB2 = sp.Add(*[argi for argi in gamma_xzLinked.args if argi.has(u2)]).subs(u2, 1)
# gB3 = sp.Add(*[argi for argi in gamma_xzLinked.args if argi.has(theta2)]).subs(theta2, 1)

# #montagem da matriz Bb, para flexão
# Bb = sp.Matrix([tB0, tB1, tB2, tB3])

# #montagem da matriz Bs, para cisalhamento
# Bs = sp.Matrix([gB0, gB1, gB2, gB3])

# #relações constitutivas para a flexão e o cisalhamento
# E = sp.Symbol('E') #módulo de elasticidade
# G = sp.Symbol('G') #módulo de elasticidade transversal
# Iy = sp.Symbol('Iy') #inércia da seção transversal em Y (fora do plano da viga)
# A = sp.Symbol('A') #área da seção transversal

# Db = E*Iy
# Ds = G*A

# #integrando e calculando as matrizes de rigidez !!!!! será que tem que multiplicar pelo determinante do jabociano L/2?????
# KbI = sp.integrate( Bb * Bb.T, (r, -1, 1) )*L*sp.Rational(1, 2)
# KsI = sp.integrate( Bs * Bs.T, (r, -1, 1) )*L*sp.Rational(1, 2)

# Kb = Db*KbI
# Ks = Ds*KsI

# #Determinação do vetor de forças nodais equivalentes para cargas distribuídas constantes (carga positiva no Z positivo)
# #Usando somente as funções de interpolação de w, lembrando que u = [u0, theta0, u2, theta2, u4, theta4] = [u0, u1, u2, u3, u4, u5], portanto, u1, u3 e u5 são de rotação!
# # Nb = sp.Matrix([ wNu0, wNtheta0, wNu2, wNtheta2 ])
# q = sp.Symbol('q')

# Fq_cte = q*L*sp.Matrix([sp.Rational(1, 2), L**2/12, sp.Rational(1, 2), -L**2/12])

# #Determinação do vetor de forças nodais equivalentes para cargas distribuídas com máximo em -1 (carga positiva no Z positivo) ?????
# # Fq_tri = sp.expand(sp.integrate( (sp.Rational(1, 2)*q + sp.Rational(1, 2))*Nb, (r, -L*0.5, L*0.5) ))

# #Determinação do vetor de forças nodais equivalentes para cargas distribuídas com máximo em +1 (carga positiva no Z positivo) ????
# # Fq_trf = sp.expand(sp.integrate( (-sp.Rational(1, 2)*q + sp.Rational(1, 2))*Nb, (r, -L*0.5, L*0.5) ))

# #Determinação dos vetores para cálculo dos esforços de momento e corte (basta multiplicá-las pelos deslocamentos calculados para se obter os esforços)
# #r deve ser um np.linspace() pois são os pontos onde o momento é calculado
# M = Db*Bb
# Q = Ds*Bs
# ### NÃO DEU CERTO!!!


# ## CONVERSÕES --------------------------------------------------------------------------------------
# #convertendo as matrizes de rigidez para funções lambda
# Keb = sp.utilities.lambdify((E, Iy, L), Kb, "numpy")
# Kes = sp.utilities.lambdify((G, A, L), Ks, "numpy")

# #convertendo os vetores de força nodal equivalente para funções lambda
# Feq = sp.utilities.lambdify((q, L), Fq_cte, "numpy")
# # Feqti = sp.utilities.lambdify((q, L), Fq_tri, "numpy")
# # Feqtf = sp.utilities.lambdify((q, L), Fq_trf, "numpy")

# #convertendo os vetores para cálculo dos esforços
# Me = sp.utilities.lambdify((E, Iy, L, r), M, "numpy")
# Qe = sp.utilities.lambdify((G, A, L, r), Q, "numpy")

# # ## apagando o resto não utilizado, somente irão restar as funçoes acima!
# # del A, B, Bb, Bs, C, Coefs, D, Ds, Db, E, Fq_cte, Fq_tri, Fq_trf, G, Iy, Kb, KbI, Ks, KsI, L, M, Mat_Coef, N0, N1, N2, N3, N4, Nb, Ns, Q, U, dtheta_dr, gB0, gB1, gB2, gB3, gB4, gB5
# # del gamma_xz, gamma_xz3, gamma_xz_cte, gamma_xz_cub, gamma_xz_cub_coefs, gamma_xz_cub_vetor, gamma_xz_linear, gamma_xz_linear_coefs, gamma_xz_linear_vetor, gamma_xz_quad, gamma_xz_quad_coefs
# # del gamma_xz_quad_vetor, gamma_xz_quar, gamma_xz_quar_coefs, gamma_xz_quar_vetor, incognitas, matriz_coeficientes, q, r, r0, r1, r2, r3, r4, tB0, tB1, tB2, tB3, tB4, tB5, theta, theta0
# # del theta1, theta2, theta2c, theta3, theta4, thetaNtheta0, thetaNtheta2, thetaNtheta4, thetaNu0, thetaNu2, thetaNu4, u0, u1, u2, u3, u4, vetorB, w, w3, wNtheta0, wNtheta2, wNtheta4
# # del wNu0, wNu2, wNu4
# # ## ---------------------------------------------------------------------------------------------

# ### Resolvendo uma viga

# #material
# E = 20000. #kN/cm2
# G = 7700. #kN/cm2

# #seção transversal
# Iy = 10.*20.**3/12 #20 cm de base e 40 cm de altura
# A = 10.*20.

# #comprimento da viga
# L = 500. #cm

# #carga
# q = -0.01 #kN/cm

# nosRestringidos = np.array([0, 1]) #1 elemento, nós com apoios (primeiro e útlimo)
# # nosRestringidos = np.array([0]) #1 elemento, nós com apoio primeiro
# # nosRestringidos = np.array([0, 4]) #2 elemento2, nós com apoios (primeiro e útlimo)

# #quantidade de elementos na viga
# nelems = 1
# nnos = 2*nelems - (nelems - 1)
# nGLs = nnos*2 #número de graus de liberdade totais
# GL = np.arange(0, nGLs, dtype=int).reshape(int(nGLs/2), 2) #graus de liberdade em matriz para cada nó
# # IE = quantidade de elementos linhas x nó inicial ao final colunas
# IE = np.tile(np.arange(0, 2), nelems).reshape(nelems, 2) + np.arange(0, nelems)[:, np.newaxis]

# GLsR = nosRestringidos*2 #somente apoios simples, graus de liberdade restringidos
# # GLsR = np.concatenate((nosRestringidos*2, nosRestringidos*2 + 1))  #engastado, graus de liberdade restringidos
# GLsL = np.delete(np.arange(0, nGLs), GLsR, axis=0) #graus de liberdade livres

# #matrizes de rigidez, iguais pois os elementos tem comprimentos iguais
# L = L/nelems
# kbe = Keb(E, Iy, L)
# kse = Kes(G, A, L)

# #vetor de forças nodais equivalentes do elemento com carga distribuída constante
# fqe = Feq(q, L)

# #montagem da matriz de rigidez global
# IDs = []
# for e in range(0, nelems):
#     IDs.append( np.array([ GL[IE[e, 0], 0], 
#                             GL[IE[e, 0], 1], 
#                             GL[IE[e, 1], 0], 
#                             GL[IE[e, 1], 1] ]) )

# #K já sai com a soma da parcela de flaxão e da parcela de cisalhamento
# K = np.zeros((GL.size, GL.size))
# for e in range(0, nelems):
#     for i in range(0, 4):
#         for j in range(0, 4):
#             K[ IDs[e][i], IDs[e][j] ] += kbe[i, j] + kse[i, j]

# F = np.zeros(GL.size)
# for e in range(0, nelems):
#     for i in range(0, 4):
#         F[ IDs[e][i] ] += fqe[i]

# Ku = K[GLsL,:][:, GLsL]
# Kr = K[GLsR,:][:, GLsL]

# Fu = F[GLsL]
# Fr = F[GLsR]

# U = np.linalg.solve(Ku, Fu)
# Ra = np.matmul(Kr, U) - Fr




# # ###!!!!!!!!!!!!!!!! continuar no item 2.8.4 página 68 do Onate











