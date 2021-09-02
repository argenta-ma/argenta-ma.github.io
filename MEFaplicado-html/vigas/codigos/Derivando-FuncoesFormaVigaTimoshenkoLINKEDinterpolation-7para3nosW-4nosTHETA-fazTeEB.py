#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:54:46 2020

Funções de forma para a viga de Timoshenko usando linked interpolation.

Onate, volume 2, 2.8.3, página 66 do PDF.

Usando 7 pontos para obter uma interpolação a sexta e depois linkando com a condição
de que gamma_xz deve desaparecer para vigas esbeltas (de Euler-Bernouilli) para obter
uma viga de Timshenko de 3 nós somente para o U, e mantendo theta com 4 nós!

Para os deslocamentos u:

                s
               ^
               |
r -- r -- r -- r -- r -- r -- r   -> r

r0   r1   r2   r3   r4   r5   r6


Final: 

      s
     ^
     |
r -- r -- r   -> r

r0   r3   r6


Para as rotações:

         s
        ^
        |
r -- r --- r -- r   -> r

r0   r1    r2   r3



Graus de liberdade no final:
    
              ^ s
              |
   ,      ,   |     ,      ,
 1(o ---2(o - o - 4(o -- 6(o  --> r
   ^          ^            ^
   |0         |3           |5

  no0   no1  no2   no3    no4


Translação: 0, 3, 5 -> obtidos com o linked interpolation!
Rotação: 1, 2, 4, 6 -> interpolação original sem modificações!


Batem os momentos e os cortes!!!
FAZ TIMOSHENKO E EULER BERNOUILLI COM 1 ELEMENTO!!!!!!!!!

Testado com biapoiado e biengastado.

Será que se fizer direto com 4 nós para o theta e 3 nós para o u, também não daria certo???

@author: markinho
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


#para viga
L = sp.Symbol('L')

#elemento padrão vai de -1 até 1 em r
#elemento padrão vai de -L/2 até L/2 em r
r0 = -L*sp.Rational(1, 2)
r1 = -L*sp.Rational(1, 3) #objetivo: eliminar esse pelo link
r2 = -L*sp.Rational(1, 6) #objetivo: eliminar esse pelo link
r3 = 0.
r4 = L*sp.Rational(1, 6) #objetivo: eliminar esse pelo link
r5 = L*sp.Rational(1, 3) #objetivo: eliminar esse pelo link
r6 = L*sp.Rational(1, 2)

#interpolação para o w com 3 nós
s0 = -L*sp.Rational(1, 2)
s1 = -L*sp.Rational(1, 3)
s2 = L*sp.Rational(1, 3)
s3 = L*sp.Rational(1, 2)


#somente para os graus de liberdade de deslocamentos
theta0 = sp.Symbol('theta0')
theta1 = sp.Symbol('theta1')
theta2 = sp.Symbol('theta2')
theta3 = sp.Symbol('theta3')


u0 = sp.Symbol('u0')
u1 = sp.Symbol('u1')
u2 = sp.Symbol('u2')
u3 = sp.Symbol('u3')
u4 = sp.Symbol('u4')
u5 = sp.Symbol('u5')
u6 = sp.Symbol('u6')


Mat_Coefu = sp.Matrix([[1, r0, r0**2, r0**3, r0**4, r0**5, r0**6],
                      [1, r1, r1**2, r1**3, r1**4, r1**5, r1**6],
                      [1, r2, r2**2, r2**3, r2**4, r2**5, r2**6],
                      [1, r3, r3**2, r3**3, r3**4, r3**5, r3**6],
                      [1, r4, r4**2, r4**3, r4**4, r4**5, r4**6],
                      [1, r5, r5**2, r5**3, r5**4, r5**5, r5**6],
                      [1, r6, r6**2, r6**3, r6**4, r6**5, r6**6]])

Mat_Coeftheta = sp.Matrix([[1, s0, s0**2, s0**3],
                           [1, s1, s1**2, s1**3],
                           [1, s2, s2**2, s2**3],
                           [1, s3, s3**2, s3**3]])


THETA = sp.Matrix([theta0, theta1, theta2, theta3])
U = sp.Matrix([u0, u1, u2, u3, u4, u5, u6])

CoefsU = Mat_Coefu.inv() * U
CoefsTHETA = Mat_Coeftheta.inv() * THETA

Atheta = CoefsTHETA[0]
Btheta = CoefsTHETA[1]
Ctheta = CoefsTHETA[2]
Dtheta = CoefsTHETA[3]

Du = CoefsU[0]
Eu = CoefsU[1]
Fu = CoefsU[2]
Gu = CoefsU[3]
Hu = CoefsU[4]
Iu = CoefsU[5]
Ju = CoefsU[6]

r = sp.Symbol('r')

Nst = sp.expand(Atheta + Btheta*r + Ctheta*r**2 + Dtheta*r**3)
Nsu = sp.expand(Du + Eu*r + Fu*r**2 + Gu*r**3 + Hu*r**4 + Iu*r**5 + Ju*r**6)

N0t = sp.Add(*[argi for argi in Nst.args if argi.has(theta0)]).subs(theta0, 1)
N1t = sp.Add(*[argi for argi in Nst.args if argi.has(theta1)]).subs(theta1, 1)
N2t = sp.Add(*[argi for argi in Nst.args if argi.has(theta2)]).subs(theta2, 1)
N3t = sp.Add(*[argi for argi in Nst.args if argi.has(theta2)]).subs(theta3, 1)

N0u = sp.Add(*[argi for argi in Nsu.args if argi.has(u0)]).subs(u0, 1)
N1u = sp.Add(*[argi for argi in Nsu.args if argi.has(u1)]).subs(u1, 1)
N2u = sp.Add(*[argi for argi in Nsu.args if argi.has(u2)]).subs(u2, 1)
N3u = sp.Add(*[argi for argi in Nsu.args if argi.has(u3)]).subs(u3, 1)
N4u = sp.Add(*[argi for argi in Nsu.args if argi.has(u4)]).subs(u4, 1)
N5u = sp.Add(*[argi for argi in Nsu.args if argi.has(u5)]).subs(u5, 1)
N6u = sp.Add(*[argi for argi in Nsu.args if argi.has(u6)]).subs(u6, 1)


# #geração dos gráficos --------------------------------------------------------------
# #convertendo para função python
# nN0 = sp.utilities.lambdify([r, L], N0u, "numpy")
# nN1 = sp.utilities.lambdify([r, L], N1u, "numpy")
# nN2 = sp.utilities.lambdify([r, L], N2u, "numpy")
# # nN3 = sp.utilities.lambdify([r, L], N3, "numpy")
# # nN4 = sp.utilities.lambdify([r, L], N4, "numpy")
# # nN5 = sp.utilities.lambdify([r, L], N5, "numpy")
# # nN6 = sp.utilities.lambdify([r, L], N6, "numpy")

# L = 1.
# r = np.linspace(-L/2., L/2, 100)

# plt.plot(r, nN0(r, L), label="N0")
# plt.plot(r, nN1(r, L), label="N1")
# plt.plot(r, nN2(r, L), label="N2")
# # plt.plot(r, nN3(r, L), label="N3")
# # plt.plot(r, nN4(r, L), label="N4")
# # plt.plot(r, nN5(r, L), label="N5")
# # plt.plot(r, nN6(r, L), label="N6")
# plt.title('Deslocamentos')
# plt.legend(loc='best')
# plt.show()

#montando o w e o theta ----------------------------------------------------------------------------------
w = Nsu
theta = Nst
gamma_xz = sp.expand(-sp.diff(w, r) + theta)

#obtendo apenas os termos independentes
gamma_xz_cte = (gamma_xz + sp.O(r**1)).removeO()
#dos termos independentes
# theta2c = gamma_xz_cte + theta2 #já sai um valor para theta2: the average slope equals the rotation at the mid-node, which is a physical condition for slender beams!!! Onate pag. 67.

#obtendo somente os lineares
gamma_xz_linear = sp.collect(gamma_xz, r, evaluate=False)[r]

#obtendo somente os termos quadráticos
gamma_xz_quad = sp.collect(gamma_xz, r**2, evaluate=False)[r**2]

#obtendo somente os termos cubicos
gamma_xz_cub = sp.collect(gamma_xz, r**3, evaluate=False)[r**3]

#obtendo somente os termos quarquicos
gamma_xz_quar = sp.collect(gamma_xz, r**4, evaluate=False)[r**4]

#obtendo somente os termos a quinta
gamma_xz_qui = sp.collect(gamma_xz, r**5, evaluate=False)[r**5]

#obtendo somente os termos a sexta
# gamma_xz_sex = sp.collect(gamma_xz, r**6, evaluate=False)[r**6]


#isolar das equações acima, u1, u2, u4, u5 para resolver Ax = B
incognitas = sp.Matrix([u1, u2, u4, u5])

incognitas = sp.solve([gamma_xz_quad, gamma_xz_cub, gamma_xz_quar, gamma_xz_qui], [u1, u2, u4, u5])

# #substituindo em theta2c - Melhorou o resultado do momento!!!!! Tem que fazer!!! Fez o corte ser constante e não zero!!! NÃO DEIXA CERTO OS DESLOCAMENTOS!!!! NÃO USAR!!!
# # theta2c_subs = sp.expand(theta2c.subs({u1: incognitas[u1], u3: incognitas[u3]}))
# #substituindo novamente em w, theta e gamma_xy para obter as interpolações dos deslocamentos verticais, rotações e da deformação de cisalhamento
wLinked = sp.expand(w.subs({u1:incognitas[u1], u2:incognitas[u2], u4:incognitas[u4], u5:incognitas[u5]}))
thetaLinked = theta

# obtendo as funções de interpolação para cada um dos três nós para w3, theta3 e gamma_xz3
#esses uso para interpolar as cargas!
wNu0 = sp.Add(*[argi for argi in wLinked.args if argi.has(u0)]).subs(u0, 1)
wNu3 = sp.Add(*[argi for argi in wLinked.args if argi.has(u3)]).subs(u3, 1)
wNu6 = sp.Add(*[argi for argi in wLinked.args if argi.has(u6)]).subs(u6, 1)
wNtheta0 = sp.Add(*[argi for argi in wLinked.args if argi.has(theta0)]).subs(theta0, 1)
wNtheta1 = sp.Add(*[argi for argi in wLinked.args if argi.has(theta1)]).subs(theta1, 1) # É IGUAL A ZERO!!! Ou seja, wLinked não é função de theta2!!!
wNtheta2 = sp.Add(*[argi for argi in wLinked.args if argi.has(theta2)]).subs(theta2, 1)
wNtheta3 = sp.Add(*[argi for argi in wLinked.args if argi.has(theta3)]).subs(theta3, 1)

# # # thetaNu0 = sp.Add(*[argi for argi in thetaLinked.args if argi.has(u0)]).subs(u0, 1)
# # # thetaNu2 = sp.Add(*[argi for argi in thetaLinked.args if argi.has(u2)]).subs(u2, 1)
# # # thetaNu4 = sp.Add(*[argi for argi in thetaLinked.args if argi.has(u4)]).subs(u4, 1)
# # # thetaNtheta0 = sp.Add(*[argi for argi in thetaLinked.args if argi.has(theta0)]).subs(theta0, 1)
# # # thetaNtheta2 = sp.Add(*[argi for argi in thetaLinked.args if argi.has(theta2)]).subs(theta2, 1)
# # # thetaNtheta4 = sp.Add(*[argi for argi in thetaLinked.args if argi.has(theta4)]).subs(theta4, 1)

# # # # # # Não existe aqui!!
# # # gamma_xzNu0 = sp.Add(*[argi for argi in gamma_xz3.args if argi.has(u0)]).subs(u0, 1)
# # # gamma_xzNu2 = sp.Add(*[argi for argi in gamma_xz3.args if argi.has(u2)]).subs(u2, 1)
# # # gamma_xzNu4 = sp.Add(*[argi for argi in gamma_xz3.args if argi.has(u4)]).subs(u4, 1)
# # # gamma_xzNtheta0 = sp.Add(*[argi for argi in gamma_xz3.args if argi.has(theta0)]).subs(theta0, 1)
# # # gamma_xzNtheta2 = sp.Add(*[argi for argi in gamma_xz3.args if argi.has(theta2)]).subs(theta2, 1)
# # # gamma_xzNtheta4 = sp.Add(*[argi for argi in gamma_xz3.args if argi.has(theta4)]).subs(theta4, 1)

# # # ## !!!! AS FUNÇÕES PARA THETA E GAMMA SÃO AS MESMAS, em outas palavras, o campo de interpolação para o cisalhamento é o mesmo das rotações!

# # # #geração dos gráficos -------------------------------------------------------------- Resultados interessantes!!!!
# # # #convertendo para função python
# # # wN0 = sp.utilities.lambdify([r, L], wNu0, "numpy")
# # # wN2 = sp.utilities.lambdify([r, L], wNu2, "numpy")
# # # wN4 = sp.utilities.lambdify([r, L], wNu4, "numpy")
# # # wthetaN0 = sp.utilities.lambdify([r, L], wNtheta0, "numpy")
# # # wthetaN2 = sp.utilities.lambdify([r, L], wNtheta2, "numpy")
# # # wthetaN4 = sp.utilities.lambdify([r, L], wNtheta4, "numpy")

# # # thetawN0 = sp.utilities.lambdify([r, L], thetaNu0, "numpy")
# # # thetawN2 = sp.utilities.lambdify([r, L], thetaNu2, "numpy")
# # # thetawN4 = sp.utilities.lambdify([r, L], thetaNu4, "numpy")
# # # thetathetaN0 = sp.utilities.lambdify([r, L], thetaNtheta0, "numpy")
# # # thetathetaN2 = sp.utilities.lambdify([r, L], thetaNtheta2, "numpy")
# # # thetathetaN4 = sp.utilities.lambdify([r, L], thetaNtheta4, "numpy")

# # # # Não existe aqui!!
# # # gamma_xz_wN0 = sp.utilities.lambdify([r, L], gamma_xzNu0, "numpy")
# # # gamma_xz_wN2 = sp.utilities.lambdify([r, L], gamma_xzNu2, "numpy")
# # # gamma_xz_wN4 = sp.utilities.lambdify([r, L], gamma_xzNu4, "numpy")
# # # gamma_xz_thetaN0 = sp.utilities.lambdify([r, L], gamma_xzNtheta0, "numpy")
# # # gamma_xz_thetaN2 = sp.utilities.lambdify([r, L], gamma_xzNtheta2, "numpy")
# # # gamma_xz_thetaN4 = sp.utilities.lambdify([r, L], gamma_xzNtheta4, "numpy")

# # # L = 1.
# # # r = np.linspace(-L/2., L/2, 100)

# # # # w
# # # # plt.plot(r, wN0(r, L), label="wN0")
# # # # plt.plot(r, wN2(r, L), label="wN2")
# # # # plt.plot(r, wN4(r, L), label="wN4")

# # # # plt.plot(r, thetawN0(r, L), label="wthetaN0")
# # # # plt.plot(r, thetawN2(r, L), label="wthetaN2")
# # # # plt.plot(r, thetawN4(r, L), label="wthetaN4")

# # # # theta
# # # # plt.plot(r, wthetaN0(r, L), label="thetawN0")
# # # # plt.plot(r, wthetaN2(r, L), label="thetawN2")
# # # # plt.plot(r, wthetaN4(r, L), label="thetawN4")

# # # # plt.plot(r, thetathetaN0(r, L), label="thetaN0")
# # # # plt.plot(r, thetathetaN2(r, L), label="thetaN2")
# # # # plt.plot(r, thetathetaN4(r, L), label="thetaN4")

# # # # # gamma ## Não existe aqui!!
# # # # plt.plot(r, gamma_xz_wN0(r, L), label="gamma_xz_wN0")
# # # # plt.plot(r, gamma_xz_wN2(r, L), label="gamma_xz_wN2")
# # # # plt.plot(r, gamma_xz_wN4(r, L), label="gamma_xz_wN4")

# # # # plt.plot(r, gamma_xz_thetaN0(r, L), label="gamma_xz_thetaN0")
# # # # plt.plot(r, gamma_xz_thetaN2(r, L), label="gamma_xz_thetaN2")
# # # # plt.plot(r, gamma_xz_thetaN4(r, L), label="gamma_xz_thetaN4")

# # # plt.title('Deslocamentos')
# # # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# # # plt.show()

# # # # # # # #-------------------------------------------------------------------------------------

# # Derivação do elemento de Timoshenko e suas matrizes de rigidez
dtheta_dr = sp.diff(thetaLinked, r)
gamma_xzLinked = -sp.diff(wLinked, r) + thetaLinked

###### Derivação das matrizes de rigidez

#extraindo as derivadas das funções de interpolação para theta
#nó 1
tB0 = sp.Add(*[argi for argi in dtheta_dr.args if argi.has(u0)]).subs(u0, 1)
tB1 = sp.Add(*[argi for argi in dtheta_dr.args if argi.has(theta0)]).subs(theta0, 1)
#no2 theta
tB2 = sp.Add(*[argi for argi in dtheta_dr.args if argi.has(theta1)]).subs(theta1, 1)
#nó 3 meio, só u
tB3 = sp.Add(*[argi for argi in dtheta_dr.args if argi.has(u3)]).subs(u3, 1)
#nó 4 theta
tB4 = sp.Add(*[argi for argi in dtheta_dr.args if argi.has(theta2)]).subs(theta2, 1)
#nó 5
tB5 = sp.Add(*[argi for argi in dtheta_dr.args if argi.has(u6)]).subs(u6, 1)
tB6 = sp.Add(*[argi for argi in dtheta_dr.args if argi.has(theta3)]).subs(theta3, 1)

#extraindo as derivadas das funções de interpolação para gamma_xz
#nó 1
gB0 = sp.Add(*[argi for argi in gamma_xzLinked.args if argi.has(u0)]).subs(u0, 1)
gB1 = sp.Add(*[argi for argi in gamma_xzLinked.args if argi.has(theta0)]).subs(theta0, 1)
#nó 2 theta
gB2 = sp.Add(*[argi for argi in gamma_xzLinked.args if argi.has(theta1)]).subs(theta1, 1)
#no 3 meio, só u
gB3 = sp.Add(*[argi for argi in gamma_xzLinked.args if argi.has(u3)]).subs(u3, 1)
#no 4 theta
gB4 = sp.Add(*[argi for argi in gamma_xzLinked.args if argi.has(theta2)]).subs(theta2, 1)
#nó 4
gB5 = sp.Add(*[argi for argi in gamma_xzLinked.args if argi.has(u6)]).subs(u6, 1)
gB6 = sp.Add(*[argi for argi in gamma_xzLinked.args if argi.has(theta3)]).subs(theta3, 1)

#montagem da matriz Bb, para flexão
Bb = sp.Matrix([tB0, tB1, tB2, tB3, tB4, tB5, tB6])

#montagem da matriz Bs, para cisalhamento
Bs = sp.Matrix([gB0, gB1, gB2, gB3, gB4, gB5, gB6])

#relações constitutivas para a flexão e o cisalhamento
E = sp.Symbol('E') #módulo de elasticidade
G = sp.Symbol('G') #módulo de elasticidade transversal
Iy = sp.Symbol('Iy') #inércia da seção transversal em Y (fora do plano da viga)
A = sp.Symbol('A') #área da seção transversal

Db = E*Iy
Ds = G*A

#integrando e calculando as matrizes de rigidez !!!!! será que tem que multiplicar pelo determinante do jabociano L/2?????
KbI = sp.integrate( Bb * Bb.T, (r, -L*sp.Rational(1, 2), L*sp.Rational(1, 2)) )#*L*sp.Rational(1, 2)
KsI = sp.integrate( Bs * Bs.T, (r, -L*sp.Rational(1, 2), L*sp.Rational(1, 2)) )#*L*sp.Rational(1, 2)

Kb = Db*KbI
Ks = Ds*KsI

#Determinação do vetor de forças nodais equivalentes para cargas distribuídas constantes (carga positiva no Z positivo)
#Usando somente as funções de interpolação de w, lembrando que u = [u0, theta0, u2, theta2, u4, theta4] = [u0, u1, u2, u3, u4, u5], portanto, u1, u3 e u5 são de rotação!
Nb = sp.Matrix([ wNu0, wNtheta0, wNtheta1, wNu3, wNtheta2, wNu6, wNtheta3 ])
q = sp.Symbol('q')

Fq_cte = q*sp.integrate( Nb, (r, -L*0.5, L*0.5) )

#Determinação do vetor de forças nodais equivalentes para cargas distribuídas com máximo em -1 (carga positiva no Z positivo)
# Fq_tri = sp.expand(sp.integrate( (sp.Rational(1, 2)*q + sp.Rational(1, 2))*Nb, (r, -L*0.5, L*0.5) ))

#Determinação do vetor de forças nodais equivalentes para cargas distribuídas com máximo em +1 (carga positiva no Z positivo)
# Fq_trf = sp.expand(sp.integrate( (-sp.Rational(1, 2)*q + sp.Rational(1, 2))*Nb, (r, -L*0.5, L*0.5) ))

#Determinação dos vetores para cálculo dos esforços de momento e corte (basta multiplicá-las pelos deslocamentos calculados para se obter os esforços)
#r deve ser um np.linspace() pois são os pontos onde o momento é calculado
M = Db*Bb
# Q = sp.diff(M, r) #testando com a derivada dos momentos !!!!!FUNCIONA!!!!
Q = Ds*Bs ### Está zerando o corte! Não sera com a substituição do theta2 pela solução dos termos independentes!!


## CONVERSÕES --------------------------------------------------------------------------------------
#convertendo as matrizes de rigidez para funções lambda
Keb = sp.utilities.lambdify((E, Iy, L), Kb, "numpy")
Kes = sp.utilities.lambdify((G, A, L), Ks, "numpy")

#convertendo os vetores de força nodal equivalente para funções lambda
Feq = sp.utilities.lambdify((q, L), Fq_cte, "numpy")
# Feqti = sp.utilities.lambdify((q, L), Fq_tri, "numpy")
# Feqtf = sp.utilities.lambdify((q, L), Fq_trf, "numpy")

#convertendo os vetores para cálculo dos esforços
# Me = sp.utilities.lambdify((E, Iy, L, r), M, "numpy")
# Qe = sp.utilities.lambdify((G, A, L, r), Q, "numpy")

# ## apagando o resto não utilizado, somente irão restar as funçoes acima!
# del A, B, Bb, Bs, C, Coefs, D, Ds, Db, E, Fq_cte, Fq_tri, Fq_trf, G, Iy, Kb, KbI, Ks, KsI, L, M, Mat_Coef, N0, N1, N2, N3, N4, Nb, Ns, Q, U, dtheta_dr, gB0, gB1, gB2, gB3, gB4, gB5
# del gamma_xz, gamma_xz3, gamma_xz_cte, gamma_xz_cub, gamma_xz_cub_coefs, gamma_xz_cub_vetor, gamma_xz_linear, gamma_xz_linear_coefs, gamma_xz_linear_vetor, gamma_xz_quad, gamma_xz_quad_coefs
# del gamma_xz_quad_vetor, gamma_xz_quar, gamma_xz_quar_coefs, gamma_xz_quar_vetor, incognitas, matriz_coeficientes, q, r, r0, r1, r2, r3, r4, tB0, tB1, tB2, tB3, tB4, tB5, theta, theta0
# del theta1, theta2, theta2c, theta3, theta4, thetaNtheta0, thetaNtheta2, thetaNtheta4, thetaNu0, thetaNu2, thetaNu4, u0, u1, u2, u3, u4, vetorB, w, wLinked, wNtheta0, wNtheta2, wNtheta4
# del wNu0, wNu2, wNu4
# ## ---------------------------------------------------------------------------------------------

### Resolvendo uma viga simplesmente apoiada com 1 elemento

#material
Ev = 20000. #kN/cm2
Gv = 7700. #kN/cm2

#seção transversal
base = 10. #cm
altura = 100. #cm
Iyv = base*altura**3/12
Av = base*altura

#comprimento da viga
Lv = 500. #cm

#carga
qv = -0.01 #kN/cm


# Graus de liberdade no final:
#               ^ s
#               |
#    ,      ,   |     ,      ,
#  1(o ---2(o - o - 4(o -- 6(o  --> r
#    ^          ^            ^
#    |0         |3           |5

#   no0   no1  no2  no3     no4

#quantidade de elementos na viga
nelems = 1
nnos = 5*nelems - (nelems - 1)
nGLs = nnos*2 - nelems*3 #número de graus de liberdade totais
#graus de liberdade em matriz para cada elemento: FAZ O PAPEL DE GL E IE (incidência), ASSIM COMO DO ID, o indexador!!! PARA VIGAS!!!
GL = np.tile(np.arange(0, 7), nelems).reshape(nelems, 7) + 5*np.arange(0, nelems)[:, np.newaxis]

#identificando os nós com apoios
nosRestringidos = np.array([0, nnos-1]) #1 elemento, nós com apoios (primeiro e útlimo)
# nosRestringidos = np.array([0]) #1 elemento, nós com apoio primeiro
# nosRestringidos = np.array([0, 4]) #2 elemento2, nós com apoios (primeiro e útlimo)

#colocando os tipos dos apoios
# GLsR = np.array([GL.flatten()[0], GL.flatten()[-2]]) #somente apoios simples, graus de liberdade restringidos
GLsR = np.array([GL.flatten()[0], GL.flatten()[1], GL.flatten()[-2], GL.flatten()[-1]])  #engastado nas extremidade, graus de liberdade restringidos
# GLsR = np.array([GL[0], GL[1]])  #em balanço, graus de liberdade restringidos

GLsL = np.delete(np.arange(0, nGLs), GLsR, axis=0) #graus de liberdade livres

#matrizes de rigidez, iguais pois os elementos tem comprimentos iguais
Le = Lv/nelems
kbe = Keb(Ev, Iyv, Le)
kse = Kes(Gv, Av, Le)

#vetor de forças nodais equivalentes do elemento com carga distribuída constante
fqe = Feq(qv, Le)

#montagem da matriz de rigidez global
#K já sai com a soma da parcela de flaxão e da parcela de cisalhamento
K = np.zeros((GL.size, GL.size))
for e in range(0, nelems):
    for i in range(0, 7):
        for j in range(0, 7):
            K[ GL[e, i], GL[e, j] ] += kbe[i, j] + kse[i, j]

F = np.zeros(GL.size)
for e in range(0, nelems):
    for i in range(0, 7):
        F[ GL[e, i] ] += fqe[i]

Ku = K[GLsL,:][:, GLsL]
Kr = K[GLsR,:][:, GLsL]

Fu = F[GLsL]
Fr = F[GLsR]

U = np.linalg.solve(Ku, Fu)
Ra = np.matmul(Kr, U) - Fr

ug = np.zeros(nGLs)
ug[GLsL] = U
ug = ug[:, np.newaxis]

uge = []
MomentosF = []
CortesF = []
for e in range(0, nelems):
    uge.append( ug[GL[e]] )
    momento = M.T*ug[GL[e]]
    MomentosF.append(sp.utilities.lambdify((E, Iy, L, r), momento[0], "numpy"))
    corte = Q.T*ug[GL[e]]
    # CortesF.append(sp.utilities.lambdify((E, Iy, L, r), corte[0], "numpy"))
    CortesF.append(sp.utilities.lambdify((G, A, L, r), corte[0], "numpy"))

pontosdGrafico = 100
MomentosVal = []
CortesVal = []
rl = np.linspace(-Le*0.5, Le*0.5, pontosdGrafico) #momentos e cortes avaliam localmente, mas plotam no global!!!
for e in range(0, nelems):
    MomentosVal.append(-MomentosF[e](Ev, Iyv, Le, rl))
    CortesVal.append(CortesF[e](Gv, Av, Le, rl))
    # CortesVal.append(CortesF[e](Ev, Iyv, Le, rl))

MomentosTodos = np.array(MomentosVal).reshape(nelems*pontosdGrafico)
rT = np.linspace(-Lv*0.5, Lv*0.5, nelems*pontosdGrafico)
plt.plot(rT, MomentosTodos)
plt.show()

CortesTodos = np.array(CortesVal).reshape(nelems*pontosdGrafico)
plt.plot(rT, CortesTodos)
plt.show()

###!!!!!!!!!!!!!!!! continuar no item 2.8.4 página 68 do Onate





