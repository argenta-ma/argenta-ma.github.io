#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:46:37 2019

Funções de forma ok!

VERIFICAR POIS FOI ALTERADA A ORDEM DOS NÓS NAS FUNÇÕES DE INTERPOLAÇÃO!!!

@author: markinho
"""

import sympy as sp
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'

import meshio

#elemento padrão
l = sp.Symbol('l')

r1 = 1
s1 = 1

r2 = -1
s2 = 1

r3 = -1
s3 = -1

r4 = 1
s4 = -1

u1 = sp.Symbol('u1')
u2 = sp.Symbol('u2')
u3 = sp.Symbol('u3')
u4 = sp.Symbol('u4')

#polinomio completo do segundo grau completo
#c0 + c1 x1 + c2 x2 + c3 x1**2 + c4 x1 x2 + c5 x2**2

Mat_Coef = sp.Matrix([[1, r1, s1, r1*s1],  #no1
                      [1, r2, s2, r2*s2],  #no2
                      [1, r3, s3, r3*s3],  #no3
                      [1, r4, s4, r4*s4]]) #no4

U = sp.Matrix([u1, u2, u3, u4])

Coefs = Mat_Coef.inv() * U

A = Coefs[0]
B = Coefs[1]
C = Coefs[2]
D = Coefs[3]

r = sp.Symbol('r')
s = sp.Symbol('s')

Ns = sp.expand(A + B*r + C*s + D*r*s)

N1 = sp.Add(*[argi for argi in Ns.args if argi.has(u1)]).subs(u1, 1)
N2 = sp.Add(*[argi for argi in Ns.args if argi.has(u2)]).subs(u2, 1)
N3 = sp.Add(*[argi for argi in Ns.args if argi.has(u3)]).subs(u3, 1)
N4 = sp.Add(*[argi for argi in Ns.args if argi.has(u4)]).subs(u4, 1)

N = sp.Matrix([N1, N2, N3, N4])

##grafico das funcoes de forma -------------------------------------------------------------------
nN1 = sp.utilities.lambdify([r, s], N1, "numpy")
nN2 = sp.utilities.lambdify([r, s], N2, "numpy")
nN3 = sp.utilities.lambdify([r, s], N3, "numpy")
nN4 = sp.utilities.lambdify([r, s], N4, "numpy")
#
#rl = np.linspace(-1., 1., 30)
#sl = np.linspace(-1., 1., 30)
#
#rm, sm = np.meshgrid(rl, sl)
#
##plotagem com o matplotlib -------------------------------------------------------------------------------
#fig = plt.figure()
##ax = Axes3D(fig)
#
#ax = fig.add_subplot(2, 2, 1, projection='3d')
#surf = ax.plot_surface(rm, sm, nN1(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(2, 2, 2, projection='3d')
#surf = ax.plot_surface(rm, sm, nN2(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(2, 2, 3, projection='3d')
#surf = ax.plot_surface(rm, sm, nN3(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#ax = fig.add_subplot(2, 2, 4, projection='3d')
#surf = ax.plot_surface(rm, sm, nN4(rm, sm), cmap=cm.jet, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.7)
#
#plt.show()

## plotagem usando plotly -----------------------------------------------------------------------------
#dados1 = plty.graph_objs.Surface(z=list(nN1(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados2 = plty.graph_objs.Surface(z=list(nN2(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados3 = plty.graph_objs.Surface(z=list(nN3(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#dados4 = plty.graph_objs.Surface(z=list(nN4(rm, sm)), x=list(rm), y=list(sm), colorscale='Jet')
#
#fig = plty.subplots.make_subplots(rows=2, cols=2,
#    specs=[[{'type': 'surface'}, {'type': 'surface'}],
#           [{'type': 'surface'}, {'type': 'surface'}]])
#
#fig.add_trace(dados1, row=1, col=1)
#fig.add_trace(dados2, row=1, col=2)
#fig.add_trace(dados3, row=2, col=1)
#fig.add_trace(dados4, row=2, col=2)
#
#fig.update_layout(title="Funções de forma do elemento bilinear" , autosize=True, width=900, height=700)
#
#fig.write_html('funcoesForma.html')
#fig.write_image("funcoesForma.svg")
##-------------------------------------------------------------------------------------------------------------

#primeira derivada em r
dN1r = sp.diff(N1, r)#.subs({r: r1, s: s1})
dN2r = sp.diff(N2, r)#.subs({r: r2, s: s2})
dN3r = sp.diff(N3, r)#.subs({r: r3, s: s3})
dN4r = sp.diff(N4, r)#.subs({r: r4, s: s4})
#convertendo para função lambda nuympy
ndN1r = sp.utilities.lambdify([r, s], dN1r, "numpy")
ndN2r = sp.utilities.lambdify([r, s], dN2r, "numpy")
ndN3r = sp.utilities.lambdify([r, s], dN3r, "numpy")
ndN4r = sp.utilities.lambdify([r, s], dN4r, "numpy")

#primeira derivada em s
dN1s = sp.diff(N1, s)#.subs({r: r1, s: s1})
dN2s = sp.diff(N2, s)#.subs({r: r2, s: s2})
dN3s = sp.diff(N3, s)#.subs({r: r3, s: s3})
dN4s = sp.diff(N4, s)#.subs({r: r4, s: s4})
#convertendo para função lambda nuympy
ndN1s = sp.utilities.lambdify([r, s], dN1s, "numpy")
ndN2s = sp.utilities.lambdify([r, s], dN2s, "numpy")
ndN3s = sp.utilities.lambdify([r, s], dN3s, "numpy")
ndN4s = sp.utilities.lambdify([r, s], dN4s, "numpy")

##gerando a matriz dNdx analítica ---------------------------------------------------------------------------
x1 = sp.Symbol('x1')
y1 = sp.Symbol('y1')
x2 = sp.Symbol('x2')
y2 = sp.Symbol('y2')
x3 = sp.Symbol('x3')
y3 = sp.Symbol('y3')
x4 = sp.Symbol('x4')
y4 = sp.Symbol('y4')
    
#Matriz dos nós de um elemento
Xe = sp.Matrix([[x1, y1],[x2, y2], [x3, y3], [x4, y4]])
#Matriz das derivadas das funções de interpolação do elemento padrão no sistema r s
dNds = sp.Matrix([[dN1r, dN1s], [dN2r, dN2s], [dN3r, dN3s], [dN4r, dN4s]])

#Jacobiano analítico
J = Xe.T * dNds
JI = J.inv()

#derivadas das funções de interpolação do elemento no sistema local x y
dNdx = dNds * JI

dNdrsPG1 = np.array([[ndN1r(1/np.sqrt(3), 1/np.sqrt(3)), ndN1s(1/np.sqrt(3), 1/np.sqrt(3))],
                     [ndN2r(1/np.sqrt(3), 1/np.sqrt(3)), ndN2s(1/np.sqrt(3), 1/np.sqrt(3))],
                     [ndN3r(1/np.sqrt(3), 1/np.sqrt(3)), ndN3s(1/np.sqrt(3), 1/np.sqrt(3))],
                     [ndN4r(1/np.sqrt(3), 1/np.sqrt(3)), ndN4s(1/np.sqrt(3), 1/np.sqrt(3))]])
dNdrsPG2 = np.array([[ndN1r(-1/np.sqrt(3), 1/np.sqrt(3)), ndN1s(-1/np.sqrt(3), 1/np.sqrt(3))],
                     [ndN2r(-1/np.sqrt(3), 1/np.sqrt(3)), ndN2s(-1/np.sqrt(3), 1/np.sqrt(3))],
                     [ndN3r(-1/np.sqrt(3), 1/np.sqrt(3)), ndN3s(-1/np.sqrt(3), 1/np.sqrt(3))],
                     [ndN4r(-1/np.sqrt(3), 1/np.sqrt(3)), ndN4s(-1/np.sqrt(3), 1/np.sqrt(3))]])
dNdrsPG3 = np.array([[ndN1r(-1/np.sqrt(3), -1/np.sqrt(3)), ndN1s(-1/np.sqrt(3), -1/np.sqrt(3))],
                     [ndN2r(-1/np.sqrt(3), -1/np.sqrt(3)), ndN2s(-1/np.sqrt(3), -1/np.sqrt(3))],
                     [ndN3r(-1/np.sqrt(3), -1/np.sqrt(3)), ndN3s(-1/np.sqrt(3), -1/np.sqrt(3))],
                     [ndN4r(-1/np.sqrt(3), -1/np.sqrt(3)), ndN4s(-1/np.sqrt(3), -1/np.sqrt(3))]])
dNdrsPG4 = np.array([[ndN1r(1/np.sqrt(3), -1/np.sqrt(3)), ndN1s(-1/np.sqrt(3), -1/np.sqrt(3))],
                     [ndN2r(1/np.sqrt(3), -1/np.sqrt(3)), ndN2s(-1/np.sqrt(3), -1/np.sqrt(3))],
                     [ndN3r(1/np.sqrt(3), -1/np.sqrt(3)), ndN3s(-1/np.sqrt(3), -1/np.sqrt(3))],
                     [ndN4r(1/np.sqrt(3), -1/np.sqrt(3)), ndN4s(-1/np.sqrt(3), -1/np.sqrt(3))]])


def Jnum(Xe, pg):
    r = pg[0]
    s = pg[1]
    x1 = Xe[0,0]
    y1 = Xe[1,0]
    x2 = Xe[0,1]
    y2 = Xe[1,1]
    x3 = Xe[0,2]
    y3 = Xe[1,2]
    x4 = Xe[0,3]
    y4 = Xe[1,3]
    return np.array([[x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4), x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4)],
                   [y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4), y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)]])

Xe1 = np.array([[150, 0, 0, 150], 
               [170, 170, 0, 0]])

Xe2 = np.array([[150, 0, 0, 150], 
               [340, 340, 170, 170]])

Xe3 = np.array([[150, 0, 0, 150], 
               [510, 510, 340, 340]])

Xe4 = np.array([[150, 0, 0, 150], 
               [680, 680, 510, 510]])

Xe5 = np.array([[150, 0, 0, 150], 
               [850, 850, 680, 680]])

PG = np.array([[0.5773502691896258, 0.5773502691896258],
               [-0.5773502691896258, 0.5773502691896258],
               [-0.5773502691896258, -0.5773502691896258],
               [0.5773502691896258, -0.5773502691896258]])

Xet = np.array([[90, 50, 0, 100], 
               [110, 90, 0, 30]])

J1_p1 = Jnum(Xe1, PG[0])
J1_p2 = Jnum(Xe1, PG[1])
J1_p3 = Jnum(Xe1, PG[2])
J1_p4 = Jnum(Xe1, PG[3])

J2_p1 = Jnum(Xe2, PG[0])
J2_p2 = Jnum(Xe2, PG[1])
J2_p3 = Jnum(Xe2, PG[2])
J2_p4 = Jnum(Xe2, PG[3])

J3_p1 = Jnum(Xe3, PG[0])
J3_p2 = Jnum(Xe3, PG[1])
J3_p3 = Jnum(Xe3, PG[2])
J3_p4 = Jnum(Xe3, PG[3])

J4_p1 = Jnum(Xe4, PG[0])
J4_p2 = Jnum(Xe4, PG[1])
J4_p3 = Jnum(Xe4, PG[2])
J4_p4 = Jnum(Xe4, PG[3])

J5_p1 = Jnum(Xe5, PG[0])
J5_p2 = Jnum(Xe5, PG[1])
J5_p3 = Jnum(Xe5, PG[2])
J5_p4 = Jnum(Xe5, PG[3])

Jt_p1 = Jnum(Xet, PG[0])
Jt_p2 = Jnum(Xet, PG[1])
Jt_p3 = Jnum(Xet, PG[2])
Jt_p4 = Jnum(Xet, PG[3])

JnumInv = np.linalg.inv(J1_p1)
dNdxyPG1 = np.matmul(dNdrsPG1, JnumInv)
dNdxyPG2 = np.matmul(dNdrsPG2, JnumInv)
dNdxyPG3 = np.matmul(dNdrsPG3, JnumInv)
dNdxyPG4 = np.matmul(dNdrsPG4, JnumInv)

BPG1 = np.array([[dNdxyPG1[0,0],   0, dNdxyPG1[1,0],   0, dNdxyPG1[2,0],   0, dNdxyPG1[3,0],   0],
                 [  0, dNdxyPG1[0,1],   0, dNdxyPG1[1,1],   0, dNdxyPG1[2,1],   0, dNdxyPG1[3,1]],
                 [dNdxyPG1[0,1], dNdxyPG1[0,0], dNdxyPG1[1,1], dNdxyPG1[1,0], dNdxyPG1[2,1], dNdxyPG1[2,0], dNdxyPG1[3,1], dNdxyPG1[3,0]]])
BPG2 = np.array([[dNdxyPG2[0,0],   0, dNdxyPG2[1,0],   0, dNdxyPG2[2,0],   0, dNdxyPG2[3,0],   0],
                 [  0, dNdxyPG2[0,1],   0, dNdxyPG2[1,1],   0, dNdxyPG2[2,1],   0, dNdxyPG2[3,1]],
                 [dNdxyPG2[0,1], dNdxyPG2[0,0], dNdxyPG2[1,1], dNdxyPG2[1,0], dNdxyPG2[2,1], dNdxyPG2[2,0], dNdxyPG2[3,1], dNdxyPG2[3,0]]])
BPG3 = np.array([[dNdxyPG3[0,0],   0, dNdxyPG3[1,0],   0, dNdxyPG3[2,0],   0, dNdxyPG3[3,0],   0],
                 [  0, dNdxyPG3[0,1],   0, dNdxyPG3[1,1],   0, dNdxyPG3[2,1],   0, dNdxyPG3[3,1]],
                 [dNdxyPG3[0,1], dNdxyPG3[0,0], dNdxyPG3[1,1], dNdxyPG3[1,0], dNdxyPG3[2,1], dNdxyPG3[2,0], dNdxyPG3[3,1], dNdxyPG3[3,0]]])
BPG4 = np.array([[dNdxyPG4[0,0],   0, dNdxyPG4[1,0],   0, dNdxyPG4[2,0],   0, dNdxyPG4[3,0],   0],
                 [  0, dNdxyPG4[0,1],   0, dNdxyPG4[1,1],   0, dNdxyPG4[2,1],   0, dNdxyPG4[3,1]],
                 [dNdxyPG4[0,1], dNdxyPG4[0,0], dNdxyPG4[1,1], dNdxyPG4[1,0], dNdxyPG4[2,1], dNdxyPG4[2,0], dNdxyPG4[3,1], dNdxyPG4[3,0]]])

## vetor de forças nodais equivalentes de corpo --------------------------------------------------------------
Nest = sp.Matrix([[ N[0],    0, N[1],    0, N[2],    0, N[3],    0 ], 
                  [ 0,    N[0],    0, N[1],    0, N[2],    0, N[3] ]])
Np = [ np.array(Nest.subs({r: PG[0,0], s: PG[0,1]}), dtype=float).T, 
       np.array(Nest.subs({r: PG[1,0], s: PG[1,1]}), dtype=float).T, 
       np.array(Nest.subs({r: PG[2,0], s: PG[2,1]}), dtype=float).T, 
       np.array(Nest.subs({r: PG[3,0], s: PG[3,1]}), dtype=float).T ]

fe = np.zeros((8,1))
gamma = np.array([[0 ], [-0.000025]])
for e in Np:
    fe += np.matmul(e, gamma) * 20. * 6375.
#------------------------------------------------------------------------------------------------------------
 
##geração da malha via GMSH API ------------------------------------------------------------------------------------------------------
#gmsh.initialize()
#
#gmsh.option.setNumber("General.Terminal", 1) #Should information be printed on the terminal (if available)? 1 sim 0 não
#gmsh.option.setNumber("Mesh.Algorithm", 8) #2D mesh algorithm (1: MeshAdapt, 2: Automatic, 5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay for Quads, 9: Packing of Parallelograms)
#gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3) #Mesh recombination algorithm (0: simple, 1: blossom, 2: simple full-quad, 3: blossom full-quad)
#gmsh.option.setNumber("Mesh.SecondOrderExperimental", 1) #Element order 
#gmsh.option.setNumber("Mesh.SecondOrderLinear", 1) #Should second order nodes (as well as nodes generated with subdivision algorithms) simply be created by linear interpolation?
#gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 75) #Minimum mesh element size
#gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 75) #Maximum mesh element size
#gmsh.option.setNumber("Mesh.Format", 16) #Mesh output format (1: msh, 2: unv, 10: auto, 16: vtk, 19: vrml, 21: mail, 26: pos stat, 27: stl, 28: p3d, 30: mesh, 31: bdf, 32: cgns, 33: med, 34: diff, 38: ir3, 39: inp, 40: ply2, 41: celum, 42: su2, 47: tochnog, 49: neu, 50: matlab)
#
#gmsh.model.add("parede") #Add a new model, with name "name", and set it as the current model.
#
#
#gmsh.model.geo.addPoint(0, 0, 0, 0, 1) #parametros (x, y, z, meshSize, tag) meshSize is > 0, add a meshing constraint at that point. If tag is positive, set the tag explicitly; otherwise a new tag is selected automatically.
#gmsh.model.geo.addPoint(150, 0, 0, 0, 2)
#gmsh.model.geo.addPoint(150, 850, 0, 0, 3)
#gmsh.model.geo.addPoint(0, 850, 0, 0, 4)
#
#gmsh.model.geo.addLine(1, 2, 1) #parametros (startTag, endTag, tag) Add a straight line segment between the two points with tags startTag and endTag. If tag is positive, set the tag explicitly.
#gmsh.model.geo.addLine(2, 3, 2)
#gmsh.model.geo.addLine(3, 4, 3)
#gmsh.model.geo.addLine(4, 1, 4)
#
#gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1) #parametros (curveTags, tag) Add a curve loop (a closed wire) formed by the curves curveTags. curveTags should contain (signed) tags of model enties of dimension 1 forming a closed loop: a negative tag signifies that the underlying curve is considered with reversed orientation. If tag is positive, set the tag explicitly
#
#gmsh.model.geo.addPlaneSurface([1], 1) #parametros (wireTags, tag) Add a plane surface defined by one or more curve loops wireTags. The first curve loop defines the exterior contour; additional curve loop define holes. If tag is positive, set the tag explicitly
#
#gmsh.model.geo.synchronize() #Synchronize the built-in CAD representation with the current Gmsh model.
#
#
#gmsh.model.mesh.generate(2) # Generate a mesh of the current model, up to dimension dim (0, 1, 2 or 3)
#gmsh.model.mesh.recombine() # Recombine the mesh of the current model, usei para fazer os quads
#
#teste = gmsh.model.mesh.getElements()
#
##gmsh.write('teste.vtk')
#
##mesh = gmsh_api.Mesh.from_gmsh(gmsh)
##gmsh.finalize()
#
##print(mesh.nodes)
##print(mesh.elements)
#--------------------------------------------------------------------------------------------------------------------------






#### iniciando do código numérico ---------------------------------------------------------------------------------------------------------------------
#def dNdx(Xe, pg):
#    '''
#    Função para a determinação da matriz das derivadas das funções de interpolação já no sistema x y e do jacobiano
#    
#    Parâmetros
#    ----------
#    
#    Xe: array numpy com as coordenadas de cada nó dos elementos dispostas no sentido horário, com o primeiro nó o correspondente ao segundo quadrante
#    
#    >>>
#        2 ----- 1
#        |       |
#        |       |
#        3-------4
#        
#    >>> Xe = np.array([ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ])
#    
#    pg: coordenadas do ponto de gauss utilizado
#    
#    >>> pg = np.array([ [xpg, ypg] ])
#    
#    retorna a matriz B para cada ponto de gauss
#    '''
#    r = pg[0]
#    s = pg[1]
#    x1 = Xe[0,0]
#    y1 = Xe[0,1]
#    x2 = Xe[1,0]
#    y2 = Xe[1,1]
#    x3 = Xe[2,0]
#    y3 = Xe[2,1]
#    x4 = Xe[3,0]
#    y4 = Xe[3,1]
#    
#    J = np.array([ [x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4), x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4)],
#                    [y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4), y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)]])
#    
#    dNdx = np.array([ [ (r/4 + 1/4)*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) + (s/4 + 1/4)*(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4)) - (x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))/((-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))),   (r/4 + 1/4)*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) - (s/4 + 1/4)*(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))],
#                       [(-r/4 + 1/4)*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) + (-s/4 - 1/4)*(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4)) - (x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))/((-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))), (-r/4 + 1/4)*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) - (-s/4 - 1/4)*(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))],
#                       [  (r/4 - 1/4)*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) + (s/4 - 1/4)*(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4)) - (x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))/((-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))),   (r/4 - 1/4)*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) - (s/4 - 1/4)*(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))],
#                       [(-r/4 - 1/4)*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) + (-s/4 + 1/4)*(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(-y1*(s/4 + 1/4) - y2*(-s/4 - 1/4) - y3*(s/4 - 1/4) - y4*(-s/4 + 1/4)) - (x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))/((-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))), (-r/4 - 1/4)*(x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4))) - (-s/4 + 1/4)*(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))/(-(x1*(r/4 + 1/4) + x2*(-r/4 + 1/4) + x3*(r/4 - 1/4) + x4*(-r/4 - 1/4))*(y1*(s/4 + 1/4) + y2*(-s/4 - 1/4) + y3*(s/4 - 1/4) + y4*(-s/4 + 1/4)) + (x1*(s/4 + 1/4) + x2*(-s/4 - 1/4) + x3*(s/4 - 1/4) + x4*(-s/4 + 1/4))*(y1*(r/4 + 1/4) + y2*(-r/4 + 1/4) + y3*(r/4 - 1/4) + y4*(-r/4 - 1/4)))]])
#    B1x = dNdx[0,0]
#    B1y = dNdx[0,1]
#    B2x = dNdx[1,0]
#    B2y = dNdx[1,1]
#    B3x = dNdx[2,0]
#    B3y = dNdx[2,1]
#    B4x = dNdx[3,0]
#    B4y = dNdx[3,1]
#    B = np.array([[B1x,   0, B2x,   0, B3x,   0, B4x,   0],
#                  [  0, B1y,   0, B2y,   0, B3y,   0, B4y],
#                  [B1y, B1x, B2y, B2x, B3y, B3x, B4y, B4x]])
#    return B, J
#
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
#    >>> Xe = np.array([ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ])
#    '''
#    #matriz constitutiva do material
#    D = E/(1 - nu**2) * np.array([[1, nu, 0],
#                                  [nu, 1, 0],
#                                  [0, 0, (1 - nu**2)/(2 + 2*nu)]])
#    #número de graus de liberdade por elemento
#    GLe = 8
#    #coordenadas e pesos dos pontos de Gauss
#    PG = np.array([[0.5773502691896258, 0.5773502691896258],
#                   [-0.5773502691896258, 0.5773502691896258],
#                   [-0.5773502691896258, -0.5773502691896258],
#                   [0.5773502691896258, -0.5773502691896258]])
#    wPG = np.array([[1., 1.],
#                    [1., 1.],
#                    [1., 1.],
#                    [1., 1.]])
#    Be = []
#    Ke = np.zeros((GLe, GLe))
#    for p in range(PG.shape[0]):
#        B, J = dNdx(Xe, PG[p])
#        Be.append(B)
#        Ke += np.matmul( np.matmul(np.transpose(B), D), B) * wPG[p, 0] * wPG[p, 1] * np.linalg.det(J) * t
#    return Ke, Be
#
##coordenadas dos nós da estrutura
#NOS = np.array([ [0., 0.],
#                 [150., 0.],
#                 [0., 170.],
#                 [150., 170.],
#                 [0., 340.],
#                 [150., 340.],
#                 [0., 510.],
#                 [150., 510.],
#                 [0., 680.],
#                 [150., 680.],
#                 [0., 850.],
#                 [150., 850.]])
#
##incidência dos elementos !!! DEVE SEGUIR A ORDEM DAS FUNÇÕES DE INTERPOLAÇÃO DEFINIDA NA FUNÇÃO dNdx !!!
#IE = np.array([[ 3,  2,  0,  1],
#               [ 5,  4,  2,  3],
#               [ 7,  6,  4,  5],
#               [ 9,  8,  6,  7],
#               [11, 10,  8,  9]])
#
##malha de elementos
#Xe = []
#for e in IE:
#    Xe.append( np.array([ NOS[e[0]], NOS[e[1]], NOS[e[2]], NOS[e[3]] ]) )
#    
##propriedades mecânicas do material da estrutura e espessura
#E = 33130.0468 #kN/cm2
#nu = 0.2
#t = 20. #cm
##resistência a compressão e a tração para aplicação do critério de Christensen, concreto C35
#Sc = 35. #kN/cm2
#St = 3.5 #kN/cm2
##coesão e o ângulo de atrito para os critérios de Mohr-Coulomb e Drucker-Prager (http://www.pcc.usp.br/files/text/publications/BT_00231.pdf)
#phi = 51. * np.pi/180.
#coesao = 0.00073 #kN/cm2
#
##determinação da matriz de rigidez dos elementos
#Ke1, Be1 = ke(Xe[0], E, nu, t)
#Ke2, Be2 = ke(Xe[1], E, nu, t)
#Ke3, Be3 = ke(Xe[2], E, nu, t)
#Ke4, Be4 = ke(Xe[3], E, nu, t)
#Ke5, Be5 = ke(Xe[4], E, nu, t)
#
##indexação dos graus de liberdade
#ID1 = np.repeat(IE[0]*2, 2) + np.tile(np.array([0, 1]), 4)
#ID2 = np.repeat(IE[1]*2, 2) + np.tile(np.array([0, 1]), 4)
#ID3 = np.repeat(IE[2]*2, 2) + np.tile(np.array([0, 1]), 4)
#ID4 = np.repeat(IE[3]*2, 2) + np.tile(np.array([0, 1]), 4)
#ID5 = np.repeat(IE[4]*2, 2) + np.tile(np.array([0, 1]), 4)
#
##graus de liberdade da estrutura
#GL = NOS.shape[0]*2
#DOF = GL - 4
#
##montagem da matriz de rigidez da estrutura
#K = np.zeros((GL, GL))
#for i in range(8):
#    for j in range(8):
#        K[ ID1[i], ID1[j] ] += Ke1[i, j]
#        K[ ID2[i], ID2[j] ] += Ke2[i, j]
#        K[ ID3[i], ID3[j] ] += Ke3[i, j]
#        K[ ID4[i], ID4[j] ] += Ke4[i, j]
#        K[ ID5[i], ID5[j] ] += Ke5[i, j]
#
##separação das matrizes de rigidez
#Ku = K[:DOF, :DOF]
#Kr = K[DOF:, :DOF]
#
##vetor de forças nodais
#F = np.zeros(GL)
#F[7] = -150. #kN
#F[15] = -150. #kN
#
#Fu = F[:DOF]
#Fr = F[DOF:]
#
#Uu = np.linalg.solve(Ku, Fu)
#Rr = np.matmul(Kr, Uu) - Fr
#
#U = np.zeros(GL)
#U[:DOF] = Uu
#
#Uxy = U.reshape(NOS.shape)
#
###visualização dos deslocamentos ---------------------------------------------------------------------------------------------------------------------
##fig = go.Figure(data = go.Contour(z=Uxy[:,0], x=NOS[:,0], y=NOS[:,1], colorscale='Jet', contours=dict(
##            showlabels = True, # show labels on contours
##            labelfont = dict(size = 12, color = 'white') ) ) )
##fig.update_layout(title="Deslocamentos em X", autosize=True, width=1200, height=400)
##fig.write_html('deslocamentos.html')
#
###visualização dos vetores de deslocamentos dos nós
##fig = plty.figure_factory.create_quiver(NOS[:,0], NOS[:,1], Uxy[:,0], Uxy[:,1])
##fig.write_html('deslocamentosVetor.html')
#
###geração do arquivo vtu
##pontos = NOS
##celulas = {'quad': IE}
##meshio.write_points_cells(
##        "teste.vtu",
##        pontos,
##        celulas,
##        # Optionally provide extra data on points, cells, etc.
##        point_data = {"U": Uxy},
##        # cell_data=cell_data,
##        # field_data=field_data
##        )
#
##-----------------------------------------------------------------------------------------------------------------------------------------------------
#
##determinação dos deslocamentos por elemento
#Ue = []
#Ue.append( U[ID1] )
#Ue.append( U[ID2] )
#Ue.append( U[ID3] )
#Ue.append( U[ID4] )
#
#
##determinação das deformações por ponto de Gauss -----------------------------------------------------------------------------------------------------
#epsilon1 = []
#epsilon2 = []
#epsilon3 = []
#epsilon4 = []
#for b in range(4): #range na quantidade de pontos de Gauss
#    epsilon1.append( np.matmul(Be1[b], Ue[0]) )
#    epsilon2.append( np.matmul(Be2[b], Ue[1]) )
#    epsilon3.append( np.matmul(Be3[b], Ue[2]) )
#    epsilon4.append( np.matmul(Be4[b], Ue[3]) )
#
##determinação das tensões por ponto de Gauss ---------------------------------------------------------------------------------------------------------
##matriz constitutiva do material
#D = E/(1 - nu**2) * np.array([[1, nu, 0],
#                              [nu, 1, 0],
#                              [0, 0, (1 - nu**2)/(2 + 2*nu)]])
#
#sigma1 = []
#sigma2 = []
#sigma3 = []
#sigma4 = []
#for b in range(4): #range na quantidade de pontos de Gauss
#    sigma1.append( np.matmul(D, epsilon1[b]) )
#    sigma2.append( np.matmul(D, epsilon2[b]) )
#    sigma3.append( np.matmul(D, epsilon3[b]) )
#    sigma4.append( np.matmul(D, epsilon4[b]) )
#
##cálculo das tensões principais nos pontos de Gauss---------------------------------------------------------------------------------------------------
##tensão principal máxima, tensão principal mínima, ângulo das tensões principais, tensão máxima de cisalhamento, tensão equivalente de von Mises
#sigmaPP1 = []
#sigmaPP2 = []
#sigmaPP3 = []
#sigmaPP4 = []
#
#def principaisPG(sigmas):
#    '''
#    Função para a determinação da tensão principal 1 (sigmaMAX), tensão principal 2 (sigmaMIN), 
#    ângulo das tensões principais, tensão máxima de cisalhamento, tensão equivalente de von Mises, de Christensen para materiais frágeis com
#    sigmaC <= 0.5 sigmaT (que deve ser menor que 1), de Morh-Coulomb de Drucker-Prager para a tensão fora do plano igual a zero
#    
#    sigmas é um array de uma dimensão contendo sigma_x, sigma_y e tau_xy
#    
#    retorna um array de uma dimensão com as quantidades acima
#    '''
#    sigma_x = sigmas[0]
#    sigma_y = sigmas[1]
#    tay_xy = sigmas[2]
#    
#    sigmaMAX = (sigma_x + sigma_y)/2 + np.sqrt( ((sigma_x - sigma_y)/2)**2 + tay_xy**2 )
#    sigmaMIN = (sigma_x + sigma_y)/2 - np.sqrt( ((sigma_x - sigma_y)/2)**2 + tay_xy**2 )
#    theta = 1./2. * np.arctan( 2*tay_xy/(sigma_x - sigma_y) )
#    tauMAX = (sigmaMAX - sigmaMIN)/2
#    sigmaEQvM = np.sqrt( sigmaMAX**2 - sigmaMAX*sigmaMIN + sigmaMIN**2 )
#    sigmaEQc = (1/St - 1/Sc)*(sigmaMAX + sigmaMIN) + 1/(St*Sc)*(sigmaMAX**2 - sigmaMAX*sigmaMIN + sigmaMIN**2)
#    sigmaEQmc = 2*( (sigmaMAX + sigmaMIN)/2.*np.sin(phi) + coesao*np.cos(phi) )/(sigmaMAX - sigmaMIN)
#    A = 2*1.4142135623730951*np.sin(phi)/(3 - np.sin(phi))
#    B = 3.*coesao*np.cos(phi)/np.sin(phi)
#    sigmaEQdp = ( (sigmaMAX - sigmaMIN)**2 + sigmaMAX**2 + sigmaMIN**2 )/( A**2*(sigmaMAX + sigmaMIN + B)**2 )
#    
#    return np.array([ sigmaMAX, sigmaMIN, theta, tauMAX, sigmaEQvM, sigmaEQc, sigmaEQmc, sigmaEQdp ])
#
#for p in range(4):
#    sigmaPP1.append( principaisPG(sigma1[p]) )
#    sigmaPP2.append( principaisPG(sigma2[p]) )
#    sigmaPP3.append( principaisPG(sigma3[p]) )
#    sigmaPP4.append( principaisPG(sigma4[p]) )
#
##cálculo das tensões nos nós, interpolando com as funções de interpolação dos elementos





