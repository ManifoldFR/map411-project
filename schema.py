
# -*-coding:utf-8 -*
#/Users/cheikh/Applications/Python

# Implémentations et tests numériques

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numba
import numpy as np
from scipy.integrate import quad

print(mpl.use('Qt4Agg'))

gridsize = 20 # taille I de la grille
h = 1./gridsize # Pas de discrétisation de [0,1]

# Définition des éléments finis P1
def phi(x,i=None):
	if i is None:
		return np.maximum(0,1-np.absolute(x))
	else:
		return phi((x-i*h)/h)

def dPhi(x,i):
	if(x >= (i-1)*h and x <= (i+1)*h):
		if(x <= i*h):
			return 1./h
		else:
			return -1./h
	else :
		return 0.

def quad_trapz(g,a,b,p = 10):      # Calcul de l'intégrale de g entre a et b

    step = (b-a)/(p+1)
    first_term = 0.5*(g(a)+g(b))
    second_term = np.sum(g(a+k*step) for k in range(1,p))
    return step*(first_term + second_term)


# matrices de rigidité D et de masse M définies dans le problème
def build_matrices():
	print("\nComputing mass and rigidity matrices...")

	rigidity_matrix = np.zeros((gridsize+1,gridsize+1))
	mass_matrix = np.zeros((gridsize+1,gridsize+1))

	iter_integr = 20

	for i in range(gridsize+1):
		for j in range(gridsize+1):
			rigidity_matrix[i,j] = quad_trapz(
				lambda x: dPhi(x,i)*dPhi(x,j),0,1,iter_integr)
			mass_matrix[i,j] = quad_trapz(
				lambda x: phi(x,i)*phi(x,j),0,1,iter_integr)

	print(np.array_repr(mass_matrix,max_line_width=160,precision=4))
	return mass_matrix,rigidity_matrix

mass_matrix,rigidity_matrix = build_matrices()

print("\nComputing tensorial representation of RHS function...")

# choix des composantes f1 et f2 de f sous forme séparée
# une seule composante tensorielle est retenue
def f1(x):
	return x*(1-x)
def f2(y):
	return y*(1-y)

F1 = np.zeros((gridsize+1,))
F2 = np.zeros((gridsize+1,))

for i in range(gridsize+1):
	def produit1(x):
		return f1(x)*phi(x,i)
	def produit2(x):
		return f2(x)*phi(x,i)

	F1[i] = quad_trapz(produit1,0,1)
	F2[i] = quad_trapz(produit2,0,1)

@numba.jit
def A(V):   # A(V) est l'application M(V) dans l'énoncé
	return np.dot(np.dot(V.T,np.dot(rigidity_matrix,V)),mass_matrix) + np.dot(np.dot(V.T,np.dot(mass_matrix,V)),rigidity_matrix) + np.dot(np.dot(V.T,np.dot(mass_matrix,V)),mass_matrix)

def F(V,F1,F2,Rlist,Slist):
	VTD = np.dot(V.T,rigidity_matrix)
	VTM = np.dot(V.T,mass_matrix)

	# Calcule un nouveau terme de la somme
	def Fterm(V,R,S):
			return np.dot(np.dot(VTD,S),np.dot(mass_matrix,R)) +\
				np.dot(np.dot(VTM,S),np.dot(rigidity_matrix,R)) +\
				np.dot(np.dot(VTM,S),np.dot(mass_matrix,R))

	dim = gridsize + 1
	res = np.dot(np.dot(V.T,F2),F1)
	n = len(Rlist)
	for k in range(1,n):
		res -= Fterm(V,Rlist[k],Slist[k])
	return res

def G(V,F1,F2,Rlist,Slist):
	return F(V,F2,F1,Slist,Rlist)

# Méthode de point fixe; part des matrices aléatoires initialisées
def fixed_point(F1,F2,Rlist,Slist,m):
	S = np.array(np.random.rand(gridsize+1,)) # S est choisie aléatoirement dans le parallépipède [0,1]^{I+1}
	#print(np.array2string(S,max_line_width=160))
	R = S # R est ainsi initialisée
	for _ in range(m):
		R_prev = R
		S_prev = S

		R = np.dot(np.linalg.inv(A(S)),F(S,F1,F2,Rlist,Slist))
		S = np.dot(np.linalg.inv(A(R)),G(R,F1,F2,Rlist,Slist))

	# La précision est la taille de la différence entre les deux dernières itérées
	prec = max(np.linalg.norm(R-R_prev),np.linalg.norm(S-S_prev))
	return R,S,prec

def run():
	Rlist = []
	Slist = []

	m = 20
	n = 10
	for i in range(n+1):
		R,S,prec = fixed_point(F1,F2,Rlist,Slist,m)
		print("Précision du point fixe",i,":",prec)
		Rlist.append(R)
		Slist.append(S)


	return Rlist,Slist

## Construction de la solution
def build_sol(run_results):
	Rlist,Slist = run_results
	dim = gridsize + 1
	steps = len(Rlist)

	# Construit la fonction somme(g_i*phi_i)
	def build_base(x,G):
		return np.sum((g*phi(x,i) for i,g in enumerate(G)))

	def u(x,y):
		somme = np.sum((build_base(x,Rlist[n])*build_base(y,Slist[n]) for n in range(steps)))
		return somme

	return u


def graphe_sol():
	fig = plt.figure(0,figsize=(12,8))
	ax0 = fig.add_subplot(221)
	ax0.grid(True)
	ax1 = fig.add_subplot(222,projection='3d')
	ax1.grid(True)
	ax2 = fig.add_subplot(223)
	ax2.grid(True)
	ax3 = fig.add_subplot(224,projection='3d')
	ax3.grid(True)

	intervalle = np.linspace(0,1,gridsize,endpoint=True)
	domain = np.meshgrid(intervalle,intervalle)

	solFunction = build_sol(run())
	solVals = solFunction(*domain)

	fVals = (lambda x,y: f1(x)*f2(y))(*domain)

	ax0.imshow(solVals,extent=[0,1,0,1])
	ax1.plot_wireframe(*domain,solVals)
	ax2.imshow(fVals,extent=[0,1,0,1])
	ax3.plot_wireframe(*domain,fVals)

	return fig

fig = graphe_sol()
plt.show()
