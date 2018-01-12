
# -*-coding:utf-8 -*
#/Users/cheikh/Applications/Python

# Implémentations et tests numériques

import matplotlib.pyplot as p
import numpy as np
from scipy.integrate import quad
from scipy import misc

# Définitions des paramétres 
n = 10
m = 10  # méthide des points fixes
P = 1
I = 10

# choix des fonction f1 et f2 pour le choix de f sous forme séparée
print "----------------------------------------------------------"
def f1(x):
	return x**2
def f2(y):
	return y**2


h=1./I # Pas de discrétisation de l'intervalle [0,1]


les_phi = []  # liste qui contiendra fonctions phi i

def phi_0(x):
	if(x >=0 and x <= h):
		return 1. - x/h
	else:
		return 0
les_phi.append(phi_0)

for i in range(1,I):

	def phi_i(x):
		if(x >= (i-1)*h and x <= (i+1)*h):
			if(x <= i*h):
				return 1. - (i*h - x)/h
			else:
				return 1. - (x - i*h)/h
			
		else :
			return 0.

	les_phi.append(phi_i)

	
def phi_I(x):
	if(x >= (I-1)*h and x <= 1):
		return 1. + (x - I*h)/h
	else :
		return 0
les_phi.append(phi_I)



les_phi_derivate = []	  # Liste qui contiendra les dérivées des fonction phi i 

def phi_0_derivate(x):
	if(x >=0 and x <= h):
		return -1./h
	else:
		return 0
les_phi_derivate.append(phi_0_derivate)

for i in range(1,I):

	def phi_i_derivate(x):
		if(x >= (i-1)*h and x <= (i+1)*h):
			if(x <= i*h):
				return 1./h
			else:
				return -1./h
			
		else :
			return 0.

	les_phi_derivate.append(phi_i_derivate)

def phi_I(x):
	if(x >= (I-1)*h and x <= 1):
		return 1./h
	else :
		return 0
les_phi_derivate.append(phi_i_derivate)




def quad_trapz(g,a,b,p = 5):      # Calcucl de l'intégrale de g entre a et b

    step = (b-a)/(p+1)
    first_term = 0.5*(g(a)+g(b))
    second_term = np.sum(g(a+k*step) for k in range(1,p))
    return step*(first_term + second_term)


D_ = []  # arrays des matrices D et M définies dans le problème
M_ = []

for i in range(I+1):
	D_.append([])
	M_.append([])

for i in range(I+1):
	for j in range(I+1):
		def f(x):
			fi = les_phi_derivate[i]
			fj = les_phi_derivate[j]
			return fi(x) * fj(x)
		def g(x):
			gi = les_phi[i]
			gj = les_phi[j]
			return gi(x) * gj(x)

		D_[i].append(quad_trapz(f,0,1,p=5))
		M_[i].append(quad_trapz(g,0,1,p=5))
		



D = np.array(D_)  # Matrices D et M
M = np.array(M_)




F1_ = []   # Arrays des F_alpha
F2_ = []



for i in range(I+1):
	def produit1(x):
		return f1(x) * les_phi[i](x)
	def produit2(x):
		return f2(x) * les_phi[i](x)

	F1_.append(quad_trapz(produit1,0,1,p = 5))
	F2_.append(quad_trapz(produit2,0,1,p = 5))


F1 = np.array(F1_) # Matrices colonnes F_alpha
F2 = np.array(F2_)



# Quelques outils de calculs de matrice en chaine 
def produit_matriciel4(A,B,C,D): # produit de 4 matrices
	return np.dot(np.dot(A,B),np.dot(C,D)) # il y'a potentiellement un calcul matriciel plus efficace mais on ne va pas s'en soucier pour l'instant

def produit_matriciel5(A,B,C,D,E): # prodiit de 5 matrices
	return np.dot(produit_matriciel4(A,B,C,D),E)


def A(V):   # A(V) est l'application M(V) dans l'énoncé
	return np.dot(np.dot(V.T,np.dot(D,V)),M) + np.dot(np.dot(V.T,np.dot(M,V)),D) + np.dot(np.dot(V.T,np.dot(M,V)),M)


def F(V):
	return np.dot(np.dot(V.T,F2),F1) 


def G(V):
	return np.dot(np.dot(V.T,F1),F2)


# Algorithme récurisif pour le calcul des vecteurs colonnes R et S avec n et m fixés 

S0 = np.array(np.random.rand(I+1,)) # So est choisie aléatoirement
R0 = S0 # R0 est ainsi initialisée
Fval = F(S0)
Gval = G(R0)
print A(S0)
i = 0 
while(i <= n):
	
	S0_before = S0
	R0_before = R0
	k = 0
	while(k <= m):
		R0 = np.dot(np.linalg.inv(A(S0)),F(S0))
		S0 = np.dot(np.linalg.inv(A(R0)),G(R0))
		R0 = np.dot(np.linalg.inv(A(S0)),F(S0))
		k += 1
	
	Fval = Fval - produit_matriciel5(S0.T,D,S0_before,M,R0_before) - produit_matriciel5(S0.T,M,S0_before,D,R0_before) - produit_matriciel5(S0.T,M,S0_before,M,R0_before)
	Gval = Gval - produit_matriciel5(R0.T,D,R0_before,M,S0_before) - produit_matriciel5(R0.T,M,R0_before,D,S0_before) - produit_matriciel5(R0.T,M,R0_before,M,S0_before)
	i+=1



print ("ok")