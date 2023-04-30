import matplotlib.pyplot as plt
import numpy as np

epsilon = 10 ** -3
f = lambda x : x**2 - c
df = lambda x : 2*x


def newton(c, x0):
    X = [x0]
    e = epsilon
    while e >= epsilon:
        X.append(X[-1] - f(X[-1])/df(X[-1]))
        e = abs(X[-1] - X[-2])
    return X


def quasi_newton(c, x0, x1):
    X = [x0, x1]
    e = epsilon
    while e >= epsilon:
        X.append(X[-1] - f(X[-1]) * (X[-1] - X[-2])/(f(X[-1]) - f(X[-2])))
        e = abs(X[-1] - X[-2])
    return X


print("\n+" + "-"*73 + "+")
print("|          Calcul itératif de la racine carée d'un nombre positif c       |")
print("|                             en partant de c                             |")
print("+" + "-"*73 + "+\n")

c = float(input("Saisir c : "))
print()
x0 = float(input("Saisir x0 : "))

print("\n+" + "-"*68 + "+")
print("|                           Méthode de Newton                        |")
print("+" + "-"*68 + "+\n")

X = newton(c, x0)
print("Les itérations :")
for i, x in enumerate(X):
    print("x" + str(i), "=", x)
print("\nErreur du resultat |c - x" + str(i) + "²| : ", abs(c - X[-1]**2))

for i in range(len(X) - 1):
    plt.plot([X[i], X[i]], [0, f(X[i])], "k--")
    plt.plot([X[i], X[i+1]], [f(X[i]), 0], color="b")
    plt.scatter([X[i], X[i]], [0, f(X[i])], color="r")
    
x = np.linspace(0, max(X), 1000)
plt.plot(x, f(x), "k")
plt.title("Méthode de Newton")
plt.show()

print("\n+" + "-"*68 + "+")
print("|                        Méthode de Quasi-Newton                     |")
print("+" + "-"*68 + "+\n")
X = quasi_newton(c, x0, x0 + 0.1)
print("Les itérations :")
for i, x in enumerate(X):
    print("x" + str(i), "=", x)
print("\nErreur du resultat |c - x" + str(i) + "²| : ", abs(c - X[-1]**2))

for i in range(len(X) - 2):
    plt.plot([X[i], X[i]], [0, f(X[i])], "k--")
    plt.plot([X[i], X[i+2]], [f(X[i]), 0], color="b")
    plt.scatter([X[i], X[i]], [0, f(X[i])], color="r")
    
x = np.linspace(0, max(X), 1000)
plt.plot(x, f(x), "k")
plt.title("Méthode de Quasi-Newton")
plt.show()

#%%


f1 = lambda x : np.exp(x - 1) - x + 1
f2 = lambda x : [(20 - np.sqrt(25 - x**2))/5, (20 + np.sqrt(25 - x**2))/5]

def trace():
    X = np.linspace(-6, 4, 101)
    plt.plot(X, f1(X), "k")

    X = np.linspace(-5, 5, 101)
    Y1, Y2 = f2(X)
    plt.plot(X, Y1, "b")
    plt.plot(X, Y2, "b")  
    
#%%


e1 = lambda X : f1(X[0]) - X[1]
de1dx = lambda X : np.exp(X[0] - 1) - 1
de1dy = lambda X : -1

e2 = lambda X : X[0]**2/25 + (X[1] - 4)**2 - 1
de2dx = lambda X : 2*X[0]/25
de2dy = lambda X : 2*(X[1] - 4)

f = lambda X : np.array([[e1(X)], [e2(X)]])
jf_newton = lambda X : np.array([[de1dx(X), de1dy(X)],
                                 [de2dx(X), de2dy(X)]])
jf_quasi_newton = lambda X0, X1 : np.array([[(e1([X0[0], 0]) - e1([X1[0], 0]))/(X0[0] - X1[0]), (e1([0, X0[1]]) - e1([0, X1[1]]))/(X0[1] - X1[1])],
                                            [(e2([X0[0], 0]) - e2([X1[0], 0]))/(X0[0] - X1[0]), (e2([0, X0[1]]) - e2([0, X1[1]]))/(X0[1] - X1[1])]])

#%%

def triangu(M):
    n, _ = M.shape
    for k in range(n-1):
        for i in range(k+1, n):
            c = M[i, k]/M[k, k]
            M[i, k:n+1] -= c*M[k, k:n+1]

def remonte(M):
    n, _ = M.shape
    x = np.zeros((n, 1))
    x[-1] = M[-1, -1]/M[-1, -2]
    for i in range(n-2, -1, -1):
        x[i] = (M[i, n] - np.sum(M[i, i+1:-1]*x[i+1:, 0]))/M[i, i]
    return x
    
def gauss(A, B):
    M = np.concatenate((A, B), axis=1)
    triangu(M)
    return remonte(M)

#%%

def newton_2(X0):
    X = [X0]
    e = epsilon
    while e >= epsilon:
        dX = gauss(jf_newton(X[-1]), -f(X[-1]))
        X.append(X[-1] + dX[:, 0])        
        e = np.linalg.norm(X[-1] - X[-2])
    return np.array(X)

def quasi_newton_2(l0):
    X = list(l0)
    e = epsilon
    while e >= epsilon:
        dX = gauss(jf_quasi_newton(X[-1], X[-2]), -f(X[-1]))
        X.append(X[-1] + dX[:, 0])
        e = np.linalg.norm(X[-1] - X[-2])
    return np.array(X)

#%%

print("\n+" + "-"*73 + "+")
print("|              Résolution itérative d'un système non linéaire             |")
print("+" + "-"*73 + "+\n")

trace()
plt.title("Système non linéaire")
plt.show()

#%%

print("\n+" + "-"*68 + "+")
print("|                           Méthode de Newton                        |")
print("+" + "-"*68 + "+\n")

X0 = np.array([[6, 8],
               [6, 1],
               [-2, 8],
               [-1, 1]])

for x0 in X0:
    X = newton_2(x0)
    plt.plot(X[:, 0], X[:, 1], linestyle="--", color="r")
    plt.scatter(X[:, 0], X[:, 1], color="r")
    print("En partant du point ",x0,", ",X.shape[0]," opérations faites, ||Xn - Xn-1|| = ", np.linalg.norm(X[-1] - X[-2]))
    print(X,"\n")

trace()
plt.title("Système non linéaire - Newton")
plt.show()

#%%

print("\n+" + "-"*68 + "+")
print("|                        Méthode de Quasi-Newton                     |")
print("+" + "-"*68 + "+\n")

d = 0.1
L0 = np.array([[[6, 8], [6+d, 8+d]],
               [[6, 1], [6+d, 1+d]],
               [[-2, 8], [-2+d, 8+d]],
               [[-1, 1], [-1+d, 1+d]]])

for l0 in L0:
    X = quasi_newton_2(l0)
    plt.plot(X[:, 0], X[:, 1], linestyle="--", color="r")
    plt.scatter(X[:, 0], X[:, 1], color="r")
    print("En partant du point ",x0,", ",X.shape[0]," opérations faites, ||Xn - Xn-1|| = ", np.linalg.norm(X[-1] - X[-2]))
    print(X,"\n")

trace() 
plt.title("Système non linéaire - Quasi-Newton")
plt.show()








