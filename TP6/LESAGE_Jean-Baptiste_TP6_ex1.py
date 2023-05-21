import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import odeint

F = lambda X, t : np.array([X[1], -3*X[1] - 40*X[0]])


def Euler(X0, F, a, b, N):
    Ysol = [X0]
    T = np.linspace(a, b, N)
    pas = (b - a)/(N - 1)
    for k in range(1, N):
        Ysol.append(Ysol[k-1] + pas*F(Ysol[k-1], T[k-1]))
    return np.array(Ysol)


def Euler_modif(X0, F ,a ,b ,N):
    Ysol = [X0]
    T = np.linspace(a, b, N)
    pas = (b - a)/(N - 1)
    for k in range(1, N):
        Ysol.append(Ysol[k-1] + pas*F(Ysol[k-1] + pas*F(Ysol[k-1], T[k-1])/2, T[k-1] + pas/2))
    return np.array(Ysol)


def Adams_Moulton_2(X0, F, a, b, N):
    Ysol = [X0]
    pas = (b - a)/(N - 1)    
    for k in range(1,N):
        y = Ysol[-1][0]
        dy = Ysol[-1][1]
        ddy = -3*dy-40*y
        h = (y + pas/2*(1/(1 + 3/2*pas)*(dy + pas/2*ddy) + dy))*(1 + 3/2*pas)/(1 + 3/2*pas + (pas/2)**2*40)
        s = (dy + pas/2*(-40*h + ddy))/(1 + 3/2*pas)
        Ysol.append((h, s))
    return np.array(Ysol)


def aff(met, X):
    print("Itération d'" + met +" :")
    for k in range(N):
        print("y" + str(k) + " =", X[0, k])
    print()
    
    plt.title("Résolution d'un système d'ED par" + met +  " avec un pas = " + str(N - 1))
    plt.plot(T, X[0], 'r-.o', label="y_"+met)
    plt.plot(t, X_exact[0], 'r', label="y_exact")
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.title("Résolution d'un système d'ED par RK1 avec un pas = " + str(N - 1))
    plt.plot(T, X[1], 'b-.o', label="dy_"+met)
    plt.plot(t, X_exact[1], 'b', label="dy_exact")
    plt.grid()
    plt.legend()
    plt.show()


def main():
    N = int(input("Nombre de pas : ")) + 1
    print()

    X0, a, b = (1, 1/3), 0, 1

    t = np.linspace(a, b, 101)
    T = np.linspace(a, b, N)

    X_exact = odeint(F, X0, t).T

    X1 = Euler(X0, F, a, b, N).T
    aff('RK1', X1)
    X2 = Euler_modif(X0, F, a, b, N).T
    aff('RK2', X2)
    X3 = Adams_Moulton_2(X0, F, a, b, N).T
    aff('AM2', X3)

    X = np.array((X1, X2, X3))[:, 0]
    err = np.round(np.abs(X - odeint(F, X0, T).T[0]), 8)  # TODO: Remplacer par résultat courbe analytique - résultat obtenu

    print("\nTableau des erreurs :")
    print('\ty0' + ''.join(['         y' + str(i) for i in range(1, N)]))
    for met_name, e in zip(['RK1', 'RK2', 'AM2'], err):
        print(met_name, str(e)[1:-1])


if __name__ == '__main__':
    main()
