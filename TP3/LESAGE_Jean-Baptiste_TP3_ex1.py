import numpy as np
import matplotlib.pyplot as plt

#%%
path = './linear_systems_to_solve/'
# path = '//intram.ensam.eu/Cluny/TP/tp-infomath/MENUM/linear_systems_to_solve/sysLinDiagDominante_'
N_max, epsilon = 100, 10e-6


def import_mat(n):
    f = open(path + 'sysLinDiagDominante_' + str(n), 'r')
    lines = f.readlines()
    f.close()
    
    a = np.array([[float(cara) for cara in line.split(" ") if cara != ''] for line in lines[1:n+1]])
    b = np.array([[float(cara)] for cara in lines[n+2].split(" ") if cara != ''])
    x = np.array([[float(cara)] for cara in lines[n+4].split(" ") if cara != ''])
    
    return a, b, x


def jacobi(a, b, x0=None):
    n, _ = a.shape
    e, iteration = epsilon, 0
    x = [np.zeros((n, 1), float) if x0 is None else x0]
    a_ = a.copy()
    np.fill_diagonal(a_, 0)
    D = np.diag(a)[:, np.newaxis]
    while iteration < N_max and e >= epsilon:
        x.append((b - a_.dot(x[-1])) / D)
        e = np.linalg.norm(x[-1] - x[-2])
        iteration += 1
    return x


def gauss_seidel(a, b, x0=None):
    n, _ = a.shape
    e, iteration = epsilon, 0
    x = [np.zeros((n, 1), float) if x0 is None else x0]
    while iteration < N_max and e >= epsilon:
        x.append(x[-1].copy())
        for i in range(n):
            s = 0
            for j in range(i):
                s += a[i, j] * x[-1][j]
            for j in range(i+1, n):
                s += a[i, j] * x[-2][j]
            x[-1][i] = (b[i] - s) / a[i, i]
        e = np.linalg.norm(x[-1] - x[-2])
        iteration += 1
    return x


def relaxation(a, b, x0=None, w=1.2):
    n, _ = a.shape
    e, iteration = epsilon, 0
    x = [np.zeros((n, 1), float) if x0 is None else x0]
    while iteration < N_max and e >= epsilon:
        x.append(x[-1].copy())
        for i in range(n):
            s = 0
            for j in range(i):
                s += a[i, j] * x[-1][j]
            for j in range(i+1, n):
                s += a[i, j] * x[-2][j]
            x[-1][i] = (b[i] - s) / a[i, i]
            x[-1][i] = x[-2][i] + w*(x[-1][i] - x[-2][i])
        e = np.linalg.norm(x[-1] - x[-2])
        iteration += 1
    return x


def change_array(x):
    xx = np.empty((2, x.shape[1] * 2))
    xx[:, ::2] = xx[:, 1::2] = x
    return xx[0, 1:], xx[1, :-1]


def aff(a, b, x, method=None):
    n, _ = a.shape
    
    print('\nFin de', method, 'en', len(x), 'itérations')
    print('Justesse :', np.linalg.norm(a.dot(x[-1]) - b))
    
    if n == 2:
        plt.figure()
        
        s = np.linspace(-10, 15, 2)
        plt.plot(s, (b[0] - a[0, 0] * s) / a[0, 1], 'tab:blue')
        plt.plot(s, (b[1] - a[1, 0] * s) / a[1, 1], 'tab:blue')
        
        x = change_array(np.array(x).T[0])
        plt.plot(x[0], x[1], 'r.--')
        
        plt.title(method)
        plt.show()


def main():
    print('+' + '-'*51 + '+')
    print("| Résolution d'un système choisi par l'utilisateur |")
    print('+' + '-'*51 + '+')

    n = int(input("Saisir la dimension du système : "))
    a, b, x = import_mat(n)

    print('-'*14 + ' Importation du système ' + '-'*15)

    print("\nA :\n", a, sep='')
    print("\nB :\n", b, sep='')
    print("\nX :\n", x, sep='')
    print()

    #%%

    print('+' + '-'*51 + '+')
    print("|          Résolution par méthode itérative         |")
    print('+' + '-'*51 + '+')

    if n == 2:
        x0 = np.array([[15.], [10.]])
    else:
        x0 = None

    aff(a, b, jacobi(a, b, x0), 'Jacobi')
    aff(a, b, gauss_seidel(a, b, x0), 'Gauss-Seidel')
    aff(a, b, relaxation(a, b, x0), 'Relaxation')


if __name__ == '__main__':
    main()
