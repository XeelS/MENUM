import numpy as np
import matplotlib.pyplot as plt

#%%

global path, epsillon, N_max
path = './linear_systems_to_solve/'
# path = '//intram.ensam.eu/Cluny/TP/tp-infomath/MENUM/linear_systems_to_solve/sysLinDiagDominante_'
N_max, epsillon = 100, 10e-6

def import_mat(n):
    f = open(path + 'sysLinDiagDominante_' + str(n), 'r')
    lines = f.readlines()
    f.close()
    
    A = np.array([[float(cara) for cara in line.split(" ") if cara != ''] for line in lines[1:n+1]])
    B = np.array([[float(cara)] for cara in lines[n+2].split(" ") if cara != ''])
    X = np.array([[float(cara)] for cara in lines[n+4].split(" ") if cara != ''])
    
    return A, B, X


def jacobi(A, B, x0=None):
    n, _ = A.shape
    e, N = epsillon, 0
    X = [np.zeros((n, 1), float) if x0 is None else x0]
    A_ = A.copy()
    np.fill_diagonal(A_, 0)
    D = np.diag(A)[:, np.newaxis]
    while N < N_max and e >= epsillon:
        X.append((B - A_.dot(X[-1]))/D)
        e = np.linalg.norm(X[-1] - X[-2])
        N += 1
    return X


def gauss_seidel(A, B, x0=None):
    n, _ = A.shape
    e, N = epsillon, 0
    X = [np.zeros((n, 1), float) if x0 is None else x0]
    while N < N_max and e >= epsillon:
        X.append(X[-1].copy())
        for i in range(n):
            s = 0
            for j in range(i):
                s += A[i, j]*X[-1][j]
            for j in range(i+1, n):
                s += A[i, j]*X[-2][j]
            X[-1][i] = (B[i] - s)/A[i, i]
        e = np.linalg.norm(X[-1] - X[-2])
        N += 1
    return X


def relaxation(A, B, x0=None, w=1.2):
    n, _ = A.shape
    e, N = epsillon, 0
    X = [np.zeros((n, 1), float) if x0 is None else x0]
    while N < N_max and e >= epsillon:
        X.append(X[-1].copy())
        for i in range(n):
            s = 0
            for j in range(i):
                s += A[i, j]*X[-1][j]
            for j in range(i+1, n):
                s += A[i, j]*X[-2][j]
            X[-1][i] = (B[i] - s)/A[i, i]
            X[-1][i] = X[-2][i] + w*(X[-1][i] - X[-2][i])
        e = np.linalg.norm(X[-1] - X[-2])
        N += 1
    return X


def change_array(X):
    XX = np.empty((2, X.shape[1]*2))
    XX[:, ::2] = XX[:, 1::2] = X
    return XX[0, 1:], XX[1, :-1]


def aff(A, B, X, method=None):
    n, _ = A.shape
    
    print('\nFin de', method, 'en', len(X), 'itérations')
    print('Justesse :', np.linalg.norm(A.dot(X[-1]) - B))
    
    if n == 2:
        plt.figure()
        
        S = np.linspace(-10, 15, 2)
        plt.plot(S, (B[0] - A[0, 0]*S)/A[0, 1], 'tab:blue')
        plt.plot(S, (B[1] - A[1, 0]*S)/A[1, 1], 'tab:blue')
        
        X = change_array(np.array(X).T[0])
        plt.plot(X[0], X[1], 'r.--')
        
        plt.title(method)
        plt.show()


def main():
    print('+' + '-'*51 + '+')
    print("| Résolution d'un système choisi par l'utilisateur |")
    print('+' + '-'*51 + '+')

    n = int(input("Saisir la dimension du système : "))
    A, B, X = import_mat(n)

    print('-'*14 + ' Importation du système ' + '-'*15)

    print("\nA :\n", A, sep='')
    print("\nB :\n", B, sep='')
    print("\nX :\n", X, sep='')
    print()

    #%%

    print('+' + '-'*51 + '+')
    print("|          Résolution par méthode itérative         |")
    print('+' + '-'*51 + '+')

    if n == 2:
        x0 = np.array([[15.], [10.]])
    else:
        x0 = None

    aff(A, B, jacobi(A, B, x0), 'Jacobi')
    aff(A, B, gauss_seidel(A, B, x0), 'Gauss-Seidel')
    aff(A, B, relaxation(A, B, x0), 'Relaxation')


if __name__ == '__main__':
    main()
