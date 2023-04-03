import numpy as np
import numpy.linalg as npl
import time
import matplotlib.pyplot as plt

path: str = 'C:/Users/Uxus/PycharmProjects/MENUM/TP 1'
# path: str = "//intram.ensam.eu/Cluny/TP/tp-infomath/MENUM/linear_systems_to_solve/"

#%%


def import_mat(n):
    """Importation d'une matrice de dimension n à partir du fichier correspondant.
    :type n: int
    """

    file = open(path + '/linear_systems_to_solve/sysLin_' + str(n), 'r')
    lines = file.readlines()
    file.close()

    a = np.array([[float(char) for char in lines.split(' ') if char != ''] for lines in lines[1:n+1]])
    b = np.array([[float(char)] for char in lines[n + 2].split(' ') if char != ''])
    x = np.array([[float(char)] for char in lines[n + 4].split(' ') if char != ''])
    det = float(lines[n+6])

    return a, b, x, det


def determinant(matrice):
    """Calcul du déterminant d'une matrice."""

    n, _ = matrice.shape
    if n > 2:
        s = 0
        for j in range(n):
            s += (-1) ** j * matrice[0, j] * determinant(np.delete(matrice[1:], j, axis=1))
        return s
    elif n == 2:
        return matrice[0, 0] * matrice[1, 1] - matrice[0, 1] * matrice[1, 0]
    else:
        return matrice[0, 0]


cramer = lambda a, b: np.array([[determinant(np.concatenate((a[:, :i], b, a[:, i+1:]), axis=1))] for i in range(n)])/determinant(a)

#%%

print('+' + '-'*51 + '+')
print("| Résolution d'un système choisi par l'utilisateur |")
print('+' + '-'*51 + '+')

n = int(input("Saisir la dimension du système : "))
a, b, x, det = import_mat(n)

print('-'*14 + ' Importation du système ' + '-'*15, sep='')

print("\nA :\n", a, sep='')
print("\nB :\n", b, sep='')
print("\nX :\n", x, sep='')
print("\nDéterminant :", det)
print()

#%%

print('-'*15 + ' Calcul du déterminant ' + '-'*15)

d = determinant(a)
print("\nDéterminant calculé : ", d)
print('Erreur du déterminant calculé : ', np.abs(d-det))
print()

#%%

print('-'*20 + ' Calcul de x ' + '-'*20)

solution = cramer(a, b)
print("\nSolution calculée :\n", solution, sep='')
print('Erreur de la solution calculée : ', np.linalg.norm(a.dot(solution)-b))
print()

#%%

print('+' + '-'*51 + '+')
print("|    Résolution des systèmes de dimension 2 à 8    |")
print('+' + '-'*51 + '+')

temps_execution = []
for n in range(2, 9):
    a, b, _, _ = import_mat(n)
    print('(' + str(n) + ') ... ', end='')
    start = time.time()
    cramer(a, b)
    temps_execution.append(time.time()-start)
    print('OK')

plt.grid()
plt.plot(range(2, 9), temps_execution)
plt.title('Méthode de Cramer')
plt.xlabel('Dimension du système')
plt.ylabel('Temps de calcul (en s)')
plt.show()
