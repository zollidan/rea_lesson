# Практическая работа 1
# NumPy + Matplotlib
# Задание выполнено одним файлом.

import math
import numpy as np
import matplotlib.pyplot as plt


# Задание 1
# Построить архимедову спираль.
# Столбцы таблицы: n, phi, x, y.
# n: целые числа от 0 до 96.
# Формулы:
# phi = n * pi / 16
# r = phi
# x = r * cos(phi)
# y = r * sin(phi)

n1 = np.arange(0, 97)
phi1 = n1 * math.pi / 16
r1 = phi1
x1 = r1 * np.cos(phi1)
y1 = r1 * np.sin(phi1)

archimedes_table = np.column_stack((n1, phi1, x1, y1))

print('Задание 1. Архимедова спираль')
print('n\tphi\t\tx\t\ty')
for row in archimedes_table:
    print(f'{int(row[0])}\t{row[1]:.4f}\t\t{row[2]:.4f}\t\t{row[3]:.4f}')

plt.figure(figsize=(6, 6))
plt.plot(x1, y1)
plt.title('Архимедова спираль')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')
plt.show()


# Задание 2
# Построить улитку Паскаля.
# Столбцы таблицы: n, phi, x, y.
# n: целые числа от 0 до 24.
# Формулы:
# phi = n * pi / 12
# r = 1 + 2 * cos(phi)
# x = r * cos(phi)
# y = r * sin(phi)

n2 = np.arange(0, 25)
phi2 = n2 * math.pi / 12
r2 = 1 + 2 * np.cos(phi2)
x2 = r2 * np.cos(phi2)
y2 = r2 * np.sin(phi2)

pascal_table = np.column_stack((n2, phi2, x2, y2))

print('\nЗадание 2. Улитка Паскаля')
print('n\tphi\t\tx\t\ty')
for row in pascal_table:
    print(f'{int(row[0])}\t{row[1]:.4f}\t\t{row[2]:.4f}\t\t{row[3]:.4f}')

plt.figure(figsize=(6, 6))
plt.plot(x2, y2)
plt.title('Улитка Паскаля')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')
plt.show()
