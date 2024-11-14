import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#цільовa функція
def func(x1, x2):
    return np.cos(x1) * np.cos(x2) * np.exp(-(x1**2 + x2**2))

#Обмеження для x1 та x2
x_min, x_max = -np.pi, 3 * np.pi
y_min, y_max = -np.pi, 3 * np.pi

#Пошук
K = 10000  #Кількість випадкових точок
N = 100  #Кількість повторень пошуку
#Для пошуку максимуму
best_x1, best_x2 = None, None
best_value = -float('inf')
#Для пошуку мінімуму
#best_x1, best_x2 = 1, 1
#best_value = float('inf')

#Випадковий пошук
for i in range(N):
    current_best_value = -float('inf')
    for _ in range(K):
        x1_rand = random.uniform(x_min, x_max)
        x2_rand = random.uniform(y_min, y_max)
        value = func(x1_rand, x2_rand)

        if value > current_best_value:
            current_best_value = value
            current_best_x1 = x1_rand
            current_best_x2 = x2_rand
    
    #Оновлюємо глобальний максимум, якщо знайшли кращий
    if current_best_value > best_value:
        best_value = current_best_value
        best_x1 = current_best_x1
        best_x2 = current_best_x2

#Малюємо графік
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x1 = np.linspace(x_min, x_max, 400)
x2 = np.linspace(y_min, y_max, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = func(X1, X2)

ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')

ax.scatter(best_x1, best_x2, func(best_x1, best_x2), color='red', s=100, label='Found maximum')

ax.set_title('Plot of f(x1, x2) = cos(x1) cos(x2) exp(-(x1^2 + x2^2))')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

plt.legend()
plt.show()


print(f"Max value found at x1 = {best_x1}, x2 = {best_x2}")
print(f"f(x1, x2): {best_value}")
