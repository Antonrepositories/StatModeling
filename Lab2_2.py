import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Параметри моделювання
T = 1.0         # кінцевий час
N = 500         # кількість кроків у часі
dt = T / N      # крок по часу
n_realizations = 1000  # кількість реалізацій
level = 1.0     # заданий рівень для пошуку часу виходу

# 1. Генерація вінерівського процесу
time = np.linspace(0, T, N)
dW = np.sqrt(dt) * np.random.randn(n_realizations, N)
W = np.cumsum(dW, axis=1)

# 2. Оцінка середнього значення та дисперсії
mean_W = np.mean(W, axis=0)
var_W = np.var(W, axis=0)

# Графік середнього та дисперсії
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(time, mean_W, label="Середнє значення")
plt.title("Середнє значення Вінерівського процесу")
plt.xlabel("Час")
plt.ylabel("Середнє значення")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time, var_W, label="Дисперсія")
plt.title("Дисперсія Вінерівського процесу")
plt.xlabel("Час")
plt.ylabel("Дисперсія")
plt.legend()
plt.show()

# 3. Знаходження емпіричного закону розподілу часу першого виходу
first_exit_times = []
for i in range(n_realizations):
    crossing_points = np.where(W[i, :] >= level)[0]
    if len(crossing_points) > 0:
        first_exit_time = time[crossing_points[0]]
        first_exit_times.append(first_exit_time)

# Побудова гістограми для емпіричного закону розподілу часу першого виходу
plt.hist(first_exit_times, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
plt.title("Емпіричний закон розподілу часу першого виходу за рівень")
plt.xlabel("Час першого виходу")
plt.ylabel("Частота")
plt.show()
