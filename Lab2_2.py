import numpy as np
import matplotlib.pyplot as plt

# Параметри
T = 1      
N = 1000   
a = 0.1    
num_realizations = 100  

#M = [10000, 100000, 1000000]
M = 10000
t = np.linspace(0, T, N)

def w2_process(M, t):
    eta_0 = np.random.normal()
    eta_1 = np.random.normal(size=(M,))
    eta_2 = np.random.normal(size=(M,))
    W2 = eta_0 * t + np.sqrt(2) * np.sum(
        [
            eta_1[i] * np.sin(2 * np.pi * (i + 1) * t) / (2 * np.pi * (i + 1)) +
            eta_2[i] * (1 - np.cos(2 * np.pi * (i + 1) * t)) / (2 * np.pi * (i + 1))
            for i in range(M)
        ],
        axis=0
    )
    return W2

# Список для збереження всіх реалізацій та часу першого виходу
all_realizations = []
first_exit_times = []

# Генерація реалізацій та обчислення часу першого виходу
for _ in range(num_realizations):
    W = w2_process(M, t)
    all_realizations.append(W)
    first_exit_time = next((t[i] for i in range(N) if W[i] >= a), 0)
    first_exit_times.append(first_exit_time)

all_realizations = np.array(all_realizations)
mean_process = np.mean(all_realizations, axis=0)
variance_process = np.var(all_realizations, axis=0)

# Вибір однієї реалізації для побудови графіка виходу
W_example = all_realizations[0]
first_exit_example = next((t[i] for i in range(N) if W_example[i] >= a), None)

# Побудова графіка для однієї реалізації з позначенням часу виходу
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, W_example, color='red', label='W(t)')
plt.axhline(y=a, color='green', linestyle='-', label=f'Рівень a={a}')
if first_exit_example:
    plt.axvline(x=first_exit_example, color='green', linestyle='--', label=f'Час виходу t={first_exit_example:.3f}')
plt.xlabel('Час t')
plt.ylabel('W(t)')
plt.title(f'Вихід процесу за рівень a={a}')
plt.legend()
plt.grid()

# Побудова графіка середнього та дисперсії
plt.subplot(2, 1, 2)
plt.plot(t, mean_process, color='blue', label='Середнє значення процесу')
plt.plot(t, variance_process, color='green', label='Дисперсія процесу')
plt.xlabel('Час t')
plt.title('Середнє та дисперсія за реалізаціями процесу')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

print("Час першого виходу для кожної реалізації:")
for i, exit_time in enumerate(first_exit_times, 1):
    print(f"Реалізація {i}: Час виходу = {exit_time}")


#first_exit_times = np.array(first_exit_times)  

sorted_times = np.sort(first_exit_times)

# Розрахунок CDF
cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)

# Побудова графіка CDF
plt.figure(figsize=(10, 6))
plt.plot(sorted_times, cdf, marker="o", linestyle="-", color="blue", label="Емпірична CDF")
plt.xlabel("Час першого виходу")
plt.ylabel("Ймовірність")
plt.title(f"Емпірична функція розподілу часу першого виходу за рівень a={a}")
plt.grid()
plt.legend()
plt.show()


# Побудова гістограми всіх значень
plt.figure(figsize=(10, 6))
plt.hist(all_realizations.flatten(), bins=30, color='skyblue', edgecolor='blue', alpha=0.7)
plt.title('Гістограма значень W2 для всіх реалізацій')
plt.xlabel('W2 значення')
plt.ylabel('Частота')
plt.grid()
plt.show()