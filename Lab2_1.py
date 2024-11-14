import numpy as np
import matplotlib.pyplot as plt
import math

lambda_rate = 0.5
n_events = 100    

#Генерація стрибків
inter_arrival_times = -np.log(1 - np.random.rand(n_events)) / lambda_rate
jump_times = np.cumsum(inter_arrival_times)

#Побудова графіка процесу
plt.figure(figsize=(10, 6))
plt.step(jump_times, np.arange(1, n_events + 1), where='post')
plt.xlabel("Час")
plt.ylabel("Кількість подій")
plt.title("Реалізація Пуассонівського процесу")
plt.grid()
plt.show()

#Побудова гістограм розподілів
event_times = [jump_times[0], jump_times[1], jump_times[-1]]
event_labels = ['Перша подія', 'Друга подія', f'Остання подія ({n_events}-та)']

#Побудова гістограми часу появи першої, другої та останньої подій
plt.figure(figsize=(10, 6))
plt.bar(event_labels, event_times, color=['blue', 'green', 'red'], alpha=0.7)
plt.xlabel("Події")
plt.ylabel("Час появи події")
plt.title("Час появи першої, другої та останньої події процесу")
plt.grid(axis='y')

#Гістограма інтервалу між подіями
plt.figure(figsize=(10, 6))
plt.hist(inter_arrival_times, bins=30, color='green', alpha=0.7)
plt.xlabel("Інтервал між подіями")
plt.ylabel("Частота")
plt.title("Гістограма інтервалів між подіями")
plt.grid()
plt.show()

#Гістограма кількості появи рівно n подій за фіксований інтервал часу
lambda_rate = 0.5  
n = 10             
T = n / lambda_rate 
M = 10000         
event_counts = []
for _ in range(M):
    inter_arrival_times = -np.log(1 - np.random.rand(n * 2)) / lambda_rate
    jump_times = np.cumsum(inter_arrival_times)
    event_counts.append(np.sum(jump_times <= T))

n_event_frequency = event_counts.count(n) / M

theoretical_probability = (math.exp(-lambda_rate * T) * (lambda_rate * T)**n) / math.factorial(n)

print(f"Частота появи рівно {n} подій (змодельована): {n_event_frequency:.4f}")
print(f"Теоретична ймовірність P(N(T) = {n}): {theoretical_probability:.4f}")

plt.figure(figsize=(10, 6))
plt.hist(event_counts, bins=range(0, max(event_counts)+1), color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(n, color='red', linestyle='dashed', linewidth=1.5, label=f'n = {n}')
plt.xlabel("Кількість подій за час T")
plt.ylabel("Частота")
plt.title(f"Гістограма частоти кількості подій за час T = {T}")
plt.legend()
plt.grid(axis='y')
plt.show()