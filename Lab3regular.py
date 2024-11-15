import numpy as np
import matplotlib.pyplot as plt

P_regular = np.array([
    [0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3],
    [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
])

initial_distribution_regular = [1 / 7] * 7

def simulate_regular_chain(P, initial_distribution, steps=10, num_realizations=300):
    trajectories = []
    
    for _ in range(num_realizations):
        state = np.random.choice(range(len(P)), p=initial_distribution)
        trajectory = [state]
        
        for _ in range(steps):
            state = np.random.choice(range(len(P)), p=P[state])
            trajectory.append(state)
        
        trajectories.append(trajectory)
    
    return trajectories

trajectories_regular = simulate_regular_chain(P_regular, initial_distribution_regular, steps=10)

#Теоретичний стаціонарний розподіл
eigvals, eigvecs = np.linalg.eig(P_regular.T)
stationary_distribution_theoretical = eigvecs[:, np.isclose(eigvals, 1)].flatten()

stationary_distribution_theoretical = np.real(stationary_distribution_theoretical)
stationary_distribution_theoretical /= stationary_distribution_theoretical.sum()

#Теоретичний час перебування в кожному стані
mean_time_theoretical = 1 / stationary_distribution_theoretical

print("Теоретичний стаціонарний розподіл:")
print(stationary_distribution_theoretical)

print("\nТеоретичний час перебування в кожному стані:")
for state, time in zip(range(len(P_regular)), mean_time_theoretical):
    print(f"Стан {state}: {time:.2f} кроки")

#Експериментальна матриця переходів
transition_counts = np.zeros_like(P_regular)
for trajectory in trajectories_regular:
    for i in range(len(trajectory) - 1):
        transition_counts[trajectory[i], trajectory[i+1]] += 1

#Нормалізація для отримання ймовірностей
transition_probabilities_experimental = transition_counts / transition_counts.sum(axis=1, keepdims=True)

#Експериментальний час перебування в кожному стані
state_durations = np.zeros((len(P_regular), len(trajectories_regular)))
for idx, trajectory in enumerate(trajectories_regular):
    for state in range(len(P_regular)):
        state_durations[state, idx] = trajectory.count(state)

#Середній час перебування в кожному стані
mean_state_durations = np.mean(state_durations, axis=1)

#Експериментальний стаціонарний розподіл
state_visit_counts = np.sum(state_durations, axis=1)
state_visit_probabilities = state_visit_counts / (len(trajectories_regular) * len(trajectories_regular[0]))

print("\nЕкспериментальний стаціонарний розподіл:")
print(state_visit_probabilities)

print("\nЕкспериментальна матриця переходів:")
print(transition_probabilities_experimental)

print("\nЕкспериментальний час перебування в кожному стані (середнє значення):")
for state, duration in zip(range(len(P_regular)), mean_state_durations):
    print(f"Стан {state}: {duration:.2f} кроки")

# Побудова графіка реалізацій регулярного ланцюга
plt.figure(figsize=(10, 6))
for i, trajectory in enumerate(trajectories_regular[:7]):  
    plt.plot(range(1, len(trajectory) + 1), trajectory, marker='o', label=f'Ряд {i+1}')
plt.title("Реалізації регулярного ланцюга Маркова")
plt.xlabel("Крок")
plt.ylabel("Стан")
plt.legend()
plt.grid(True)
plt.show()
