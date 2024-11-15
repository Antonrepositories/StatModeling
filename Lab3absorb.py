import numpy as np
import matplotlib.pyplot as plt

#Задання матриці переходів та вектора початкових станів для поглинаючого ланцюга 
states = 7
absorbing_states = [5, 6]
non_absorbing_states = [0, 1, 2, 3, 4]

P_absorbing = np.array([
    [0.1, 0.6, 0.2, 0.1, 0.0, 0.0, 0.0],
    [0.3, 0.0, 0.5, 0.2, 0.0, 0.0, 0.0],
    [0.4, 0.0, 0.0, 0.4, 0.0, 0.1, 0.1],
    [0.3, 0.0, 0.2, 0.0, 0.4, 0.1, 0.0],
    [0.0, 0.3, 0.2, 0.4, 0.0, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
])

initial_state_absorbing = [1, 0, 0, 0, 0, 0, 0]  

def simulate_absorbing_chain(P, initial_state, num_realizations=300):
    absorbing_times = []
    absorption_probabilities = [0] * states
    trajectories = []
    transition_counts = np.zeros((states, states))  #Матриця переходів експериментальна

    for _ in range(num_realizations):
        state = np.random.choice(range(states), p=initial_state)
        trajectory = [state]
        steps = 0
        
        while state not in absorbing_states:
            next_state = np.random.choice(range(states), p=P[state])
            transition_counts[state, next_state] += 1 
            state = next_state
            trajectory.append(state)
            steps += 1
            
        absorbing_times.append(steps)
        absorption_probabilities[state] += 1
        trajectories.append(trajectory)
    
    absorption_probabilities = [p / num_realizations for p in absorption_probabilities]
    
    #Обчислення експериментальної матриці переходів
    transition_probabilities = transition_counts / transition_counts.sum(axis=1, keepdims=True)

    transition_probabilities = np.nan_to_num(transition_probabilities)
    
    return absorbing_times, absorption_probabilities, trajectories, transition_probabilities

absorbing_times, absorption_probabilities, trajectories_absorbing, experiment_matrix = simulate_absorbing_chain(
    P_absorbing, initial_state_absorbing
)

# Теоретичні характеристики
Q = P_absorbing[non_absorbing_states, :][:, non_absorbing_states]
R = P_absorbing[non_absorbing_states, :][:, absorbing_states]
I = np.eye(len(non_absorbing_states))
N = np.linalg.inv(I - Q)

mean_absorption_time_theoretical = np.sum(N, axis=1)
absorption_probabilities_theoretical = np.dot(N, R)

print("Теоретичні характеристики:")
print("Час поглинання для кожного стану (не-поглинаючі стани):")
for state, time in zip(non_absorbing_states, mean_absorption_time_theoretical):
    print(f"Стан {state}: {time:.2f} кроків")

print("\nЙмовірність поглинання для кожного не-поглинаючого стану:")
for state, probability in zip(non_absorbing_states, absorption_probabilities_theoretical):
    print(f"Стан {state}: {probability[0]:.4f}")


print("\nЕкспериментальні характеристики:")
print("Час поглинання (середнє значення):", np.mean(absorbing_times))
print("Ймовірності поглинання для кожного стану:")
for state, probability in enumerate(absorption_probabilities):
    print(f"Стан {state}: {probability:.4f}")

print("\nЕкспериментальна матриця переходів:")
print(experiment_matrix)

# Побудова графіка реалізацій поглинаючого ланцюга
plt.figure(figsize=(10, 6))
for i, trajectory in enumerate(trajectories_absorbing[:7]):
    plt.plot(range(1, len(trajectory) + 1), trajectory, marker='o', label=f'Ряд{i+1}')
plt.title("Реалізації поглинаючого ланцюга Маркова")
plt.xlabel("Крок")
plt.ylabel("Стан")
plt.legend(loc="best")
plt.grid(True)
plt.show()