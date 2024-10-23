import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

N = 1000

#Генерація ВВ
data = [random.random() for _ in range(N)]

#середнє та дисперсія
mean_empirical = np.mean(data)
variance_empirical = np.var(data)

#Теоретичні значення
mean_theoretical = 0.5
variance_theoretical = 1/12

print(f"Empirical mean: {mean_empirical}")
print(f"Theoretical mean: {mean_theoretical}")
print(f"Empirical variance: {variance_empirical}")
print(f"Theoretical variance: {variance_theoretical}")

#Критерій Пірсона
observed_values, bin_edges = np.histogram(data, bins=10, density=False)
expected_values = [N / 10] * 10  #Очікувана кількість у кожному інтервалі

chi2_stat, p_value = stats.chisquare(observed_values, expected_values)

print(f"Xi-squared test: {chi2_stat}")
print(f"P-value: {p_value}")
# Візуалізація гістограми
plt.hist(data, bins=10, density=True, alpha=0.6, color='g', label='Empirical Data')
plt.title('Гістограма')
plt.xlabel('Значення')
plt.legend()
plt.show()
