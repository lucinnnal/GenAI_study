import os 
import numpy as np
import matplotlib.pyplot as plt

# print(os.path.dirname(__file__)) prints the current directory
path = os.path.join(os.path.dirname(__file__), 'height.txt')
xs = np.loadtxt(path)

# MLE parameter estimation for normal distributions are sample mean and sample std
mu = np.mean(xs)
sigma = np.std(xs)
samples = np.random.normal(mu, sigma, 10000) # Sample from MLE estimation distribution

# Comparison with Sample data
plt.hist(xs, bins='auto', density=True, alpha=0.7, label='original')
plt.hist(samples, bins='auto', density=True, alpha=0.7, label='generated')
plt.xlabel('Height(cm)')
plt.ylabel('Probability Density')
plt.legend()
plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
plt.show()
