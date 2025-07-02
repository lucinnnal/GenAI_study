import os
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__), 'old_faithful.txt')
xs = np.loadtxt(path)
print(xs.shape) # (272, 2)

# initialize parameters for GMM -> k=2
# phi: mixing coefficients, mu: means, cov: covariance matrices
phis = np.array([0.5, 0.5])
mus = np.array([[0.0, 50.0], [0.0, 100.0]])
covs = np.array([np.eye(2), np.eye(2)]) # Identity matrices for covariance matrices

K = len(phis)  # # of Gaussian distributions in GMM
N = len(xs)  # # of data points
MAX_ITERS = 100 # Maximum number of iterations for EM algorithm
THRESHOLD = 1e-4 # Convergence threshold for EM algorithm -> 매개변수 갱신 후 로그 가능도의 변화량이 threshold보다 작으면 갱신 종료

# Probability density function for multivariate normal distribution
def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    d = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** d * det)
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
    return y

# Probability density function for Gaussian Mixture Model (GMM)
def gmm(x, phis, mus, covs):
    K = len(phis)
    y = 0
    for k in range(K):
        phi, mu, cov = phis[k], mus[k], covs[k]
        y += phi * multivariate_normal(x, mu, cov)
    return y

# Mean log likelihood of the GMM -> 알고리즘 종료를 위한 변화량 체크를 위해 
def likelihood(xs, phis, mus, covs):
    eps = 1e-8 # preventing log function receiving 0
    L = 0 
    N = len(xs)
    for x in xs:
        y = gmm(x, phis, mus, covs)
        L += np.log(y + eps)
    return L / N

# E-M Algorithm
current_likelihood = likelihood(xs, phis, mus, covs)

for iter in range(MAX_ITERS):
    # E-step -> qn(zn=k) for all data n and for all latent variable z
    qs = np.zeros((N, K)) # q(z=k) for N datas (N, K)
    for n in range(N):
        x = xs[n]
        for k in range(K):
            phi, mu, cov = phis[k], mus[k], covs[k]
            qs[n, k] = phi * multivariate_normal(x, mu, cov)
        qs[n] /= gmm(x, phis, mus, covs)   

    # M-step
    qs_sum = qs.sum(axis=0)

    for k in range(K):
        # phi(k) update
        phis[k] = qs_sum[k] / N

        # mu(k) update
        sum = 0
        for n in range(N):
            sum += qs[n, k] * xs[n]
        mus[k] = sum / qs_sum[k]

        # cov(k) update
        sum = 0
        for n in range(N):
            z = xs[n] - mus[k] # (2,)
            z = z[:, np.newaxis] # (2, 1) 
            sum += qs[n, k] * z @ z.T # (2, 2)
        covs[k] = sum / qs_sum[k]

    # Termination Checking
    print(f'Step {iter + 1} mean likelihood : {current_likelihood}')

    new_likelihood = likelihood(xs, phis, mus, covs)
    diff = np.abs(new_likelihood - current_likelihood)

    if diff < THRESHOLD:
        print("Stop EM Algorithm")
        break
    current_likelihood = new_likelihood

"""
# Parameter Updated GMM visualize with contour line
def plot_contour(w, mus, covs):
    x = np.arange(1, 6, 0.1)
    y = np.arange(40, 100, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])

            for k in range(len(mus)):
                mu, cov = mus[k], covs[k]
                Z[i, j] += w[k] * multivariate_normal(x, mu, cov)
    plt.contour(X, Y, Z, label='Trained GMM Contour')

plt.scatter(xs[:,0], xs[:,1], label='Sample Data')
plot_contour(phis, mus, covs)
plt.xlabel('Eruptions(Min)')
plt.ylabel('Waiting(Min)')
plt.legend()
plt.savefig('trained_gmm_contour.png')
plt.show()
"""

# Generation : sample from Trained GMM 
N = 500
new_xs = np.zeros((N, 2))
for n in range(N):
    k = np.random.choice(2, p=phis)
    mu, cov = mus[k], covs[k]
    new_xs[n] = np.random.multivariate_normal(mu, cov)

# visualize
plt.scatter(xs[:,0], xs[:,1], alpha=0.7, label='original')
plt.scatter(new_xs[:,0], new_xs[:,1], alpha=0.7, label='generated')
plt.legend()
plt.xlabel('Eruptions(Min)')
plt.ylabel('Waiting(Min)')
plt.savefig('gmm_generated_samples.png')
plt.show()