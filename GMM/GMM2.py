import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

np.random.seed(0)

N = 1000  # 班级总人数
N_male = 750  # 男生人数
N_female = 250  # 女生人数

mu_male = 130  # 男生身高均值
sigma_male = 8  # 男生身高标准差

mu_female = 165  # 女生身高均值
sigma_female = 5  # 女生身高标准差

male_heights = np.random.normal(mu_male, sigma_male, N_male)
female_heights = np.random.normal(mu_female, sigma_female, N_female)

heights = np.concatenate([male_heights, female_heights])
np.random.shuffle(heights)

def gaussian_mixture_pdf(x, weights, means, stds):
    pdf = np.zeros_like(x)
    for w, mu, sigma in zip(weights, means, stds):
        pdf += w * norm.pdf(x, mu, sigma)
    return pdf

# 初始化模型参数
weights = np.ones(2) / 2  # 权重
means = np.array([120, 165])  # 均值
stds = np.array([10, 10])  # 标准差

log_likelihoods = []
tol = 1e-6
max_iter = 1000
for i in range(max_iter):
    # E步：计算后验概率
    likelihoods = np.zeros((N, 2))
    for j in range(2):
        likelihoods[:, j] = norm.pdf(heights, means[j], stds[j])
    likelihoods *= weights
    likelihoods /= likelihoods.sum(axis=1, keepdims=True)

    # 计算对数似然函数值
    log_likelihood = np.log(likelihoods.sum(axis=1)).sum()
    log_likelihoods.append(log_likelihood)

    # 判断收敛
    if i > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
        break

    # M步：更新模型参数
    weights = likelihoods.mean(axis=0)
    means = (likelihoods * heights[:, np.newaxis]).sum(axis=0) / likelihoods.sum(axis=0)
    stds = np.sqrt((likelihoods * (heights[:, np.newaxis] - means) ** 2).sum(axis=0) / likelihoods.sum(axis=0))

# 输出结果
print("Weights: ", weights)
print("Means: ", means)
print("Stds: ", stds)

# 绘制数据分布和拟合的混合高斯分布
x = np.linspace(100, 200, 1000)
pdf = gaussian_mixture_pdf(x, weights, means, stds)

plt.figure(figsize=(8, 5))
plt.hist(heights, bins=20, density=True, alpha=0.5)
plt.plot(x, pdf, label="Fitted Gaussian Mixture")
plt.legend()
plt.xlabel("Height (cm)")
plt.ylabel("Density")
plt.show()

