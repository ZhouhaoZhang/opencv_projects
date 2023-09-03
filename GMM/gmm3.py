# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt

# 设置随机数种子
np.random.seed(2023)

# 定义混合高斯分布的参数
mu1 = 170  # 男生身高均值
sigma1 = 10  # 男生身高标准差
mu2 = 160  # 女生身高均值
sigma2 = 8  # 女生身高标准差
alpha = 0.6  # 男生占比

# 生成100个人的身高数据
n = 100  # 样本数量
z = np.random.binomial(1, alpha, n)  # 隐变量，表示性别
x = np.zeros(n)  # 观测变量，表示身高
for i in range(n):
    if z[i] == 1:  # 如果是男生
        x[i] = np.random.normal(mu1, sigma1)  # 按照男生身高分布生成数据
    else:  # 如果是女生
        x[i] = np.random.normal(mu2, sigma2)  # 按照女生身高分布生成数据


# 定义EM算法类
class EM:
    def __init__(self, x):
        self.x = x  # 观测数据
        self.n = len(x)  # 样本数量

        # 初始化参数
        self.mu1 = np.mean(x) - 5  # 第一个分量的均值
        self.sigma1 = np.std(x) - 2  # 第一个分量的标准差
        self.mu2 = np.mean(x) + 5  # 第二个分量的均值
        self.sigma2 = np.std(x) + 2  # 第二个分量的标准差
        self.alpha = 0.5  # 第一个分量的权重

        self.log_likelihoods = []  # 记录对数似然值

    def e_step(self):
        # 计算每个样本属于第一个分量和第二个分量的概率（后验概率）
        p1 = self.alpha * (1 / (np.sqrt(2 * np.pi) * self.sigma1)) * np.exp(- (self.x - self.mu1) ** 2 / (2 * self.sigma1 ** 2))
        p2 = (1 - self.alpha) * (1 / (np.sqrt(2 * np.pi) * self.sigma2)) * np.exp(- (self.x - self.mu2) ** 2 / (2 * self.sigma2 ** 2))

        w1 = p1 / (p1 + p2)
        w2 = p2 / (p1 + p2)

        return w1, w2

    def m_step(self, w1, w2):
        # 更新参数

        self.mu1 = np.sum(w1 * self.x) / np.sum(w1)
        self.sigma1 = np.sqrt(np.sum(w1 * (self.x - self.mu1) ** 2) / np.sum(w1))

        self.mu2 = np.sum(w2 * self.x) / np.sum(w2)
        self.sigma2 = np.sqrt(np.sum(w2 * (self.x - self.mu2) ** 2) / np.sum(w2))

        self.alpha = np.mean(w1)

    def log_likelihood(self):
        # 计算对数似然值
        p1 = self.alpha * (1 / (np.sqrt(2 * np.pi) * self.sigma1)) * np.exp(- (self.x - self.mu1) ** 2 / (2 * self.sigma1 ** 2))
        p2 = (1 - self.alpha) * (1 / (np.sqrt(2 * np.pi) * self.sigma2)) * np.exp(- (self.x - self.mu2) ** 2 / (2 * self.sigma2 ** 2))

        return np.sum(np.log(p1 + p2))

    def fit(self, max_iter=100, tol=1e-4):
        # 迭代优化参数
        for i in range(max_iter):
            w1, w2 = self.e_step()  # E步
            self.m_step(w1, w2)  # M步
            ll = self.log_likelihood()  # 计算对数似然值
            self.log_likelihoods.append(ll)  # 记录对数似然值

            if i > 0 and np.abs(ll - self.log_likelihoods[-2]) < tol:  # 判断是否收敛
                break

        return self.mu1, self.sigma1, self.mu2, self.sigma2, self.alpha

    def plot(self):
        # 可视化数据直方图和拟合曲线

        plt.hist(self.x, bins=20, density=True)  # 画出数据直方图

        x_axis = np.linspace(np.min(self.x), np.max(self.x), 1000)  # 定义x轴范围

        y_axis1 = self.alpha * (1 / (np.sqrt(2 * np.pi) * self.sigma1)) * np.exp(- (x_axis - self.mu1) ** 2 / (2 * self.sigma1 ** 2))  # 计算第一个分量的概率密度函数
        y_axis2 = (1 - self.alpha) * (1 / (np.sqrt(2 * np.pi) * self.sigma2)) * np.exp(- (x_axis - self.mu2) ** 2 / (2 * self.sigma2 ** 2))  # 计算第二个分量的概率密度函数
        y_axis = y_axis1 + y_axis2  # 计算混合高斯分布的概率密度函数

        plt.plot(x_axis, y_axis, color='red')  # 画出拟合曲线

        plt.xlabel('Height')
        plt.ylabel('Density')

        plt.show()

    def plot_log_likelihood(self):
        # 可视化对数似然值和迭代次数的关系

        plt.plot(range(len(self.log_likelihoods)), self.log_likelihoods)

        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood')

        plt.show()


# 创建EM算法对象
em = EM(x)

# 优化参数
mu1, sigma1, mu2, sigma2, alpha = em.fit()

# 打印参数
print('mu1 =', mu1)
print('sigma1 =', sigma1)
print('mu2 =', mu2)
print('sigma2 =', sigma2)
print('alpha =', alpha)

# 画出数据直方图和拟合曲线
em.plot()

# 画出对数似然值和迭代次数的关系
em.plot_log_likelihood()
