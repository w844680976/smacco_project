import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义多目标优化问题的目标函数
def objective_function(x):
    f1 = x**2
    f2 = (x - 2)**2
    f3 = (x - 1)**3
    return np.array([f1, f2, f3])

# 随机生成样本数据
np.random.seed(42)
X = np.random.uniform(0, 4, 200)
Y = np.array([objective_function(x) for x in X])

# 识别非支配解
def identify_pareto(Y):
    is_pareto = np.ones(Y.shape[0], dtype=bool)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[0]):
            if all(Y[j] <= Y[i]) and any(Y[j] < Y[i]):
                is_pareto[i] = 0
                break
    return is_pareto

pareto_mask = identify_pareto(Y)
pareto_front = Y[pareto_mask]

# 绘制所有样本点和帕累托前沿
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 所有解
ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], label='Solutions', alpha=0.5)

# 帕累托前沿
ax.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], color='r', label='Pareto Front', marker='o')

# 图形设置
ax.set_xlabel('Objective 1')
ax.set_ylabel('Objective 2')
ax.set_zlabel('Objective 3')
ax.legend()
plt.title('Pareto Front Example in 3D')

plt.show()