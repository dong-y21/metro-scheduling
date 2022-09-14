import matplotlib.pyplot as plt
import numpy as np

def visual_pareto_2d(pareto_real, pareto_appr):
    plt.title('Pareto Front')
    plt.xlabel('$f_1(x)$')
    plt.ylabel('$f_2(x)$')
    #ax1.set_xlim(0, 1)
    #ax1.set_ylim(0, 1.2)
    plt.scatter(pareto_real[:, 0], pareto_real[:, 1], c='b', marker='.', linewidths=0.5, label='Real')  # 绘制散点,其他参数可以设置
    plt.scatter(pareto_appr[:, 0], pareto_appr[:, 1], c='r', marker='x', linewidths=1, label='NSGA-II')     #绘制散点,其他参数可以设置
    plt.legend(loc='upper right')
    plt.show()


def visual_vals(type, data):
    x = np.linspace(1, len(data), len(data))
    plt.plot(x, data)
    plt.title(type)
    plt.xlabel('Generations')
    plt.ylabel('Vals')
    plt.show()


def get_plt(type, data):
    x = np.linspace(1, len(data), len(data))
    plt.plot(x, data)
    plt.title(type)
    plt.xlabel('Generations')
    plt.ylabel('Values')
    return plt