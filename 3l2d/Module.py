import numpy as np
from random import sample
from numpy.random import rand, seed, choice, randint
from math import ceil, floor
from Parameters import *


'''算法函数'''


# 初始化种群
def initial_population(num_populations, num_services, bounds_headway, time_span):
    seed(12)
    num_all_services = sum(num_services)
    num_lines = len(num_services)
    populations = np.zeros((num_populations, num_all_services * 2), dtype=int)
    for p in populations:
        start = 0
        end = 0
        for l in range(num_lines):
            start = end
            end += num_services[l]
            h_l = np.array(p[start:end], dtype=float)
            h_l += time_span / num_services[l]
            error = 0
            remain = h_l[0] % 1
            lack = 1 - remain
            for i in range(len(h_l)):
                if error + remain > 0.999999999:
                    h_l[i] = ceil(h_l[i])
                    error -= lack
                else:
                    h_l[i] = floor(h_l[i])
                    error += remain
            h_l = np.array(h_l, dtype=int)
            for _ in range(4):
                h_l = mutation_single_point(h_l, bounds_headway[l])
            p[start:end] = h_l
        for l in range(num_lines):
            start = end
            end += num_services[l]
            p[start:end] = p[start - num_all_services: end - num_all_services].copy()
    seed()
    return populations


# N元锦标赛选择个体
def select_tournament(vals, N):
    num = len(vals)
    index = sample(range(num), N)    # 随机产生N个不重复编号
    sample_vals = [vals[i] for i in index]      # 提取编号对应的val
    min_val = np.where(sample_vals == min(sample_vals))[0]     # 得到最小val对应的所有位置
    sample_min = [index[i] for i in min_val]    # 提取所有最小val对应的index
    sample_index = choice(sample_min)   # 得到其中一个对应的index
    return sample_index    # 返回锦标赛选择的个体编号


# 按线交叉
def cross_by_line(p, q):
    for i in range(len(start_pos)):
        start = start_pos[i]
        end = end_pos[i]
        if rand() < 0.5:
            p[start: end], q[start: end] = q[start: end].copy(), p[start: end].copy()
    return p, q


# 个体变异
def mutation(p, mutation_epsilon):
    for i in range(len(start_pos)):
        if rand() >= mutation_epsilon:
            continue
        start = start_pos[i]
        end = end_pos[i]
        mutation_type = randint(8)
        if mutation_type < 5:
            p[start: end] = mutation_single_point(p[start: end].copy(), bounds_headway[i % num_lines])
        elif mutation_type < 6:
            p[start: end] = mutation_reverse(p[start: end].copy())
        elif mutation_type < 7:
            p[start: end] = mutation_shift(p[start: end].copy())
        else:
            p[start: end] = mutation_exchange(p[start: end].copy())
    return p


# 单点变异
def mutation_single_point(p, bound):
    num = len(p)
    index = choice(range(num))
    add = round(rand(1)[0])
    if add:
        while True:
            if p[index] < bound[1]:
                p[index] += 1
                next = index + 1
                while True:
                    if next == num:
                        next = 0
                    if p[next] > bound[0]:
                        p[next] -= 1
                        return p
                    next += 1
            else:
                index += 1
                if index == num:
                    index = 0
    else:
        while True:
            if p[index] > bound[0]:
                p[index] -= 1
                next = index + 1
                while True:
                    if next == num:
                        next = 0
                    if p[next] < bound[1]:
                        p[next] += 1
                        return p
                    next += 1
            else:
                index += 1
                if index == num:
                    index = 0


# 翻转变异
def mutation_reverse(p):
    num = len(p)
    position_1, position_2 = sample(range(num + 1), 2)
    if position_1 > position_2:
        position_1, position_2 = position_2, position_1
    p[position_1: position_2] = np.flipud(p[position_1: position_2])
    return p


# 平移变异
def mutation_shift(p):
    num = len(p)
    dist = choice(range(1, num))
    p = np.roll(p, dist)
    return p


# 交换变异
def mutation_exchange(p):
    num = len(p)
    position_1, position_2 = sample(range(num), 2)
    p[position_1], p[position_2] = p[position_2], p[position_1]
    return p


# 计算交叉率，变异率
def get_epsilon(gen):
    begin_generation = 0.1 * num_generations
    end_generation = 0.8 * num_generations
    if gen < begin_generation:
        return cross_epsilon_begin, mutation_epsilon_begin
    elif gen < end_generation:
        cross_epsilon = cross_epsilon_begin + (cross_epsilon_end - cross_epsilon_begin) * (gen - begin_generation) / (end_generation - begin_generation)
        mutation_epsilon = mutation_epsilon_begin + (mutation_epsilon_end - mutation_epsilon_begin) * (gen - begin_generation) / (end_generation - begin_generation)
        return cross_epsilon, mutation_epsilon
    else:
        return cross_epsilon_end, mutation_epsilon_end

initial_populations = initial_population(num_populations, num_services, bounds_headway, time_span)   # 初始化种群
