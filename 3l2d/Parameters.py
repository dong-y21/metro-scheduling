from math import floor
from cv2 import UMatData_USER_ALLOCATED
import numpy as np
from numpy.random import rand, seed
import random
from multiprocessing import Pool
import os, time


'''模型函数'''


# 功能函数,计算总站数
def calculate_all_stations(num_lines, num_stations):
    num_all_stations = 0
    for i in range(num_lines):
        num_all_stations += num_stations[i]
    return num_all_stations


# 辅助函数,计算阶乘
def jiecheng(i):
    ans = 1
    if i <= 1:
        return 1
    else:
        for j in range(2, i + 1):
            ans *= j
    return ans


# 辅助函数,计算各到达人数的概率
def initial_probability(OD_lambda):
    probability_lambda = []
    sum_p = 0
    for i in range(10):
        p = (OD_lambda ** i) / (jiecheng(i)) * (np.e ** -OD_lambda)
        sum_p += p
        probability_lambda.append(sum_p)
    return probability_lambda


# 功能函数,初始化运行时间矩阵
def initial_travelMatrix(num_lines, num_stations, bounds):
    travel_Matrix = []
    random.seed(8)
    for i in range(num_lines):
        t_line = [0]
        for j in range(1, num_stations[i]):
            t = random.randint(bounds[i][0], bounds[i][1])
            t_line.append(t)
        travel_Matrix.append(t_line)
    for i in range(num_lines):
        t_line = travel_Matrix[i][1:] + [0]
        travel_Matrix.append(t_line)
    return travel_Matrix


# 功能函数,初始化停站时间矩阵
def initial_dwellMatrix(num_lines, num_stations):
    dwell_Matrix = []
    for i in range(num_lines):
        dwell_line = []
        for j in range(num_stations[i]):
            d = 2
            dwell_line.append(d)
        dwell_Matrix.append(dwell_line)
    for i in range(num_lines):
        dwell_line = dwell_Matrix[i].copy()
        dwell_Matrix.append(dwell_line)
    return dwell_Matrix


# 辅助函数,生成某个时间的到达人数矩阵
def calculate_ODMatrix(num_all_stations, OD_lambda, time):
    seed(time)
    ODMatrix = np.zeros((num_all_stations, num_all_stations))  # OD矩阵,采用泊松分布随机生成到站人数
    probability_lambda = initial_probability(OD_lambda)
    max_person = len(probability_lambda)
    for i in range(num_all_stations):
        for j in range(num_all_stations):
            r = rand()
            ODMatrix[i][j] = max_person
            for k in range(len(probability_lambda)):
                if r <= probability_lambda[k]:
                    ODMatrix[i][j] = k
                    break
    return ODMatrix


# 功能函数,计算整个仿真跨度内的所有到达人数矩阵,设置了种子保持不变性
def calculate_all_ODMatrix(num_all_stations, OD_lambda, time_span):
    all_ODMatrix = []
    for t in range(time_span):
        OD_Matrix = calculate_ODMatrix(num_all_stations, OD_lambda, t)
        all_ODMatrix.append(OD_Matrix)
    all_ODMatrix = np.array(all_ODMatrix, dtype=int)
    return all_ODMatrix


# 辅助函数,计算从始发站到各个站的时间
def calculate_begin_time(travel_Matrix, dwell_Matrix):
    time_from_start = []
    for i in range(num_lines):
        line_starts = [0]
        start_time = 0
        for j in range(1, len(travel_Matrix[i])):
            start_time += dwell_Matrix[i][j - 1]
            start_time += travel_Matrix[i][j]
            line_starts.append(start_time)
        time_from_start.append(line_starts)
    for i in range(num_lines, 2 * num_lines):
        line_starts = [0]
        start_time = 0
        for j in reversed(range(len(travel_Matrix[i]) - 1)):
            start_time += dwell_Matrix[i][j + 1]
            start_time += travel_Matrix[i][j]
            line_starts.insert(0, start_time)
        time_from_start.append(line_starts)
    return time_from_start


# 计算分界节点位置
def calculate_boundary_pos(num_services):
    num_lines = len(num_services)
    start_pos = [0]
    end_pos = []
    end = 0
    for l in range(num_lines):
        end += num_services[l]
        end_pos.append(end)
    for l in range(num_lines):
        end += num_services[l]
        end_pos.append(end)
    start_pos = start_pos + end_pos[:-1]
    return start_pos, end_pos


# 辅助函数,计算每条线的站点偏移量
def calculate_start_station_index(num_stations):
    num_lines = len(num_stations)
    start_station_index = []
    for line in range(num_lines):
        start_station_index.append(sum(num_stations[:line]))
    return start_station_index


# 计算站台属于哪条线
def station_in_line(s, num_stations):
    line = 0
    while s >= num_stations[line]:
        s -= num_stations[line]
        line += 1
    return line


# 辅助函数,计算每个站能到达的站（建议直接手动给出矩阵）
def calculate_reachable_stations(num_stations, num_all_stations, transfer_stations):
    reachable_stations = []
    for s in range(num_all_stations):
        stations = [[], []]
        if s < num_stations[0]:     # 在第一条线上
            j = s + 1
            while j < num_stations[0]:
                stations[0].append(j)
                j += 1
            if s < transfer_stations[0][1]:     # 能换乘到第二条线
                j = start_station_index[1]
                while j < start_station_index[2]:
                    stations[0].append(j)
                    j += 1
            if s < transfer_stations[0][2]:     # 能换乘到第三条线
                j = start_station_index[2]
                while j < num_all_stations:
                    stations[0].append(j)
                    j += 1
        elif s < start_station_index[2]:    # 在第二条线上
            if s < transfer_stations[1][0]:     # 能换乘到第一条线
                j = 0
                while j < num_stations[0]:
                    stations[0].append(j)
                    j += 1
            j = s + 1
            while j < start_station_index[2]:
                stations[0].append(j)
                j += 1
            if s < transfer_stations[1][2]:     # 能换乘到第三条线
                j = start_station_index[2]
                while j < num_all_stations:
                    stations[0].append(j)
                    j += 1
        else:   # 在第三条线上
            if s < transfer_stations[2][0]:     # 能换乘到第一条线
                j = 0
                while j < num_stations[0]:
                    stations[0].append(j)
                    j += 1
            j = s + 1
            if s < transfer_stations[2][1]:     # 能换乘到第二条线
                j = start_station_index[1]
                while j < start_station_index[2]:
                    stations[0].append(j)
                    j += 1
            j = s + 1
            while j < num_all_stations:
                stations[0].append(j)
                j += 1
        reachable_stations.append(stations)
    for s in reversed(range(num_all_stations)):
        if s >= start_station_index[2]:     # 在第三条线上
            j = s - 1
            while j >= start_station_index[2]:
                reachable_stations[s][1].insert(0, j)
                j -= 1
            if s > transfer_stations[2][1]:     # 能换乘到第二条线
                j = start_station_index[2] - 1
                while j >= start_station_index[1]:
                    reachable_stations[s][1].insert(0, j)
                    j -= 1
            if s > transfer_stations[2][0]:     # 能换乘到第一条线
                j = start_station_index[1] - 1
                while j >= start_station_index[0]:
                    reachable_stations[s][1].insert(0, j)
                    j -= 1
        elif s >= start_station_index[1]:   # 在第二条线上
            if s > transfer_stations[1][2]:     # 能换乘到第三条线
                j = num_all_stations - 1
                while j >= start_station_index[2]:
                    reachable_stations[s][1].insert(0, j)
                    j -= 1
            j = s - 1
            while j >= start_station_index[1]:
                reachable_stations[s][1].insert(0, j)
                j -= 1
            if s > transfer_stations[1][0]:     # 能换乘到第一条线
                j = start_station_index[1] - 1
                while j >= start_station_index[0]:
                    reachable_stations[s][1].insert(0, j)
                    j -= 1  
        else:   # 在第一条线上
            if s > transfer_stations[0][2]:     # 能换乘到第三条线
                j = num_all_stations - 1
                while j >= start_station_index[2]:
                    reachable_stations[s][1].insert(0, j)
                    j -= 1
            j = s - 1
            if s > transfer_stations[0][1]:     # 能换乘到第二条线
                j = start_station_index[2] - 1
                while j >= start_station_index[1]:
                    reachable_stations[s][1].insert(0, j)
                    j -= 1
            j = s - 1
            while j >= start_station_index[0]:
                reachable_stations[s][1].insert(0, j)
                j -= 1
    return reachable_stations


# 初始化换乘矩阵（建议直接手动设置矩阵）
def initial_transfer_Matrix(num_stations, num_all_stations,  transfer_stations):
    transfer_Matrix = np.zeros((num_all_stations, num_all_stations), dtype=int) - 1
    for s in range(start_station_index[1]):     # 在第一条线上
        if s in transfer_stations[0]:
            continue
        for e in range(start_station_index[1], start_station_index[2]):     # 到第二条线
            if e == post_transfer_stations[0][1]:
                continue
            transfer_Matrix[s][e] = post_transfer_stations[0][1]
        for e in range(start_station_index[2], num_all_stations):     # 到第三条线
            if e == post_transfer_stations[0][2]:
                continue
            transfer_Matrix[s][e] = post_transfer_stations[0][2]
    for s in range(start_station_index[1], start_station_index[2]):     # 在第二条线上
        if s in transfer_stations[1]:
            continue
        for e in range(start_station_index[0], start_station_index[1]):     # 到第一条线
            if e == post_transfer_stations[1][0]:
                continue
            transfer_Matrix[s][e] = post_transfer_stations[1][0]
        for e in range(start_station_index[2], num_all_stations):     # 到第三条线
            if e == post_transfer_stations[1][2]:
                continue
            transfer_Matrix[s][e] = post_transfer_stations[1][2]
    for s in range(start_station_index[2], num_all_stations):     # 在第三条线上
        if s in transfer_stations[2]:
            continue
        for e in range(start_station_index[0], start_station_index[1]):     # 到第一条线
            if e == post_transfer_stations[2][0]:
                continue
            transfer_Matrix[s][e] = post_transfer_stations[2][0]
        for e in range(start_station_index[1], start_station_index[2]):     # 到第二条线
            if e == post_transfer_stations[2][1]:
                continue
            transfer_Matrix[s][e] = post_transfer_stations[2][1]
    return transfer_Matrix


# 计算距离换乘站的时间
def calculate_time_to_transfer(travel_Matrix, dwell_Matrix, transfer_stations, transfer_time):
    time_to_transfer = []
    line_index = []
    start_line = 0
    for s in range(start_station_index[1]):     # 第一条线上
        station_index = [-1]
        target_line = 1     # 换乘到第二条线
        t = transfer_time[start_line][target_line]
        j = s + 1
        while j < transfer_stations[start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[start_line]]
            t += dwell_Matrix[start_line][j - start_station_index[start_line]]
            j += 1
        if j == transfer_stations[start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[start_line]]
        if t > transfer_time[start_line][target_line]:
            station_index.append(t)
        else:
            station_index.append(-1)
        target_line = 2     # 换乘到第三条线
        t = transfer_time[start_line][target_line]
        j = s + 1
        while j < transfer_stations[start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[start_line]]
            t += dwell_Matrix[start_line][j - start_station_index[start_line]]
            j += 1
        if j == transfer_stations[start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[start_line]]
        if t > transfer_time[start_line][target_line]:
            station_index.append(t)
        else:
            station_index.append(-1)

        line_index.append(station_index)
    time_to_transfer.append(line_index)

    line_index = []
    start_line = 1
    for s in range(start_station_index[1], start_station_index[2]):     # 第二条线上
        station_index = []
        target_line = 0     # 换乘到第一条线
        t = transfer_time[start_line][target_line]
        j = s + 1
        while j < transfer_stations[start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[start_line]]
            t += dwell_Matrix[start_line][j - start_station_index[start_line]]
            j += 1
        if j == transfer_stations[start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[start_line]]
        if t > transfer_time[start_line][target_line]:
            station_index.append(t)
        else:
            station_index.append(-1)
        station_index.append(-1)
        target_line = 2     # 换乘到第三条线
        t = transfer_time[start_line][target_line]
        j = s + 1
        while j < transfer_stations[start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[start_line]]
            t += dwell_Matrix[start_line][j - start_station_index[start_line]]
            j += 1
        if j == transfer_stations[start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[start_line]]
        if t > transfer_time[start_line][target_line]:
            station_index.append(t)
        else:
            station_index.append(-1)

        line_index.append(station_index)
    time_to_transfer.append(line_index)

    line_index = []
    start_line = 2
    for s in range(start_station_index[2], num_all_stations):     # 第三条线上
        station_index = []
        target_line = 0     # 换乘到第一条线
        t = transfer_time[start_line][target_line]
        j = s + 1
        while j < transfer_stations[start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[start_line]]
            t += dwell_Matrix[start_line][j - start_station_index[start_line]]
            j += 1
        if j == transfer_stations[start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[start_line]]
        if t > transfer_time[start_line][target_line]:
            station_index.append(t)
        else:
            station_index.append(-1)
        target_line = 1     # 换乘到第二条线
        t = transfer_time[start_line][target_line]
        j = s + 1
        while j < transfer_stations[start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[start_line]]
            t += dwell_Matrix[start_line][j - start_station_index[start_line]]
            j += 1
        if j == transfer_stations[start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[start_line]]
        if t > transfer_time[start_line][target_line]:
            station_index.append(t)
        else:
            station_index.append(-1)
        station_index.append(-1)
        line_index.append(station_index)
    time_to_transfer.append(line_index)
    
    line_index = []     # 反方向
    actual_start_line = 0
    start_line = actual_start_line + num_lines
    for s in range(start_station_index[1]):     # 第一条线上
        station_index = [-1]
        target_line = 1     # 换乘到第二条线
        t = transfer_time[actual_start_line][target_line]
        j = s - 1
        while j > transfer_stations[actual_start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[actual_start_line]]
            t += dwell_Matrix[start_line][j - start_station_index[actual_start_line]]
            j -= 1
        if j == transfer_stations[actual_start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[actual_start_line]]
        if t > transfer_time[actual_start_line][target_line]:
            station_index.append(t)
        else:
            station_index.append(-1)
        target_line = 2     # 换乘到第三条线
        t = transfer_time[actual_start_line][target_line]
        j = s - 1
        while j > transfer_stations[actual_start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[actual_start_line]]
            t += dwell_Matrix[start_line][j - start_station_index[actual_start_line]]
            j -= 1
        if j == transfer_stations[actual_start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[actual_start_line]]
        if t > transfer_time[actual_start_line][target_line]:
            station_index.append(t)
        else:
            station_index.append(-1)

        line_index.append(station_index)
    time_to_transfer.append(line_index)

    line_index = []
    actual_start_line = 1
    start_line = actual_start_line + num_lines
    for s in range(start_station_index[1], start_station_index[2]):     # 第二条线上
        station_index = []
        target_line = 0     # 换乘到第一条线
        t = transfer_time[actual_start_line][target_line]
        j = s - 1
        while j > transfer_stations[actual_start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[actual_start_line]]
            t += dwell_Matrix[start_line][j - start_station_index[actual_start_line]]
            j -= 1
        if j == transfer_stations[actual_start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[actual_start_line]]
        if t > transfer_time[actual_start_line][target_line]:
            station_index.append(t)
        else:
            station_index.append(-1)
        station_index.append(-1)
        target_line = 2     # 换乘到第三条线
        t = transfer_time[actual_start_line][target_line]
        j = s - 1
        while j > transfer_stations[actual_start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[actual_start_line]]
            t += dwell_Matrix[start_line][j - start_station_index[actual_start_line]]
            j -= 1
        if j == transfer_stations[actual_start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[actual_start_line]]
        if t > transfer_time[actual_start_line][target_line]:
            station_index.append(t)
        else:
            station_index.append(-1)

        line_index.append(station_index)
    time_to_transfer.append(line_index)

    line_index = []
    actual_start_line = 2
    start_line = actual_start_line + num_lines
    for s in range(start_station_index[2], num_all_stations):     # 第三条线上
        station_index = []
        target_line = 0     # 换乘到第一条线
        t = transfer_time[actual_start_line][target_line]
        j = s - 1
        while j > transfer_stations[actual_start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[actual_start_line]]
            t += dwell_Matrix[start_line][j - start_station_index[actual_start_line]]
            j -= 1
        if j == transfer_stations[actual_start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[actual_start_line]]
        if t > transfer_time[actual_start_line][target_line]:
            station_index.append(t)
        else:
            station_index.append(-1)
        target_line = 1     # 换乘到第二条线
        t = transfer_time[actual_start_line][target_line]
        j = s - 1
        while j > transfer_stations[actual_start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[actual_start_line]]
            t += dwell_Matrix[start_line][j - start_station_index[actual_start_line]]
            j -= 1
        if j == transfer_stations[actual_start_line][target_line]:
            t += travel_Matrix[start_line][j - start_station_index[actual_start_line]]
        if t > transfer_time[actual_start_line][target_line]:
            station_index.append(t)
        else:
            station_index.append(-1)
        station_index.append(-1)
        line_index.append(station_index)
    time_to_transfer.append(line_index)
    return time_to_transfer


# 更新策略计算一个编码对应的总等待时间
def calculate_pwt(p, num_services, num_stations, all_ODMatrix, transfer_Matrix,
                    time_from_start, reachable_stations, time_to_transfer, time_span, 
                    train_capacity, start_station_index, val_pos):
    pwt = 0
    OD_Matrix = all_ODMatrix.copy()
    train_arrive_station = []   # 最近到达的站序号
    station_begin_time = [] # 每个站还有未上车乘客的起始时间
    station_end_time = []   # 每个站最后一班车驶离的时间
    event_trains = []   # 各时刻有事件发生的列车号
    longest_service_time = 0    # 从始发站到最远站的时间
    passengers_onboard = []  # 每辆车上当前人数
    passengers_destination = [] # 车上乘客去往各目的站的人数
    for line, services in enumerate(num_services):
        train_arrive_station.append([-1] * services)
        station_begin_time.append(time_from_start[line].copy())
        station_end_time.append([time_from_start[line][i] + dwell_Matrix[line][i] + time_span for i in range(num_stations[line])])
        longest_service_time = time_from_start[line][-1] if longest_service_time < time_from_start[line][-1] else longest_service_time
        passengers_onboard.append([0] * services)
        passengers_destination.append([[0] * num_all_stations for _ in range(services)])
    for line, services in enumerate(num_services):
        train_arrive_station.append([num_stations[line]] * services)
        line += num_lines
        station_begin_time.append(time_from_start[line].copy())
        station_end_time.append([time_from_start[line][i] + dwell_Matrix[line][i] + time_span for i in range(num_stations[line % num_lines])])
        longest_service_time = time_from_start[line][0] if longest_service_time < time_from_start[line][0] else longest_service_time
        passengers_onboard.append([0] * services)
        passengers_destination.append([[0] * num_all_stations for _ in range(services)])
    service_span = time_span + longest_service_time     # 总服务时长

    event_trains = [[[] for _ in range(num_lines * 2)] for _ in range(service_span)]    # 建立event_trains[time][line]的索引
    for line in range(num_lines):  # 每条线
        start_p_index = start_pos[line]    # 每条线的起始发车间隔h对应序号
        service_begin_time = dwell_Matrix[line][0]  # 每辆车的发车时间
        for service in range(num_services[line]):  # 每趟车
            service_begin_time += p[start_p_index + service]    # 计算该车次的发车时刻
            event_trains[service_begin_time][line].append(service)  # 记录该时刻该线路发车的车次号
    for line in range(num_lines, num_lines * 2):  # 每条线
        start_p_index = start_pos[line]    # 每条线的起始发车间隔h对应序号
        service_begin_time = dwell_Matrix[line][-1]  # 每辆车的发车时间
        for service in range(num_services[line % num_lines]):  # 每趟车
            service_begin_time += p[start_p_index + service]    # 计算该车次的发车时刻
            event_trains[service_begin_time][line].append(service)  # 记录该时刻该线路发车的车次号
    for t in range(service_span):
        for line in range(num_lines):
            if len(event_trains[t][line]) == 0:
                continue
            else:
                service_list = event_trains[t][line]
                for service in service_list:
                    train_arrive_station[line][service] += 1    # 更新列车到达站的线上编号
                    arrive_station = train_arrive_station[line][service]    # 列车到达站的线上编号
                    begin_t = station_begin_time[line][arrive_station]  # 计算上车乘客的起始时间
                    index_in_all = start_station_index[line] + arrive_station    # 到达站的总编号
                    consider_stations = reachable_stations[index_in_all][0]    # 可达站的总编号集合
                    passengers_available = train_capacity[line][0] - passengers_onboard[line][service] \
                                                + passengers_destination[line][service][index_in_all] # 可以上车的人数
                    passengers_onboard[line][service] -= passengers_destination[line][service][index_in_all]    # 更新车上人数
                    for consider_t in range(begin_t, t):
                        passengers_waiting = 0
                        for dest in consider_stations:
                            passengers_waiting += OD_Matrix[consider_t][index_in_all][dest] # 计算该站当前时刻总等待人数
                        if passengers_waiting <= passengers_available:  # 总等待人数小于可上车人数
                            passengers_available -= passengers_waiting  # 更新可上车人数
                            passengers_onboard[line][service] += passengers_waiting # 更新车上人数
                            for dest in consider_stations:
                                pwt += OD_Matrix[consider_t][index_in_all][dest] * (t - consider_t - 0.5)   # 更新总等待时间
                                if transfer_Matrix[index_in_all][dest] > 0:    # 如果需要换乘
                                    transfer_station = transfer_Matrix[index_in_all][dest]  # 换乘后到达的站总编号
                                    target_line = station_in_line(transfer_station, num_stations)   # 换乘后到达线的编号
                                    arrive_time = t + time_to_transfer[line][arrive_station][target_line]    # 到达换乘站的时刻
                                    # 更新换乘站的OD矩阵
                                    OD_Matrix[arrive_time][transfer_station][dest] += OD_Matrix[consider_t][index_in_all][dest]
                                    # 更新下车矩阵，在当前线的换乘站下车
                                    passengers_destination[line][service][transfer_stations[line][target_line]] += OD_Matrix[consider_t][index_in_all][dest]
                                else:
                                    # 更新下车矩阵，在当前线的换乘站下车
                                    if dest in transfer_stations and dest not in transfer_stations[line]:   # 目的地是另一条线的换乘站
                                        passengers_destination[line][service][transfer_stations[line][target_line]] += OD_Matrix[consider_t][index_in_all][dest]
                                    else:   # 更新下车矩阵，在当前线的目标站下车
                                        passengers_destination[line][service][dest] += OD_Matrix[consider_t][index_in_all][dest]
                                OD_Matrix[consider_t][index_in_all][dest] = 0   # 更新候车人数
                            station_begin_time[line][arrive_station] = consider_t + 1   # 更新该站未上车乘客起始时间
                        else:   # 候车人数大于可上车人数
                            percent = float(passengers_available) / passengers_waiting     # 计算上车比例
                            error = 0   # 累积小数缺口
                            for dest in consider_stations:
                                num_onboard = OD_Matrix[consider_t][index_in_all][dest] * percent  # 按比例计算上车人数
                                error += num_onboard % 1    # 对小数部分进行累积
                                num_onboard = floor(num_onboard)    # 取整
                                if error > 0.999999999:     # 累积超过1则上车人数加1
                                    num_onboard += 1
                                    error -= 1
                                pwt += num_onboard * (t - consider_t - 0.5)   # 更新总等待时间
                                if transfer_Matrix[index_in_all][dest] > 0:    # 如果需要换乘
                                    transfer_station = transfer_Matrix[index_in_all][dest]  # 换乘后到达的站总编号
                                    target_line = station_in_line(transfer_station, num_stations)   # 换乘后到达线的编号
                                    arrive_time = t + time_to_transfer[line][arrive_station][target_line]    # 到达换乘站的时刻
                                    # 更新换乘站的OD矩阵
                                    OD_Matrix[arrive_time][transfer_station][dest] += num_onboard
                                    # 更新下车矩阵，在当前线的换乘站下车
                                    passengers_destination[line][service][transfer_stations[line][target_line]] += num_onboard
                                else:
                                    # 更新下车矩阵，在当前线的换乘站下车
                                    if dest in transfer_stations and dest not in transfer_stations[line]:   # 目的地是另一条线的换乘站
                                        passengers_destination[line][service][transfer_stations[line][target_line]] += num_onboard
                                    else:   # 更新下车矩阵，在当前线的目标站下车
                                        passengers_destination[line][service][dest] += num_onboard
                                OD_Matrix[consider_t][index_in_all][dest] -= num_onboard   # 更新候车人数
                            station_begin_time[line][arrive_station] = consider_t   # 更新计算上车乘客的起始时间
                            break
                    if arrive_station < num_stations[line] - 2:
                        next_station = arrive_station + 1
                        next_event_time = t + travel_Matrix[line][next_station] + dwell_Matrix[line][next_station]
                        event_trains[next_event_time][line].append(service)
        for line in range(num_lines, num_lines * 2):
            if len(event_trains[t][line]) == 0:
                continue
            else:
                service_list = event_trains[t][line]
                for service in service_list:
                    train_arrive_station[line][service] -= 1    # 更新列车到达站的线上编号
                    arrive_station = train_arrive_station[line][service]    # 列车到达站的线上编号
                    begin_t = station_begin_time[line][arrive_station]  # 计算上车乘客的起始时间
                    index_in_all = start_station_index[line % num_lines] + arrive_station    # 到达站的总编号
                    consider_stations = reachable_stations[index_in_all][1]    # 可达站的总编号集合
                    passengers_available = train_capacity[line % num_lines][1] - passengers_onboard[line][service] \
                                                + passengers_destination[line][service][index_in_all] # 可以上车的人数
                    passengers_onboard[line][service] -= passengers_destination[line][service][index_in_all]    # 更新车上人数
                    for consider_t in range(begin_t, t):
                        passengers_waiting = 0
                        for dest in consider_stations:
                            passengers_waiting += OD_Matrix[consider_t][index_in_all][dest] # 计算该站当前时刻总等待人数
                        if passengers_waiting <= passengers_available:  # 总等待人数小于可上车人数
                            passengers_available -= passengers_waiting  # 更新可上车人数
                            passengers_onboard[line][service] += passengers_waiting # 更新车上人数
                            for dest in consider_stations:
                                pwt += OD_Matrix[consider_t][index_in_all][dest] * (t - consider_t - 0.5)   # 更新总等待时间
                                if transfer_Matrix[index_in_all][dest] > 0:    # 如果需要换乘
                                    transfer_station = transfer_Matrix[index_in_all][dest]  # 换乘后到达的站总编号
                                    target_line = station_in_line(transfer_station, num_stations)   # 换乘后到达线的编号
                                    arrive_time = t + time_to_transfer[line][arrive_station][target_line]    # 到达换乘站的时刻
                                    # 更新换乘站的OD矩阵
                                    OD_Matrix[arrive_time][transfer_station][dest] += OD_Matrix[consider_t][index_in_all][dest]
                                    # 更新下车矩阵，在当前线的换乘站下车
                                    passengers_destination[line][service][transfer_stations[line % num_lines][target_line]] += OD_Matrix[consider_t][index_in_all][dest]
                                else:
                                    # 更新下车矩阵，在当前线的换乘站下车
                                    if dest in transfer_stations and dest not in transfer_stations[line % num_lines]:   # 目的地是另一条线的换乘站
                                        passengers_destination[line][service][transfer_stations[line % num_lines][target_line]] += OD_Matrix[consider_t][index_in_all][dest]
                                    else:   # 更新下车矩阵，在当前线的目标站下车
                                        passengers_destination[line][service][dest] += OD_Matrix[consider_t][index_in_all][dest]
                                OD_Matrix[consider_t][index_in_all][dest] = 0   # 更新候车人数
                            station_begin_time[line][arrive_station] = consider_t + 1   # 更新该站未上车乘客起始时间
                        else:   # 候车人数大于可上车人数
                            percent = float(passengers_available) / passengers_waiting     # 计算上车比例
                            error = 0   # 累积小数缺口
                            for dest in consider_stations:
                                num_onboard = OD_Matrix[consider_t][index_in_all][dest] * percent  # 按比例计算上车人数
                                error += num_onboard % 1    # 对小数部分进行累积
                                num_onboard = floor(num_onboard)    # 取整
                                if error > 0.999999999:     # 累积超过1则上车人数加1
                                    num_onboard += 1
                                    error -= 1
                                pwt += num_onboard * (t - consider_t - 0.5)   # 更新总等待时间
                                if transfer_Matrix[index_in_all][dest] > 0:    # 如果需要换乘
                                    transfer_station = transfer_Matrix[index_in_all][dest]  # 换乘后到达的站总编号
                                    target_line = station_in_line(transfer_station, num_stations)   # 换乘后到达线的编号
                                    arrive_time = t + time_to_transfer[line][arrive_station][target_line]    # 到达换乘站的时刻
                                    # 更新换乘站的OD矩阵
                                    OD_Matrix[arrive_time][transfer_station][dest] += num_onboard
                                    # 更新下车矩阵，在当前线的换乘站下车
                                    passengers_destination[line][service][transfer_stations[line % num_lines][target_line]] += num_onboard
                                else:
                                    # 更新下车矩阵，在当前线的换乘站下车
                                    if dest in transfer_stations and dest not in transfer_stations[line % num_lines]:   # 目的地是另一条线的换乘站
                                        passengers_destination[line][service][transfer_stations[line % num_lines][target_line]] += num_onboard
                                    else:   # 更新下车矩阵，在当前线的目标站下车
                                        passengers_destination[line][service][dest] += num_onboard
                                OD_Matrix[consider_t][index_in_all][dest] -= num_onboard   # 更新候车人数
                            station_begin_time[line][arrive_station] = consider_t   # 更新计算上车乘客的起始时间
                            break
                    if arrive_station > 1:
                        next_station = arrive_station - 1
                        next_event_time = t + travel_Matrix[line][next_station] + dwell_Matrix[line][next_station]
                        event_trains[next_event_time][line].append(service)
    for line in range(num_lines):   # 计算所有未上车乘客的等待时间
        for station in range(num_stations[line]):
            index_in_all = start_station_index[line] + station    # 站的总编号
            begin_t = station_begin_time[line][station]
            consider_stations = reachable_stations[index_in_all][0]    # 可达站的总编号集合
            end_t = station_end_time[line][station]
            for consider_t in range(begin_t, end_t):
                for dest in consider_stations:
                    pwt += OD_Matrix[consider_t][index_in_all][dest] * (end_t - consider_t - 0.5)   # 更新总等待时间
    for line in range(num_lines):   # 计算所有未上车乘客的等待时间
        for station in range(num_stations[line]):
            index_in_all = start_station_index[line] + station    # 站的总编号
            begin_t = station_begin_time[line][station]
            consider_stations = reachable_stations[index_in_all][1]    # 可达站的总编号集合
            end_t = station_end_time[line + num_lines][station]
            for consider_t in range(begin_t, end_t):
                for dest in consider_stations:
                    pwt += OD_Matrix[consider_t][index_in_all][dest] * (end_t - consider_t - 0.5)   # 更新总等待时间
    return (val_pos, pwt)


# 功能函数,计算种群中所有个体的等待时间目标函数
def calculate_PWTs(populations, num_services, num_stations, transfer_Matrix, all_ODMatrix, time_from_start, 
                        reachable_stations, time_to_transfer, time_span, train_capacity, start_station_index):
    vals = []
    for p in populations:
        val = calculate_pwt(p, num_services, num_stations, all_ODMatrix, transfer_Matrix, time_from_start, 
                                reachable_stations, time_to_transfer, time_span, train_capacity, start_station_index)
        vals.append(val)
    return vals


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_config(path):
    with open(os.path.join(path, 'config.txt'),'w') as f:
        msg = 'Model Configs' + '\n'
        msg += 'unit: {}'.format(unit) + '\n'
        msg += 'num_lines: {}'.format(num_lines) + '\n'
        msg += 'num_stations: {}'.format(num_stations) + '\n'
        msg += 'transfer_stations: {}'.format(transfer_stations) + '\n'
        msg += 'transfer_time: {}'.format(transfer_time) + '\n'
        msg += 'bounds_travel:\n{}'.format(bounds_travel) + '\n'
        msg += 'bounds_headway:\n{}'.format(bounds_headway) + '\n'
        msg += 'num_services: {}'.format(num_services) + '\n'
        msg += 'train_capacity: {}'.format(train_capacity) + '\n'
        msg += 'time_span: {}'.format(time_span) + '\n'
        msg += 'OD_lambda: {}'.format(OD_lambda) + '\n'

        msg += '\nAlgorithm Configs' + '\n'
        msg += 'num_populations: {}'.format(num_populations) + '\n'
        msg += 'num_generations: {}'.format(num_generations) + '\n'
        msg += 'cross_epsilon_begin: {}'.format(cross_epsilon_begin) + '\n'
        msg += 'cross_epsilon_end: {}'.format(cross_epsilon_end) + '\n'
        msg += 'mutation_epsilon_begin: {}'.format(mutation_epsilon_begin) + '\n'
        msg += 'mutation_epsilon_end: {}'.format(mutation_epsilon_end) + '\n'
        f.write(msg)
    f.close()


'''模型参数设置'''
unit = 10   # 最小单元10s
num_lines = 3   # 线路数
num_stations = [5, 8, 5]   # 每条线的站点数
transfer_stations = np.array([[-1, 2, 2], 
                              [7, -1, 10], 
                              [15, 15, -1]], dtype=int)  # 各线路上的换乘站序号transfer_stations[line_A][line_B] = index of all

post_transfer_stations = np.array([[-1, 7, 7], 
                              [2, -1, 15], 
                              [10, 10, -1]], dtype=int)  # 各线路上的换乘后站序号transfer_stations[line_A][line_B] = index of all

transfer_time = np.array([[-1, 9, 9], 
                          [9, -1, 8], 
                          [8, 8, -1]], dtype=int)   # 换乘步行时间transfer_time[line_A][line_B] = t
                            
bounds_travel = np.array([[15, 24],
                          [15, 24], 
                          [15, 24]], dtype=int)    # 各线路运行时间上下界bounds_travel[line] = [low, high]

bounds_headway = np.array([[9, 18],
                           [9, 18],
                           [9, 18]], dtype=int)     # 各线路发车间隔上下界bounds_headway[line] = [low, high]

num_services = [26, 27, 26]     # 每条线路上的服务车次限制

OD_lambda = 5     # 泊松分布参数

time_span = 360     # 仿真时长,1小时

train_capacity = [[5000, 5000], [5000, 5000], [5000, 5000]]    # 列车容量c[line][direction]


'''算法参数设置'''
num_populations = 50     # 种群规模
num_generations = 800   # 迭代次数
cross_epsilon_begin = 0.2   # 初始交叉率
cross_epsilon_end = 0.8     # 终止交叉率
mutation_epsilon_begin = 0.9    # 初始变异率
mutation_epsilon_end = 0.4      # 终止变异率


'''生成模型'''
num_all_stations = calculate_all_stations(num_lines, num_stations)      # 计算总站数

start_station_index = calculate_start_station_index(num_stations)   # 计算每条线的站点偏移量

travel_Matrix = initial_travelMatrix(num_lines, num_stations, bounds_travel)    # 初始化运行时间矩阵

dwell_Matrix = initial_dwellMatrix(num_lines, num_stations)     # 初始化停站时间矩阵

all_ODMatrix = calculate_all_ODMatrix(num_all_stations, OD_lambda, int(time_span * 1.5))   # 初始化仿真时间段内的到达矩阵

time_from_start = calculate_begin_time(travel_Matrix, dwell_Matrix)     # 计算从始发站到各站的时间

reachable_stations = calculate_reachable_stations(num_stations, num_all_stations, transfer_stations)    # 计算每个站的可达站点

transfer_Matrix = initial_transfer_Matrix(num_stations, num_all_stations, transfer_stations)    # 计算换乘矩阵

time_to_transfer = calculate_time_to_transfer(travel_Matrix, dwell_Matrix, transfer_stations, transfer_time)   # 计算从离站到到达换乘站的时间

start_pos, end_pos = calculate_boundary_pos(num_services)   # 计算分界位置
'''
print('\ntransfer_Matrix:')
for i in range(len(transfer_Matrix)):
    print(transfer_Matrix[i])

print('\nreachable_stations:')
for i in range(len(reachable_stations)):
    print(reachable_stations[i])

print('\ntravel_Matrix:')
for i in range(len(travel_Matrix)):
    print(travel_Matrix[i])

print('\ntime_from_start:')
for i in range(len(time_from_start)):
    print(time_from_start[i])

print('\ntime_to_transfer:')
for i in range(len(time_to_transfer)):
    print(time_to_transfer[i])

print('\nstart_station_index:')
print(start_station_index)

print('\nstart/end pos:')
print(start_pos, end_pos)

print('\ndwell_Matrix:')
for i in range(len(dwell_Matrix)):
    print(dwell_Matrix[i])
'''