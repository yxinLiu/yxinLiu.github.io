import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import math
import random

matplotlib.rcParams['font.family'] = 'STSong'
np.set_printoptions(linewidth=400)
np.set_printoptions(threshold=np.inf)
"""
伪代码：
输入：destory算子的集合D，repair算子的集合I，初始化算子权值P（初始均为1）
初始化：生成初试解X_init，令X_current = X_best = X_init
	初始化destory算子和repair算子的概率，分别为1/2，1/2
	设置初始温度T和计数器 j = 1，j_max = 10
while 未满足迭代终止条件：
	依概率选择destory算子和repair算子，作用于X_current,得新解X_new
	if X_new 优于 X_current ：
		X_current = X_new
		if X_new 优于 X_best
			X_best = X_new
	else:
		if 满足接受准则：
			X_current = X_new
	if j == j_max:
		j = 1
		根据自适应权重调整策略更新算子权值P
	else:
		j += 1
"""

"""
破坏操作 有两种算子
"""
def destory_operators(distance, current_solution, destory_number, city_number):
    temp_solution = copy.copy(current_solution)
    ## 破坏list
    destory_list = []
    ## 随机移除算子
    if destory_number == 0: # 从0~city_number中随机取n个不重复的数，返回片段
        ## 多用于截取列表的指定长度的随机数
        temp_0 = random.sample(range(0, city_number), int(np.random.randint(2, 6)))
        for pp in temp_0:
            destory_list.append(temp_solution[pp])
        temp_0.sort()
        temp_0.reverse()
        for pp in temp_0:
            del temp_solution[pp]

    ## 最差距离移除算子
    if destory_number == 1:
        ## 将temp_distance
        temp_distance = np.zeros(city_number)
        temp_distance[0] = distance[current_solution[-1]][current_solution[0]] + distance[current_solution[0]][current_solution[1]]
        temp_distance[-1] = distance[current_solution[-2]][current_solution[-1]] + distance[current_solution[-1]][current_solution[0]]
        for h in range(1, city_number - 1):
            temp_distance[h] = distance[current_solution[h - 1]][current_solution[h]] + distance[current_solution[h]][current_solution[h + 1]]
        Inf = 0
        temp = []
        for i in range(int(np.random.randint(2, 6))):
            temp_2 = np.argmax(temp_distance)#最大索引
            ## 存储索引
            temp.append(temp_2)
            ## 存储被移除的客户点标号
            destory_list.append(temp_solution[temp_2])
            ## 避免再次被选中
            temp_distance[temp_2] = Inf
        temp.sort()
        temp.reverse()
        ## 移除操作
        for i in temp:
            del temp_solution[i]
    return temp_solution, destory_list

"""
修复操作 也是两种
"""
def repair_operators(distance,temp_solution,destory_list,city_number,u,repair_number):
    ## 贪婪插入，找到适应度最小的
    if repair_number == 0:
        for temp_1 in destory_list:
            temp_value = 100000000000000  # 用于记录邻域操作后最好的邻域解
            for f in range(len(temp_solution) + 1):
                temp_route = temp_solution.copy()
                temp_route.insert(f, temp_1)  # 将城市编号插入
                if f == 0:
                    temp1 = distance[temp_route[-1]][temp_route[0]] + distance[temp_route[0]][temp_route[1]] - distance[temp_route[-1]][temp_route[1]]
                elif f == len(temp_solution):
                    temp1 = distance[temp_route[-2]][temp_route[-1]] + distance[temp_route[-1]][temp_route[0]] - distance[temp_route[-2]][temp_route[0]]
                else:
                    temp1 = distance[temp_route[f-1]][temp_route[f]] + distance[temp_route[f]][temp_route[f+1]] -  distance[temp_route[f-1]][temp_route[f+1]]
                if temp1 < temp_value:
                    temp_value = temp1
                    temp_route_new = temp_route.copy()
            temp_solution = temp_route_new.copy()

    ## 随机扰动+贪婪
    if repair_number == 1:
        temp_max = 0
        for i in range(city_number+1):
            for j in range(city_number + 1):
                if distance[i][j] > temp_max:
                    temp_max = distance[i][j]
        for temp_1 in destory_list:
            temp_value = 100000000000000  # 用于记录邻域操作后最好的邻域解
            for f in range(len(temp_solution) + 1):
                temp_route = temp_solution.copy()
                temp_route.insert(f, temp_1)  # 将城市编号插入
                if f == 0:
                    temp1 = distance[temp_route[-1]][temp_route[0]] + distance[temp_route[0]][temp_route[1]] -  distance[temp_route[-1]][temp_route[1]] + temp_max*u*np.random.uniform(-1,1)
                elif f == len(temp_solution):
                    temp1 = distance[temp_route[-2]][temp_route[-1]] + distance[temp_route[-1]][temp_route[0]] - distance[temp_route[-2]][temp_route[0]] + temp_max*u*np.random.uniform(-1,1)
                else:
                    temp1 = distance[temp_route[f-1]][temp_route[f]] + distance[temp_route[f]][temp_route[f+1]] - distance[temp_route[f-1]][temp_route[f+1]] + temp_max*u*np.random.uniform(-1,1)
                if temp1 < temp_value:
                    temp_value = temp1
                    temp_route_new = temp_route.copy()
            temp_solution = temp_route_new.copy()
    temp_value = get_total_distance(temp_solution)
    return temp_solution, temp_value

# ——————————————————自适应大规模邻域搜索主程序——————————————————
def ALNS(distance, city_number, destory_size, repair_size, destory_weight, repair_weight, j_max, iterations, u, alpha, T, theta_1, theta_2, theta_3):
    ## 初始解的生成
    initial_solution = [i for i in range(1,city_number+1)]
    current_value = get_total_distance(initial_solution)
    best_value = current_value
    current_solution = initial_solution.copy()
    ## 全局最优的记录
    best_record = [current_value]
    # 将初始解赋值给当前解和最优解
    best_solution = initial_solution.copy()

    ## destory_size，repair_size分别是破坏算子数和修复算子数
    P_destory = np.array([1 / destory_size] * destory_size)  # 破坏因子选择概率
    P_repair = np.array([1 / repair_size] * repair_size)  # 修复因子选择概率
    time_destory = np.array([0] * destory_size)  # 记录destory算子选中次数
    time_repair = np.array([0] * repair_size)  # 记录destory算子选中次数
    score_destory = np.array([0] * destory_size)  # 记录destory算子分数
    score_repair = np.array([0] * repair_size)  # 记录destory算子分数

    j = 0
    k = 1
    while k <= iterations:
        k += 1
        ## 选择破坏因子
        temp_D = np.cumsum(P_destory)
        temp_probability_D = np.random.rand()
        if temp_probability_D == 0:
            temp_probability_D += 0.000001
        for i in range(destory_size):
            ## 如果随机因子在0-0.3之间，那么选择第0个破坏算子，如果随机因子在两个算子之间，那么选择第i个。
            if i == 0:
                if 0 < temp_probability_D <= temp_D[0]:
                    destory_number = i

            else:
                if temp_D[i-1] < temp_probability_D <= temp_D[i]:
                    destory_number = i

        # 把destory的选择次数加1
        time_destory[destory_number] += 1
        #-------破坏操作，输入当前解、随机的数，输出破坏后路径和移除列表-------
        temp_solution, destory_list = destory_operators(distance, current_solution, destory_number, city_number)

        #-----选择修复因子-------
        temp_P = np.cumsum(P_repair)#累加概率
        temp_probability_P = np.random.rand()
        if temp_probability_P == 0:
            temp_probability_P += 0.000001
        for i in range(repair_size):
            ## 如果随机因子在0-0.3之间，那么选择第0个破坏算子，如果随机因子在两个算子之间，那么选择第i个。
            if i == 0:
                if 0 < temp_probability_P <= temp_P[0]:
                    repair_number = i

            else:
                if temp_P[i - 1] < temp_probability_P <= temp_P[i]:
                    repair_number = i

        # 把time_repair的选择次数加1
        time_repair[repair_number] += 1
        #--------修复操作,输入破坏后路径和移除列表，输出新路径及其适应度-----------
        new_solution, new_value = repair_operators(distance,temp_solution,destory_list,city_number,u,repair_number)

        if new_value < current_value:
            current_solution = new_solution.copy()
            if new_value < best_value:
                best_value = new_value
                best_solution = new_solution.copy()
                ## 将破坏算子的destory_number加分
                score_destory[destory_number] += theta_1
                score_repair[repair_number] += theta_1
            else:
                score_destory[destory_number] += theta_2
                score_repair[repair_number] += theta_2

        else:
            if np.random.rand() < T:
                current_solution = new_solution.copy()
                score_destory[destory_number] += theta_3
                score_repair[repair_number] += theta_3
        j += 1
        ## 存储最优解
        best_record.append(best_value)

        # ---------更新权值概率---------
        if j == j_max:
            for o in range(destory_size):
                if time_destory[o] == 0:
                    destory_weight[o] = destory_weight[o]*alpha
                else:
                    destory_weight[o] = destory_weight[o]*alpha + (1-alpha)*score_destory[o]/time_destory[o]
            sum_destory_weight = np.sum(destory_weight)
            P_destory = destory_weight/sum_destory_weight

            # 修复算子的权重参数
            for o in range(repair_size):
                if time_repair[o] == 0:
                    repair_weight[o] = repair_weight[o]
                else:
                    #                    print(score_repair)
                    repair_weight[o] = repair_weight[o] * (1 - alpha) + alpha * score_repair[o] / time_repair[o]
            sum_repair_weight = np.sum(repair_weight)
            P_repair = repair_weight / sum_repair_weight

            ## 参数初始化
            time_destory = np.array([0] * destory_size)
            time_repair = np.array([0] * repair_size)
            score_destory = np.array([0] * destory_size)
            score_repair = np.array([0] * repair_size)
            j = 0
    return best_solution, best_value, best_record


if "__main__" == __name__:

    ## 全局参数
    destory_size = 2  # 破坏算子数
    repair_size = 2  # 修复算子数
    j_max = 50  #内层迭代次数
    iterations = j_max * 20  # 外层迭代次数
    destory_weight = np.array([1] * destory_size, dtype=np.float64)  # 破坏算子权重
    repair_weight = np.array([1] * repair_size, dtype=np.float64)  # 修复算子权重
    theta_1 = 20  # 新解优于最优解的分数
    theta_2 = 12  # 新解优于当前解的分数
    theta_3 = 8  # 接受不优于当前解的新解分数
    alpha = 0.95  # 相当于蚁群算法的挥发因子
    T = 0.2  # 接受概率
    u = 0.1  # 噪音参数

    ## 数据载入
    city_name = []
    city_condition = []
    with open('data.txt', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\n')[0]
            line = line.split(',')
            city_name.append(line[0])
            city_condition.append([float(line[1]), float(line[2])])
    city_condition = np.array(city_condition)

    ## 距离矩阵
    city_number = len(city_name)
    distance = np.zeros([city_number + 1, city_number + 1])
    for i in range(1, city_number + 1):
        for j in range(1, city_number + 1):
            distance[i][j] = math.sqrt((city_condition[i - 1][0] - city_condition[j - 1][0]) ** 2 + (
                        city_condition[i - 1][1] - city_condition[j - 1][1]) ** 2)
    """
    总距离,适应度计算
    """
    def get_total_distance(path_new):
        path_value = 0
        for i in range(city_number - 1):
            # count为30，意味着回到了开始的点，此时的值应该为0.
            path_value += distance[path_new[i]][path_new[i + 1]]
        path_value += distance[path_new[-1]][path_new[0]]
        return path_value

    start = time.time()
    best_solution, best_solution_value, best_record = ALNS(distance, city_number, destory_size, repair_size,
                                                           destory_weight, repair_weight, j_max, iterations, u, alpha,
                                                           T, theta_1, theta_2, theta_3)
    end = time.time()
    print("总用时", end - start)

    # 用来显示结果
    print("VLNS_TSP")
    print("最优解", best_solution)
    print("最优值", best_solution_value)
    plt.plot(np.array(best_record))
    plt.ylabel("bestvalue")
    plt.xlabel("t")

    # 路线图绘制
    fig = plt.figure()
    ax2 = fig.add_subplot()
    # ax2.set_title('最佳路线图')
    x = []
    y = []
    path = []
    for i in range(len(city_name)):
        x.append(city_condition[best_solution[i] - 1][0])
        y.append(city_condition[best_solution[i] - 1][1])
        path.append(best_solution[i])
    x.append(x[0])
    y.append(y[0])
    path.append(path[0])
    for i in range(len(x)):
        plt.annotate(path[i], xy=(x[i], y[i]), xytext=(x[i] + 0.3, y[i] + 0.3))
    plt.plot(x, y, '-o')
    plt.show()









