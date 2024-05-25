# -*- coding: utf-8 -*-
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文


def calFitness(line, dis_matrix):
    '''
    计算路径距离，即评价函数
    输入：line-路径，dis_matrix-城市间距离矩阵；
    输出：路径距离-dis_sum
    '''
    dis_sum = 0
    dis = 0
    for i in range(len(line) - 1):
        dis = dis_matrix.loc[line[i], line[i + 1]]  # 计算距离
        dis_sum = dis_sum + dis
    dis = dis_matrix.loc[line[-1], line[0]]
    dis_sum = dis_sum + dis
    return round(dis_sum, 1)


def tournament_select(pops, popsize, fits, tournament_size):
    '''
    #锦标赛选择
    '''


    new_pops = []

    while len(new_pops) < len(pops):
        tournament_list = random.sample(range(0, popsize), tournament_size)
        tournament_fit = [fits[i] for i in tournament_list]
        # 转化为df方便索引
        tournament_df = pd.DataFrame([tournament_list, tournament_fit]).transpose().sort_values(by=1).reset_index(drop=True)
        # 选出获胜者
        pop = pops[int(tournament_df.iloc[0, 0])]
        new_pops.append(pop)

    return new_pops


def crossover(popsize, parent1_pops, parent2_pops, pc, tabu_list, par_pops):
    '''
    #顺序交叉
    输入：popsize-种群大小,parent1_pops-父代,parent2_pops-母代,pc-交叉概率,tabu_list-全局禁忌表,par_pops-上一代种群；
    输出：交叉后的种群-child_pops
    '''
    child_pops = []

    for i in range(popsize):
        flag = True
        while flag:

            # 初始化
            child = [None] * len(parent1_pops[i])
            parent1 = parent1_pops[i]
            parent2 = parent2_pops[i]

            if random.random() >= pc:
                child = parent1.copy()  # 随机生成一个
                random.shuffle(child)

            else:
                # parent1 -> child
                start_pos = random.randint(0, len(parent1) - 1)
                end_pos = random.randint(0, len(parent1) - 1)
                if start_pos > end_pos: start_pos, end_pos = end_pos, start_pos
                child[start_pos:end_pos + 1] = parent1[start_pos:end_pos + 1].copy()
                # parent2 -> child
                list1 = list(range(end_pos + 1, len(parent2)))
                list2 = list(range(0, start_pos))
                list_index = list1 + list2
                j = -1
                for i in list_index:
                    for j in range(j + 1, len(parent2)):
                        if parent2[j] not in child:
                            child[i] = parent2[j]
                            break

            if (child not in tabu_list) & (child not in child_pops) & (child not in par_pops):  # 判断是否满足禁忌表
                child_pops.append(child)
                flag = False

    return child_pops


def mutate(pops, pm, tabu_list, par_pops):
    '''
    #基本位变异，成对变异
    输入：pops-交叉后种群,pm-变异概率,tabu_list-群居禁忌表,par_pops-上一代种群；
    输出：变异后种群-pops_mutate
    '''
    pops_mutate = []
    for i in range(len(pops)):
        flag = True
        while flag:
            pop = pops[i].copy()
            t = random.randint(1, 10)  # 随机多次成对变异
            count = 0
            while count < t:
                if random.random() < pm:
                    mut_pos1 = random.randint(0, len(pop) - 1)
                    mut_pos2 = random.randint(0, len(pop) - 1)
                    if mut_pos1 != mut_pos2: pop[mut_pos1], pop[mut_pos2] = pop[mut_pos2], pop[mut_pos1]
                count += 1

            if (pop not in tabu_list) & (pop not in pops_mutate) & (pop not in pops) & (
                    pop not in par_pops):  # 全局禁忌、当前禁忌、交叉后种群禁忌，上一代种群禁忌
                pops_mutate.append(pop)
                flag = False

    return pops_mutate


# 画路径图
def draw_path(line, CityCoordinates):
    x, y = [], []
    for i in line:
        Coordinate = CityCoordinates[i]
        x.append(Coordinate[0])
        y.append(Coordinate[1])
    x.append(x[0])
    y.append(y[0])

    plt.plot(x, y, 'r-', color='#4169E1', alpha=0.8, linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    # 参数
    # CityNum = 50#城市数量
    # MinCoordinate = 0#二维坐标最小值
    # MaxCoordinate = 101#二维坐标最大值

    # GA参数
    generation = 100  # 迭代次数
    popsize = 100  # 种群大小
    tournament_size = 5  # 锦标赛小组大小
    pc = 0.95  # 交叉概率
    pm = 0.1  # 变异概率

    # TS参数
    tabu_limit = 100  # 禁忌长度
    tabu_list = []  # 禁忌表
    tabu_time = []  # 禁忌次数

    # 随机生成城市数据,城市序号为0,1，2,3...
    # CityCoordinates = [(random.randint(MinCoordinate,MaxCoordinate+1),random.randint(MinCoordinate,MaxCoordinate+1)) for i in range(CityNum)]
    # CityCoordinates = [(88, 16),(42, 76),(5, 76),(69, 13),(73, 56),(100, 100),(22, 92),(48, 74),(73, 46),(39, 1),(51, 75),(92, 2),(101, 44),(55, 26),(71, 27),(42, 81),(51, 91),(89, 54),(33, 18),(40, 78)]
    CityCoordinates = [(71, 71), (68, 71), (19, 41), (9, 67), (22, 34), (15, 2), (60, 56), (36, 38), (18, 92), (96, 27),
                       (71, 85), (24, 70), (12, 31), (77, 88), (59, 49), (27, 87), (94, 97), (37, 42), (32, 78),
                       (65, 57), (96, 47), (95, 86), (61, 80), (55, 7), (94, 74), (39, 6), (62, 43), (34, 11), (18, 89),
                       (79, 16), (100, 99), (76, 39), (35, 51), (74, 71), (59, 48), (98, 1), (35, 98), (82, 91),
                       (0, 64), (56, 48), (89, 8), (69, 54), (3, 72), (79, 16), (66, 88), (80, 15), (56, 88), (30, 57),
                       (67, 86), (75, 4)]

    # 计算城市之间的距离
    dis_matrix = pd.DataFrame(data=None, columns=range(len(CityCoordinates)), index=range(len(CityCoordinates)))
    for i in range(len(CityCoordinates)):
        xi, yi = CityCoordinates[i][0], CityCoordinates[i][1]
        for j in range(len(CityCoordinates)):
            xj, yj = CityCoordinates[j][0], CityCoordinates[j][1]
            dis_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)

    iteration = 0
    # 初始化,随机构造
    pops = [random.sample([i for i in list(range(len(CityCoordinates)))], len(CityCoordinates)) for j in range(popsize)]

    # 计算适应度
    fits = [None] * popsize
    for i in range(popsize):
        fits[i] = calFitness(pops[i], dis_matrix)

    # 保留当前最优
    best_fit = min(fits)
    best_pop = pops[fits.index(best_fit)]

    # 禁忌
    tabu_list.append(best_pop)
    tabu_time.append(tabu_limit)

    print('初代最优值 %.1f' % (best_fit))
    best_fit_list = []
    best_fit_list.append(best_fit)

    while iteration <= generation:
        # 锦标赛赛选择
        pop1 = tournament_select(pops, popsize, fits, tournament_size)
        pop2 = tournament_select(pops, popsize, fits, tournament_size)
        child_pops = crossover(popsize, pop1, pop2, pc, tabu_list, pops)  # 交叉
        child_pops = mutate(child_pops, pm, tabu_list, pops)  # 变异

        child_fits = [None] * popsize
        for i in range(popsize):
            child_fits[i] = calFitness(child_pops[i], dis_matrix)

        # 一对一生存者竞争
        for i in range(popsize):
            if fits[i] > child_fits[i]:
                fits[i] = child_fits[i]
                pops[i] = child_pops[i]

        # 更新禁忌表
        tabu_time = [x - 1 for x in tabu_time]
        if 0 in tabu_time:
            tabu_list.remove(tabu_list[tabu_time.index(0)])
            tabu_time.remove(0)

        if best_fit > min(fits):
            best_fit = min(fits)
            best_pop = pops[fits.index(best_fit)]
        # 添加禁忌
        tabu_list.append(best_pop)
        tabu_time.append(tabu_limit)

        best_fit_list.append(best_fit)
        print('第%d代最优值 %.1f' % (iteration, best_fit))
        iteration += 1

    # 路径顺序
    print(best_pop)
    # #画图
    draw_path(best_pop, CityCoordinates)
    # 迭代图
    iters = list(range(len(best_fit_list)))
    plt.plot(iters, best_fit_list, 'r-', color='#4169E1', alpha=0.8, linewidth=0.8)
    plt.xlabel('迭代次数')
    plt.ylabel('最优解')
    plt.show()
