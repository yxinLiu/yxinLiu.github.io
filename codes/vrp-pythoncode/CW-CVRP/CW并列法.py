# -*- coding: utf-8 -*-
"""
Created on Sun May 23 00:23:31 2021

@author: Administrator
"""
# -*- coding: utf-8 -*-
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文


def calDistance(CityCoordinates):
    '''
    计算城市间距离
    输入：CityCoordinates-城市坐标；
    输出：城市间距离矩阵-dis_matrix
    '''
    dis_matrix = pd.DataFrame(data=None, columns=range(len(CityCoordinates)), index=range(len(CityCoordinates)))
    for i in range(len(CityCoordinates)):
        xi, yi = CityCoordinates[i][0], CityCoordinates[i][1]
        for j in range(len(CityCoordinates)):
            xj, yj = CityCoordinates[j][0], CityCoordinates[j][1]
            dis_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)
    return dis_matrix


def indexfind(point1, point2, carLines):
    '''
    定位链接点在哪辆车【连接点表示已在车辆上的点，比如车辆0-1-2-3-0，合并点3-4，那么这里把3称为链接点】
    输入：point1和point2-合并点，carLines-所有车辆服务点
    输出：连接点位置
    '''
    for i in range(len(carLines)):
        if (point1 in carLines[i]) | (point2 in carLines[i]):
            return i


def linkfind(carline, point1, point2):
    '''
    返回车辆中链接点的位置，连接点，和待合并点
    输入：point1和point2-合并点，carLine-车辆服务点
    输出：车辆中链接点的位置，连接点，和待合并点
    '''
    left = carline[0]
    right = carline[-1]
    if point1 == left:
        return 0, point1, point2
    elif point2 == left:
        return 0, point2, point1
    elif point1 == right:
        return -1, point1, point2
    else:
        return -1, point2, point1


def draw_path(car_routes, CityCoordinates):
    '''
    #画路径图
    输入：line-路径，CityCoordinates-城市坐标；
    输出：路径图
    '''
    for route in car_routes:
        x, y = [], []
        for i in route:
            Coordinate = CityCoordinates[i]
            x.append(Coordinate[0])
            y.append(Coordinate[1])
        x.append(x[0])
        y.append(y[0])
        plt.plot(x, y, 'o-', alpha=0.8, linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    car_load = 50

    # 0表示配送中心，1-31表示需求点
    points = [(50, 50), (96, 24), (40, 5), (49, 8), (13, 7), (29, 89), (48, 30), (84, 39), (14, 47), (2, 24), (3, 82),
              (65, 10), (98, 52), (84, 25), (41, 69), (1, 65),
              (51, 71), (75, 83), (29, 32), (83, 3), (50, 93), (80, 94), (5, 42), (62, 70), (31, 62), (19, 97),
              (91, 75), (27, 49), (23, 15), (20, 70), (85, 60), (98, 85)]
    demand = [0, 16, 11, 6, 1, 7, 2, 6, 9, 6, 8, 4, 7, 10, 3, 2, 8, 9, 1, 4, 8, 2, 4, 8, 4, 5, 2, 10, 5, 2, 7, 9]
    dis_matrix = calDistance(points)  # 计算城市间距离

    # 计算合并减少的里程
    dis_com = pd.DataFrame(data=None, columns=["point1", "point2", "save_dis"])
    for i in range(1, len(points) - 1):
        for j in range(i + 1, len(points)):
            detal = dis_matrix.iloc[0, i] + dis_matrix.iloc[0, j] - dis_matrix.iloc[i, j]
            dis_com = dis_com.append({"point1": i, "point2": j, "save_dis": detal}, ignore_index=True)
    dis_com = dis_com.sort_values(by="save_dis", ascending=False).reset_index(drop=True)  # 排序

    carLines = [[]]  # 记录分车
    unfinished_point = []  # 在车辆两端的点
    finished_point = []  # 记录已完成车辆
    carDemands = [0]  # 记录车辆装载量

    # 并列
    for i in range(len(dis_com)):
        if not carLines[-1]:  # 列表为空时
            carLines[0].append(int(dis_com.loc[i, 'point1']))
            carLines[0].append(int(dis_com.loc[i, 'point2']))
            carDemands[0] = demand[int(dis_com.loc[i, 'point1'])] + demand[int(dis_com.loc[i, 'point2'])]
            unfinished_point.append(int(dis_com.loc[i, 'point1']))  # 全局
            unfinished_point.append(int(dis_com.loc[i, 'point2']))
            continue
        if ((int(dis_com.loc[i, 'point1']) in unfinished_point) & (int(dis_com.loc[i, 'point2']) in unfinished_point)) \
                | (int(dis_com.loc[i, 'point1']) in finished_point) | (int(dis_com.loc[i, 'point2']) in finished_point):
            continue  # 两点都装车，或有一点已完成

        elif ((int(dis_com.loc[i, 'point1']) not in unfinished_point) & (
                int(dis_com.loc[i, 'point2']) not in unfinished_point)):  # 两点都不在，新的车
            carLines.append([int(dis_com.loc[i, 'point1']), int(dis_com.loc[i, 'point2'])])
            carDemands.append(demand[int(dis_com.loc[i, 'point1'])] + demand[int(dis_com.loc[i, 'point2'])])
            unfinished_point.append(int(dis_com.loc[i, 'point1']))
            unfinished_point.append(int(dis_com.loc[i, 'point2']))

        else:  # 一点已装车且允许再衔接其他点，一点未装车，
            car_index = indexfind(int(dis_com.loc[i, 'point1']), int(dis_com.loc[i, 'point2']), carLines)  # 查看在哪辆车
            link_index, link_point, point = linkfind(carLines[car_index], int(dis_com.loc[i, 'point1']),
                                                     int(dis_com.loc[i, 'point2']))  # 确定链接位置和链接点
            if carDemands[car_index] + demand[point] <= car_load:
                carDemands[car_index] += demand[point]
                if link_index == 0:
                    unfinished_point.remove(link_point)
                    unfinished_point.append(point)
                    finished_point.append(link_point)
                    carLines[car_index].insert(0, point)
                else:
                    unfinished_point.remove(link_point)
                    unfinished_point.append(point)
                    finished_point.append(link_point)
                    carLines[car_index].append(point)
                    continue

    for i in carLines:
        i.append(0)
        i.insert(0, 0)

    draw_path(carLines, points)  # 画路径图
