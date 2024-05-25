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

    carLine, carLines = [], []  # 记录分车
    finished_point = []  # 记录已完成车辆
    carDemand, carDemands = [], []  # 记录车辆装载量

    # 序贯
    while True:
        for i in range(len(dis_com)):
            if not carLine:  # 列表为空时直接合并
                carLine.append(int(dis_com.loc[i, 'point1']))
                carLine.append(int(dis_com.loc[i, 'point2']))
                carDemand = demand[int(dis_com.loc[i, 'point1'])] + demand[int(dis_com.loc[i, 'point2'])]
                finished_point.append(int(dis_com.loc[i, 'point1']))  # 全局
                finished_point.append(int(dis_com.loc[i, 'point2']))
                continue

            if ((int(dis_com.loc[i, 'point1']) in finished_point) & (int(dis_com.loc[i, 'point2']) in finished_point)) \
                    | ((int(dis_com.loc[i, 'point1']) not in carLine) & (
                    int(dis_com.loc[i, 'point2']) not in carLine)):  # 两点都已完成，或两点都不在当前车辆服务的点中
                continue

            else:  # 一点在车上，一点不在车上，
                if int(dis_com.loc[i, 'point1']) == carLine[0]:
                    if carDemand + demand[int(dis_com.loc[i, 'point2'])] <= car_load:
                        carDemand += demand[int(dis_com.loc[i, 'point2'])]
                        carLine.insert(0, int(dis_com.loc[i, 'point2']))
                        finished_point.append(int(dis_com.loc[i, 'point2']))
                        continue
                    else:
                        carDemands.append(carDemand)
                        carLine = [0] + carLine + [0]
                        carLines.append(carLine)
                        carLine = []
                        carDemand = 0

                elif int(dis_com.loc[i, 'point1']) == carLine[-1]:
                    if carDemand + demand[int(dis_com.loc[i, 'point2'])] <= car_load:
                        carDemand += demand[int(dis_com.loc[i, 'point2'])]
                        carLine.append(int(dis_com.loc[i, 'point2']))
                        finished_point.append(int(dis_com.loc[i, 'point2']))
                        continue
                    else:
                        carDemands.append(carDemand)
                        carLine = [0] + carLine + [0]
                        carLines.append(carLine)
                        carLine = []
                        carDemand = 0

                elif int(dis_com.loc[i, 'point2']) == carLine[0]:
                    if carDemand + demand[int(dis_com.loc[i, 'point1'])] <= car_load:
                        carDemand += demand[int(dis_com.loc[i, 'point1'])]
                        carLine.insert(0, int(dis_com.loc[i, 'point1']))
                        finished_point.append(int(dis_com.loc[i, 'point1']))
                        continue
                    else:
                        carDemands.append(carDemand)
                        carLine = [0] + carLine + [0]
                        carLines.append(carLine)
                        carLine = []
                        carDemand = 0

                elif int(dis_com.loc[i, 'point2']) == carLine[-1]:
                    if carDemand + demand[int(dis_com.loc[i, 'point1'])] <= car_load:
                        carDemand += demand[int(dis_com.loc[i, 'point1'])]
                        carLine.append(int(dis_com.loc[i, 'point1']))
                        finished_point.append(int(dis_com.loc[i, 'point1']))
                        continue
                    else:
                        carDemands.append(carDemand)
                        carLine = [0] + carLine + [0]
                        carLines.append(carLine)
                        carLine = []
                        carDemand = 0
                else:  # 一点不在，一点在线路中间，无法链接
                    continue

            # 更新减少里程列表
            dis_com = dis_com[
                ~(dis_com['point1'].isin(finished_point) | dis_com['point2'].isin(finished_point))].reset_index(
                drop=True)

            break  # 跳出for循环

        if sorted(finished_point) == list(range(1, len(points))):
            # 最后一辆车
            carDemands.append(carDemand)
            carLine = [0] + carLine + [0]
            carLines.append(carLine)
            carDemand = 0
            carLine = []
            break

        if dis_com.empty:  # 打补丁，存在列表空了但节点未全部服务的情况
            carLine = list(set(list(range(1, len(points)))).difference(set(sorted(finished_point))))
            for i in carLine: carDemand += demand[i]
            carLine = [0] + carLine + [0]
            carLines.append(carLine)
            break

    draw_path(carLines, points)  # 画路径图
