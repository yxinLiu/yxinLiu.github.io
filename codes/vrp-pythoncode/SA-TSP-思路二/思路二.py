# -*- coding: utf-8 -*-
"""
模拟退火算法法求解TSP问题
随机在（0,101）二维平面生成20个点
距离最小化
"""
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文


#计算路径距离，即评价函数
def calFitness(line,dis_matrix):
    dis_sum = 0
    dis = 0
    for i in range(len(line)):
        if i<len(line)-1:
            dis = dis_matrix.loc[line[i],line[i+1]]#计算距离
            dis_sum = dis_sum+dis
        else:
            dis = dis_matrix.loc[line[i],line[0]]
            dis_sum = dis_sum+dis
    return round(dis_sum,1)


def greedy(CityCoordinates,dis_matrix,p):
    '''贪婪策略构造初始解，基于概率扰动'''
    #更改dis_matrix格式
    dis_matrix = dis_matrix.astype('float64')
    for i in range(len(CityCoordinates)):dis_matrix.loc[i,i]=math.pow(10,10)
    line = []#初始化，存放路径
    now_city = random.randint(0,len(CityCoordinates)-1)#随机生成出发城市
    line.append(now_city)#添加当前城市到路径
    dis_matrix.loc[:,now_city] = math.pow(10,10)#更新距离矩阵，已途径城市不再被取出

    i = 1#当前已排好城市个数
    j = 0#记录当前城市已遍历临近城市次数
    while len(CityCoordinates)-i>0:
        next_city = dis_matrix.loc[now_city,:].idxmin()#距离该城市最近的城市
        j += 1
        if j == len(CityCoordinates)-i:#是否为当前可遍历最后一个城市,是则直接添加——防止所有城市都不被取出的情况
            line.append(next_city)#添加进路径
            dis_matrix.loc[:,next_city] = math.pow(10,10)#更新距离矩阵
            now_city = next_city#更新当前城市
            i += 1
            j = 0#重置
        else:
            if random.random() < p:
                line.append(next_city)#添加进路径
                dis_matrix.loc[:,next_city] = math.pow(10,10)#更新距离矩阵
                now_city = next_city#更新当前城市
                i += 1
                j = 0#重置
            else:
                #不接受当前城市作为下一个城市
                dis_matrix.loc[now_city,next_city] = math.pow(10,10)#更新距离矩阵
    return line


#画路径图
def draw_path(line,CityCoordinates):
    x,y= [],[]
    for i in line:
        Coordinate = CityCoordinates[i]
        x.append(Coordinate[0])
        y.append(Coordinate[1])
    x.append(x[0])
    y.append(y[0])
    plt.plot(x, y,'r-', color='#4169E1', alpha=0.8, linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    #参数
    CityNum = 20#城市数量
    MinCoordinate = 0#二维坐标最小值
    MaxCoordinate = 101#二维坐标最大值

    #SA参数
    Tend = 0.1
    T = 100
    beta = 0.98
    #p = 1/(1+math.exp(-0.03*T)) #S型曲线

    best_value = math.pow(10,10)#较大的初始值，存储最优解
    best_line = []#存储最优路径

    #随机生成城市数据,城市序号为0,1，2,3...
    # CityCoordinates = [(random.randint(MinCoordinate,MaxCoordinate),random.randint(MinCoordinate,MaxCoordinate)) for i in range(CityNum)]
    CityCoordinates = [(88, 16),(42, 76),(5, 76),(69, 13),(73, 56),(100, 100),(22, 92),(48, 74),(73, 46),(39, 1),(51, 75),(92, 2),(101, 44),(55, 26),(71, 27),(42, 81),(51, 91),(89, 54),(33, 18),(40, 78)]

    #计算城市间距离
    dis_matrix = pd.DataFrame(data=None,columns=range(len(CityCoordinates)),index=range(len(CityCoordinates)))
    for i in range(len(CityCoordinates)):
        xi,yi = CityCoordinates[i][0],CityCoordinates[i][1]
        for j in range(len(CityCoordinates)):
            xj,yj = CityCoordinates[j][0],CityCoordinates[j][1]
            dis_matrix.iloc[i,j] = round(math.sqrt((xi-xj)**2+(yi-yj)**2),2)

    #循环
    while T >= Tend:
        p = 1/(1+math.exp(-0.03*T))#概率
        line = greedy(CityCoordinates,dis_matrix,p)#路径
        value = calFitness(line,dis_matrix)#路径距离
        if value < best_value:#优于最优解
            best_value,best_line = value,line#更新最优解
            print("当前最优解%.1f" % (best_value))
        T *= beta#更新T

    #路径顺序
    print(best_line)
    #画路径图
    draw_path(best_line,CityCoordinates)