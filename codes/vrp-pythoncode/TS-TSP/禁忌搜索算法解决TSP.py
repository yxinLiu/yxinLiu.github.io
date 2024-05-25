# -*- coding: utf-8 -*-
"""
禁忌搜索算法求解TSP问题
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


def traversal_search(line,dis_matrix,tabu_list):
    #邻域随机遍历搜索
    traversal = 0#搜索次数
    traversal_list = []#存储局部搜索生成的解,也充当局部禁忌表
    traversal_value = []#存储局部解对应路径距离
    while traversal <= traversalMax:
        pos1,pos2 = random.randint(0,len(line)-1),random.randint(0,len(line)-1)#交换点
        #复制当前路径，并交换生成新路径
        new_line = line.copy()
        new_line[pos1],new_line[pos2]=new_line[pos2],new_line[pos1]
        new_value = calFitness(new_line,dis_matrix)#当前路径距离
        #新生成路径不在全局禁忌表和局部禁忌表中，为有效搜索，否则继续搜索
        if (new_line not in traversal_list) & (new_line not in tabu_list):
            traversal_list.append(new_line)
            traversal_value.append(new_value)
            traversal += 1

    return min(traversal_value),traversal_list[traversal_value.index(min(traversal_value))]


def greedy(CityCoordinates,dis_matrix):
    '''贪婪策略构造初始解'''
    #dateframe某列的str类型转为float
    dis_matrix = dis_matrix.astype('float64')
    for i in range(len(CityCoordinates)):dis_matrix.loc[i,i]=math.pow(10,10)
    line = []#初始化
    now_city = random.randint(0,len(CityCoordinates)-1)#随机生成出发城市
    line.append(now_city)#添加当前城市到路径
    dis_matrix.loc[:,now_city] = math.pow(10,10)#更新距离矩阵，已经过城市不再被取出
    for i in range(len(CityCoordinates)-1):
        next_city = dis_matrix.loc[now_city,:].idxmin()#距离最近的城市
        line.append(next_city)#添加进路径
        dis_matrix.loc[:,next_city] = math.pow(10,10)#更新距离矩阵
        now_city = next_city#更新当前城市

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
    plt.savefig('result3-贪婪构造.png')
    plt.show()
#迭代图
def draw_path2(best_fit_list):
    iters = list(range(len(best_fit_list)))
    plt.plot(iters, best_fit_list, 'r-', color='#4169E1', alpha=0.8, linewidth=0.8)
    plt.xlabel('迭代次数')
    plt.savefig('result4-贪婪构造.png')
    plt.show()

if __name__ == '__main__':
    #参数
    CityNum = 20#城市数量
    MinCoordinate = 0#二维坐标最小值
    MaxCoordinate = 101#二维坐标最大值

    #TS参数
    tabu_limit = 50 #禁忌长度，该值应小于(CityNum*(CityNum-1)/2）
    iterMax = 5000#迭代次数
    traversalMax = 100#每一代局部搜索次数

    tabu_list = [] #禁忌表
    tabu_time = [] #禁忌次数
    best_value = math.pow(10,10)#较大的初始值10的10次方，存储最优解
    best_line = []#存储最优路径


    #随机生成城市数据,城市序号为0,1，2,3...
    # CityCoordinates = [(random.randint(MinCoordinate,MaxCoordinate),random.randint(MinCoordinate,MaxCoordinate)) for i in range(CityNum)]
    CityCoordinates = [(88, 16),(42, 76),(5, 76),(69, 13),(73, 56),(100, 100),(22, 92),(48, 74),(73, 46),(39, 1),(51, 75),(92, 2),(101, 44),(55, 26),(71, 27),(42, 81),(51, 91),(89, 54),(33, 18),(40, 78)]
    #计算城市之间的距离
    dis_matrix = pd.DataFrame(data=None,columns=range(len(CityCoordinates)),index=range(len(CityCoordinates)))
    for i in range(len(CityCoordinates)):
        xi,yi = CityCoordinates[i][0],CityCoordinates[i][1]
        for j in range(len(CityCoordinates)):
            xj,yj = CityCoordinates[j][0],CityCoordinates[j][1]
            dis_matrix.iloc[i,j] = round(math.sqrt((xi-xj)**2+(yi-yj)**2),2)

    # #初始化,随机构造
    # line = list(range(len(CityCoordinates)));random.shuffle(line) #20个
    # value = calFitness(line,dis_matrix)#初始路径距离
    # 贪婪构造
    line = greedy(CityCoordinates,dis_matrix)
    value = calFitness(line,dis_matrix)#初始路径距离

    #存储当前最优
    best_value,best_line = value,line
     # draw_path(best_line,CityCoordinates)
    best_value_list = []
    best_value_list.append(best_value)
    #更新禁忌表
    tabu_list.append(line) #一整条路径存储在禁忌表中
    tabu_time.append(tabu_limit)

    itera = 0
    new_value_list = []
    #开始迭代
    while itera <= iterMax:
        new_value,new_line = traversal_search(line,dis_matrix,tabu_list)
        new_value_list.append(new_value)
        if new_value < best_value:#优于最优解
            best_value,best_line = new_value,new_line#更新最优解
            best_value_list.append(best_value)
        line,value = new_line,new_value#更新当前解

        #更新禁忌表
        tabu_time = [x-1 for x in tabu_time]
        #达到了禁忌次数
        if 0 in tabu_time:
            tabu_list.remove(tabu_list[tabu_time.index(0)])
            tabu_time.remove(0)

        tabu_list.append(line)
        tabu_time.append(tabu_limit)
        print('第%d代最优值 %.1f' % (itera, best_value))
        itera +=1

    #路径顺序
    print(best_line)
    #画路径图
    draw_path(best_line,CityCoordinates)
    #迭代图
    draw_path2(new_value_list)