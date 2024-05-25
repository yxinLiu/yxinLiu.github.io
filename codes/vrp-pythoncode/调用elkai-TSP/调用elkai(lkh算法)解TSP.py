# -*- coding: utf-8 -*-
"""
目前这个包能够达到城市数量N=315 cities以内的tsp(有回路)问题达到最优解
https://pypi.org/project/elkai/
"""


import numpy as np
import elkai
import math
import matplotlib.pyplot as plt
import scipy as sp
import scipy.spatial as ssp

"""
data.txt数据载入
"""
city_name = []
city_condition = []
with open('data.txt','r',encoding='UTF-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split('\n')[0]
        line = line.split(',')
        city_name.append(line[0])
        city_condition.append([float(line[1]), float(line[2])])
city_condition = np.array(city_condition)


    

#计算距离矩阵
def genDistanceMat(x,y):
    X=np.array([x,y])
    distMat=ssp.distance.pdist(X.T)
    sqdist=ssp.distance.squareform(distMat)
    return sqdist
x,y = city_condition[:,0],city_condition[:,1]
#计算距离矩阵
distance = genDistanceMat(x, y)

#传入解方案，计算解成本
def cal_fitness(solutionnew):
    valuenew = np.sum(distance[solutionnew[:-1],solutionnew[1:len(solutionnew)]])
    return valuenew
        


solutionbest = elkai.solve_int_matrix(distance*1000)
#solutionbest = elkai.solve_float_matrix(distance,runs=10)#允许浮点距离
solutionbest.append(0)
print("最优解方案:",solutionbest)
print("最优解总长度:",cal_fitness(solutionbest))
#solutionbest.append(0)
#路线图绘制
fig2 = plt.figure(2)
x = []
y = []
txt = []
#将x，y坐标按顺序存进来
solutionbest.append(0)
for i in range(len(solutionbest)):
    x.append(float(city_condition[int(solutionbest[i])][0]))
    y.append(float(city_condition[int(solutionbest[i])][1]))

#x.append(x[0])
#y.append(y[0])
#txt.append(txt[0])
##plt.rcParams['font.sans-serif']=['SimHei']
plt.scatter(x,y)
for i in range(len(x)):
    plt.annotate(solutionbest[i], xy = (x[i], y[i]), xytext = (x[i]+0.3, y[i]+0.3)) # 这里xy是需要标记的坐标，xytext是对应的标签坐标
plt.plot(x,y)

plt.show()