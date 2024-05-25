# -*- coding: utf-8 -*-
#    liuyx's first gp for CARP in python
from __future__ import division
import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import sys
import readfile


# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
def max(left, right):
    if left>=right:
        return left
    else:
        return right

pset = gp.PrimitiveSet("MAIN", 6)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.atan2, 2)
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))
# change the name of arguments from ARG0 to x
pset.renameArguments(ARG0='demand')
pset.renameArguments(ARG1="load")
pset.renameArguments(ARG2="cost")
pset.renameArguments(ARG3="depotCost")
pset.renameArguments(ARG4="satisfied")
pset.renameArguments(ARG5="heuristicValue")

# creating fitness function and individual
# 适应度表现为base模块中的Fitness基类，个体类表现为一个gp.PrimitiveTree
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

#register some parameters，使用register将自定函数填充到工具箱base.Toolbox()当中，后可通过toolbox.name调用。
#过程中所需参数动态绑定
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
# list和toolbox.individual为tool.initRepeat的参数，剩余一个参数在下面使用的时候传入，即n=300。
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def floyd(verticeNum, arcList):
    nodeNum=verticeNum + 1
    dis=[None] * nodeNum
    for i in range(nodeNum):
        dis[i]=[None] * nodeNum
        for j in range(nodeNum):
            if i == j:
                dis[i][j] = 0
            else:
                dis[i][j] = 9999
    for i in range(len(arcList)):
        head = arcList[i].getHead()
        tail = arcList[i].getTail()
        dis[head][tail] = arcList[i].getSerCost()
    for k in range(nodeNum):
        for i in range(nodeNum):
            for j in range(nodeNum):
                if dis[i][j] > dis[i][k] + dis[k][j]:
                    dis[i][j] = dis[i][k] + dis[k][j]
    return dis

def calMaxCost(unEdgeList, capacity, vehiclePosition, depot, dis):
    maxCost = 0
    for i in range(len(unEdgeList)):
        thisEdgeCost = 0
        head = unEdgeList[i].getHead()
        if capacity >= unEdgeList[i].getDemand():
            thisEdgeCost = unEdgeList[i].getDeadCost() + dis[vehiclePosition][head]
        else:
            thisEdgeCost = unEdgeList[i].getDeadCost() + dis[vehiclePosition][depot] + dis[depot][head]
        if(thisEdgeCost > maxCost):
            maxCost = thisEdgeCost
    return maxCost

def calMaxDepotCost(unEdgeList, dis, depot):
    maxDepotCost = 0
    for i in range(len(unEdgeList)):
        thisDepotCost = 0
        tail = unEdgeList[i].getTail()
        thisDepotCost = dis[tail][depot]
        if thisDepotCost > maxDepotCost:
            maxDepotCost = thisDepotCost
    return maxDepotCost

def evalRules(individual):
    # compile: Transform the tree expression in a callable function
    # 这里将individual的树形表现形式转换为可执行的形式
    func = toolbox.compile(expr=individual)
    # 如果把读文件放在这里，那么每评价一个个体，都需要读一次文件
    rf = readfile.graph()
    numNodes, numReq, arcList, capacity, depot = rf.read()
    # initialization
    tourCost = 0
    vehiclePosition = depot
    load = capacity
    unEdgeList=arcList
    unEdgeTotalNum = len(unEdgeList) / 2.0
    dis=floyd(numNodes, arcList)
    currentDemand = 0
    currentLoad = 0
    currentCost = 0
    currentDepotCost = 0
    currentSatisfied = 0
    currentHeuristicValue = 0

    # begin the evaluation
    while(len(unEdgeList) > 0):
        minHValue = sys.float_info
        minHValueEdge = -1

        maxCost = calMaxCost(unEdgeList, capacity, vehiclePosition, depot, dis)
        maxDepotCost = calMaxDepotCost(unEdgeList, dis, depot)
        # 不明白这里下面为什么要减1再除以2
        currentUnEdgeNum = len(unEdgeList) / 2.0

        for i in range(len(unEdgeList)):
            head = unEdgeList[i].getHead()
            tail = unEdgeList[i].getTail()
            currentDemand = unEdgeList[i].getDemand() / capacity
            currentLoad = load / capacity

            routingCost = 0
            if (load >= unEdgeList[i].getDemand()):
                routingCost = dis[vehiclePosition][head]
            else:
                routingCost = dis[vehiclePosition][depot] + dis[depot][head]
            currentCost = (unEdgeList[i].getDeadCost() +routingCost) / maxCost

            currentDepotCost = dis[tail][depot] / maxDepotCost
            currentSatisfied = (unEdgeTotalNum - currentUnEdgeNum) / unEdgeTotalNum
            currentHeuristicValue = unEdgeList[i].getHValue()

            hv = func(currentDemand,currentLoad,currentCost,currentDepotCost,currentSatisfied,currentHeuristicValue)
            if math.isinf(hv):
                hv = 1
            elif math.isnan(hv):
                hv = 0
                #print "2here"
            else:
                hv = hv
            unEdgeList[i].setHValue(hv)

            if(unEdgeList[i].getHValue() < minHValue):
                minHValue = unEdgeList[i].getHeuristicValue()
                minHValueEdge = i

        if minHValueEdge == -1:
            print("Does not find the next edge to serve!")
            return
        else:
            if load < unEdgeList[minHValueEdge].getDemand():
                tourCost = tourCost + dis[vehiclePosition][depot]
                load = capacity
                vehiclePosition = depot
            else:
                tourCost = tourCost + dis[vehiclePosition][unEdgeList[minHValueEdge].getHead()] + unEdgeList[minHValueEdge].getCost()
                load = load - unEdgeList[minHValueEdge].getDemand()
                vehiclePosition = unEdgeList[minHValueEdge].getTail()
                del unEdgeList[minHValueEdge]
                if minHValueEdge % 2 == 0:
                    del unEdgeList[minHValueEdge]
                else:
                    del unEdgeList[minHValueEdge-1]

    tourCost = tourCost + dis[vehiclePosition][depot]
    #注意这个逗号，即使是单变量优化问题，也需要返回tuple（元组类型）
    return tourCost,


toolbox.register("evaluate", evalRules)
toolbox.register("select", tools.selTournament, tournsize=7)
# 将gp.cxOnePoint函数命名为"mate"，放入工具箱中，执行时，调用工具箱的"mate"，便会直接调用gp.cxOnePoint，进行交叉
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=2, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

def main():
    random.seed(318)

    # 设置种群数量
    pop = toolbox.population(n=500)
    # 名人堂（Hall Of Fame，HOF）最多存储一个个体
    hof = tools.HallOfFame(1)

    # 设置统计项，注册到统计容器mstats中，在后面能够通过log观察运行过程中每一代的数据
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    # 创建一个有两个统计目标的统计对象
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    # 注册统计目标的统计项
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.2, 0.5, 60, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    print(hof.items[0])
    print(hof.keys[0])
    return pop, log, hof

if __name__ == "__main__":
    main()
