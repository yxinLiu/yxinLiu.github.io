# -*- coding: utf-8 -*-
#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

#For explanation: https://deap.readthedocs.io/en/master/examples/gp_symbreg.html   #--yuxin

import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
# change the name of arguments from ARG0 to x
pset.renameArguments(ARG0='x')

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


def evalSymbReg(individual, points):
    # compile: Transform the tree expression in a callable function
    # 这里将individual的树形表现形式转换为可执行的形式
    func = toolbox.compile(expr=individual)

    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x ** 4 - x ** 3 - x ** 2 - x) ** 2 for x in points)
    #注意这个逗号，即使是单变量优化问题，也需要返回tuple（元组类型）
    return math.fsum(sqerrors) / len(points),


toolbox.register("evaluate", evalSymbReg, points=[x / 10. for x in range(-10, 10)])
toolbox.register("select", tools.selTournament, tournsize=3)
# 将gp.cxOnePoint函数命名为"mate"，放入工具箱中，执行时，调用工具箱的"mate"，便会直接调用gp.cxOnePoint，进行交叉
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main():
    random.seed(318)
    # 设置种群数量
    pop = toolbox.population(n=300)
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

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    print hof.items[0]
    print hof.keys[0]
    return pop, log, hof

if __name__ == "__main__":
    main()
