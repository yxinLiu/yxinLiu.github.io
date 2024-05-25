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
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))
# change the name of arguments from ARG0 to x
pset.renameArguments(ARG0='x')

# creating fitness function and individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

#register some parameters
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

arrayX = [-2.00, -1.75, -1.50, -1.25, -1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75,
            2.00, 2.25, 2.50, 2.75]

arrayY = [37.00000, 24.16016, 15.06250, 8.91016, 5.00000, 2.72266, 1.56250, 1.09766, 1.00000, 1.03516, 1.06250,
            1.03516, 1.00000, 1.09766, 1.56250, 2.72266, 5.00000, 8.91016, 15.06250, 24.16016]

def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    sum = 0
    for x, y in zip(arrayX, arrayY):
        result = abs(func(x)-y)
        sum = sum + result
    return sum,


toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))


def main():
    random.seed(318)

    pop = toolbox.population(1024)
    hof = tools.HallOfFame(2)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 300, stats=mstats,
                                   halloffame=hof, verbose=True)

    # print log
    print hof.items[0]
    print hof.keys[0]
    print hof.items[1]
    print hof.keys[1]
    return pop, log, hof

if __name__ == "__main__":
    main()