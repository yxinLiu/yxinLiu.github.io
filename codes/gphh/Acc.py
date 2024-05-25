import pandas as pd
import numpy as np
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import operator
import datetime
import csv
from sklearn.model_selection import train_test_split
import data

print("Acc")
starttime = datetime.datetime.now()
print("starttime",starttime)

def protectedDiv(left, right):     # protected
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def If(f1,f2,f3):
    if(f1<0):
        return f2
    else:
        return f3


pset1 = gp.PrimitiveSet("MAIN", data.feature_num, prefix="f")
pset1.addPrimitive(operator.add, 2)
pset1.addPrimitive(operator.sub, 2)
pset1.addPrimitive(operator.mul, 2)
pset1.addPrimitive(protectedDiv, 2)
pset1.addPrimitive(If, 3)
pset1.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset1, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset1)


def dict_det(table):
    vector = np.array(table).tolist()
    values = vector
    keys = [i for i in range(len(table))]
    dictioanary = {k: v for k, v in zip(keys, values)}
    return dictioanary


def evalClassification(individual, list_train, list_target):
    tp = 0  # true positive
    fp = 0  # false positive
    fn = 0  # false negative
    tn = 0  # true negative

    majority_class = []
    minority_class = []

    dict_train = dict_det(list_train)
    func = toolbox.compile(expr=individual)
    for keys, values in dict_train.items():
        progOut = func(*values)
        if progOut <= 0:
            majority_class.append(keys)
        else:
            minority_class.append(keys)

    dict_target = dict_det(list_target)
    for i in majority_class:
        for key_r, value_r in dict_target.items():
            if (key_r == i) and (value_r == 0):
                tp=tp+1
            elif(key_r == i) and (value_r == 1):   
                fp=fp+1

    for i in minority_class:
        for key_r, value_r in dict_target.items():
            if (key_r == i) and (value_r == 1):    
                tn=tn+1
            elif (key_r == i) and (value_r == 0):    
                fn=fn+1
    return (tp+tn)/(tp+tn+fp+fn),



toolbox.register("evaluate", evalClassification, list_train=data.train_X, list_target=data.train_Y)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=2, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset1)


toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

def prediction(hof, test, label):
    tp = 0  # true positive
    fp = 0  # false positive
    fn = 0  # false negative
    tn = 0  # true negative
    majority_class = []
    minority_class = []
    func = toolbox.compile(expr=hof)
    dict_validation= dict_det(test)
    for keys, values in dict_validation.items():
        progOut = func(*values)
        if progOut <= 0:
            majority_class.append(keys)
        else:
            minority_class.append(keys)

    dict_target = dict_det(label)

    for i in majority_class:   # majoriy is 0; minority is 1 
        for key_r, value_r in dict_target.items():
            if (key_r == i) and (value_r == 0):
                tp=tp+1
            elif(key_r == i) and (value_r == 1):
                fp=fp+1

    for i in minority_class:
        for key_r, value_r in dict_target.items():
            if (key_r == i) and (value_r == 1):
                tn=tn+1
            elif (key_r == i) and (value_r == 0):
                fn=fn+1
    t1=tp / (tp + fn)
    t2=tn / (tn + fp)
    t3=(tp + tn) / (tp + tn + fp + fn)
    t4= 0.5 * (tp / (tp + fn)) + 0.5 * (tn / (tn + fp))
    
    print("majority", tp / (tp + fn))
    print("minority", tn / (tn + fp))
    print("overall accuracty", (tp + tn) / (tp + tn + fp + fn))
    print("balanced accuracy", 0.5 * (tp / (tp + fn)) + 0.5 * (tn / (tn + fp)))

    return t1,t2,t3,t4

def eaSimple1(population, toolbox, cxpb, mutpb, frac_elitist, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
       # print(fit)


    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        Elism=tools.selBest(population, frac_elitist)
        offspring = toolbox.select(population, len(population)-len(Elism))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)


        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            #print(fit)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring+Elism

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    return population, logbook

def main(seed):
    random.seed(seed)
    pop = toolbox.population(1024)
    hof1 = tools.HallOfFame(2)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = eaSimple1(pop, toolbox, 0.8, 0.2, 1, 50, stats=mstats, halloffame=hof1, verbose=True)
    print("ACC result")
    print("training accuracy of fitness", hof1[0].fitness)
    endtime = datetime.datetime.now()
    time = endtime - starttime
    print("ACC time", time)

    t1=prediction(hof1[0], data.test_X, data.test_Y)



    with open('acc_result1.csv', mode='w') as employee_file:
        result_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow([str(hof1[0].fitness), str(time), str(t1[0]),str(t1[1]),str(t1[2]),str(t1[3])])

    return hof1



