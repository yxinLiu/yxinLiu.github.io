import geatpy as ea
import numpy as np
import matplotlib.pyplot as plt

places = [[35.0, 35.0], [41.0, 49.0], [35.0, 17.0], [55.0, 45.0], [55.0, 20.0], [15.0, 30.0], [25.0, 30.0], [20.0, 50.0], [10.0, 43.0], [55.0, 60.0], [30.0, 60.0], [20.0, 65.0], [50.0, 35.0], [30.0, 25.0], [15.0, 10.0], [30.0, 5.0], [10.0, 20.0], [5.0, 30.0], [20.0, 40.0], [15.0, 60.0]]

# 定义问题

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = len(places) - 1 # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [1] * Dim  # 决策变量下界
        ub = [Dim] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 新增一个属性存储旅行地坐标
        self.places = np.array(places)

    def aimFunc(self, pop):  # 目标函数
        x = pop.Phen.copy()  # 得到决策变量矩阵
        # 添加最后回到出发地
        X = np.hstack([x, x[:, [0]]]).astype(int)
        ObjV = []  # 存储所有种群个体对应的总路程
        for i in range(pop.sizes):
            journey = self.places[X[i], :]  # 按既定顺序到达的地点坐标
            distance = np.sum(np.sqrt(np.sum(np.diff(journey.T) ** 2, 0)))  # 计算总路程
            ObjV.append(distance)
        pop.ObjV = np.array([ObjV]).T

if __name__ == '__main__':
    """================================实例化问题对象============================"""
    problem = MyProblem()  # 生成问题对象
    """==================================种群设置=============================="""
    Encoding = 'P'  # 编码方式，采用排列编码
    NIND = 50  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.soea_SEGA_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 500 # 最大进化代数
    myAlgorithm.mutOper.Pm = 0.5  # 变异概率
    myAlgorithm.logTras = 0  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """===========================调用算法模板进行种群进化========================"""
    [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    BestIndi.save()  # 把最优个体的信息保存到文件中
    """==================================输出结果=============================="""
    print('评价次数：%s' % myAlgorithm.evalsNum)
    print('时间已过 %s 秒' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('最短路程为：%s' % BestIndi.ObjV[0][0])
        print('最佳路线为：')
        best_journey = np.hstack([0, BestIndi.Phen[0, :], 0])
        for i in range(len(best_journey)):
            print(int(best_journey[i]), end=' ')
        print()
        # 绘图
        plt.figure()
        plt.plot(problem.places[best_journey.astype(int), 0], problem.places[best_journey.astype(int), 1], c='black')
        plt.plot(problem.places[best_journey.astype(int), 0], problem.places[best_journey.astype(int), 1], 'o',
                 c='black')
        for i in range(len(best_journey)):
            plt.text(problem.places[int(best_journey[i]), 0], problem.places[int(best_journey[i]), 1],
                     chr(int(best_journey[i]) + 65), fontsize=20)
        plt.grid(True)
        plt.xlabel('x坐标')
        plt.ylabel('y坐标')
        plt.savefig('roadmap.svg', dpi=600, bbox_inches='tight')
    else:
        print('没找到可行解。')