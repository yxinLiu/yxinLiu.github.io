import math
import random
import pandas as pd
import matplotlib.pyplot as plt #只用于绘图
from matplotlib.pylab import mpl #包含numpy和pyplot的常用函数，方便计算和绘图
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文

# 适应度计算
def calFitness(line,dis_matrix): #line指的种群
    dis_sum = 0 #距离综合，作为返回值
    dis = 0
    for i in range(len(line)):
        if i<len(line)-1:
            dis = dis_matrix.loc[line[i],line[i+1]] #loc（）：取行和列
            dis_sum = dis_sum+dis
        else:
            dis = dis_matrix.loc[line[i],line[0]]
            dis_sum = dis_sum+dis
    return round(dis_sum,1) #round（）：保留一位小数

# 锦标赛选择
def tournament_select(pops,popsize,fits,tournament_size):
    new_pops,new_fits = [],[] #新的种群和适应度值
    while len(new_pops)<len(pops): #直到选满所需的种群个数
        tournament_list = random.sample(range(0,popsize),tournament_size) #在100个个体中选择5个竞争者
        tournament_fit = [fits[i] for i in tournament_list] #取出竞争者的适应度值
        #转化为df方便索引
            #transpose（）：将矩阵转置
            #reset_index():重置索引，因为有时候对dataframe做处理后索引可能是乱的。
            #sort_values(by=1):从低到高排序
        tournament_df = pd.DataFrame([tournament_list,tournament_fit]).transpose().sort_values(by=1).reset_index(drop=True)
        #选出获胜者
            #.iloc(i，j):取第i行第j行
            #append（）：添加到末尾
        fit = tournament_df.iloc[0,1] #第0行第1列fit最小
        pop = pops[int(tournament_df.iloc[0,0])] #对应的（0，0）就是fit最小的个体
        new_pops.append(pop)
        new_fits.append(fit)
    return new_pops,new_fits #返回更新后的最优个体

# 顺序交叉
def crossover(popsize,parent1_pops,parent2_pops,pc):
    child_pops = []
    for i in range(popsize):
        #初始化
        child = [None]*len(parent1_pops[i])
            #两个种群各出一个父代个体
        parent1 = parent1_pops[i]
        parent2 = parent2_pops[i]
            #不变异概率，则不交叉
        if random.random() >= pc:
                ##随机生成一个（或者随机保留父代中的一个）
            child = parent1.copy()
            random.shuffle(child) #shuffle（）：重新排列里面的元素
        else:
            #parent1，选择start~end区间里的
                #randint（）：产生均匀分布的随机整数
            start_pos = random.randint(0,len(parent1)-1)
            end_pos = random.randint(0,len(parent1)-1)
                #若start>end，则调换位置
            if start_pos>end_pos:
                tem_pop = start_pos
                start_pos = end_pos
                end_pos = tem_pop
            child[start_pos:end_pos+1] = parent1[start_pos:end_pos+1].copy()
            # parent2 -> child，选择#end~100和0~start两个区间里的
            list1 = list(range(end_pos+1,len(parent2)))
            list2 = list(range(0,start_pos))
            list_index = list1+list2 #两个区间合在一起
            j = -1
            for i in list_index:
                for j in range(j+1,len(parent2)):
                    if parent2[j] not in child:
                        child[i] = parent2[j]
                        break
        child_pops.append(child)
    return child_pops

# 基本位置变异，交叉得到的child_pops继续进行变异
# 变异就是交换位置
def mutate(pops,pm):
    pops_mutate = []
    for i in range(len(pops)):
        pop = pops[i].copy()
        #随机多次成对变异
        t = random.randint(1,5)
        count = 0
        while count < t:
            if random.random() < pm: #在变异概率内
                    mut_pos1 = random.randint(0,len(pop)-1)
                    mut_pos2 = random.randint(0,len(pop)-1)
                    if mut_pos1 != mut_pos2:
                        tem = pop[mut_pos1]
                        pop[mut_pos1] = pop[mut_pos2]
                        pop[mut_pos2] = tem
            pops_mutate.append(pop)
            count +=1
    return pops_mutate

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
    plt.savefig('result1.png')
    plt.show()

if __name__ == '__main__':
    #参数
    CityNum = 20#城市数量
    MinCoordinate = 0#二维坐标最小值
    MaxCoordinate = 101#二维坐标最大值

    #GA参数
    generation = 100  #迭代次数
    popsize = 100   #种群大小
    tournament_size = 5 #锦标赛小组大小
    pc = 0.95   #交叉概率
    pm = 0.1    #变异概率

    #随机生成城市坐标,城市序号为0,1,2,3...
    #CityCoordinates = [(random.randint(MinCoordinate,MaxCoordinate),random.randint(MinCoordinate,MaxCoordinate)) for i in range(CityNum)]

    #随机生成的一个例子
    CityCoordinates = [(88, 16),(42, 76),(5, 76),(69, 13),(73, 56),(100, 100),(22, 92),(48, 74),(73, 46),(39, 1),(51, 75),(92, 2),(101, 44),(55, 26),(71, 27),(42, 81),(51, 91),(89, 54),(33, 18),(40, 78)]

    #计算城市之间的距离
    dis_matrix = pd.DataFrame(data=None,columns=range(len(CityCoordinates)),index=range(len(CityCoordinates))) #距离矩阵（数据，列，行）
    for i in range(len(CityCoordinates)):
        xi,yi = CityCoordinates[i][0],CityCoordinates[i][1]
        for j in range(len(CityCoordinates)):
            xj,yj = CityCoordinates[j][0],CityCoordinates[j][1]
            dis_matrix.iloc[i,j] = round(math.sqrt((xi-xj)**2+(yi-yj)**2),2) #算距离

    iteration = 0
    #初始化,根据种群大小随机构造个体（会有重复的）
        #list（）得到里面具体的值
        #sample（）从指定序列中随机获取指定长度的片断
    pops = [random.sample([i for i in list(range(len(CityCoordinates)))],len(CityCoordinates)) for j in range(popsize)]

    #第一次计算适应度
    fits = [None]*popsize #定义数组，相当于[0,0,...0]（种群大小为100，即100个0）
    for i in range(popsize):
        fits[i] = calFitness(pops[i],dis_matrix) #调用函数

    #保留当前最优
    best_fit = min(fits) #最小的适应度值
    best_pop = pops[fits.index(best_fit)] #对应的最好的种群个体
    print('初代最优值 %.1f' % (best_fit))
    best_fit_list = []
    best_fit_list.append(best_fit) #append（）：在列表末尾加入最好的适应度值

    #开始循环迭代
    while iteration <= generation:
        #锦标赛选择
            #弄出两个种群，用来交叉
        pop1,fits1 = tournament_select(pops,popsize,fits,tournament_size)
        pop2,fits2 = tournament_select(pops,popsize,fits,tournament_size)
        #交叉
        child_pops = crossover(popsize,pop1,pop2,pc)
        #变异
        child_pops = mutate(child_pops,pm)
        #计算子代适应度
        child_fits = [None]*popsize
        for i in range(popsize):
            child_fits[i] = calFitness(child_pops[i],dis_matrix)
        #一对一生存者竞争
        #好的子代会替换坏的父代
        for i in range(popsize):
            if fits[i] > child_fits[i]:
                fits[i] = child_fits[i]
                pops[i] = child_pops[i]
        #更新最好fit
        if best_fit>min(fits):
            best_fit = min(fits)
            best_pop = pops[fits.index(best_fit)]

        best_fit_list.append(best_fit)

        print('第%d代最优值 %.1f' % (iteration, best_fit))
        iteration += 1

    #路径顺序
    print(best_pop)
    #路径图
    draw_path(best_pop,CityCoordinates)
    #迭代图
    iters = list(range(len(best_fit_list)))
    plt.plot(iters, best_fit_list, 'r-', color='#4169E1', alpha=0.8, linewidth=0.8)
    plt.xlabel('迭代次数')
    plt.savefig('result2.png')
    plt.show()