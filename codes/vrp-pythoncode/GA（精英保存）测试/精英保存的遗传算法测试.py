import geatpy as ea
import numpy as np
import time

"""-----------------------------目标函数-------------------------------"""

def aimfunc(Phen):
    x1 = Phen[:,[0]]
    x2 = Phen[:,[1]]
    return np.sin(x1+x2)+(x1-x2)**2-1.5*x1+2.5*x2+1

"""-------------------参数设置--主要是区域生成器参数，也就是染色体编码规则矩阵---------------------"""

Encoding = 'BG'#BG表示采用二进制或格雷编码
varTypes = np.array([0,0])#也就是变量的连续与否，0连续，1离散；这里有两个数，表示两个变量

x1 = [-1.5,4]#变量1的范围
x2 = [-3,4]#变量2的范围
ranges = np.vstack([x1,x2]).T#表示列表纵向叠加，转换为np数组，然后转置，后续有变量可继续放

b1 = [1,1]#对应变量1的两个边界，1表示包含边界
b2 = [1,1]
borders = np.vstack([b1,b2]).T

precisions = [5,5]#决策变量的编码精度，表示解码后能表示的决策变量的精度可达到小数点后5位,在一定程度上影响编码的长度

codes = [1,1]#代表变量的编码方式，(0:二进制 | 1:格雷)
scales = [0,0]# 0表示采用算术刻度，1表示采用对数刻度

FieldD = ea.crtfld(Encoding, varTypes, ranges, borders, precisions, codes, scales)

"""根据后面的初始化函数，设置遗传算法初始化参数"""
Nind = 20


"""初始化算法参数"""

start_time = time.time()#开始计时
Chrom = ea.crtpc(Encoding, Nind, FieldD) #种群染色体，初始化函数，结果为编码过后的含有NIND行的个体的矩阵

Phen = ea.bs2real(Chrom, FieldD)#解码为十进制后的矩阵，2xNIND矩阵

ObjV = aimfunc(Phen)#计算初始目标值

best_ind = np.argmin(ObjV)#计算最优(最小)值的序号
# print(best_ind)


"""进化迭代参数初始化设置"""

MAXGEN = 100 #最大遗传代数
maxormins = [1]#表示目标函数是最小化，元素为-1则表示对应的目标函数是最大化
selectStyle = 'sus' # 采用随机抽样选择
recStyle = recStyle = 'xovdp' # 采用两点交叉
RecOpt=0.7 #交叉概率
mutStyle = 'mutbin' # 采用二进制染色体的变异算子
pm = 1 # 整条染色体的变异概率（每一位的变异概率=pm/染色体长度）


"""记录矩阵"""

Lind = int(np.sum(FieldD[0, :])) # 计算染色体长度等价于lens = len(Chrom[0])
obj_trace = np.zeros((MAXGEN, 2)) # 定义目标函数值记录器
var_trace = np.zeros((MAXGEN, Lind)) #染色体记录器，记录历代最优个体的染色体


"""进化迭代"""

for gen in range(MAXGEN):
    FitnV = ea.ranking(ObjV)
    Selch = Chrom[ea.selecting(selectStyle, FitnV, GGAP=Nind-1),:]#选择
    Selch = ea.recombin(recStyle, Selch, RecOpt)#重组
    Selch = ea.mutate(mutStyle, Encoding, Selch, pm)#变异

    #将父代精英个体与子代染色体进行合并，得到新一代种群染色体
    Chrom = np.vstack([Chrom[best_ind,:],Selch])
    Phen = ea.bs2real(Chrom,FieldD)# 对种群进行解码(二进制转十进制)
    ObjV = aimfunc(Phen)# 求种群个体的目标函数值
    best_ind = np.argmin(ObjV)#最优值索引

    #进行记录
    obj_trace[gen,0]=np.sum(ObjV)/ObjV.shape[0]#记录当代种群的目标函数均值
    obj_trace[gen,1] = ObjV[best_ind]#记录当代种群最优个体目标函数值
    var_trace[gen,:]=Chrom[best_ind,:] #记录当代种群最优个体的染色体

# 进化完成
end_time = time.time() # 结束计时
ea.trcplot(obj_trace, [['种群个体平均目标函数值','种群最优个体目标函数值']]) # 绘制图像
# print(help(ea.mutate))
"""============================输出结果============================"""

best_gen = np.argmin(obj_trace[:, [1]])
print('最优解的目标函数值：', obj_trace[best_gen, 1])
variable = ea.bs2real(var_trace[[best_gen], :], FieldD)
#解码得到表现型（即对应的决策变量值）
print('最优解的决策变量值为：')
for i in range(variable.shape[1]):
    print('x'+str(i)+'=',variable[0, i])
print('用时：', end_time - start_time, '秒')