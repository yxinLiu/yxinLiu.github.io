#内置函数
import math
import random
import copy

#第三方库
import numpy as np
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt

# Sol()类，表示一个可行解
class Sol():
    def __init__(self):
        self.nodes_seq=None#需求节点seq_no有序排列集合，对应TSP的解
        self.obj=None#优化目标值
        self.routes=None#车辆路径集合，对应CVRP的解
# Node()类，表示一个网络节点
class Node():
    def __init__(self):
        self.id=0#节点id
        self.name=''#节点名称
        self.seq_no=0#节点映射id，基地节点为-1，需求节点从0编号
        self.x_coord=0#节点x坐标
        self.y_coord=0#节点y坐标
        self.demand=0#节点需求
# Model()类，存储算法参数
class Model():
    def __init__(self):
        self.best_sol=None#全局最优解，值类型为Sol()
        self.node_list=[]#节点集合，值类型为Node()
        self.node_seq_no_list=[]#节点映射id集合
        self.depot=None#车辆基地，值类型为Node()
        self.number_of_nodes=0#需求节点数量
        self.opt_type=0#优化目标类型，0：最小车辆数，1：最小行驶距离
        self.vehicle_cap=80#车辆容量
        self.distance = {}#网络弧距离
        self.rand_d_max=0.4#随机破坏程度上限
        self.rand_d_min=0.1#随机破坏程度下限
        self.worst_d_max=5#最坏破坏程度上限
        self.worst_d_min=20#最坏破坏程度下限
        self.regret_n=5#次优位置个数
        self.r1=30#算子奖励1
        self.r2=18#算子奖励2
        self.r3=12#算子奖励3
        self.rho=0.6#算子权重衰减系数
        self.d_weight=np.ones(2)*10#破坏算子权重
        self.d_select=np.zeros(2)#破坏算子被选中次数/每轮
        self.d_score=np.zeros(2)#破坏算子被奖励得分/每轮
        self.d_history_select=np.zeros(2)#破坏算子历史共计被选中次数
        self.d_history_score=np.zeros(2)#破坏算子历史共计被奖励得分
        self.r_weight=np.ones(3)*10#修复算子权重
        self.r_select=np.zeros(3)#修复算子被选中次数/每轮
        self.r_score=np.zeros(3)#修复算子被奖励得分/每轮
        self.r_history_select = np.zeros(3)#修复算子历史共计被选中次数
        self.r_history_score = np.zeros(3)#修复算子历史共计被奖励得分
        self.demand_dict={}#车辆载重，画图用

# （1）文件读取
def readXlsxFile(filepath,model):
    # It is recommended that the vehicle depot data be placed in the first line of xlsx file
    node_seq_no = -1#the depot node seq_no is -1,and demand node seq_no is 0,1,2,...
    df = pd.read_excel(filepath)
    for i in range(df.shape[0]):
        node=Node()
        node.id=node_seq_no
        node.seq_no=node_seq_no
        node.x_coord= df['x_coord'][i]
        node.y_coord= df['y_coord'][i]
        node.demand=df['demand'][i]
        model.demand_dict[node.id] = node
        if df['demand'][i] == 0:
            model.depot=node
        else:
            model.node_list.append(node)
            model.node_seq_no_list.append(node_seq_no)
        try:
            node.name=df['name'][i]
        except:
            pass
        try:
            node.id=df['id'][i]
        except:
            pass
        node_seq_no=node_seq_no+1
    model.number_of_nodes=len(model.node_list)

# （2）初始化参数，计算网格弧距离，以便后续算子使用
def initParam(model):
    for i in range(model.number_of_nodes):
        for j in range(i+1,model.number_of_nodes):
            d=math.sqrt((model.node_list[i].x_coord-model.node_list[j].x_coord)**2+
                        (model.node_list[i].y_coord-model.node_list[j].y_coord)**2)
            model.distance[i,j]=d
            model.distance[j,i]=d

# （3）初始解
def genInitialSol(node_seq):
    node_seq=copy.deepcopy(node_seq)
    random.seed(0)
    random.shuffle(node_seq)
    return node_seq

# （4）目标值计算，和前面一样
def splitRoutes(nodes_seq,model):
    num_vehicle = 0
    vehicle_routes = []
    route = []
    remained_cap = model.vehicle_cap
    for node_no in nodes_seq:
        if remained_cap - model.node_list[node_no].demand >= 0:
            route.append(node_no)
            remained_cap = remained_cap - model.node_list[node_no].demand
        else:
            vehicle_routes.append(route)
            route = [node_no]
            num_vehicle = num_vehicle + 1
            remained_cap =model.vehicle_cap - model.node_list[node_no].demand
    vehicle_routes.append(route)
    return num_vehicle,vehicle_routes
def calDistance(route,model):
    distance=0
    depot=model.depot
    for i in range(len(route)-1):
        distance+=model.distance[route[i],route[i+1]]
    first_node=model.node_list[route[0]]
    last_node=model.node_list[route[-1]]
    distance+=math.sqrt((depot.x_coord-first_node.x_coord)**2+(depot.y_coord-first_node.y_coord)**2)
    distance+=math.sqrt((depot.x_coord-last_node.x_coord)**2+(depot.y_coord - last_node.y_coord)**2)
    return distance
def calObj(nodes_seq,model):
    num_vehicle, vehicle_routes = splitRoutes(nodes_seq, model)
    if model.opt_type==0:
        return num_vehicle,vehicle_routes
    else:
        distance=0
        for route in vehicle_routes:
            distance+=calDistance(route,model)
        return distance,vehicle_routes

# （5）定义destroy算子（破坏算子）
#这里实现两种destroy
#随机从当前解中移除一定比例的需求节点
def createRandomDestory(model):
    d=random.uniform(model.rand_d_min,model.rand_d_max)
    reomve_list=random.sample(range(model.number_of_nodes),int(d*model.number_of_nodes))
    return reomve_list

#从当前解中移除一定比例引起目标函数增幅较大的需求节点
def createWorseDestory(model,sol):
    deta_f=[]
    for node_no in sol.nodes_seq:
        nodes_seq_=copy.deepcopy(sol.nodes_seq)
        nodes_seq_.remove(node_no)
        obj,vehicle_routes=calObj(nodes_seq_,model)
        deta_f.append(sol.obj-obj)
    sorted_id = sorted(range(len(deta_f)), key=lambda k: deta_f[k], reverse=True)
    d=random.randint(model.worst_d_min,model.worst_d_max)
    remove_list=sorted_id[:d]
    return remove_list

# （6）定义repair算子（修复算子）
#实现三种repair
#将被移除的需求节点随机插入已分配节点序列中
def createRandomRepair(remove_list,model,sol):
    unassigned_nodes_seq=[]
    assigned_nodes_seq = []
    # remove node from current solution
    for i in range(model.number_of_nodes):
        if i in remove_list:
            unassigned_nodes_seq.append(sol.nodes_seq[i])
        else:
            assigned_nodes_seq.append(sol.nodes_seq[i])
    # insert
    for node_no in unassigned_nodes_seq:
        index=random.randint(0,len(assigned_nodes_seq)-1)
        assigned_nodes_seq.insert(index,node_no)
    new_sol=Sol()
    new_sol.nodes_seq=copy.deepcopy(assigned_nodes_seq)
    new_sol.obj,new_sol.routes=calObj(assigned_nodes_seq,model)
    return new_sol

#根据被移除的需求节点插入已分配节点序列中每个可能位置的目标函数增量大小，
#依次选择目标函数增量最小的需求节点与插入位置组合，
# 直到所有被移除的需求节点都重新插入为止
# （可简单理解为，依次选择使目标函数增量最小的需求节点与其最优的插入位置）；
#有个公式在VRP的笔记里
#包括create和find两个函数
def createGreedyRepair(remove_list,model,sol):
    unassigned_nodes_seq = []
    assigned_nodes_seq = []
    # remove node from current solution
    for i in range(model.number_of_nodes):
        if i in remove_list:
            unassigned_nodes_seq.append(sol.nodes_seq[i])
        else:
            assigned_nodes_seq.append(sol.nodes_seq[i])
    #insert
    while len(unassigned_nodes_seq)>0:
        insert_node_no,insert_index=findGreedyInsert(unassigned_nodes_seq,assigned_nodes_seq,model)
        assigned_nodes_seq.insert(insert_index,insert_node_no)
        unassigned_nodes_seq.remove(insert_node_no)
    new_sol=Sol()
    new_sol.nodes_seq=copy.deepcopy(assigned_nodes_seq)
    new_sol.obj,new_sol.routes=calObj(assigned_nodes_seq,model)
    return new_sol

def findGreedyInsert(unassigned_nodes_seq,assigned_nodes_seq,model):
    best_insert_node_no=None
    best_insert_index = None
    best_insert_cost = float('inf')
    assigned_nodes_seq_obj,_=calObj(assigned_nodes_seq,model)
    for node_no in unassigned_nodes_seq:
        for i in range(len(assigned_nodes_seq)):
            assigned_nodes_seq_ = copy.deepcopy(assigned_nodes_seq)
            assigned_nodes_seq_.insert(i, node_no)
            obj_, _ = calObj(assigned_nodes_seq_, model)
            deta_f = obj_ - assigned_nodes_seq_obj
            if deta_f<best_insert_cost:
                best_insert_index=i
                best_insert_node_no=node_no
                best_insert_cost=deta_f
    return best_insert_node_no,best_insert_index

#计算被移除节点插回到已分配节点序列中n个次优位置时其目标函数值与最优位置的目标函数值的差之和，
# 然后选择差之和最大的需求节点及其最优位置。
# （可简单理解为，优先选择n个次优位置与最优位置距离较远的需求节点及其最优位置。）
#包括create和find两个函数
#公式在笔记里
def createRegretRepair(remove_list,model,sol):
    unassigned_nodes_seq = []
    assigned_nodes_seq = []
    # remove node from current solution
    for i in range(model.number_of_nodes):
        if i in remove_list:
            unassigned_nodes_seq.append(sol.nodes_seq[i])
        else:
            assigned_nodes_seq.append(sol.nodes_seq[i])
    # insert
    while len(unassigned_nodes_seq)>0:
        insert_node_no,insert_index=findRegretInsert(unassigned_nodes_seq,assigned_nodes_seq,model)
        assigned_nodes_seq.insert(insert_index,insert_node_no)
        unassigned_nodes_seq.remove(insert_node_no)
    new_sol = Sol()
    new_sol.nodes_seq = copy.deepcopy(assigned_nodes_seq)
    new_sol.obj, new_sol.routes = calObj(assigned_nodes_seq, model)
    return new_sol

def findRegretInsert(unassigned_nodes_seq,assigned_nodes_seq,model):
    opt_insert_node_no = None
    opt_insert_index = None
    opt_insert_cost = -float('inf')
    for node_no in unassigned_nodes_seq:
        n_insert_cost=np.zeros((len(assigned_nodes_seq),3))
        for i in range(len(assigned_nodes_seq)):
            assigned_nodes_seq_=copy.deepcopy(assigned_nodes_seq)
            assigned_nodes_seq_.insert(i,node_no)
            obj_,_=calObj(assigned_nodes_seq_,model)
            n_insert_cost[i,0]=node_no
            n_insert_cost[i,1]=i
            n_insert_cost[i,2]=obj_
        n_insert_cost= n_insert_cost[n_insert_cost[:, 2].argsort()]
        deta_f=0
        for i in range(1,model.regret_n):
            deta_f=deta_f+n_insert_cost[i,2]-n_insert_cost[0,2]
        if deta_f>opt_insert_cost:
            opt_insert_node_no = int(n_insert_cost[0, 0])
            opt_insert_index=int(n_insert_cost[0,1])
            opt_insert_cost=deta_f
    return opt_insert_node_no,opt_insert_index

# （7）随机选择算子
# 采用轮盘赌法选择destory和repair算子。
def selectDestoryRepair(model):
    d_weight=model.d_weight
    d_cumsumprob = (d_weight / sum(d_weight)).cumsum()
    d_cumsumprob -= np.random.rand()
    destory_id= list(d_cumsumprob > 0).index(True)

    r_weight=model.r_weight
    r_cumsumprob = (r_weight / sum(r_weight)).cumsum()
    r_cumsumprob -= np.random.rand()
    repair_id = list(r_cumsumprob > 0).index(True)
    return destory_id,repair_id

# （8）执行destory算子
# 根据被选择的destory算子返回需要被移除的节点index序列。
def doDestory(destory_id,model,sol):
    if destory_id==0:
        reomve_list=createRandomDestory(model)
    else:
        reomve_list=createWorseDestory(model,sol)
    return reomve_list

# （9）执行repair算子
# 根据被选择的repair算子对当前接进行修复操作。
def doRepair(repair_id,reomve_list,model,sol):
    if repair_id==0:
        new_sol=createRandomRepair(reomve_list,model,sol)
    elif repair_id==1:
        new_sol=createGreedyRepair(reomve_list,model,sol)
    else:
        new_sol=createRegretRepair(reomve_list,model,sol)
    return new_sol

# （10）重置算子得分
# 在每执行pu次destory和repair，重置算子的得分和被选中次数。
def resetScore(model):

    model.d_select = np.zeros(2)
    model.d_score = np.zeros(2)

    model.r_select = np.zeros(3)
    model.r_score = np.zeros(3)

# （11）更新算子权重
"""
对于算子权重的更新有两种策略，一种是每执行依次destory和repair更新一次，
另一种是每执行pu次destory和repair更新一次权重。前者，能够保证权重及时得到更新，
但却需要更多的计算时间；后者，通过合理设置pu参数，节省了计算时间，同时又不至于权
重更新太滞后。这里采用后者更新策略。
"""
def updateWeight(model):

    for i in range(model.d_weight.shape[0]):
        if model.d_select[i]>0:
            model.d_weight[i]=model.d_weight[i]*(1-model.rho)+model.rho*model.d_score[i]/model.d_select[i]
        else:
            model.d_weight[i] = model.d_weight[i] * (1 - model.rho)
    for i in range(model.r_weight.shape[0]):
        if model.r_select[i]>0:
            model.r_weight[i]=model.r_weight[i]*(1-model.rho)+model.rho*model.r_score[i]/model.r_select[i]
        else:
            model.r_weight[i] = model.r_weight[i] * (1 - model.rho)
    model.d_history_select = model.d_history_select + model.d_select
    model.d_history_score = model.d_history_score + model.d_score
    model.r_history_select = model.r_history_select + model.r_select
    model.r_history_score = model.r_history_score + model.r_score

# （12）绘制收敛曲线
def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.savefig('result1.png')
    plt.show()

# （13）绘制收运路线图
def plotRoutes(model):
    for route in model.best_sol.routes:
        x_coord=[model.depot.x_coord]
        y_coord=[model.depot.y_coord]
        for node_id in route[0:-1]:
            x_coord.append(model.demand_dict[node_id].x_coord)
            y_coord.append(model.demand_dict[node_id].y_coord)
        x_coord.append(model.depot.x_coord)
        y_coord.append(model.depot.y_coord)
        plt.grid()
        
        plt.plot(x_coord,y_coord,marker='o',color='black',linewidth=0.5,markersize=5)
        # plt.plot(x_coord,y_coord,marker='o',color='orange',linewidth=0.5,markersize=5)
        # plt.plot(x_coord,y_coord,marker='o',color='b',linewidth=0.5,markersize=5)
    plt.xlabel('x_coord')
    plt.ylabel('y_coord')
    plt.savefig('result2.png')
    plt.show()

# （14）输出结果到Excel
def outPut(model):
    work=xlsxwriter.Workbook('result.xlsx')
    worksheet=work.add_worksheet()
    worksheet.write(0,0,'opt_type')
    worksheet.write(1,0,'obj')
    if model.opt_type==0:
        worksheet.write(0,1,'number of vehicles')
    else:
        worksheet.write(0, 1, 'drive distance of vehicles')
    worksheet.write(1,1,model.best_sol.obj)
    for row,route in enumerate(model.best_sol.routes):
        worksheet.write(row+2,0,'v'+str(row+1))
        r=[str(i)for i in route]
        worksheet.write(row+2,1, '-'.join(r))
    work.close()

# （15）主函数
#主函数中设置两层循环，外层循环由epochs参数控制，内层循环由pu参数控制，
# 每执行一次内层循环更新一次算子权重。
# 对于邻域新解的接受准则有很多类型，比如RW, GRE,SA,TA,OBA,GDA，
# 这里采用TA准则，即f(newx)-f(x)<T时接受新解，T按照系数 φ衰减，初始T取为当前解*0.2。
# 另外，在定义destory算子时，这里没有指定具体的破坏程度数值，
# 而是采用一个区间，每次随机选择一个破坏程度对当前解进行破坏。
# 更多算法细节可以参考文末资料1，关于加速技巧可以参考文末资料4，5。

def run(filepath,rand_d_max,rand_d_min,worst_d_min,worst_d_max,regret_n,r1,r2,r3,rho,phi,epochs,pu,v_cap,opt_type):
    """
    :param filepath: Xlsx 数据表路径
    :param rand_d_max: 随机破坏程度上限
    :param rand_d_min: 随机破坏程度下限
    :param worst_d_max: 最坏破坏程度上限
    :param worst_d_min: 最坏破坏程度下限
    :param regret_n:  次优位置个数
    :param r1: score if the new solution is the best one found so far.算子奖励1
    :param r2: score if the new solution improves the current solution.算子奖励2
    :param r3: score if the new solution does not improve the current solution, but is accepted.算子奖励3
    :param rho: reaction factor of action weight算子权重衰减系数
    :param phi: 阈值降低因子
    :param epochs:迭代次数
    :param pu: the frequency of weight adjustment
    :param v_cap: 车载容量
    :param opt_type: 优化目标类型，0：最小车辆数，1：最小行驶距离
    :return:
    """
    model=Model()
    model.rand_d_max=rand_d_max
    model.rand_d_min=rand_d_min
    model.worst_d_min=worst_d_min
    model.worst_d_max=worst_d_max
    model.regret_n=regret_n
    model.r1=r1
    model.r2=r2
    model.r3=r3
    model.rho=rho
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    readXlsxFile(filepath, model)#数据读取
    initParam(model)#模型初始化
    history_best_obj = []
    sol = Sol()#可行解初始化
    sol.nodes_seq = genInitialSol(model.node_seq_no_list)
    sol.obj, sol.routes = calObj(sol.nodes_seq, model)
    model.best_sol = copy.deepcopy(sol)
    history_best_obj.append(sol.obj)
    for ep in range(epochs):#算法优化迭代开始
        T=sol.obj*0.2
        resetScore(model)#重置算子得分
        for k in range(pu):
            destory_id,repair_id=selectDestoryRepair(model)#随机选择算子
            model.d_select[destory_id]+=1
            model.r_select[repair_id]+=1
            reomve_list=doDestory(destory_id,model,sol)#执行destory算子
            new_sol=doRepair(repair_id,reomve_list,model,sol)#执行repair算子
            if new_sol.obj<sol.obj:
                sol=copy.deepcopy(new_sol)
                if new_sol.obj<model.best_sol.obj:
                    model.best_sol=copy.deepcopy(new_sol)
                    model.d_score[destory_id]+=model.r1
                    model.r_score[repair_id]+=model.r1
                else:
                    model.d_score[destory_id]+=model.r2
                    model.r_score[repair_id]+=model.r2
            elif new_sol.obj-sol.obj<T:
                sol=copy.deepcopy(new_sol)
                model.d_score[destory_id] += model.r3
                model.r_score[repair_id] += model.r3
            T=T*phi
            print("%s/%s:%s/%s， best obj: %s" % (ep,epochs,k,pu, model.best_sol.obj))
            history_best_obj.append(model.best_sol.obj)
        updateWeight(model)#更新算子权重

    plotObj(history_best_obj)#收敛图绘制
    plotRoutes(model)#收运路线绘制
    outPut(model)#保存最优结果
    print("random destory weight is {:.3f}\tselect is {}\tscore is {:.3f}".format(model.d_weight[0],
                                                                        model.d_history_select[0],
                                                                        model.d_history_score[0]))
    print("worse destory weight is {:.3f}\tselect is {}\tscore is {:.3f} ".format(model.d_weight[1],
                                                                        model.d_history_select[1],
                                                                        model.d_history_score[1]))
    print("random repair weight is {:.3f}\tselect is {}\tscore is {:.3f}".format(model.r_weight[0],
                                                                       model.r_history_select[0],
                                                                       model.r_history_score[0]))
    print("greedy repair weight is {:.3f}\tselect is {}\tscore is {:.3f}".format(model.r_weight[1],
                                                                       model.r_history_select[1],
                                                                       model.r_history_score[1]))
    print("regret repair weight is {:.3f}\tselect is {}\tscore is {:.3f}".format(model.r_weight[2],
                                                                       model.r_history_select[2],
                                                                       model.r_history_score[2]))

if __name__=='__main__':
    file = './cvrp.xlsx'
    run(filepath=file,rand_d_max=0.4,rand_d_min=0.1,
        worst_d_min=5,worst_d_max=20,regret_n=5,r1=30,r2=20,r3=10,rho=0.4,
        phi=0.9,epochs=100,pu=5,v_cap=80,opt_type=1)

