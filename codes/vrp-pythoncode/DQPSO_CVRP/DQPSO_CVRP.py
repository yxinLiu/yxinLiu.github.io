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
        self.sol_list=[]#可行解集合，值类型为Sol()
        self.best_sol=None#全局最优解，值类型为Sol()
        self.node_list=[]#节点集合，值类型为Node()
        self.node_seq_no_list=[]#节点映射id集合
        self.depot=None#车辆基地，值类型为Node()
        self.number_of_nodes=0#需求节点数量
        self.opt_type=0#优化目标类型，0：最小车辆数，1：最小行驶距离
        self.vehicle_cap=0#车辆容量
        self.popsize=100#种群规模
        self.pl=[]#个体历史最优位置
        self.pg=None#群体历史最优位置
        self.mg=None#群体历史平均最优位置
        self.alpha=1.0#扩张-收缩因子
        self.demand_dict={}#车辆载重，画图用
#(1)文件读取
def readXlsxFile(filepath,model):
    # It is recommended that the vehicle depot data be placed in the first line of xlsx file
    node_seq_no = -1 #the depot node seq_no is -1,and demand node seq_no is 0,1,2,...
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

#(2)初始化种群
def genInitialSol(model):
    node_seq=copy.deepcopy(model.node_seq_no_list)
    best_sol=Sol()
    best_sol.obj=float('inf')
    mg=[0]*model.number_of_nodes
    for i in range(model.popsize):
        seed = int(random.randint(0, 10))
        random.seed(seed)
        random.shuffle(node_seq)
        sol=Sol()
        sol.nodes_seq= copy.deepcopy(node_seq)
        sol.obj,sol.routes=calObj(sol.nodes_seq,model)
        model.sol_list.append(sol)
        #init the optimal position of each particle
        model.pl.append(sol.nodes_seq)
        #init the average optimal position of particle population
        mg=[mg[k]+node_seq[k]/model.popsize for k in range(model.number_of_nodes)]
        #init the optimal position of particle population
        if sol.obj<best_sol.obj:
            best_sol=copy.deepcopy(sol)
    model.best_sol=best_sol
    model.pg=best_sol.nodes_seq
    model.mg=mg

#(3)首先将粒子看做连续空间中的点进行位置更新，然后对位置分量取整离散化。
# 在具体操作时需要注意两个问题：1）粒子位置分量的值为应整数，不能超出需求节点seq_no范围，即[0, number_of_nodes-1]；
# 2）粒子位置分量的值具有唯一性，且刚好覆盖需求节点的seq_no值。满足以上条件时，更新后的粒子才是TSP、CVRP的可行解。
#位置更新，这里采用与DPSO算法相同的处理策略。
def adjustRoutes(nodes_seq,model):
    all_node_list=copy.deepcopy(model.node_seq_no_list)
    repeat_node=[]
    for id,node_no in enumerate(nodes_seq):
        if node_no in all_node_list:
            all_node_list.remove(node_no)
        else:
            repeat_node.append(id)
    for i in range(len(repeat_node)):
        nodes_seq[repeat_node[i]]=all_node_list[i]
    return nodes_seq
def updatePosition(model):
    alpha=model.alpha
    pg=model.pg
    mg=model.mg
    mg_=[0]*model.number_of_nodes  #update optimal position of each particle for next iteration
    for id, sol in enumerate(model.sol_list):
        x=sol.nodes_seq
        pl = model.pl[id]
        pi=[]
        for k in range(model.number_of_nodes): #calculate pi(ep+1)
            phi = random.random()
            pi.append(phi*pl[k]+(1-phi)*pg[k])
        #calculate x(ep+1)
        if random.random()<=0.5:
            X=[min(int(pi[k]+alpha*abs(mg[k]-x[k])*math.log(1/random.random())),model.number_of_nodes-1)
               for k in range(model.number_of_nodes)]
        else:
            X=[min(int(pi[k]-alpha*abs(mg[k]-x[k])*math.log(1/random.random())),model.number_of_nodes-1)
               for k in range(model.number_of_nodes)]

        X= adjustRoutes(X, model)
        X_obj, X_routes = calObj(X,model)
        # update pl
        if X_obj < sol.obj:
            model.pl[id] = copy.deepcopy(X)
        # update pg,best_sol
        if X_obj < model.best_sol.obj:
            model.best_sol.obj = copy.deepcopy(X_obj)
            model.best_sol.nodes_seq = copy.deepcopy(X)
            model.best_sol.routes = copy.deepcopy(X_routes)
            model.pg = copy.deepcopy(X)
        mg_ = [mg_[k] + model.pl[id][k] / model.popsize for k in range(model.number_of_nodes)]
        model.sol_list[id].nodes_seq = copy.deepcopy(X)
        model.sol_list[id].obj = copy.deepcopy(X_obj)
        model.sol_list[id].routes = copy.deepcopy(X_routes)
    # update mg
    model.mg=copy.deepcopy(mg_)

#(4)目标值计算目标值计算依赖 " splitRoutes " 函数对TSP可行解分割得到车辆行驶路线和所需车辆数，
# " calDistance " 函数计算行驶距离。
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
        from_node=model.node_list[route[i]]
        to_node=model.node_list[route[i+1]]
        distance+=math.sqrt((from_node.x_coord-to_node.x_coord)**2+(from_node.y_coord-to_node.y_coord)**2)
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
#(5)绘制收敛曲线,路径规划图
def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #show chinese
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.savefig('result1.png')
    plt.show()
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
#(6)输出结果
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
#(7)主函数
def run(filepath,epochs,popsize,alpha,v_cap,opt_type):
    """
    :param filepath: 数据路径
    :type str
    :param epochs:迭代次数
    :type int
    :param popsize:种群大小
    :type int
    :param alpha:控制参数,(0,1]
    :type float,
    :param v_cap:车辆载重
    :type float
    :param opt_type:优化目标类型，0：最小车辆数，1：最小行驶距离
    :type int,0 or 1
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    model.alpha=alpha
    model.popsize=popsize
    readXlsxFile(filepath,model)#数据读取
    history_best_obj=[]
    genInitialSol(model)#算法初始化
    history_best_obj.append(model.best_sol.obj)
    for ep in range(epochs):
        updatePosition(model)#粒子位置和速度更新
        history_best_obj.append(model.best_sol.obj)
        print("%s/%s: best obj: %s"%(ep,epochs,model.best_sol.obj))
    plotObj(history_best_obj)#收敛图绘制
    plotRoutes(model)#收运路线绘制
    outPut(model)#保存最优结果
if __name__=='__main__':
    file='./cvrp.xlsx'
    run(filepath=file,epochs=100,popsize=150,alpha=0.8,v_cap=80,opt_type=1)
