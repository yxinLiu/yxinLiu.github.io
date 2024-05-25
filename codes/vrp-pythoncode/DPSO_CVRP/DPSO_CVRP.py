import pandas as pd
import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt

# Sol()类表示一个可行解
class Sol():
    def __init__(self):
        self.nodes_seq=None#需求节点seq_no有序排列集合，对应TSP的解
        self.obj=None#优化目标值
        self.routes=None#车辆路径集合，对应CVRP的解

#Node()类，表示一个网络节点
class Node():
    def __init__(self):
        self.id=0
        self.name=''
        self.seq_no=0#节点映射id，基地节点为-1，需求节点从0编号
        self.x_coord=0#x坐标
        self.y_coord=0#y坐标
        self.demand=0#需求量
        
#Model()类，存储算法参数
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
        self.pl=[]#可行解历史最优位置
        self.pg=None#全局最优解历史最优位置
        self.v=[]#可行解更新速度
        self.Vmax = 5#最大速度
        self.w=0.8#惯性权重
        self.c1=2#学习因子
        self.c2=2#学习因子
        self.demand_dict={}

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

#（2）初始化种群
def genInitialSol(model,popsize):
    node_seq=copy.deepcopy(model.node_seq_no_list)
    best_sol=Sol()
    best_sol.obj=float('inf')
    for i in range(popsize):
        seed = int(random.randint(0, 10))
        random.seed(seed)
        random.shuffle(node_seq)
        sol=Sol()
        sol.nodes_seq= copy.deepcopy(node_seq)
        sol.obj,sol.routes=calObj(sol.nodes_seq,model)
        model.sol_list.append(sol)
        model.v.append([model.Vmax]*model.number_of_nodes)
        model.pl.append(sol.nodes_seq)
        if sol.obj<best_sol.obj:
            best_sol=copy.deepcopy(sol)
    model.best_sol=best_sol
    model.pg=best_sol.nodes_seq

#（3）速度及位置更新
# 粒子速度更新公式采取标准形式。在更新粒子位置时需要注意两个问题：
# 1）粒子位置分量的值为应整数，不能超出需求节点seq_no范围，即[0, number_of_nodes-1]；
# 2）粒子位置分量的值具有唯一性，且刚好覆盖需求节点的seq_no值。满足以上条件时，更新后的粒子才是TSP、CVRP的可行解。
# 为了保证这两个条件，本文采取以下策略：
# 在更新粒子位置时，首先，将更新后粒子位置分量的值转换为int型，即向下取整，且不超过最大seq_no；
# 其次，统计未被覆盖的需求节点的seq_no，然后依次赋值给粒子位置分量中小于0或重复出现的分量。
# 为避免冗余循环操作，这里在更新粒子位置后，随即更新粒子（可行解）的属性。

def updatePosition(model):
    w=model.w
    c1=model.c1
    c2=model.c2
    pg = model.pg
    for id,sol in enumerate(model.sol_list):
        x=sol.nodes_seq
        v=model.v[id]
        pl=model.pl[id]
        r1=random.random()
        r2=random.random()
        new_v=[]
        for i in range(model.number_of_nodes):
            v_=w*v[i]+c1*r1*(pl[i]-x[i])+c2*r2*(pg[i]-x[i])
            if v_>0:
                new_v.append(min(v_,model.Vmax))
            else:
                new_v.append(max(v_,-model.Vmax))
        new_x=[min(int(x[i]+new_v[i]),model.number_of_nodes-1) for i in range(model.number_of_nodes) ]
        new_x=adjustRoutes(new_x,model)
        model.v[id]=new_v

        new_x_obj,new_x_routes=calObj(new_x,model)
        if new_x_obj<sol.obj:
            model.pl[id]=copy.deepcopy(new_x)
        if new_x_obj<model.best_sol.obj:
            model.best_sol.obj=copy.deepcopy(new_x_obj)
            model.best_sol.nodes_seq=copy.deepcopy(new_x)
            model.best_sol.routes=copy.deepcopy(new_x_routes)
            model.pg=copy.deepcopy(new_x)
        model.sol_list[id].nodes_seq = copy.deepcopy(new_x)
        model.sol_list[id].obj = copy.deepcopy(new_x_obj)
        model.sol_list[id].routes = copy.deepcopy(new_x_routes)

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

#（4）计算目标函数：
# 目标值计算依赖 " splitRoutes " 函数对TSP可行解分割得到车辆行驶路线和所需车辆数，
#  " calDistance " 函数计算行驶距离。
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

# （5）绘制收敛曲线
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

def run(filepath,epochs,popsize,Vmax,v_cap,opt_type,w,c1,c2):
    """
    :param filepath: Xlsx file path文件路径
    :param epochs: Iterations迭代次数
    :param popsize: Population size种群规模
    :param v_cap: Vehicle capacity车辆容量
    :param Vmax :Max 最大速度
    :param opt_type: 优化目标类型，0：最小车辆数，1：最小行驶距离
    :param w: 惯性权重
    :param c1:学习因子
    :param c2:学习因子
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    model.w=w
    model.c1=c1
    model.c2=c2
    model.Vmax = Vmax
    readXlsxFile(filepath,model)
    history_best_obj=[]
    genInitialSol(model,popsize)
    history_best_obj.append(model.best_sol.obj)
    for ep in range(epochs):
        updatePosition(model)
        history_best_obj.append(model.best_sol.obj)
        print("%s/%s: best obj: %s"%(ep,epochs,model.best_sol.obj))
    plotObj(history_best_obj)
    plotRoutes(model)
    outPut(model)

if __name__=='__main__':
    file='./cvrp.xlsx'
    run(filepath=file,epochs=100,popsize=150,Vmax=2,v_cap=80,opt_type=1,w=0.9,c1=1,c2=5)




