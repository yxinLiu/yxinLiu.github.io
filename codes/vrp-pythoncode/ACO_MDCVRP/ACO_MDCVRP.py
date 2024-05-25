# 微信公众号：学长带你飞
# 主要更新方向：1、车辆路径问题求解算法
#              2、学术写作技巧
#              3、读书感悟
# @Author  : Jack Hao
# @File    : TS_MDCVRP.py
# @Software: PyCharm

# 内嵌函数
import math
import random
import csv
import copy

# 第三方库
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt

# 可行解类
class Sol():
    def __init__(self):
        self.node_id_list=None#需求节点id有序排列集合，对应TSP的解
        self.obj=None#优化目标值
        self.fit=None#适应度值
        self.routes=None#车辆路径集合，对应CVRP的解

# 表示一个网络点类
class Node():
    def __init__(self):
        self.id=0#节点id
        self.x_coord=0#节点x轴坐标
        self.y_coord=0#节点y轴坐标
        self.demand=0#需求量
        self.depot_capacity=15#车场车队规模

# 存储算法参数类
class Model():
    def __init__(self):
        self.best_sol=None#全局最优解
        self.demand_dict = {}#需求节点集合（字典）
        self.depot_dict = {}#车场节点集合（字典）
        self.demand_id_list = []#需求节点id集合
        self.opt_type=0#优化目标类型，0：最小车辆数，1：最小行驶距离
        self.vehicle_cap=0#车辆容量
        self.distance_matrix={}#网络弧距离
        self.popsize=100#蚁群规模
        self.alpha=2#信息启发式因子
        self.beta=3#期望启发式因子
        self.Q=100#信息素总量
        self.tau0=10#路径初始信息素
        self.rho=0.5#信息素挥发系数
        self.tau={}#网络弧信息素

# 1、数据读取：车场数据，网络点数据
def readCsvFile(demand_file,depot_file,model):
    with open(demand_file,'r') as f:
        demand_reader=csv.DictReader(f)
        for row in demand_reader:
            node = Node()
            node.id = int(row['id'])
            node.x_coord = float(row['x_coord'])
            node.y_coord = float(row['y_coord'])
            node.demand = float(row['demand'])
            model.demand_dict[node.id] = node
            model.demand_id_list.append(node.id)

    with open(depot_file,'r') as f:
        depot_reader=csv.DictReader(f)
        for row in depot_reader:
            node = Node()
            node.id = row['id']
            node.x_coord=float(row['x_coord'])
            node.y_coord=float(row['y_coord'])
            node.depot_capacity=float(row['capacity'])
            model.depot_dict[node.id] = node

# 2、计算距离矩阵、初始化路径信息素
def initDistanceTau(model):
    for i in range(len(model.demand_id_list)):
        from_node_id = model.demand_id_list[i]
        for j in range(i+1,len(model.demand_id_list)):
            to_node_id=model.demand_id_list[j]
            dist=math.sqrt( (model.demand_dict[from_node_id].x_coord-model.demand_dict[to_node_id].x_coord)**2
                            +(model.demand_dict[from_node_id].y_coord-model.demand_dict[to_node_id].y_coord)**2)
            model.distance_matrix[from_node_id,to_node_id]=dist
            model.distance_matrix[to_node_id,from_node_id]=dist
            model.tau[from_node_id,to_node_id]=model.tau0
            model.tau[to_node_id,from_node_id]=model.tau0
        for _,depot in model.depot_dict.items():
            dist = math.sqrt((model.demand_dict[from_node_id].x_coord - depot.x_coord) ** 2
                             + (model.demand_dict[from_node_id].y_coord -depot.y_coord)**2)
            model.distance_matrix[from_node_id, depot.id] = dist
            model.distance_matrix[depot.id, from_node_id] = dist

# 3、目标值计算
"""
适应度计算依赖
" splitRoutes “函数对TSP可行解分割得到车辆行驶路线和所需车辆数，
在得到各车辆行驶路线后调用
” selectDepot “函数，在满足车场车队规模条件下分配最近车场，
” calDistance "函数计算行驶距离。
"""
def selectDepot(route,depot_dict,model):
    min_in_out_distance=float('inf')
    index=None
    for _,depot in depot_dict.items():
        if depot.depot_capacity>0:
            in_out_distance=model.distance_matrix[depot.id,route[0]]+model.distance_matrix[route[-1],depot.id]
            if in_out_distance<min_in_out_distance:
                index=depot.id
                min_in_out_distance=in_out_distance
    if index is None:
        print("there is no vehicle to dispatch")
    route.insert(0,index)
    route.append(index)
    depot_dict[index].depot_capacity=depot_dict[index].depot_capacity-1
    return route,depot_dict

def splitRoutes(node_id_list,model):
    num_vehicle = 0
    vehicle_routes = []
    route = []
    remained_cap = model.vehicle_cap
    depot_dict=copy.deepcopy(model.depot_dict)
    for node_id in node_id_list:
        if remained_cap - model.demand_dict[node_id].demand >= 0:
            route.append(node_id)
            remained_cap = remained_cap - model.demand_dict[node_id].demand
        else:
            route,depot_dict=selectDepot(route,depot_dict,model)
            vehicle_routes.append(route)
            route = [node_id]
            num_vehicle = num_vehicle + 1
            remained_cap =model.vehicle_cap - model.demand_dict[node_id].demand

    route, depot_dict = selectDepot(route, depot_dict, model)
    vehicle_routes.append(route)

    return num_vehicle,vehicle_routes

def calRouteDistance(route,model):
    distance=0
    for i in range(len(route)-1):
        from_node=route[i]
        to_node=route[i+1]
        distance +=model.distance_matrix[from_node,to_node]
    return distance

def calObj(node_id_list,model):
    num_vehicle, vehicle_routes = splitRoutes(node_id_list, model)
    if model.opt_type==0:
        return num_vehicle,vehicle_routes
    else:
        distance = 0
        for route in vehicle_routes:
            distance += calRouteDistance(route, model)
        return distance,vehicle_routes


# 4、位置更新
"""
在更新蚂蚁位置时，调用 " searchNextNode "函数，
依据网络弧信息素浓度及启发式信息搜索蚂蚁下一个可能访问的节点。
"""
def movePosition(model):
    sol_list=[]
    local_sol=Sol()
    local_sol.obj=float('inf')
    for k in range(model.popsize):
        #随机初始化蚂蚁为止
        nodes_id=[int(random.randint(0,len(model.demand_id_list)-1))]
        all_nodes_id=copy.deepcopy(model.demand_id_list)
        all_nodes_id.remove(nodes_id[-1])
        #确定下一个访问节点
        while len(all_nodes_id)>0:
            next_node_no=searchNextNode(model,nodes_id[-1],all_nodes_id)
            nodes_id.append(next_node_no)
            all_nodes_id.remove(next_node_no)
        sol=Sol()
        sol.node_id_list=nodes_id
        sol.obj,sol.routes=calObj(nodes_id,model)
        sol_list.append(sol)
        if sol.obj<local_sol.obj:
            local_sol=copy.deepcopy(sol)
    model.sol_list=copy.deepcopy(sol_list)
    if local_sol.obj<model.best_sol.obj:
        model.best_sol=copy.deepcopy(local_sol)

def searchNextNode(model,current_node_id,SE_List):
    prob=np.zeros(len(SE_List))
    for i,node_id in enumerate(SE_List):
        eta=1/model.distance_matrix[current_node_id,node_id]
        tau=model.tau[current_node_id,node_id]
        prob[i]=((eta**model.alpha)*(tau**model.beta))
    #采用轮盘法选择下一个访问节点
    cumsumprob=(prob/sum(prob)).cumsum()
    cumsumprob -= np.random.rand()
    next_node_id= SE_List[list(cumsumprob > 0).index(True)]
    return next_node_id

# 5、信息素更新
"""
这里采用蚁周模型对网络弧信息素进行更新，在具体更新时可根据
可行解的node_id_list属性（TSP问题的解）对所经过的网络弧信息素进行更新。
"""
def upateTau(model):
    rho=model.rho
    for k in model.tau.keys():
        model.tau[k]=(1-rho)*model.tau[k]
    #根据解的node_id_list属性更新路径信息素（TSP问题的解）
    for sol in model.sol_list:
        nodes_id=sol.node_id_list
        for i in range(len(nodes_id)-1):
            from_node_id=nodes_id[i]
            to_node_id=nodes_id[i+1]
            model.tau[from_node_id,to_node_id]+=model.Q/sol.obj

# 6、绘制收敛曲线
def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #显示中文
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.savefig('result1.png')
    plt.show()

# 7、输出结果到excel
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

# 8、绘制车辆路线
def plotRoutes(model):
    for route in model.best_sol.routes:
        x_coord=[model.depot_dict[route[0]].x_coord]
        y_coord=[model.depot_dict[route[0]].y_coord]
        for node_id in route[1:-1]:
            x_coord.append(model.demand_dict[node_id].x_coord)
            y_coord.append(model.demand_dict[node_id].y_coord)
        x_coord.append(model.depot_dict[route[-1]].x_coord)
        y_coord.append(model.depot_dict[route[-1]].y_coord)
        plt.grid()
        if route[0]=='d1':
            plt.plot(x_coord,y_coord,marker='o',color='black',linewidth=0.5,markersize=5)
        elif route[0]=='d2':
            plt.plot(x_coord,y_coord,marker='o',color='orange',linewidth=0.5,markersize=5)
        else:
            plt.plot(x_coord,y_coord,marker='o',color='b',linewidth=0.5,markersize=5)
    plt.xlabel('x_coord')
    plt.ylabel('y_coord')
    plt.savefig('result2.png')
    plt.show()

# 9、主函数
def run(demand_file,depot_file,Q,tau0,alpha,beta,rho,epochs,v_cap,opt_type,popsize):
    """
    :param demand_file: 需求点文件路径
    :param depot_file: 车场数据文件路径
    :param Q: 总信息素
    :param tau0: 路径初始信息素
    :param alpha: 信息启发式因子
    :param beta: 期望启发式因子
    :param rho: 信息素挥发系数
    :param epochs: 迭代次数
    :param v_cap: 车辆容量
    :param opt_type:优化目标类型，0：最小车辆数，1：最小行驶距离
    :param popsize:蚁群规模
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    model.alpha=alpha
    model.beta=beta
    model.Q=Q
    model.tau0=tau0
    model.rho=rho
    model.popsize=popsize
    sol=Sol()
    sol.obj=float('inf')
    model.best_sol=sol
    history_best_obj = []
    readCsvFile(demand_file,depot_file,model)
    initDistanceTau(model)
    for ep in range(epochs):
        movePosition(model)
        upateTau(model)
        history_best_obj.append(model.best_sol.obj)
        print("%s/%s， best obj: %s" % (ep,epochs, model.best_sol.obj))
    plotObj(history_best_obj)
    plotRoutes(model)
    outPut(model)

if __name__=='__main__':
    demand_file = './demand.csv'
    depot_file = './depot.csv'
    run(demand_file,depot_file,Q=10,tau0=10,alpha=1,beta=5,rho=0.1,epochs=100,v_cap=60,opt_type=1,popsize=60)



