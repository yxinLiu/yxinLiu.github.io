# -*- coding: utf-8 -*-
# @Author  : Jack Hao
# @File    : SA_MDVRPTW.py
# 更多调度问题可关注微信公众号：学长带你飞
# 专注于调度相关问题：车间调度和车辆路径优化


# 内含库
import math
import random
import csv
import sys
import copy

# 第三方库
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt

#可行解类
class Sol():
    def __init__(self):
        self.obj=None#目标函数值
        self.node_id_list=[]#需求点的id列表
        self.cost_of_distance=None#距离成本
        self.cost_of_time=None#时间成本
        self.route_list=[]#车辆路径集合
        self.timetable_list=[]#节点访问时间集合

#网络节点类
class Node():
    def __init__(self):
        self.id=0#物理节点ID
        self.x_coord=0#x坐标
        self.y_cooord=0#y坐标
        self.demand=0#点需求量
        self.depot_capacity=0#基地车队规模
        self.start_time=0#最早到达时间
        self.end_time=1440#最晚到达时间
        self.service_time=0#服务时间

#存储算法参数类
class Model():
    def __init__(self):
        self.best_sol=None#最优解类
        self.demand_dict={}#需求节点
        self.depot_dict={}#车场集合
        self.depot_id_list=[]#车场节点id集合
        self.demand_id_list=[]#需求节点id集合
        self.distance_matrix={}#距离矩阵
        self.time_matrix={}#节点旅行时间矩阵
        self.number_of_demands=0#需求节点数量
        self.vehicle_cap=0#车辆容量
        self.vehicle_speed=1#车速
        self.opt_type=1#优化目标类型，0：最小旅行距离，1：最小时间成本

#1、数据读取
def readCSVFile(demand_file,depot_file,model):
    with open(demand_file,'r') as f:
        demand_reader=csv.DictReader(f)
        for row in demand_reader:
            node = Node()
            node.id = int(row['id'])
            node.x_coord = float(row['x_coord'])
            node.y_coord = float(row['y_coord'])
            node.demand = float(row['demand'])
            node.start_time=float(row['start_time'])
            node.end_time=float(row['end_time'])
            node.service_time=float(row['service_time'])
            model.demand_dict[node.id] = node
            model.demand_id_list.append(node.id)
        model.number_of_demands=len(model.demand_id_list)

    with open(depot_file, 'r') as f:
        depot_reader = csv.DictReader(f)
        for row in depot_reader:
            node = Node()
            node.id = row['id']
            node.x_coord = float(row['x_coord'])
            node.y_coord = float(row['y_coord'])
            node.depot_capacity = float(row['capacity'])
            node.start_time=float(row['start_time'])
            node.end_time=float(row['end_time'])
            model.depot_dict[node.id] = node
            model.depot_id_list.append(node.id)

#2、计算距离矩阵，时间矩阵
def calDistanceTimeMatrix(model):
    for i in range(len(model.demand_id_list)):
        from_node_id = model.demand_id_list[i]
        for j in range(i + 1, len(model.demand_id_list)):
            to_node_id = model.demand_id_list[j]
            dist = math.sqrt((model.demand_dict[from_node_id].x_coord - model.demand_dict[to_node_id].x_coord) ** 2
                             + (model.demand_dict[from_node_id].y_coord - model.demand_dict[to_node_id].y_coord) ** 2)
            model.distance_matrix[from_node_id, to_node_id] = dist
            model.distance_matrix[to_node_id, from_node_id] = dist
            model.time_matrix[from_node_id,to_node_id] = math.ceil(dist/model.vehicle_speed)
            model.time_matrix[to_node_id,from_node_id] = math.ceil(dist/model.vehicle_speed)
        for _, depot in model.depot_dict.items():
            dist = math.sqrt((model.demand_dict[from_node_id].x_coord - depot.x_coord) ** 2
                             + (model.demand_dict[from_node_id].y_coord - depot.y_coord) ** 2)
            model.distance_matrix[from_node_id, depot.id] = dist
            model.distance_matrix[depot.id, from_node_id] = dist
            model.time_matrix[from_node_id,depot.id] = math.ceil(dist/model.vehicle_speed)
            model.time_matrix[depot.id,from_node_id] = math.ceil(dist/model.vehicle_speed)

#3、目标值计算
# 适应度计算依赖" splitRoutes "函数对有序节点序列行解分割得到车辆行驶路线，同时在得到各车辆形式路线后在满足车场车队规模条件下分配最近车场，之后调用 " calTravelCost "函数确定车辆访问各路径节点的到达和离开时间点，并计算旅行距离成本和旅行时间成本。
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
        sys.exit(0)
    route.insert(0,index)
    route.append(index)
    depot_dict[index].depot_capacity=depot_dict[index].depot_capacity-1
    return route,depot_dict

def calTravelCost(route_list,model):
    timetable_list=[]
    cost_of_distance=0
    cost_of_time=0
    for route in route_list:
        timetable=[]
        for i in range(len(route)):
            if i == 0:
                depot_id=route[i]
                next_node_id=route[i+1]
                travel_time=model.time_matrix[depot_id,next_node_id]
                departure=max(0,model.demand_dict[next_node_id].start_time-travel_time)
                timetable.append((departure,departure))
            elif 1<= i <= len(route)-2:
                last_node_id=route[i-1]
                current_node_id=route[i]
                current_node = model.demand_dict[current_node_id]
                travel_time=model.time_matrix[last_node_id,current_node_id]
                arrival=max(timetable[-1][1]+travel_time,current_node.start_time)
                departure=arrival+current_node.service_time
                timetable.append((arrival,departure))
                cost_of_distance += model.distance_matrix[last_node_id, current_node_id]
                cost_of_time += model.time_matrix[last_node_id, current_node_id]+ current_node.service_time\
                                + max(current_node.start_time - timetable[-1][1] - travel_time, 0)
            else:
                last_node_id = route[i - 1]
                depot_id=route[i]
                travel_time = model.time_matrix[last_node_id,depot_id]
                departure = timetable[-1][1]+travel_time
                timetable.append((departure,departure))
                cost_of_distance +=model.distance_matrix[last_node_id,depot_id]
                cost_of_time+=model.time_matrix[last_node_id,depot_id]
        timetable_list.append(timetable)
    return timetable_list,cost_of_time,cost_of_distance

def extractRoutes(node_id_list,Pred,model):
    depot_dict=copy.deepcopy(model.depot_dict)
    route_list = []
    route = []
    label = Pred[node_id_list[0]]
    for node_id in node_id_list:
        if Pred[node_id] == label:
            route.append(node_id)
        else:
            route, depot_dict=selectDepot(route,depot_dict,model)
            route_list.append(route)
            route = [node_id]
            label = Pred[node_id]
    route, depot_dict = selectDepot(route, depot_dict, model)
    route_list.append(route)
    return route_list

def splitRoutes(node_id_list,model):
    depot=model.depot_id_list[0]
    V={id:float('inf') for id in model.demand_id_list}
    V[depot]=0
    Pred={id:depot for id in model.demand_id_list}
    for i in range(len(node_id_list)):
        n_1=node_id_list[i]
        demand=0
        departure=0
        j=i
        cost=0
        while True:
            n_2 = node_id_list[j]
            demand = demand + model.demand_dict[n_2].demand
            if n_1 == n_2:
                arrival = max(model.demand_dict[n_2].start_time,model.depot_dict[depot].start_time + model.time_matrix[depot, n_2])
                departure = arrival + model.demand_dict[n_2].service_time
                if model.opt_type == 0:
                    cost=model.distance_matrix[depot,n_2]*2
                else:
                    cost=model.time_matrix[depot,n_2]*2
            else:
                n_3=node_id_list[j-1]
                arrival= max(departure+model.time_matrix[n_3,n_2],model.demand_dict[n_2].start_time)
                departure=arrival+model.demand_dict[n_2].service_time
                if model.opt_type == 0:
                    cost=cost-model.distance_matrix[n_3,depot]+model.distance_matrix[n_3,n_2]+model.distance_matrix[n_2,depot]
                else:
                    cost=cost-model.time_matrix[n_3,depot]+model.time_matrix[n_3,n_2]\
                         +max(model.demand_dict[n_2].start_time-arrival,0)+model.time_matrix[n_2,depot]
            if demand<=model.vehicle_cap and departure<= model.demand_dict[n_2].end_time:
                if departure+model.time_matrix[n_2,depot] <= model.depot_dict[depot].end_time:
                    n_4=node_id_list[i-1] if i-1>=0 else depot
                    if V[n_4]+cost <= V[n_2]:
                        V[n_2]=V[n_4]+cost
                        Pred[n_2]=i-1
                    j=j+1
            else:
                break
            if j==len(node_id_list):
                break
    route_list= extractRoutes(node_id_list,Pred,model)
    return len(route_list),route_list

def calObj(sol,model):

    node_id_list=copy.deepcopy(sol.node_id_list)
    num_vehicle, sol.route_list = splitRoutes(node_id_list, model)
    # travel cost
    sol.timetable_list,sol.cost_of_time,sol.cost_of_distance =calTravelCost(sol.route_list,model)
    if model.opt_type == 0:
        sol.obj=sol.cost_of_distance
    else:
        sol.obj=sol.cost_of_time

#4、初始解生成
def genInitialSol(node_id_list):
    node_id_list=copy.deepcopy(node_id_list)
    random.seed(0)
    random.shuffle(node_id_list)
    return node_id_list

#5、定义邻域生成算子
def createActions(n):
    action_list=[]
    nswap=n//2
    for i in range(nswap):
        action_list.append([1,i,i+nswap])
    for i in range(0,nswap,2):
        action_list.append([2,i,i+nswap])
    for i in range(0,n,4):
        action_list.append([3,i,i+3])
    return action_list
#6、生成邻域
def doAction(node_id_list,action):
    node_id_list_=copy.deepcopy(node_id_list)
    if action[0]==1:
        index_1=action[1]
        index_2=action[2]
        node_id_list_[index_1],node_id_list_[index_2]=node_id_list_[index_2],node_id_list_[index_1]
        return node_id_list_
    elif action[0]==2:
        index_1 = action[1]
        index_2 = action[2]
        temporary=[node_id_list_[index_1],node_id_list_[index_1+1]]
        node_id_list_[index_1]=node_id_list_[index_2]
        node_id_list_[index_1+1]=node_id_list_[index_2+1]
        node_id_list_[index_2]=temporary[0]
        node_id_list_[index_2+1]=temporary[1]
        return node_id_list_
    elif action[0]==3:
        index_1=action[1]
        index_2=action[2]
        node_id_list_[index_1:index_2+1]=list(reversed(node_id_list_[index_1:index_2+1]))
        return node_id_list_

#7、绘制收敛曲线
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

#8、输出结果
def outPut(model):
    work=xlsxwriter.Workbook('result.xlsx')
    worksheet=work.add_worksheet()
    worksheet.write(0, 0, 'cost_of_time')
    worksheet.write(0, 1, 'cost_of_distance')
    worksheet.write(0, 2, 'opt_type')
    worksheet.write(0, 3, 'obj')
    worksheet.write(1,0,model.best_sol.cost_of_time)
    worksheet.write(1,1,model.best_sol.cost_of_distance)
    worksheet.write(1,2,model.opt_type)
    worksheet.write(1,3,model.best_sol.obj)
    worksheet.write(2,0,'vehicleID')
    worksheet.write(2,1,'route')
    worksheet.write(2,2,'timetable')
    for row,route in enumerate(model.best_sol.route_list):
        worksheet.write(row+3,0,'v'+str(row+1))
        r=[str(i)for i in route]
        worksheet.write(row+3,1, '-'.join(r))
        r=[str(i)for i in model.best_sol.timetable_list[row]]
        worksheet.write(row+3,2, '-'.join(r))
    work.close()

#9、绘制车辆路线
def plotRoutes(model):
    for route in model.best_sol.route_list:
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

#10、主函数
def run(demand_file,depot_file,T0,Tf,deltaT,v_cap,opt_type):
    """
    :param demand_file: Demand文件路径
    :param depot_file: Depot文件路径
    :param T0: 初始温度
    :param Tf: 终止温度
    :param deltaT: 温度下降步长或下降比例
    :param v_cap: 车辆容量
    :param opt_type: 优化类型:0:最小化旅行距离,1:最小化行驶时间
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    readCSVFile(demand_file,depot_file,model)
    calDistanceTimeMatrix(model)
    action_list=createActions(len(model.demand_id_list))
    history_best_obj=[]
    sol=Sol()
    sol.node_id_list=genInitialSol(model.demand_id_list)
    calObj(sol,model)
    model.best_sol=copy.deepcopy(sol)
    history_best_obj.append(sol.obj)
    Tk=T0
    nTk=len(action_list)
    while Tk>=Tf:
        for i in range(nTk):
            new_sol = Sol()
            new_sol.node_id_list = doAction(sol.node_id_list, action_list[i])
            calObj(new_sol, model)
            detla_f=new_sol.obj-sol.obj
            if detla_f<0 or math.exp(-detla_f/Tk)>random.random():
                sol=copy.deepcopy(new_sol)
            if sol.obj<model.best_sol.obj:
                model.best_sol=copy.deepcopy(sol)
        if deltaT<1:
            Tk=Tk*deltaT
        else:
            Tk = Tk - deltaT
        history_best_obj.append(model.best_sol.obj)
        print("Current temperature：%s，local obj:%s best obj: %s" % (Tk,sol.obj,model.best_sol.obj))
    plotObj(history_best_obj)
    plotRoutes(model)
    outPut(model)

if __name__=='__main__':
    demand_file='./demand.csv'
    depot_file='./depot.csv'
    run(demand_file=demand_file,depot_file=depot_file,T0=6000,Tf=0.001,deltaT=0.9,v_cap=80,opt_type=0)