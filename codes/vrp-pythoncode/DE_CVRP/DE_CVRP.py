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
        self.vehicle_cap=0#车辆容量
        self.sol_list=[]#可行解集合，值类型为Sol()
        self.Cr=0.5#交叉概率
        self.F=0.5#缩放因子
        self.popsize=4*self.number_of_nodes#种群规模
        self.demand_dict={}#车辆载重，画图用
# （1）文件读取
def readXlsxFile(filepath,model):
    # It is recommended that the vehicle depot data be placed in the first line of xlsx file
    node_seq_no =-1 #the depot node seq_no is -1,and demand node seq_no is 0,1,2,...
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
# （2）初始种群
def genInitialSol(model):
    nodes_seq=copy.deepcopy(model.node_seq_no_list)
    for i in range(model.popsize):
        seed=int(random.randint(0,10))
        random.seed(seed)
        random.shuffle(nodes_seq)
        sol=Sol()
        sol.nodes_seq=copy.deepcopy(nodes_seq)
        sol.obj,sol.routes=calObj(nodes_seq,model)
        model.sol_list.append(sol)
        if sol.obj<model.best_sol.obj:
            model.best_sol=copy.deepcopy(sol)
# （3）目标值计算
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
# （4）突变操作
#Differential mutation; mutation strategies: DE/rand/1/bin
def muSol(model,v1):
    x1=model.sol_list[v1].nodes_seq
    while True:
        v2=random.randint(0,model.number_of_nodes-1)
        if v2!=v1:
            break
    while True:
        v3=random.randint(0,model.number_of_nodes-1)
        if v3!=v2 and v3!=v1:
            break
    x2=model.sol_list[v2].nodes_seq
    x3=model.sol_list[v3].nodes_seq
    mu_x=[min(int(x1[i]+model.F*(x2[i]-x3[i])),model.number_of_nodes-1) for i in range(model.number_of_nodes) ]
    return mu_x
# （5）交叉操作
#在交叉完成后，为保证解的可行性还应调用 " adjustRoutes" 函数进行适当调整。
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
#Differential Crossover
def crossSol(model,vx,vy):
    cro_x=[]
    for i in range(model.number_of_nodes):
        if random.random()<model.Cr:
            cro_x.append(vy[i])
        else:
            cro_x.append(vx[i])
    cro_x=adjustRoutes(cro_x,model)
    return cro_x
# （6）绘制收敛曲线
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
# （7）绘制收运路线图 
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
# （8）输出结果到Excel
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
# （9）主函数
#根据经验设定种群规模为问题维度的4倍，主函数设定两层循环，
# 外层循环由epochs参数控制，内层循环由种群规模控制。
def run(filepath,epochs,Cr,F,popsize,v_cap,opt_type):
    """
    :param filepath: Xlsx 数据文件路径
    :param epochs: 迭代次数
    :param Cr: 交叉概率
    :param F:  缩放因子
    :param popsize: 种群规模
    :param v_cap:车载容量
    :param opt_type: 优化目标类型，0：最小车辆数，1：最小行驶距离
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.Cr=Cr
    model.F=F
    model.popsize=popsize
    model.opt_type=opt_type

    readXlsxFile(filepath,model)
    best_sol = Sol()
    best_sol.obj = float('inf')
    model.best_sol = best_sol
    genInitialSol(model)
    history_best_obj = []
    for ep in range(epochs):
        for i in range(popsize):
            v1=random.randint(0,model.number_of_nodes-1)
            sol=model.sol_list[v1]
            mu_x=muSol(model,v1)
            u=crossSol(model,sol.nodes_seq,mu_x)
            u_obj,u_routes=calObj(u,model)
            if u_obj<=sol.obj:
                sol.nodes_seq=copy.deepcopy(u)
                sol.obj=copy.deepcopy(u_obj)
                sol.routes=copy.deepcopy(u_routes)
                if sol.obj<model.best_sol.obj:
                    model.best_sol=copy.deepcopy(sol)
            history_best_obj.append(model.best_sol.obj)
        print("%s/%s， best obj: %s" % (ep, epochs, model.best_sol.obj))
    plotObj(history_best_obj)#收敛图绘制
    plotRoutes(model)#收运路线绘制
    outPut(model)#保存最优结果

if __name__ == '__main__':
    file = './cvrp.xlsx'
    run(filepath=file, epochs=10, Cr=0.5,F=0.5, popsize=400, v_cap=80, opt_type=1)

