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
        self.action_id=None#需求节点seq_no有序排列集合，对应TSP的解
        
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
        self.demand_dict={}#车辆载重，画图用
        self.sol_list=[]#可行解集合，值类型为Sol()
        self.distance={}#网络弧距离
        self.popsize=100#蚁群规模
        self.alpha=2#信息启发式因子
        self.beta=3#期望启发式因子
        self.Q=100#信息素总量
        self.rho=0.5#信息素挥发系数
        self.tau={}#网络弧信息素
# （1）文件读取
def readXlsxFile(filepath,model):
    node_seq_no = -1
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
# （2）初始化参数
# 在初始化参数时计算网络弧距离，同时对网络弧上的信息素浓度进行初始化，这里默认为10，也可取其他值。
def initParam(model):
    for i in range(model.number_of_nodes):
        for j in range(i+1,model.number_of_nodes):
            d=math.sqrt((model.node_list[i].x_coord-model.node_list[j].x_coord)**2+
                        (model.node_list[i].y_coord-model.node_list[j].y_coord)**2)
            model.distance[i,j]=d
            model.distance[j,i]=d
            model.tau[i,j]=10
            model.tau[j,i]=10
# （3）位置更新
# 在更新蚂蚁位置时，调用 " searchNextNode "函数，依据网络弧信息素浓度及启发式信息搜索蚂蚁下一个可能访问的节点。
def movePosition(model):
    sol_list=[]
    local_sol=Sol()
    local_sol.obj=float('inf')
    for k in range(model.popsize):
        #Random ant position
        nodes_seq=[int(random.randint(0,model.number_of_nodes-1))]
        all_nodes_seq=copy.deepcopy(model.node_seq_no_list)
        all_nodes_seq.remove(nodes_seq[-1])
        #Determine the next moving position according to pheromone
        while len(all_nodes_seq)>0:
            next_node_no=searchNextNode(model,nodes_seq[-1],all_nodes_seq)
            nodes_seq.append(next_node_no)
            all_nodes_seq.remove(next_node_no)
        sol=Sol()
        sol.nodes_seq=nodes_seq
        sol.obj,sol.routes=calObj(nodes_seq,model)
        sol_list.append(sol)
        if sol.obj<local_sol.obj:
            local_sol=copy.deepcopy(sol)
    model.sol_list=copy.deepcopy(sol_list)
    if local_sol.obj<model.best_sol.obj:
        model.best_sol=copy.deepcopy(local_sol)

def searchNextNode(model,current_node_no,SE_List):
    prob=np.zeros(len(SE_List))
    for i,node_no in enumerate(SE_List):
        eta=1/model.distance[current_node_no,node_no]
        tau=model.tau[current_node_no,node_no]
        prob[i]=((eta**model.alpha)*(tau**model.beta))
    #use Roulette to determine the next node
    cumsumprob=(prob/sum(prob)).cumsum()
    cumsumprob -= np.random.rand()
    next_node_no= SE_List[list(cumsumprob > 0).index(True)]
    return next_node_no
# （4）信息素更新
# 采用蚁周模型对网络弧信息素更新，可根据可行解的nodes_seq属性（TSP问题的解）对所经过的网络弧信息素进行更新；
# 也可根据可行解的routes属性（CVRP问题的解）对所经过的网络弧信息素进行更新。
def upateTau(model):
    rho=model.rho
    for k in model.tau.keys():
        model.tau[k]=(1-rho)*model.tau[k]
    #update tau according to sol.nodes_seq(solution of TSP)
    for sol in model.sol_list:
        nodes_seq=sol.nodes_seq
        for i in range(len(nodes_seq)-1):
            from_node_no=nodes_seq[i]
            to_node_no=nodes_seq[i+1]
            model.tau[from_node_no,to_node_no]+=model.Q/sol.obj

    #update tau according to sol.routes(solution of CVRP)
    # for sol in model.sol_list:
    #     routes=sol.routes
    #     for route in routes:
    #         for i in range(len(route)-1):
    #             from_node_no=route[i]
    #             to_node_no=route[i+1]
    #             model.tau[from_node_no,to_node_no]+=model.Q/sol.obj
# （5）目标值计算
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
# （6）绘制收敛曲线
def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei']  #show chinese
    plt.rcParams['axes.unicode_minus'] = False   # Show minus sign
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
        # for node_id in route[0:-1]:
        for node_id in route:
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
def run(filepath,Q,alpha,beta,rho,epochs,v_cap,opt_type,popsize):
    """
    :param filepath:Xlsx 数据文件路径
    :param Q:信息素总量
    :param alpha:信息启发式因子
    :param beta:期望启发式因子
    :param rho:信息素挥发系数
    :param epochs:迭代次数
    :param v_cap:车载容量
    :param opt_type:优化目标类型，0：最小车辆数，1：最小行驶距离
    :param popsize:种群大小
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    model.alpha=alpha
    model.beta=beta
    model.Q=Q
    model.rho=rho
    model.popsize=popsize
    sol=Sol()
    sol.obj=float('inf')
    model.best_sol=sol
    history_best_obj = []
    readXlsxFile(filepath,model)#读取数据
    initParam(model)#初始化参数
    for ep in range(epochs):
        movePosition(model)#更新位置
        upateTau(model)#信息素更新
        history_best_obj.append(model.best_sol.obj)
        print("%s/%s， best obj: %s" % (ep,epochs, model.best_sol.obj))
    plotObj(history_best_obj)#收敛图绘制
    plotRoutes(model)#收运路线绘制
    outPut(model)#保存最优结果
if __name__=='__main__':
    file='./cvrp.xlsx'
    run(filepath=file,Q=10,alpha=1,beta=5,rho=0.1,epochs=100,v_cap=80,opt_type=1,popsize=60)
