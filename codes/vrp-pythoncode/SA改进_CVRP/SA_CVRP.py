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
# （2）初始解
def genInitialSol(node_seq):
    node_seq=copy.deepcopy(node_seq)
    random.seed(0)
    random.shuffle(node_seq)
    return node_seq
# （3）邻域生成，这里在生成邻域时直接借用TS禁忌搜索算法中所定义的三类算子。
def createActions(n):
    action_list=[]
    nswap=n//2
    # Single point exchange
    for i in range(nswap):
        action_list.append([1, i, i + nswap])
    # Two point exchange
    for i in range(0, nswap, 2):
        action_list.append([2, i, i + nswap])
    # Reverse sequence
    for i in range(0, n, 4):
        action_list.append([3, i, i + 3])
    return action_list
def doACtion(nodes_seq,action):
    nodes_seq=copy.deepcopy(nodes_seq)
    if action[0]==1:
        index_1=action[1]
        index_2=action[2]
        temporary=nodes_seq[index_1]
        nodes_seq[index_1]=nodes_seq[index_2]
        nodes_seq[index_2]=temporary
        return nodes_seq
    elif action[0]==2:
        index_1 = action[1]
        index_2 = action[2]
        temporary=[nodes_seq[index_1],nodes_seq[index_1+1]]
        nodes_seq[index_1]=nodes_seq[index_2]
        nodes_seq[index_1+1]=nodes_seq[index_2+1]
        nodes_seq[index_2]=temporary[0]
        nodes_seq[index_2+1]=temporary[1]
        return nodes_seq
    elif action[0]==3:
        index_1=action[1]
        index_2=action[2]
        nodes_seq[index_1:index_2+1]=list(reversed(nodes_seq[index_1:index_2+1]))
        return nodes_seq
# （4）目标值计算，目标值计算依赖 " splitRoutes " 函数对TSP可行解分割得到车辆行驶路线和所需车辆数
#                           " calDistance " 函数计算行驶距离。
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
    plt.rcParams['axes.unicode_minus'] = False   # Show minus sign
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.savefig('result1.png')
    plt.show()
# （6）绘制收运路线图 
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
# （7）输出结果到Excel
def outPut(model):
    work = xlsxwriter.Workbook('result.xlsx')
    worksheet = work.add_worksheet()
    worksheet.write(0, 0, 'opt_type')
    worksheet.write(1, 0, 'obj')
    if model.opt_type == 0:
        worksheet.write(0, 1, 'number of vehicles')
    else:
        worksheet.write(0, 1, 'drive distance of vehicles')
    worksheet.write(1, 1, model.best_sol.obj)
    for row, route in enumerate(model.best_sol.routes):
        worksheet.write(row + 2, 0, 'v' + str(row + 1))
        r = [str(i) for i in route]
        worksheet.write(row + 2, 1, '-'.join(r))
    work.close()
# （8）主函数，这里实现两种退火函数，当detaT>=1时，采用定步长方式退火，当0<detaT<1时采用定比例方式退火。
#            此外，外层循环由当前温度参数控制，内层循环这里由邻域动作（算子）个数控制。也可根据具体情况采用其他循环方式。
def run(filepath,T0,Tf,detaT,v_cap,opt_type):
    """
    :param filepath: Xlsx 数据文件路径
    :param T0: 初始温度
    :param Tf: 终止温度
    :param deltaT: Step or proportion of temperature drop
    :param v_cap:  车辆载重容量
    :param opt_type: 优化目标类型，0：最小车辆数，1：最小行驶距离
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    readXlsxFile(filepath,model)#读取数据
    action_list=createActions(model.number_of_nodes)#邻域生成
    history_best_obj=[]
    sol=Sol()
    sol.nodes_seq=genInitialSol(model.node_seq_no_list)#初始解
    sol.obj,sol.routes=calObj(sol.nodes_seq,model)#目标值计算
    model.best_sol=copy.deepcopy(sol)
    history_best_obj.append(sol.obj)
    Tk=T0
    nTk=len(action_list)
    while Tk>=Tf:
        for i in range(nTk):
            new_sol = Sol()
            new_sol.nodes_seq = doACtion(sol.nodes_seq, action_list[i])#邻域生成
            new_sol.obj, new_sol.routes = calObj(new_sol.nodes_seq, model)#目标值计算
            deta_f=new_sol.obj-sol.obj
            #New interpretation of acceptance criteria
            if deta_f<0 or math.exp(-deta_f/Tk)>random.random():
                sol=copy.deepcopy(new_sol)
            if sol.obj<model.best_sol.obj:
                model.best_sol=copy.deepcopy(sol)
        if detaT<1:
            Tk=Tk*detaT
        else:
            Tk = Tk - detaT
        history_best_obj.append(model.best_sol.obj)
        print("temperature：%s，local obj:%s best obj: %s" % (Tk,sol.obj,model.best_sol.obj))
    plotObj(history_best_obj)#收敛图绘制
    plotRoutes(model)#收运路线绘制
    outPut(model)#保存最优结果
if __name__=='__main__':
    file='./cvrp.xlsx'
    run(filepath=file,T0=6000,Tf=0.001,detaT=0.9,v_cap=80,opt_type=1)
