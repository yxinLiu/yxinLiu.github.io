#内置函数
import math
import random
import copy #复制字典或列表

#第三方库
import numpy as np #用于数组计算，通常与matplotlib同时使用，代替matlab
import pandas as pd
import xlsxwriter #创建xlsx文件
import matplotlib.pyplot as plt #绘图库

# Sol()类，表示一个可行解
class Sol():
    def __init__(self):
        self.nodes_seq=None #需求节点seq_no有序排列集合，对应TSP的解
        self.obj=None #优化目标值
        self.fit=None #解的适应度
        self.routes=None #车辆路径集合，对应CVRP的解
# Node()类，表示一个网络节点
class Node():
    def __init__(self):
        self.id=0 #节点id
        self.name='' #节点名称
        self.seq_no=0 #节点映射id，基地节点为-1，需求节点从0编号
        self.x_coord=0 #节点x坐标
        self.y_coord=0 #节点y坐标
        self.demand=0 #节点需求
# Model()类，存储算法参数
class Model():
    def __init__(self):
        self.best_sol=None #全局最优解，值类型为Sol()
        self.node_list=[] #节点集合，值类型为Node()
        self.node_seq_no_list=[] #节点映射id集合
        self.depot=None #车辆基地，值类型为Node()
        self.number_of_nodes=0 #需求节点数量
        self.opt_type=0 #优化目标类型，0：最小车辆数，1：最小行驶距离
        self.vehicle_cap=0 #车辆容量
        self.demand_dict={} #车辆载重，画图用
        self.sol_list=[] #种群，值类型为Sol()
        self.pc=0.5#交叉概率
        self.pm=0.2#突变概率
        self.n_select=80#优良个体选择数量
        self.popsize=100#种群规模
# （1）文件读取
def readXlsxFile(filepath,model):
    #仓库节点最好放在第一行
    node_seq_no =-1 #仓库节点为-1
    df = pd.read_excel(filepath)
    #导入文件内容
    for i in range(df.shape[0]):
        node=Node()
        node.id=node_seq_no
        node.seq_no=node_seq_no
        node.x_coord= df['x_coord'][i]
        node.y_coord= df['y_coord'][i]
        node.demand=df['demand'][i]
        model.demand_dict[node.id] = node
        #需求为0的节点即为仓库
        if df['demand'][i] == 0:
            model.depot=node
        #其他的列入节点列表里
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
#（2）初始解生成
def genInitialSol(model):
    nodes_seq=copy.deepcopy(model.node_seq_no_list)
    #生成100个初始解
    for i in range(model.popsize):
        #随机次数随机排序节点顺序
        seed=int(random.randint(0,10))
        random.seed(seed)
        random.shuffle(nodes_seq)
        sol=Sol()
        sol.nodes_seq=copy.deepcopy(nodes_seq)
        model.sol_list.append(sol)
#（3）适应度计算
    #将TSP路径分割
def splitRoutes(nodes_seq,model):
    num_vehicle = 0
    vehicle_routes = []
    route = []
    remained_cap = model.vehicle_cap #车辆剩余容量
    for node_no in nodes_seq:
        if remained_cap - model.node_list[node_no].demand >= 0:
            #剩余容量够
            route.append(node_no)
            remained_cap = remained_cap - model.node_list[node_no].demand
        else:
            #剩余容量不够，换另一辆车从仓库出发执行
            vehicle_routes.append(route)
            route = [node_no]
            num_vehicle = num_vehicle + 1
            remained_cap =model.vehicle_cap - model.node_list[node_no].demand
    vehicle_routes.append(route)
    return num_vehicle,vehicle_routes

#总的行驶距离
def calDistance(route,model):
    distance=0
    depot=model.depot
    for i in range(len(route)-1):
        #算两点之间的距离
        from_node=model.node_list[route[i]]
        to_node=model.node_list[route[i+1]]
        distance+=math.sqrt((from_node.x_coord-to_node.x_coord)**2+(from_node.y_coord-to_node.y_coord)**2)
    first_node=model.node_list[route[0]]
    last_node=model.node_list[route[-1]]
    distance+=math.sqrt((depot.x_coord-first_node.x_coord)**2+(depot.y_coord-first_node.y_coord)**2)
    distance+=math.sqrt((depot.x_coord-last_node.x_coord)**2+(depot.y_coord - last_node.y_coord)**2)
    return distance

def calFit(model):
    #calculate fit value：fit=Objmax-obj
    Objmax=-float('inf')
    best_sol=Sol()#record the local best solution
    best_sol.obj=float('inf')
    #计算目标函数
    for sol in model.sol_list:
        nodes_seq=sol.nodes_seq
        num_vehicle, vehicle_routes = splitRoutes(nodes_seq, model) #TSP分割后
        #优化目标为车辆数目
        if model.opt_type==0:
            sol.obj=num_vehicle
            sol.routes=vehicle_routes
            if sol.obj>Objmax:
                Objmax=sol.obj
            if sol.obj<best_sol.obj:
                best_sol=copy.deepcopy(sol)
        else:
            #优化目标为距离
            distance=0
            for route in vehicle_routes:
                distance+=calDistance(route,model)
            sol.obj=distance
            sol.routes=vehicle_routes
            if sol.obj>Objmax:
                Objmax=sol.obj
            if sol.obj < best_sol.obj:
                best_sol = copy.deepcopy(sol)
    #calculate fit value
    for sol in model.sol_list:
        sol.fit=Objmax-sol.obj
    #update the global best solution
    if best_sol.obj<model.best_sol.obj:
        model.best_sol=best_sol

# （4）优良个体选择
#采用二元锦标赛法进行优良个体选择。
def selectSol(model):
    sol_list=copy.deepcopy(model.sol_list)
    model.sol_list=[]
    #选择80个优良个体
    for i in range(model.n_select):
        f1_index=random.randint(0,len(sol_list)-1)
        f2_index=random.randint(0,len(sol_list)-1)
        f1_fit=sol_list[f1_index].fit
        f2_fit=sol_list[f2_index].fit
        #选80次，每次两两对比
        if f1_fit<f2_fit:
            model.sol_list.append(sol_list[f2_index]) #fit越大，越优，因为fit=Objmax-obj
        else:
            model.sol_list.append(sol_list[f1_index])

# （5）交叉
#采用OX交叉法。（不是很明白）
def crossSol(model):
    sol_list=copy.deepcopy(model.sol_list)
    model.sol_list=[]
    while True:
        f1_index = random.randint(0, len(sol_list) - 1)
        f2_index = random.randint(0, len(sol_list) - 1)
        if f1_index!=f2_index:
            f1 = copy.deepcopy(sol_list[f1_index])
            f2 = copy.deepcopy(sol_list[f2_index])
            if random.random() <= model.pc: #在概率内
                cro1_index=int(random.randint(0,model.number_of_nodes-1))
                cro2_index=int(random.randint(cro1_index,model.number_of_nodes-1))
                #f，m，b代表父亲，母亲，孩子
                new_c1_f = []
                new_c1_m=f1.nodes_seq[cro1_index:cro2_index+1]
                new_c1_b = []
                new_c2_f = []
                new_c2_m=f2.nodes_seq[cro1_index:cro2_index+1]
                new_c2_b = []
                for index in range(model.number_of_nodes):
                    if len(new_c1_f)<cro1_index:
                        if f2.nodes_seq[index] not in new_c1_m:
                            new_c1_f.append(f2.nodes_seq[index])
                    else:
                        if f2.nodes_seq[index] not in new_c1_m:
                            new_c1_b.append(f2.nodes_seq[index])
                for index in range(model.number_of_nodes):
                    if len(new_c2_f)<cro1_index:
                        if f1.nodes_seq[index] not in new_c2_m:
                            new_c2_f.append(f1.nodes_seq[index])
                    else:
                        if f1.nodes_seq[index] not in new_c2_m:
                            new_c2_b.append(f1.nodes_seq[index])
                new_c1=copy.deepcopy(new_c1_f)
                new_c1.extend(new_c1_m)
                new_c1.extend(new_c1_b)
                f1.nodes_seq=new_c1
                new_c2=copy.deepcopy(new_c2_f)
                new_c2.extend(new_c2_m)
                new_c2.extend(new_c2_b)
                f2.nodes_seq=new_c2
                model.sol_list.append(copy.deepcopy(f1))
                model.sol_list.append(copy.deepcopy(f2))
            else:
                model.sol_list.append(copy.deepcopy(f1))
                model.sol_list.append(copy.deepcopy(f2))
            if len(model.sol_list)>model.popsize:
                break

# （6）突变
#采用二元突变
def muSol(model):
    sol_list=copy.deepcopy(model.sol_list)
    model.sol_list=[]
    while True:
        f1_index = int(random.randint(0, len(sol_list) - 1))
        f1 = copy.deepcopy(sol_list[f1_index])
        m1_index=random.randint(0,model.number_of_nodes-1)
        m2_index=random.randint(0,model.number_of_nodes-1)
        if m1_index!=m2_index:
            if random.random() <= model.pm:
                #交换位置
                node1=f1.nodes_seq[m1_index]
                f1.nodes_seq[m1_index]=f1.nodes_seq[m2_index]
                f1.nodes_seq[m2_index]=node1
                model.sol_list.append(copy.deepcopy(f1))
            else:
                model.sol_list.append(copy.deepcopy(f1))
            if len(model.sol_list)>model.popsize:
                break
# （7）绘制收敛曲线
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
# （8）绘制收运路线图 
def plotRoutes(model):
    for route in model.best_sol.routes:
        x_coord=[model.depot.x_coord]
        y_coord=[model.depot.y_coord]
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
# （9）输出结果到Excel
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

# （10）主函数
def run(filepath,epochs,pc,pm,popsize,n_select,v_cap,opt_type):
    """绿色的这部分可以显示在变量介绍里
    :param filepath:Xlsx 数据文件路径
    :param epochs:迭代次数
    :param pc:交叉概率
    :param pm:突变概率
    :param popsize:种群规模
    :param n_select:优良个体选择数量
    :param v_cap:车载容量
    :param opt_type:优化目标类型，0：最小车辆数，1：最小行驶距离
    """
    #将参数传递到model（）
    model=Model()
    model.vehicle_cap=v_cap
    model.opt_type=opt_type
    model.pc=pc
    model.pm=pm
    model.popsize=popsize
    model.n_select=n_select

    readXlsxFile(filepath,model)#数据读取
    genInitialSol(model)#初始解生成
    history_best_obj = [] #历史最好优化目标值
    best_sol=Sol()
    best_sol.obj=float('inf')
    model.best_sol=best_sol
    #开始迭代
    for ep in range(epochs):
        calFit(model)#计算适应度
        selectSol(model)#优良个体选择
        crossSol(model)#交叉
        muSol(model)#突变
        history_best_obj.append(model.best_sol.obj)
        print("%s/%s， best obj: %s" % (ep,epochs,model.best_sol.obj))
    plotObj(history_best_obj)#收敛图绘制
    plotRoutes(model)#收运路线绘制
    outPut(model)#保存最优结果

if __name__=='__main__':
    file='./cvrp.xlsx'
    run(filepath=file,epochs=100,pc=0.6,pm=0.2,popsize=100,n_select=80,v_cap=80,opt_type=1)
