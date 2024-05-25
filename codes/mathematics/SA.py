#参数
M = 15   #车辆载重
V = 15   #车辆容积
C0 = 6   #车辆启动成本
C1 = 1   #车辆变动成本

def car_line_goods(population,dgoods_df,pgoods_df,dgoods_group_df,pgoods_group_df,dis_matrix):
    #SA:T=100，Tend=0，Pt = 1-T/100，
    T = 100
    Tend = 1
    best_dgoods_list,best_pgoods_list,best_line_list = [],[],[]
    best_fit = 10000
    
    while T>Tend:
        Pt = 1-T/100
        T -= 10
        
        dgoods_group = dgoods_group_df.copy()
        pgoods_group = pgoods_group_df.copy()
        
        dgoods = dgoods_df.copy()
        pgoods = pgoods_df.copy()
        
        dgoods_list,pgoods_list,line_list,pd_list = [],[],[],[]
        tem_dgoods,tem_pgoods,tem_line = [],[],[]
        dv_sum = 0
        dm_sum = 0
        pv_sum = 0
        pm_sum = 0
        
        
        d,p = 0,0#现在已经装上车的需求点的数量
        pd_df =pd.DataFrame(columns=['loc','d_v','d_m','p_v','p_m'])
        
        while (d <= len(population)-1) | (p <= len(population)-1):
            i = population[d]
            dvi_sum = dgoods_group[dgoods_group['配送点序号']==i]['体积'].iloc[0]
            dmi_sum = dgoods_group[dgoods_group['配送点序号']==i]['重量'].iloc[0]
            pvi_sum = pgoods_group[pgoods_group['配送点序号']==i]['体积'].iloc[0]
            pmi_sum = pgoods_group[pgoods_group['配送点序号']==i]['重量'].iloc[0]
            
            dgoods_i = dgoods[dgoods['配送点序号']==i]
            pgoods_i = pgoods[pgoods['配送点序号']==i]
            
            #刚好放得下
            if (dvi_sum<=V-dv_sum) & (dmi_sum<=M-dm_sum) & (pvi_sum<=V-pv_sum) & (pmi_sum<=M-pm_sum):
                dgoods_i_list = list(dgoods_i['商品序号'])
                pgoods_i_list = list(pgoods_i['商品序号'])
                
                tem_dgoods = tem_dgoods + dgoods_i_list.copy()
                tem_pgoods = tem_pgoods + pgoods_i_list.copy()
                
                tem_line.append(i)
                dv_sum += dvi_sum
                dm_sum += dmi_sum
                
                pv_sum += pvi_sum
                pm_sum += pmi_sum
                
                dgoods_group.loc[dgoods_group['配送点序号']==i,'体积'] = 0
                dgoods_group.loc[dgoods_group['配送点序号']==i,'重量'] = 0
                dgoods = dgoods[~dgoods['商品序号'].isin(dgoods_i_list)]
            
                pgoods_group.loc[pgoods_group['配送点序号']==i,'体积'] = 0
                pgoods_group.loc[pgoods_group['配送点序号']==i,'重量'] = 0
                pgoods = pgoods[~pgoods['商品序号'].isin(pgoods_i_list)]
                
                pd_df = pd_df.append(pd.DataFrame([[i,dvi_sum,dmi_sum,pvi_sum,pmi_sum]],columns=['loc','d_v','d_m','p_v','p_m'])).reset_index(drop=True)
                
                #是否跳出循环
                if ((d== len(population)-1) & (p== len(population)-1)) | (random.random()<Pt) | ((dv_sum==V) | (dm_sum==M)) | ((pv_sum==V) | (pm_sum==M)):
                    dgoods_list.append(tem_dgoods)
                    pgoods_list.append(tem_pgoods)
                    tem_dgoods,tem_pgoods = [],[]
                    line_list.append(tem_line)
                    tem_line = []
                    pd_list.append(pd_df)
                    pd_df =pd.DataFrame(columns=['loc','d_v','d_m','p_v','p_m'])
                    dv_sum,dm_sum,pv_sum,pm_sum = 0, 0, 0, 0 
                    d += 1
                    p += 1
                    
                else:
                    d += 1
                    p += 1
                    continue
                
            elif ((dvi_sum<=V-dv_sum) & (dmi_sum<=M-dm_sum)) & ((pvi_sum>V-pv_sum) | (pmi_sum>M-pm_sum)):
                dgoods_i_list = list(dgoods_i['商品序号'])
                tem_dgoods = tem_dgoods + dgoods_i_list.copy()
                tem_line.append(i)
                dv_sum += dvi_sum
                dm_sum += dmi_sum
                dgoods_group.loc[dgoods_group['配送点序号']==i,'体积'] = 0
                dgoods_group.loc[dgoods_group['配送点序号']==i,'重量'] = 0
                dgoods = dgoods[~dgoods['商品序号'].isin(dgoods_i_list)]
                
                dgoods_list.append(tem_dgoods)
                tem_dgoods = []
                dv_sum = 0
                dm_sum = 0
                
                line_list.append(tem_line)
                tem_line = []
                
                
                #判断是否剩下一个货物
                if len(pgoods_i)==1:
                    pgoods_list.append(tem_pgoods)
                    tem_pgoods = []
                    pv_sum = 0
                    pm_sum = 0
                    pd_df = pd_df.append(pd.DataFrame([[i,dvi_sum,dmi_sum,0,0]],columns=['loc','d_v','d_m','p_v','p_m'])).reset_index(drop=True)
                    pd_list.append(pd_df)
                    pd_df =pd.DataFrame(columns=['loc','d_v','d_m','p_v','p_m'])
                
                else:
                    package_list,unpackage_list,v_pac,m_pac = package_ga(pgoods_i,v=V-pv_sum,m=M-pm_sum,pc=0.9,pm=0.1,tournament_size=2,popsize=len(pgoods_i),generations=10)
                    if (v_pac>V-pv_sum) | (m_pac>M-pm_sum):
                        pgoods_list.append(tem_pgoods)
                        tem_pgoods = []
                        pv_sum = 0
                        pm_sum = 0
                        pd_df = pd_df.append(pd.DataFrame([[i,dvi_sum,dmi_sum,0,0]],columns=['loc','d_v','d_m','p_v','p_m'])).reset_index(drop=True)
                        pd_list.append(pd_df)
                        pd_df =pd.DataFrame(columns=['loc','d_v','d_m','p_v','p_m'])
                        
                    else:
                        tem_pgoods = tem_pgoods + package_list.copy()
                        pgoods_list.append(tem_pgoods)
                        tem_pgoods = []
                        #更新
                        pgoods_group.loc[pgoods_group['配送点序号']==i,'体积'] -= v_pac
                        pgoods_group.loc[pgoods_group['配送点序号']==i,'重量'] -= m_pac
                        pgoods = pgoods[~pgoods['商品序号'].isin(package_list)]
                        pv_sum = 0
                        pm_sum = 0
                    
                        pd_df = pd_df.append(pd.DataFrame([[i,dvi_sum,dmi_sum,v_pac,m_pac]],columns=['loc','d_v','d_m','p_v','p_m'])).reset_index(drop=True)
                        pd_list.append(pd_df)
                        pd_df =pd.DataFrame(columns=['loc','d_v','d_m','p_v','p_m'])
                
                
            elif ((dvi_sum>V-dv_sum) | (dmi_sum>M-dm_sum)) & ((pvi_sum<=V-pv_sum) & (pmi_sum<=M-pm_sum)):
                pgoods_i_list = list(pgoods_i['商品序号'])
                tem_pgoods = tem_pgoods + pgoods_i_list.copy()
                tem_line.append(i)
                pv_sum += pvi_sum
                pm_sum += pmi_sum
                pgoods_group.loc[pgoods_group['配送点序号']==i,'体积'] = 0
                pgoods_group.loc[pgoods_group['配送点序号']==i,'重量'] = 0
                pgoods = pgoods[~pgoods['商品序号'].isin(pgoods_i_list)]
                
                pgoods_list.append(tem_pgoods)
                tem_pgoods = []
                pv_sum = 0
                pm_sum = 0
                
                line_list.append(tem_line)
                tem_line = []
                
                if len(dgoods_i)==1:
                    dgoods_list.append(tem_dgoods)
                    tem_dgoods = []
                    dv_sum = 0
                    dm_sum = 0
                    pd_df = pd_df.append(pd.DataFrame([[i,0,0,pvi_sum,pmi_sum]],columns=['loc','d_v','d_m','p_v','p_m'])).reset_index(drop=True)
                    pd_list.append(pd_df)
                    pd_df =pd.DataFrame(columns=['loc','d_v','d_m','p_v','p_m'])
                
                else:
                    package_list,unpackage_list,v_pac,m_pac = package_ga(dgoods_i,v=V-dv_sum,m=M-dm_sum,pc=0.9,pm=0.1,tournament_size=2,popsize=len(dgoods_i),generations=10)
                    if (v_pac>V-dv_sum) | (m_pac>M-dm_sum):
                        dgoods_list.append(tem_dgoods)
                        tem_dgoods = []
                        dv_sum = 0
                        dm_sum = 0
                        
                        pd_df = pd_df.append(pd.DataFrame([[i,0,0,pvi_sum,pmi_sum]],columns=['loc','d_v','d_m','p_v','p_m'])).reset_index(drop=True)
                        pd_list.append(pd_df)
                        pd_df =pd.DataFrame(columns=['loc','d_v','d_m','p_v','p_m'])
                        
                    else:
                        tem_dgoods = tem_dgoods + package_list.copy()
                        dgoods_list.append(tem_dgoods)
                        tem_dgoods = []
                        #更新
                        dgoods_group.loc[dgoods_group['配送点序号']==i,'体积'] -= v_pac
                        dgoods_group.loc[dgoods_group['配送点序号']==i,'重量'] -= m_pac
                        dgoods = dgoods[~dgoods['商品序号'].isin(package_list)]
                        dv_sum = 0
                        dm_sum = 0
                        
                        pd_df = pd_df.append(pd.DataFrame([[i,v_pac,m_pac,pvi_sum,pmi_sum]],columns=['loc','d_v','d_m','p_v','p_m'])).reset_index(drop=True)
                        pd_list.append(pd_df)
                        pd_df =pd.DataFrame(columns=['loc','d_v','d_m','p_v','p_m'])
                    
            elif ((dvi_sum>V-dv_sum) | (dmi_sum>M-dm_sum)) & ((pvi_sum>V-pv_sum) | (pmi_sum>M-pm_sum)):
                #d
                dv,dm,pv,pm=0,0,0,0
                if len(dgoods_i)==1:
                    dgoods_list.append(tem_dgoods)
                    tem_dgoods = []
                    dv_sum = 0
                    dm_sum = 0
                    dv,dm = 0,0
                    
                else:
                    package_list,unpackage_list,dv_pac,dm_pac = package_ga(dgoods_i,v=V-dv_sum,m=M-dm_sum,pc=0.9,pm=0.1,tournament_size=2,popsize=len(dgoods_i),generations=10)
                    if (dv_pac>V-dv_sum) | (dm_pac>M-dm_sum):
                        dgoods_list.append(tem_dgoods)
                        tem_dgoods = []
                        dv_sum = 0
                        dm_sum = 0
                        dv,dm = 0,0
                        
                    else:
                        tem_dgoods = tem_dgoods + package_list.copy()
                        dgoods_list.append(tem_dgoods)
                        tem_dgoods = []
                        #更新
                        dgoods_group.loc[dgoods_group['配送点序号']==i,'体积'] -= dv_pac
                        dgoods_group.loc[dgoods_group['配送点序号']==i,'重量'] -= dm_pac
                        dgoods = dgoods[~dgoods['商品序号'].isin(package_list)]
                        dv_sum = 0
                        dm_sum = 0
                        dv,dm = dv_pac,dm_pac
                    
                #p
                if len(pgoods_i)==1:
                    pgoods_list.append(tem_pgoods)
                    tem_pgoods = []
                    pv_sum = 0
                    pm_sum = 0
                    pv,pm = 0,0

                
                else:
                    package_list,unpackage_list,pv_pac,pm_pac = package_ga(pgoods_i,v=V-pv_sum,m=M-pm_sum,pc=0.9,pm=0.1,tournament_size=2,popsize=len(pgoods_i),generations=10)
                    if (pv_pac>V-pv_sum) | (pm_pac>M-pm_sum):
                        pgoods_list.append(tem_pgoods)
                        tem_pgoods = []
                        pv_sum = 0
                        pm_sum = 0
                        pv,pm = 0,0
                        
                        
                    else:
                        tem_pgoods = tem_pgoods + package_list.copy()
                        pgoods_list.append(tem_pgoods)
                        tem_pgoods = []
                        #更新
                        pgoods_group.loc[pgoods_group['配送点序号']==i,'体积'] -= pv_pac
                        pgoods_group.loc[pgoods_group['配送点序号']==i,'重量'] -= pm_pac
                        pgoods = pgoods[~pgoods['商品序号'].isin(package_list)]
                        pv_sum = 0
                        pm_sum = 0
                        pv,pm = pv_pac,pm_pac
                
                
                if (pv==0) & (pm==0) & (dv==0) & (dm==0):
                    line_list.append(tem_line)
                    tem_line = []
                    pd_list.append(pd_df)
                    pd_df = pd.DataFrame(columns=['loc','d_v','d_m','p_v','p_m'])
                    
                else:
                    tem_line.append(i)
                    line_list.append(tem_line)
                    tem_line = []
                    pd_df = pd_df.append(pd.DataFrame([[i,dv,dm,pv,pm]],columns=['loc','d_v','d_m','p_v','p_m'])).reset_index(drop=True)
                    pd_list.append(pd_df)
                    pd_df = pd.DataFrame(columns=['loc','d_v','d_m','p_v','p_m'])
            
        line_list,fit = car_line_optimize(line_list,dis_matrix,pd_list)
        if best_fit > fit:
            best_fit = fit
            best_dgoods_list = dgoods_list
            best_pgoods_list = pgoods_list
            best_line_list = line_list
            
    return best_dgoods_list,best_pgoods_list,best_line_list,best_fit

def package_calFitness(goods_i_df,pop,v,m):
    goods_i_df = goods_i_df.reset_index(drop=True)
    v_sum = 0
    m_sum = 0
    for j in range(len(pop)):
        if pop[j]==1:
            v_j = goods_i_df.loc[j,'体积']
            m_j = goods_i_df.loc[j,'重量']
            v_sum += v_j
            m_sum += m_j
    #计算(不满足约束的直接为负数)
    if (v_sum<=v) & (m_sum<=m):
        fit = (v_sum/v)*(m_sum/m)
       
    else:
        fit = -1
    return round(fit,3)


def car_line_optimize(car_line,dis_matrix,pd_list):
    '''
    GA优化car_line，
    '''
    car_line_opt = [-1]*len(car_line)
    fit_list = [-1]*len(car_line)
    
    for i in range(len(car_line)):
        line = car_line[i].copy()
        
        if len(line)<=1:
            line.append(0)
            car_line_opt[i] = line.copy()
            car_line_opt[i] = car_line_opt[i][car_line_opt[i].index(0):]+car_line_opt[i][:car_line_opt[i].index(0)]
            fit_list[i] = C0+calFitness([car_line_opt[i]],dis_matrix)*C1
            
        elif len(line) == 2:
            a = line[0]
            b = line[1]
            line1 = [0,a,b]
            line2 = [0,b,a]
            if pd_is_true(line1,pd_list[i]):
                fit1 = C0+calFitness([line1],dis_matrix)*C1     #计算适应度值
            else:
                fit1 = 1000000#取一个比较大的值
            if pd_is_true(line2,pd_list[i]):
                fit2 = C0+calFitness([line2],dis_matrix)*C1     #计算适应度值
            else:
                fit2 = 1000000#取一个比较大的值
            if fit1>=fit2:
                fit_list[i] = fit2
                car_line_opt[i] = line2.copy()
            else:
                fit_list[i] = fit1
                car_line_opt[i] = line1.copy()
        
        
        elif len(line) == 3:
            a = line[0]
            b = line[1]
            c = line[2]
            line_list = [[0,a,b,c],[0,a,c,b],[0,b,a,c],[0,b,c,a],[0,c,a,b],[0,c,b,a]]
            fits = []
            for line_i in line_list:
                if pd_is_true(line_i,pd_list[i]):
                    fit = C0+calFitness([line_i],dis_matrix)*C1     #计算适应度值
                else:
                    fit = 1000000#取一个比较大的值
                fits.append(fit)
            
            fit_list[i] = min(fits)
            car_line_opt[i] = line_list[fits.index(min(fits))].copy()
         
        else:
            line.append(0)
            line_opt,fit_opt = route_opt(route=line,dis_matrix=dis_matrix,pd=pd_list[i],generations=10,popsize=len(line),tournament_size=3,pc=0.9,pm=0.1)
            fit_opt = C0 + fit_opt*C1
            car_line_opt[i] = line_opt.copy()
            car_line_opt[i] = car_line_opt[i][car_line_opt[i].index(0):]+car_line_opt[i][:car_line_opt[i].index(0)]
            fit_list[i] = fit_opt
            
    return car_line_opt,round(sum(fit_list),1)


def calFitness(car_line,dis_matrix):
    """
    计算Fitness,行驶距离
    car_line是一个个体
    """
    dis_sum = 0
    dis = 0
    for line_i in car_line:
        for j in range(len(line_i)):
            if j<len(line_i)-1:
                dis = dis_matrix.loc[line_i[j],line_i[j+1]]#计算距离
                dis_sum = dis_sum+dis
            if j==len(line_i)-1:
                dis = dis_matrix.loc[line_i[j],line_i[0]]
                dis_sum = dis_sum+dis
                
    return dis_sum

def pd_is_true(route,route_pd):
    flag = 0
    v_sum = route_pd['d_v'].sum()
    m_sum = route_pd['d_m'].sum()
    route = route[route.index(0):]+route[:route.index(0)]
    
    for i in range(1,len(route)):
        v_sum = v_sum-route_pd.loc[route_pd['loc']==route[i],'d_v'].iloc[0]+route_pd.loc[route_pd['loc']==route[i],'p_v'].iloc[0]
        m_sum = m_sum-route_pd.loc[route_pd['loc']==route[i],'d_m'].iloc[0]+route_pd.loc[route_pd['loc']==route[i],'p_m'].iloc[0]
        if (v_sum<=V) & (m_sum<=M):
            continue
        else:
            break
    if i == len(route)-1:
        flag = 1
        
    return flag


def route_opt(route,dis_matrix,pd,generations,popsize,tournament_size,pc,pm):
    #初始化种群
    population = [random.sample(route,len(route)) for j in range(popsize)]
    #初始化迭代参数
    iter = 0
    
    while iter <= generations:
        #适应度
        fit = [0]*popsize
        for i in range(popsize):
            if pd_is_true(population[i],pd):
                fit[i] = calFitness([population[i]],dis_matrix)     #计算适应度值
            else:
                fit[i] = 1000000#取一个比较大的值
        
        iter +=1
    fit = [0]*popsize
    for i in range(popsize):
        
        if pd_is_true(population[i],pd):
            fit[i] = calFitness([population[i]],dis_matrix)     #计算适应度值
        else:
            fit[i] = 1000000#取一个比较大的值

    return population[fit.index(min(fit))],min(fit)


def get_fit(populations, population_start, batch_size, dgoods_df,pgoods_df,dgoods_group_df,pgoods_group_df, dis_matrix, conn_send):
    dcar_goods = [[]]*batch_size
    pcar_goods = [[]]*batch_size
    car_line = [[]]*batch_size
    fit = [0]*batch_size
    for i in range(batch_size):
        dcar_goods[i],pcar_goods[i],car_line[i],fit[i] = car_line_goods(populations[population_start+i].copy(),dgoods_df,pgoods_df,dgoods_group_df,pgoods_group_df,dis_matrix)
        
    conn_send.send((car_line,fit,dcar_goods,pcar_goods))


def greedy(loc_df1,dis_matrix):
    '''贪婪策略构造初始解'''
    #dis_matrix
    dis_matrix = dis_matrix.astype('float64')
    #dis_matrix = dis_matrix[dis_matrix.columns[1:9]].iloc[1:9]
    dis_matrix = dis_matrix[dis_matrix.columns[1:17]].iloc[1:17]
    
    #for i in range(len(loc_df1)):dis_matrix.loc[i,i]=math.pow(10,10)
    for i in range(len(loc_df1)):dis_matrix.loc[i,i]=math.pow(18,18)
    line = []#初始化
    now_city = random.randint(0+1,len(loc_df1))#随机生成出发城市
    line.append(now_city)#添加当前城市到路径
    #dis_matrix.loc[:,now_city] = math.pow(10,10)#更新距离矩阵，已经过城市不再被取出
    dis_matrix.loc[:,now_city] = math.pow(18,18)
    for i in range(len(loc_df1)-1):
        next_city = dis_matrix.loc[now_city,:].idxmin()#距离最近的城市
        line.append(next_city)#添加进路径
        #dis_matrix.loc[:,next_city] = math.pow(10,10)#更新距离矩阵
        dis_matrix.loc[:,next_city] = math.pow(18,18)
        now_city = next_city#更新当前城市
        
    return line


def merchants_ga(loc_df1,dis_matrix,dgoods_df,pgoods_df,dgoods_group_df,pgoods_group_df,GeneSequence,ChromLength,generations,popsize,tournament_size,pc,pm,cpus=4):
    #初始化种群
    # populations = [random.sample([i for i in GeneSequence],ChromLength) for j in range(popsize)]
    #贪婪
    populations = [greedy(loc_df1,dis_matrix) for j in range(popsize)]
    # population = greedy(loc_df,dis_matrix)
    # print(populations)
    
    
    #计算初代适应度值
    car_line = []   #车辆路径
    dcar_goods = []   #车辆装的货物
    pcar_goods = []
    fit = []
    processlist = []
    pipelist = []
    
    
    batch_size = int(popsize / cpus)
    for i in range(cpus):
        conn_recv, conn_send = Pipe()
        pipelist.append(conn_recv)
        processlist.append(Process(target=get_fit, args=(populations, batch_size * i, batch_size, dgoods_df,pgoods_df,dgoods_group_df,pgoods_group_df,dis_matrix, conn_send)))
        processlist[i].start()

    for i in range(cpus):
        res = pipelist[i].recv()
        processlist[i].join()
        sub_car_line = res[0]
        sub_fit = res[1]
        sub_dcar_goods = res[2]
        sub_pcar_goods = res[3]
        car_line += sub_car_line
        fit += sub_fit
        dcar_goods += sub_dcar_goods
        pcar_goods += sub_pcar_goods
    
    
    best_fit = min(fit)
    best_population,best_car_line,best_dcar_goods,best_pcar_goods = populations[fit.index(min(fit))],car_line[fit.index(min(fit))],dcar_goods[fit.index(min(fit))],pcar_goods[fit.index(min(fit))]
    print('初代最优值 %.1f' % (best_fit))
    
    time = 1
    
    fit_list = [] # 储存历史fit值，用于画图
    fit_list.append(best_fit)
    
    iter = 0
    while iter < generations:
        iter +=1
        new_dcar_goods = []
        new_pcar_goods = []
        new_car_line = []
        new_fit = []
        processlist = []
        pipelist = []
        for i in range(cpus):
            conn_recv, conn_send = Pipe()
            pipelist.append(conn_recv)
            processlist.append(Process(target=get_fit, args=(population_new, batch_size * i, batch_size, dgoods_df,pgoods_df,dgoods_group_df,pgoods_group_df, dis_matrix, conn_send)))
            processlist[i].start()
            
            
        for i in range(cpus):
            res = pipelist[i].recv()
            processlist[i].join()
            sub_car_line = res[0]
            sub_fit = res[1]
            sub_dcar_goods = res[2]
            sub_pcar_goods = res[3]
            new_car_line += sub_car_line
            new_fit += sub_fit
            new_dcar_goods += sub_dcar_goods
            new_pcar_goods += sub_pcar_goods
        
        for i in range(popsize):
            if fit[i] > new_fit[i]:
                populations[i] = population_new[i]
                fit[i] = new_fit[i]
                car_line[i] = new_car_line[i]
                dcar_goods[i] = new_dcar_goods[i]
                pcar_goods[i] = new_pcar_goods[i]
        
        
        if best_fit > min(fit):
            best_fit = min(fit)
            best_population,best_car_line,best_dcar_goods,best_pcar_goods = populations[fit.index(min(fit))],car_line[fit.index(min(fit))],dcar_goods[fit.index(min(fit))],pcar_goods[fit.index(min(fit))]
            time = 1
        else:
            time += 1
            
        if time == 20:
            break
        
        print('第%d代最优值 %.1f' % (iter, best_fit))
        print(fit)
        fit_list.append(best_fit)
        
    return best_population,best_fit,best_car_line,best_dcar_goods,best_pcar_goods,populations,fit,car_line,dcar_goods,pcar_goods,fit_list


#画路径图
def draw_path(lines,CityCoordinates):
    for i in range(len(lines)):
        line = lines[i]
        x,y= [],[]
        for i in line:
            Coordinate = CityCoordinates.iloc[i]
            x.append(Coordinate['横轴'])
            y.append(Coordinate["纵轴"])
        x.append(x[0])
        y.append(y[0])
        
        plt.plot(x, y,'o-', alpha=0.8, linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    freeze_support()
    start = time.time()
    #读入数据
    loc_df = pd.read_excel("test.xls",sheet_name='地点')
    dgoods_df = pd.read_excel("test.xls",sheet_name='delivery100')
    pgoods_df = pd.read_excel("test.xls",sheet_name='pickup100')
    
    loc_df = loc_df.sort_values(by='位置').reset_index(drop=True)
    loc_list = list(dgoods_df['配送点序号'].unique())  #dgoods_df和pgoods_df都可以
    
    dgoods_group_df = dgoods_df[['体积','重量']].groupby(dgoods_df['配送点序号']).sum().reset_index()
    pgoods_group_df = pgoods_df[['体积','重量']].groupby(pgoods_df['配送点序号']).sum().reset_index()
    
    
    
    #距离矩阵
    dis_matrix = pd.DataFrame(data=None,columns=range(len(loc_df)),index=range(len(loc_df)))
    for i in list(loc_df['位置']):
        x1, y1 = loc_df.loc[loc_df['位置']==i]['横轴'].iloc[0],loc_df.loc[loc_df['位置']==i]['纵轴'].iloc[0]
        for j in list(loc_df['位置']):
            x2,y2 = loc_df.loc[loc_df['位置']==j]['横轴'].iloc[0],loc_df.loc[loc_df['位置']==j]['纵轴'].iloc[0]
            dis_matrix.iloc[i,j] = round(math.sqrt((x1-x2)**2+(y1-y2)**2),2)
    
    #loc_df1 = loc_df[1:9].copy()
    loc_df1 = loc_df[1:17].copy()

    #画图
    iters = list(range(len(fits_list)))
    plt.plot(iters, fits_list, 'r-', color='#4169E1', alpha=0.8, linewidth=0.5)
    plt.legend(loc="upper right")
    plt.xlabel('迭代次数')
    plt.ylabel('最优解')
    plt.show()
    
    end = time.time()
    print("运行时间",end-start)
    
    for i in car_line:
        i.append(0)
    draw_path(car_line,loc_df)
        
    print("车辆路径",car_line)
    print("送货",dcar_goods)
    print("取货",pcar_goods)
    print("fits_list",fits_list)

        
    