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
        
        
        d,p = 0,0
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


def package_ga(goods_i_df,v,m,pc,pm,tournament_size,popsize,generations):
    goods_i_df = goods_i_df.reset_index(drop=True)
    Chrom = [[random.randint(0,1) for i in range(len(goods_i_df))] for j in range(popsize)]
    fit = [-1]*len(Chrom)
    
    for i in range(len(Chrom)):
        fit[i] = package_calFitness(goods_i_df,Chrom[i],v,m)
    best_fit = max(fit)
    best_pop= Chrom[fit.index(max(fit))]
    
    iter = 0
    while iter<generations:
        Chrom1,fit1 = tournament_select(Chrom,popsize,fit,tournament_size,flag=-1)
        Chrom2,fit2 = tournament_select(Chrom,popsize,fit,tournament_size,flag=-1)
        new_Chrom = package_crossover(popsize,Chrom1,fit1,Chrom2,fit2,pc,len_of_individual=len(goods_i_df))
        new_Chrom = package_mutate(new_Chrom,pm)
        iter +=1

        new_fit = [-1]*len(Chrom)
        for i in range(len(new_Chrom)):
            new_fit[i] = package_calFitness(goods_i_df,new_Chrom[i],v,m)
        
        for i in range(len(Chrom)):
            if fit[i] < new_fit[i]:
                Chrom[i] = new_Chrom[i]
                fit[i] = new_fit[i]
        
        if best_fit < max(fit):
            best_fit = max(fit)
            best_pop= Chrom[fit.index(max(fit))]
        if best_fit==1:
            break
    v_sum,m_sum = 0,0
    package_list,unpackage_list= [],[]
    for j in range(len(best_pop)):
        if best_pop[j]==0:
            unpackage_list.append(goods_i_df.loc[j,'商品序号'])
        else:
            package_list.append(goods_i_df.loc[j,'商品序号'])
            v_j = goods_i_df.loc[j,'体积']
            m_j = goods_i_df.loc[j,'重量']
            v_sum += v_j
            m_sum += m_j
            
    return package_list,unpackage_list,v_sum,m_sum


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


def tournament_select(population,popsize,fit,tournament_size,flag=0):
    new_population,new_fit = [],[]
    while len(new_population)<len(population):
        tournament_list = random.sample(range(0,popsize),tournament_size)
        tournament_fit = [fit[i] for i in tournament_list]
        #转化为df方便索引
        tournament_df = pd.DataFrame([tournament_list,tournament_fit]).transpose().sort_values(by=1).reset_index(drop=True)
        
        fittest = tournament_df.iloc[flag,1]
        new_pop = population[int(tournament_df.iloc[flag,0])]
        new_population.append(new_pop)
        new_fit.append(fittest)
        
    return new_population.copy(),new_fit.copy()


def crossover(popsize,parent1_pop,fit1,parent2_pop,fit2,pc,len_of_individual):
    child_population = []
    for i in range(popsize):
        child = [-1]*len_of_individual
        
        parent1 = parent1_pop[i].copy()
        parent2 = parent2_pop[i].copy()
        
        if random.random() >= pc:
            child = parent1.copy()#随机生成一个
            random.shuffle(child)
            
        else:
            #parent1
            start_pos = random.randint(0,len(parent1)-1)
            end_pos = random.randint(0,len(parent1)-1)
            if start_pos>end_pos:
                tem_pop = start_pos
                start_pos = end_pos
                end_pos = tem_pop
            child[start_pos:end_pos+1] = parent1[start_pos:end_pos+1].copy()
            # parent2 -> child
            list1 = list(range(end_pos+1,len(parent2)))
            list2 = list(range(0,start_pos))
            list_index = list1+list2
            j = -1
            for i in list_index:
                for j in range(j+1,len(parent2)+1):
                    if parent2[j] not in child:
                        child[i] = parent2[j]
                        break
                        
        child_population.append(child)
    return child_population


def package_crossover(popsize,parent1_pop,fit1,parent2_pop,fit2,pc,len_of_individual):
    child_population = []
    for i in range(popsize):
        child = [-1]* len_of_individual
        parent1 = parent1_pop[i].copy()
        parent2 = parent2_pop[i].copy()
        if random.random() >= pc:
            child = parent1.copy()
            random.shuffle(child)
        else:
            #parent1
            start_pos = random.randint(0,len(parent1)-1)
            end_pos = random.randint(0,len(parent1)-1)
            if start_pos>end_pos:
                tem_pop = start_pos
                start_pos = end_pos
                end_pos = tem_pop
            child[start_pos:end_pos+1] = parent1[start_pos:end_pos+1].copy()
            child[0:start_pos] = parent2[0:start_pos].copy()
            child[end_pos+1:] = parent2[end_pos+1:].copy()
            
        child_population.append(child)
    return child_population


def mutate(populations,pm):
    population_after_mutate = []
    mutate_time = 0
    for i in range(len(populations)):
        pop = populations[i].copy()
        time = random.randint(1,5)
        if random.random() < pm: 
            while mutate_time < time:
                mut_pos1 = random.randint(0,len(pop)-1)  
                mut_pos2 = random.randint(0,len(pop)-1)
                if mut_pos1 != mut_pos2:   
                    goods_tem = pop[mut_pos1]
                    pop[mut_pos1] = pop[mut_pos2]
                    pop[mut_pos2] = goods_tem
                mutate_time += 1
        population_after_mutate.append(pop)
    return population_after_mutate


def package_mutate(populations,pm):
    population_after_mutate = []
    for i in range(len(populations)):
        pop = populations[i].copy()
        for i in range(len(pop)):
            if random.random() < pm:
                if pop[i] == 0:
                    pop[i] = 1
                else:
                    pop[i] = 0
        population_after_mutate.append(pop)
        
    return population_after_mutate



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
        #进化开始
        #选择
        population1,fit1 = tournament_select(population,popsize,fit,tournament_size)
        population2,fit2 = tournament_select(population,popsize,fit,tournament_size)
        #交叉
        population = crossover(popsize,population1,fit1,population2,fit2,pc,len_of_individual=len(route))
        #变异
        population = mutate(population,pm)
        
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


def merchants_ga(dis_matrix,dgoods_df,pgoods_df,dgoods_group_df,pgoods_group_df,GeneSequence,ChromLength,generations,popsize,tournament_size,pc,pm,cpus=4):
    #初始化种群
    populations = [random.sample([i for i in GeneSequence],ChromLength) for j in range(popsize)]
    
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
        
        #进化开始
        #选择
        population1,fit1 = tournament_select(populations,popsize,fit,tournament_size)
        population2,fit2 = tournament_select(populations,popsize,fit,tournament_size)
        #交叉
        population_new = crossover(popsize,population1,fit1,population2,fit2,pc,len_of_individual=ChromLength)
        #变异
        population_new = mutate(population_new,pm)
        
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
    
    #ga参数hhhh
    ChromLength = len(loc_list)   #染色体长度，客户点位置
    GeneSequence = loc_list.copy()   #基因序列
    popsize = 40   #种群大小
    generations = 100   #进化代数
    tournament_size = 5 #锦标赛选择算子参数
    pc = 0.9   #交叉概率
    pm = 0.1   #变异概率
    
    population,fit,car_line,dcar_goods,pcar_goods,populations,fits,car_lines,dcar_goods_all,pcar_goods_all,fits_list=merchants_ga(dis_matrix,dgoods_df,pgoods_df,dgoods_group_df,pgoods_group_df,GeneSequence,ChromLength,generations,popsize,tournament_size,pc,pm)
    
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
    



        
    