# 								数学模型

## TSP

从一个城市出发，第二项为返回最初的城市

![image-20211201164048137](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20211201164048137.png)

![](https://mmbiz.qpic.cn/mmbiz_png/N76CH3yf11D1MpbqVhE5hKk646x2R9Xic2aLKDtNJwVibzVglu2jm0toibNicffQPd4Fiah3BAckWyQf1zTiboQgy0hw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

下图为初始可行解的求解算法的列表

![image-20211201171625703](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20211201171625703.png)

下图为所有返回值的情况

![image-20211201171836454](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20211201171836454.png)



路由求解器并不总是返回TSP的最优解，您可以使用一种更高级的搜索策略，称为引导局部搜索，它使求解器能够避开局部最小值，从局部最小值移开后，求解器继续进行搜索，所以设置初始解部分，有不一样的变动。



注意数据结构，使用的是**字典**！涉及字典中的取值、赋值以及增改查！

https://mp.weixin.qq.com/s/QQqtpGSgggypxSCaTNHrJg

 [TSP.py](pythoncode\TSP.py) 

 [自动钻孔机.py](pythoncode\自动钻孔机.py) 



##  VRP

### 数学模型

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/N76CH3yf11B02qyxJSiadMS2uCLcfKs0kTKDHylmIChhR93fYPmdGgZQuldlwqHFN9gYIEn7jibCLjAfibgu8luFg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



设某配送中心有M辆车, 需要对K个客户 (节点) 进行运输配送。设cij表示客户i到客户j的运输成本, 如时间、路程、花费等.取配送中心编号为0, 各客户编号为i (i=1, 2, …, K) , 定义变量如下:



![图片](https://mmbiz.qpic.cn/mmbiz_jpg/N76CH3yf11B02qyxJSiadMS2uCLcfKs0kxJGVGuDVlSZlcEdH3LhkeE5ZWrjtYaaLWhCZt3DW16vibaKd12dQXAw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/N76CH3yf11B02qyxJSiadMS2uCLcfKs0kgiaOibbVQpyajxqBfbaRRMymZ0fmRNDSDRGibXxtVOLlib0wl7W5Z5nE6A/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

第一个约束表示：保证了每个客户的运输任务仅由1 辆车完成,而所有运输任务则由 M 辆车协同完成

第二第三约束：限制了到达和离开某一客户的车辆有且仅有1辆。



https://mp.weixin.qq.com/s/VsQgaQtSVVacMGhiZwgxWA



注意：其中的距离矩阵`data['distance_matrix']`是经过处理之后的！这里采用的是Manhattan distance计算方法：(x1, y1) 和 (x2, y2) 之间的距离是：|x1 - x2| + |y1 - y2|。distance_matrix这个矩阵是需要大家另做处理的，里面的每一个数代表的是其下标之间的距离！这个地方大家可以自行DIY。

### 结果

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/N76CH3yf11B02qyxJSiadMS2uCLcfKs0kxYia6FJAdFS99quibSdtO3dcUDeSKJregBRxfibDcbHdEpOCickISF3awg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

 [VRP.py](pythoncode\VRP.py) 



## CVRP

![image-20211118152510410](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20211118152510410.png)

![image-20211118152705225](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20211118152705225.png)

https://mp.weixin.qq.com/s/WODgH1jQqhZy1JXOe3cRBg

 [CVRP.py](pythoncode\CVRP.py) 

![image-20220425191647239](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220425191647239.png)



## MDVRP

![img](https://img-blog.csdnimg.cn/20201107122013167.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoYzk2MDYwOQ==,size_16,color_FFFFFF,t_70)

![img](https://img-blog.csdnimg.cn/20201107122034533.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoYzk2MDYwOQ==,size_16,color_FFFFFF,t_70)



## VRPTW

在VRPTW问题中，除了行驶成本之外, 成本函数还要包括由于早到某个客户而引起的等待时间和客户需要的服务时间。需求点的时窗限制可以分为两种，一种是硬时窗（Hard Time Window），硬时窗要求车辆必须要在时窗内到达，早到必须等待，而迟到则拒收；另一种是软时窗（Soft Time Window），不一定要在时窗内到达，但是在时窗之外到达必须要处罚。


![image-20211118153746539](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20211118153746539.png)

![image-20211118153853609](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20211118153853609.png)

![image-20211118154456284](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20211118154456284.png)

![image-20211118154505686](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20211118154505686.png)

https://mp.weixin.qq.com/s/5sPS-2CrNVSwhob0CkzXSg

代码最重要的是数据的改变，将matix_distance矩阵变化为time_matrix,并且为每一个服务点添加了服务时间窗口。这里优化的目标是**总服务时间最小**。

 [VRPTW.py](pythoncode\VRPTW.py) （主函数部分没改）



## VRPB

VRP with Backhauls

在VRPB问题中，车辆从起点出发访问一系列客户进行送货，然后接着访问一系列客户装载货物，之后回到起点。该问题的一个关键假设是车辆必须完成送货过程然后再访问顾客进行装货。



## 设置优化策略

对复杂问题设置优化策略，本文前面的CVRP进行优化

https://mp.weixin.qq.com/s/FT9R7VH3Z0d99JDjCQJ3-A

![image-20211118154840322](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20211118154840322.png)

 [CVRP-Optimize.py](pythoncode\CVRP-Optimize.py) 



# 			==========



# 											SA

求最小值f（x）

如果*![X](https://private.codecogs.com/gif.latex?X)*是[离散](https://so.csdn.net/so/search?q=离散&spm=1001.2101.3001.7020)有限取值，那么可以通过穷取法获得问题的最优解；如果*![X](https://private.codecogs.com/gif.latex?X)*连续，但![f(x)](https://private.codecogs.com/gif.latex?f%28x%29)是凸的，那可以通过[梯度下降](https://so.csdn.net/so/search?q=梯度下降&spm=1001.2101.3001.7020)等方法获得最优解；如果*![X](https://private.codecogs.com/gif.latex?X)*连续且![f(x)](https://private.codecogs.com/gif.latex?f%28x%29)非凸，虽说根据已有的近似求解法能够找到问题解，可解是否是最优的还有待考量，很多时候若初始值选择的不好，非常容易陷入局部最优值。

![image-20220505180935240](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220505180935240.png)

主要是将热力学的理论套用到统计学上，将搜寻空间内每一点想象成空气内的分子；分子的能量，就是它本身的动能；而搜寻空间内的每一点，也像空气分子一样带有“能量”，以表示该点对命题的合适程度。演算法先以搜寻空间内一个任意点作起始：每一步先选择一个“邻居”，然后再计算从现有位置到达“邻居”的概率。若概率大于给定的阈值，则跳转到“邻居”；若概率较小，则停留在原位置不动。



模拟退火是启发示算法的一种，也是一种贪心算法，但是它的搜索过程引入了随机因素。在迭代更新可行解时，以一定的概率来接受一个比当前解要差的解，因此有可能会跳出这个局部的最优解，达到全局的最优解。以下图为例，假定初始解为左边蓝色点A，模拟退火算法会快速搜索到局部最优解B，但在搜索到局部最优解后，不是就此结束，而是会以一定的概率接受到左边的移动。也许经过几次这样的不是局部最优的移动后会到达全局最优点D，于是就跳出了局部最小值。

![img](https://img-blog.csdn.net/20180816100720863?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2h1YWh1YTE5ODkxMjIx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![image-20220505180748436](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220505180748436.png)



**三、模拟退火算法的优缺点**

模拟退火算法的应用很广泛，可以高效地求解NP完全问题，如货郎担问题(Travelling Salesman Problem，简记为TSP)、最大截问题(Max Cut Problem)、0-1背包问题(Zero One Knapsack Problem)、图着色问题(Graph Colouring Problem)等等，但其参数难以控制，不能保证一次就收敛到最优值，一般需要多次尝试才能获得（大部分情况下还是会陷入局部最优值）。观察模拟退火算法的过程，发现其主要存在如下三个参数问题：

**(1)** **温度*****T\*****的初始值设置问题** 

温度*T*的初始值设置是影响模拟退火算法全局搜索性能的重要因素之一、初始温度高，则搜索到全局最优解的可能性大，但因此要花费大量的计算时间；反之，则可节约计算时间，但全局搜索性能可能受到影响。

**(2)** **退火速度问题，即每个***T***值的迭代次数**

模拟退火算法的全局搜索性能也与退火速度密切相关。一般来说，同一温度下的“充分”搜索是相当必要的，但这也需要计算时间。循环次数增加必定带来计算开销的增大。

**(3)** **温度管理问题** 

温度管理问题也是模拟退火算法难以处理的问题之一。实际应用中，由于必须考虑计算复杂度的切实可行性等问题，常采用如下所示的降温方式：

![T=\alpha \times T , \alpha \in (0, 1)](https://private.codecogs.com/gif.latex?T%3D%5Calpha%20%5Ctimes%20T%20%2C%20%5Calpha%20%5Cin%20%280%2C%201%29)

注：为了保证较大的搜索空间，*α*一般取接近于1的值，如0.95、0.9。

[(25条消息) 模拟退火算法详细讲解（含实例python代码）_Eterbity的博客-CSDN博客_模拟退火算法](https://blog.csdn.net/weixin_48241292/article/details/109468947?ops_request_misc=%7B%22request%5Fid%22%3A%22165174422416780366578960%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=165174422416780366578960&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-109468947.142^v9^pc_search_result_control_group,157^v4^control&utm_term=模拟退火算法&spm=1018.2226.3001.4187)



# 							梯度下降算法

英文：gradient descent

不论是在线性回归还是Logistic回归中，它的主要目的是通过迭代找到目标函数的最小值，或者收敛到最小值。

数学公式

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190121203434245.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxODAwMzY2,size_16,color_FFFFFF,t_70)

α在梯度下降算法中被称作为学习率或者步长，意味着我们可以通过α来控制每一步走的距离

梯度前加一个负号，就意味着朝着梯度相反的方向前进！我们在前文提到，梯度的方向实际就是函数在此点上升最快的方向



小例子——多变量函数的梯度下降

![image-20220505161244116](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220505161244116.png)



实例

![image-20220505161420157](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220505161420157.png)

![image-20220505161908309](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220505161908309.png)



![image-20220505161930377](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220505161930377.png)

[(25条消息) 梯度下降算法原理讲解——机器学习_Ardor-Zhang的博客-CSDN博客_梯度下降法](https://blog.csdn.net/qq_41800366/article/details/86583789?ops_request_misc=%7B%22request%5Fid%22%3A%22165173251516782388084818%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=165173251516782388084818&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-86583789.142^v9^pc_search_result_control_group,157^v4^control&utm_term=梯度下降&spm=1018.2226.3001.4187)



# 									PSO

全称：discrete particle swarm optimization离散粒子群

源于对鸟群捕食的行为研究。粒子群优化算法的基本思想：是通过**群体中个体之间的协作和信息共享**来寻找最优解．
  PSO的优势：在于简单容易实现并且没有许多参数的调节。目前已被广泛应用于函数优化、神经网络训练、模糊系统控制以及其他遗传算法的应用领域。

## 1、粒子群算法的名词解释

粒子群长度：粒子群长度等于每一个参数取值范围的大小。
粒子群维度：粒子群维度等于待寻优参数的数量。
粒子群位置：粒子群位置包含参数取值的具体数值。
粒子群方向：粒子群方向表示参数取值的变化方向。
适应度函数：表征粒子对应的模型评价指标。
pbest:（局部最优）pbest的长度等于粒子群长度，表示每一个参数取值的变化过程中，到目前为止最优适应度函数值对应的取值。
gbest:（全局最优）gbest的长度为1，表示到目前为止所有适应度函数值中最优的那个对应的参数取值。
惯性因子w ww：惯性因子表示粒子保持的运动惯性。
局部学习因子c 1 {c_1}c 1 ：表示每个粒子向该粒子目前为止最优位置运动加速项的权重。
全局学习因子c 2 {c_2}c 2 ：表示每个粒子向目前为止全局最优位置运动加速项的权重。


1、基本思想

  粒子群算法通过设计一种无质量的粒子来模拟鸟群中的鸟，粒子仅具有两个属性：速度和位置。每个粒子在搜索空间中单独的搜寻最优解，并将其记为当前个体极值，并将个体极值与整个粒子群里的其他粒子共享，粒子群中的所有粒子根据自己找到的当前个体极值和整个粒子群共享的当前全局最优解来调整自己的速度和位置。下面的动图很形象地展示了PSO算法的过程：
![这里写图片描述](https://img-blog.csdn.net/20180803102329735?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RhYWlrdWFpY2h1YW4=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![更新规则](https://img-blog.csdn.net/20180803100337670?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RhYWlrdWFpY2h1YW4=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

公式(1)的第一部分称为【记忆项】，表示上次速度大小和方向的影响；公式(1)的第二部分称为【自身认知项】，是从当前点指向粒子自身最好点的一个矢量，表示粒子的动作来源于自己经验的部分；公式(1)的第三部分称为【群体认知项】，是一个从当前点指向种群最好点的矢量，反映了粒子间的协同合作和知识共享。粒子就是通过自己的经验和同伴中最好的经验来决定下一步的运动。以上面两个公式为基础，形成了PSO的标准形式。

![这里写图片描述](https://img-blog.csdn.net/20180803100428140?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RhYWlrdWFpY2h1YW4=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![这里写图片描述](https://img-blog.csdn.net/20180803102011840?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RhYWlrdWFpY2h1YW4=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



![这里写图片描述](https://img-blog.csdn.net/20180803102146843?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RhYWlrdWFpY2h1YW4=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



# 		ALNS

自适应大邻域搜索(Adaptive Large Neighborhood Search)

关于neighborhood serach，这里有好多种衍生和变种出来的胡里花俏的算法。能看到什么Large Neighborhood Serach，也可能看到Very Large Scale Neighborhood Search或者今天介绍的Adaptive Large Neighborhood Search。**对于这种名字相近，实则大有不同的概念**、

总体关系如下，VLNS分四种

ALNS由LNS拓展而来

**对于一个邻域搜索算法，当其邻域大小随着输入数据的规模大小呈指数增长的时候，**那么我们就可以称该邻域搜索算法为超大规模邻域搜索算法（Very Large Scale Neighborhood Search Algorithm，VLSNA ）。

在大型邻域搜索技术中(如变邻域搜索算法)，通过在当前解的多个邻域中寻找更满意的解，能够大大提高算法在解空间的搜索范围，但是它在使用算子时盲目地将每种算子形成的邻域结构都搜索一遍，缺少了一些**启发式信息的指导**且**时间成本**较高。

自适应大邻域搜索算法弥补了这种不足，首先ALNS算法允许在一次搜索中**搜索多个邻域**，它会根据算子的**历史表现**与**使用次数**选择下一次迭代使用的算子，通过算子间的**相互竞争**来生成当前解的邻域结构，而在这种结构中有很大概率能够找到更好的解。

![image-20220514123441578](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220514123441578.png)



**邻域搜索算法（或称为局部搜索算法）**

其在每次迭代时通过搜索当前解的“邻域”找到更优的解。 **邻域搜索算法设计中的关键是邻域结构的选择，即邻域定义的方式。** 根据以往的经验，邻域越大，局部最优解就越好，这样获得的全局最优解就越好。 但是，与此同时，邻域越大，每次迭代搜索邻域所需的时间也越长。**出于这个原因，除非能够以非常有效的方式搜索较大的邻域，否则启发式搜索也得不到很好的效果。[2]**



## 邻域

**官方一点：****所谓邻域，简单的说即是给定点附近其它点的集合。**在距离空间中，邻域一般被定义为以给定点为圆心的一个圆；而在组合优化问题中，邻域一般定义为由给定转化规则对给定的问题域上每结点进行转化所得到的问题域上结点的集合 （太难懂了 呜呜呜.....）。

**通俗一点：****邻域就是指对当前解进行一个操作(这个操作可以称之为邻域动作)可以得到的所有解的集合。**那么不同邻域的本质区别就在于邻域动作的不同了。

在LNS,解x的邻域N（x）就可以定义为：首先通过利用destroy方法破坏解x，然后利用repair方法重建解x，从而得到的一系列解的集合。

ALNS会为每个destroy和repair方法分配一个权重，通过该权重从而控制每个destroy和repair方法在搜索期间使用的频率。在搜索的过程中，**ALNS**会对各个destroy和repair方法的权重进行**动态调整**，以便获得更好的邻域和解。ALNS使用多种destroy和repair方法，然后再根据这些destroy和repair方法生成的解的质量，选择表现好的destroy和repair方法，再次生成邻域进行搜索。



## 邻域动作

**邻域动作是一个函数，通过这个函数，对当前解s，产生其相应的邻居解集合。**例如：对于一个bool型问题，其当前解为：s = 1001，**当将邻域动作定义为翻转其中一个bit时**，得到的邻居解的集合N(s)={0001,1101,1011,1000}，其中N(s) ∈ S。同理，当将邻域动作定义为互换相邻bit时，得到的邻居解的集合N(s)={0101,1001,1010}。



## **destroy和repair方法**

**一个解x经过destroy和repair方法以后，实则是相当于经过了一个邻域动作的变换。**如下图所示：



![图片](https://mmbiz.qpic.cn/mmbiz_png/mfsg1Cicib4TwlnM8ykNgph8BoWISZOibgFXLLC6OnicxDfn3uuEkibwXKYP4T8icU13ia1KicZ6CrUhUAa09ytIynibakg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

上图是三个CVRP问题的解，上左表示的是**当前解**，上右则是**经过了destroy方法以后的解**（移除了6个customers），下面表示由上右的解**经过repair方法以后最终形成的解**（重新插入了一些customers）。



**上面展示的只是生成邻域中一个解的过程而已，实际整个邻域还有很多其他可能的解。**比如在一个CVRP问题中，将destroy方法定义为：移除当前解x中**15%**的customers。假如当前的解x中有**100名customers**，那么就有**C（100,15）= 100!/(15!×85!) =2.5×10的17次方** 种移除的方式。并且，根据每一种移除方式，又可以有很多种修复的方法。这样下来，一对destroy和repair方法能生成**非常多**的邻居解，**而这些邻居解的集合，就是邻域了。**



## **LNS的具体流程**

![图片](https://mmbiz.qpic.cn/mmbiz_png/mfsg1Cicib4TwlnM8ykNgph8BoWISZOibgFuicNGmh3TMtZLwRgGzEoCA6ibevAQsUVl4kUPNxDjxVVIpHTuASExLgQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

变量x^b记录**目前为止获得的最优解**，x则表示**当前解**，而x^t是**临时解**（便于回退到之前解的状态）。

函数d（·）是**destroy方法**，而r（·）是**repair方法**。

在第2行中，初始化了全局最优解。在第4行中，算法首先用destroy方法，然后再用repair方法来获得临时解x^t。在第5行中，评估临时解x^t的好坏，并以此确定该临时解x^t是否应该成为当前新的解x（第6行）。

评估的方式因具体程序而异，可以有很多种。最简单的评估方式就只接受那些变得更优的解。注：评估准则可以参考**模拟退火的算法原理**，设置一个接受的可能性，效果也许会更佳。

第8行检查新解x是否优于全局最优解x^b。此处 c（x）表示解x的目标函数值。如果新获得的解x更优，那么第9行将会更新全局最优解x^b。在第11行中，检查终止条件。在第12行中，返回找到的全局最优解x^b。

从伪代码可以注意到，**LNS算法不会搜索解的整个邻域，而只是对该邻域进行采样搜索。也就是说，这么大的邻域是不可能一一遍历搜索的，只能采样，搜索其中的一些解而已。**



## **ALNS的具体流程**

**ALNS**对LNS进行了扩展，**它允许在一次搜索中搜索多个邻域（使用多组不同的destroy和repair方法）。**至于搜索哪个邻域呢，***\*ALNS\**会根据邻域解的质量好坏，动态进行选择调整。**

![图片](https://mmbiz.qpic.cn/mmbiz_png/mfsg1Cicib4TwlnM8ykNgph8BoWISZOibgFIF62jaJibKX9IlMHbJxIh7GRNibc5sYxqhmW2wA460dYgFM3a9JBFUtA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

上面就是ALNS伪代码。相对于LNS来说，**新增了第4行和第12行，修改了第2行。**

Ω^−和Ω^+分别表示destroy和repair方法的集合。第2行中的ρ^−和ρ^+分别表示各个destroy和repair方法的权重集合。一开始时，**所有的方法都设置相同的权重。**

第4行**根据ρ^−和ρ^+选择destroy和repair方法**。至于选择哪个方法的可能性大小，是由下面公式算出的：

![图片](https://mmbiz.qpic.cn/mmbiz_png/mfsg1Cicib4TwlnM8ykNgph8BoWISZOibgFfD4D3MAFaQE47BbUfzkKsiaIsic47Q8OESmKyiaMcBtbEWnPzIgCqRxjQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

**总的来说，权重越大，被选中的可能性越大。**(其实此处的选择方法就是遗传算法中的轮盘赌)。



除此之外，权重大小是根据destroy和repair方法的在搜索过程中的表现进行动态调整的（第12行）。具体是怎么调整的呢？这里也给大家说一说：

在ALNS完成一次迭代搜索以后，我们使用下面的函数为每组destroy和repair方法的好坏进行一个评估：[1]

![图片](https://mmbiz.qpic.cn/mmbiz_png/mfsg1Cicib4TwlnM8ykNgph8BoWISZOibgF2VGGf77dTeXDuFhjZFdblSfK0SDJiaUibmXcPkOeOKEqgk9lRn7pgfgw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

**其中，ω_1≥ω_2≥ω_3≥ω_4≥0。为自定的参数。**

假如，a和b是上次迭代中使用的destroy和repair方法。那么其权重更新如下所示：

![图片](https://mmbiz.qpic.cn/mmbiz_png/mfsg1Cicib4TwlnM8ykNgph8BoWISZOibgF5zHfepibVmMdGgDiaGeuiaxsCcX4BURyuvEFRjS5sOgZ95AFfhh8VENNg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

**其中，λ∈[0,1]，为参数。**

![图片](https://mmbiz.qpic.cn/mmbiz_png/mfsg1Cicib4TwlnM8ykNgph8BoWISZOibgF7ZlnZFah3GRibTrI9NoTsDKheoDnhxibTq0dpNBZrichpKS6xoRrDibhsA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

在一个ALNS算法中，有很多个邻域，**每个邻域都可以看做是一组destroy和repair方法生成的。**



# TS

禁忌搜索(Tabu Search,TS,又称禁忌搜寻法）,由美国科罗拉多大学教授Fred Glover在1986年左右提出的，是一个用来跳脱局部最优解的搜索方法。其先创立一个初始化的方案；基于此，算法“移动”到一相邻的方案。经过许多连续的移动过程，提高解的质量。



## TS算法原理详解

邻域

对于组合优化问题，给定任意可行解x，x∈D，D是决策变量的定义域，对于D上的一个映射：N：x∈D→N(x)∈2(D) 其中2(D)表示D的所有子集组成的集合，N(x)成为x的一个邻域，y∈N(x)称为x的一个邻居。



候选集合

候选集合一般由邻域中的邻居组成，可以将某解的所有邻居作为候选集合，也可以通过最优提取，也可以随机提取，例如某一问题的初始解是[1,2,3]，若通过两两交换法则生成候选集合，则可以是[1,3,2],[2,1,3],[3,2,1]中的一个或几个。



禁忌表

禁忌表包括禁忌对象和禁忌长度。由于在每次对当前解的搜索中，需要避免一些重复的步骤，因此将某些元素放入禁忌表中，这些元素在下次搜索时将不会被考虑，这些被禁止搜索的元素就是禁忌对象；
禁忌长度则是禁忌表所能接受的最多禁忌对象的数量，若设置的太多则可能会造成耗时较长或者算法停止，若太少则会造成重复搜索。



评价函数

用来评价当前解的好坏，TSP问题中是总旅程距离。



特赦规则

禁忌搜索算法中，迭代的某一步会出现候选集的某一个元素被禁止搜索，但是若解禁该元素，则会使评价函数有所改善，因此我们需要设置一个特赦规则，当满足该条件时该元素从禁忌表中跳出。



终止规则

一般当两次迭代得到的局部最优解不再变化，或者两次最优解的评价函数差别不大，或者迭代n次之后停止迭代，通常选择第三种方法。



## 举例详述TS算法过程

现有一架飞机，从A点出发，需要经过B,C,D,E,F之后返回A点，且每个点只能经过一次，最后返回A点，求最短路径。

该问题是一个Hamilton回路问题，其中起点和终点已经固定，因此我们可以将解形式记为，例如【A,D,C,F,E,A】,每次只需变换中间两个元素即可，现在我们将禁忌长度设置为2，候选集合长度定义为4，迭代次数为100

给定任意初始解 x1=【A,D,C,F,E,A】f(x1)=10，历史最优为10



![image-20220514163948756](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220514163948756.png)

我们发现对x1交换D和E时，f最优，此时x2=【A,E,C,F,D,A】 f(x2)=6，历史最优为6，将D-E放入禁忌表中

![image-20220514164003539](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220514164003539.png)


我们发现对x2交换C和D时，f最优，此时x3=【A,E,D,F,C,A】 f(x3)=5，历史最优为5，将D-C放入禁忌表中

![image-20220514164014575](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220514164014575.png)
此时我们发现对x3交换D和C时最优，但是由于D-C已经在禁忌表中，因此我们退而求其次，对x3交换F和D，此时x4=【A,E,F,D,C,A】 f(x4)=10，历史最优为5，将F-D放入禁忌表中，由于禁忌长度为2，因此将最先放入禁忌表中的D-E移除禁忌表

![image-20220514164024885](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220514164024885.png)


此时我们发现对x4交换D和C时最优，虽然D-C已经在禁忌表中，但是f(D-C)<历史最优5，因此满足特赦规则，现在将D-C移除禁忌表，此时x5=【A,E,F,C,D,A】 f(x5)=4，历史最优为4，然后再将D-C放入禁忌表

![image-20220514164034492](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220514164034492.png)

依次迭代下去，当迭代次数超过100时停止迭代，历史最优值即为输出解



# ACO

ACO（Ant Colony algorithm）

由意大利学者Dorigo M等人于1991年首先提出，并首先使用在解决TSP上



## 发展历史

最初为Ant System(蚂蚁系统)AS

改进的蚂蚁系统

- 精英策略的蚂蚁系统(Elitist Ant System, EAS)
- 基于排列的蚂蚁系统(Rank-based AS, ASrank )
- 最大最小蚂蚁系统(MAX-MIN Ant System, MMAS)

最后为蚁群系统（Ant Colony System）



举一个例子来进行说明：

蚂蚁从A点出发，速度相同，食物在D点，可能随机选择路线ABD或ACD。假设初始时每条分配路线一只蚂蚁，每个时间单位行走一步，本图为经过9个时间单位时的情形：走ABD的蚂蚁到达终点，而走ACD的蚂蚁刚好走到C点，为一半路程。

经过18个时间单位时的情形：走ABD的蚂蚁到达终点后得到食物又返回了起点A，而走ACD的蚂蚁刚好走到D点。![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031135312461.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)

假设蚂蚁每经过一处所留下的信息素为一个单位，则经过36个时间单位后，所有开始一起出发的蚂蚁都经过不同路径从D点取得了食物，此时ABD的路线往返了2趟，每一处的信息素为4个单位，而ACD的路线往返了一趟，每一处的信息素为2个单位，其比值为2：1

寻找食物的过程继续进行，则按信息素的指导，蚁群在ABD路线上增派一只蚂蚁（共2只），而ACD路线上仍然为一只蚂蚁。再经过36个时间单位后，两条线路上的信息素单位积累为12和4，比值为3：1。

若按以上规则继续，蚁群在ABD路线上再增派一只蚂蚁（共3只），而ACD路线上仍然为一只蚂蚁。再经过36个时间单位后，两条线路上的信息素单位积累为24和6，比值为4：1。

若继续进行，则按信息素的指导，最终所有的蚂蚁会放弃ACD路线，而都选择ABD路线。这也就是前面所提到的正反馈效应。






## 蚁群算法的基本原理:

1、蚂蚁在路径上释放信息素。

2、碰到还没走过的路口，就随机挑选一条路走。同时，释放与路径长度有关的信息素。

3、信息素浓度与路径长度成反比。后来的蚂蚁再次碰到该路口时，就选择信息素浓度较高路径。

4、最优路径上的信息素浓度越来越大。

5、最终蚁群找到最优寻食路径。



## 人工蚁群和真实蚁群

![img](https://img-blog.csdn.net/20170513193336977?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvenVvY2hhb18yMDEz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



## 蚂蚁系统

AS算法求解TSP有两大步骤：路径构建与信息素更新方式。

### 2.1 路径构建

每个蚂蚁都随机选择一个城市作为其出发城市，并维护一个路径记忆向量，用来存放该蚂蚁依次经过的城市。 蚂蚁在构建路径的每一步中，按照一个随机比例规则选 择下一个要到达的城市。
随机比例规则如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031143926849.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)



### 2.2 信息素更新

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031144153587.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)

此处的Cnn 表示最短路径的长度。



为了模拟蚂蚁在较短路径上留下更多的信息素，当所有蚂蚁到达终点时，必须把各路径的信息素浓度重新更新一次，信息素的更新也分为两个部分：
首先，每一轮过后，问题空间中的所有路径上的信息素都会发生蒸发，然后，所有的蚂蚁根据自己构建的路径长度在它们本轮经过的边上释放信息素

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020103114444456.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031144549777.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)



信息素挥发（evaporation）信息素痕迹的挥发过程是每个连接上的信息素痕迹的浓度自动逐渐减弱的过程，这个挥发过程主要用于避免算法过快地向局部最优区域集中，有助于搜索区域的扩展。
信息素增强（reinforcement）增强过程是蚁群优化算法中可选的部分，称为离线更新方式（还有在线更新方式）。这种方式可以实现 由单个蚂蚁无法实现的集中行动。基本蚁群算法的离线更新方式是在蚁群中的m只蚂蚁全部完成n城市的访问后，统一对残留信息进行更新处理。

举一个例子，是整个的过程

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031145033543.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)

矩阵D表示各城市间的距离

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031154054982.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031161738894.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031165057822.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020103116523242.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031165258457.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)



## 改进的蚁群算法

上述基本的AS算法，在不大于75城市的TSP中，结果还是较为理想的，但是当问题规模扩展时， AS的解题能力大幅度下降。进而提出了一些改进版本的AS算法，这些AS改进版本的一个共同点就是增强了蚂蚁搜索过程中对最优解的探索能力，它们之间的差异仅在于搜索控制策略方面。



### 3.1 精英策略的蚂蚁系统(Elitist Ant System, EAS)

AS算法中，蚂蚁在其爬过的边上释放与其构建路径长度成反比的信息素量，蚂蚁构建的路径越好，我们可以想象，当城市的规模较大时，问题的复杂度呈指数级增长，仅仅靠这样一个基础单一的信息素更新机制引导搜索偏向，搜索效率有瓶颈。

因而精英策略(Elitist Strategy) 被提出，通过一种“额外的手段”强化某些最有可能成为最优路径的边，让蚂蚁的搜索范围更快、更正确的收敛。

- 在算法开始后即对所有已发现的最好路径给予额外的增强，并将随后与之对应的行程记为Tb(全局最优行程)；

- 当进行信息素更新时，对这些行程予以加权，同时将经过这些行程的蚂蚁记为“精英”，从而增大较好行程的选择机会。

其信息素的更新方式如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020103117062680.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)



## 3.2 基于排列的蚂蚁系统(Rank-based AS, ASrank )

精英策略被提出后，人们提出了在精英策略的基础上，对其余边的信息素更新机制加以改善

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031171654684.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)



### 3.3 最大最小蚂蚁系统(MAX-MIN Ant System, MMAS)



#### 首先思考几个问题：

问题一：对于大规模的TSP，由于搜索蚂蚁的个数有限，而初始化时蚂蚁的分布是随机的，这会不会造成蚂蚁只搜索了所有路径中的小部分就以为找到了最好的路径，所有的蚂蚁都很快聚集在同一路径上，而真正优秀的路径并没有被搜索到呢？
问题二：当所有蚂蚁都重复构建着同一条路径的时候，意味着算法的已经进入停滞状态。此时不论是基本AS、EAS还是ASrank ， 之后的迭代过程都不再可能有更优的路径出现。这些算法收敛的 效果虽然是“单纯而快速的”，但我们都懂得欲速而不达的道理， 我们有没有办法利用算法停滞后的迭代过程进一步搜索以保证找到更接近真实目标的解呢？



#### 对于MAX-MIN Ant System

1.该算法修改了AS的信息素更新方式，只允许迭代最优蚂蚁（在本次迭代构建出最短路径的蚂蚁），或者至今最优蚂蚁释放信息素；
2.路径上的信息素浓度被限制在[MAX，MIN ]范围内；
3.另外，信息素的初始值被设为其取值上限，这样有助于增加算法初始阶段的搜索能力。
4.为了避免搜索停滞，问题空间内所有边上的信息素都会被重新初始化。



#### 改进如下

##### 改进1借鉴于精华蚂蚁系统，但又有细微的不同。

- 在EAS中，只允许至今最优的蚂蚁释放信息素，而在MMAS中，释放信息素的不仅有可能是至今最优蚂蚁，还有可能是迭代最优蚂蚁。

- 实际上，迭代最优更新规则和至今最优更新规则在MMAS中是交替使用的。
- 这两种规则使用的相对频率将会影响算法的搜索效果。
- 只使用至今最优更新规则进行信息素的更新，搜索的导向性很强，算法会很快的收敛到Tb附近；反之，如果只使用迭代最优更新规则，则算法的探索能力会得到增强，但搜索效率会下降。

##### 改进2是为了避免某些边上的信息素浓度增长过快，出现算法早熟现象。

- 蚂蚁是根据启发式信息和信息素浓度选择下一个城市的。

- 启发式信息的取值范围是确定的。
- 当信息素浓度也被限定在一个范围内以后，位于城市i的蚂蚁k选择城市j作为下一个城市的概率pk(i,j)也将被限制在一个区间内。

##### 改进3，使得算法在初始阶段，问题空间内所有边上的信息素均被初始化τmax的估计值，且信息素蒸发率非常小（在MMAS中，一般将蒸发率设为0.02）。

- 算法的初始阶段，不同边上的信息素浓度差异只会缓慢的增加，因此，MMAS在初始阶段有着较基本AS、EAS和ASrank更强搜索能力。

- 增强算法在初始阶段的探索能力有助于蚂蚁“视野开阔地”进行全局范围内的搜索，随后再逐渐缩小搜索范围，最后定格在一条全局最优路径上。

##### 改进4，当算法接近或者是进入停滞状态时，问题空间内所有边上的信息素浓度都将被初始化，从而有效的利用系统进入停滞状态后的迭代周期继续进行搜索，使算法具有更强的全局寻优能力。



### 4.蚁群系统（Ant Colony System）

上述EAS、ASrank 以及MMAS都是对AS进行了少量的修改而获得了更好的性能。1997年，蚁群算法的创始人Dorigo在Ant colony system：a cooperative learning approach to the traveling salesman problem一文中提出了一种新的具有全新机制的ACO 算法——蚁群系统，是蚁群算法发展史上的又一里程碑。



ACS与AS之间存在三方面的主要差异：

1.使用一种**伪随机比例规则**选择下一个城市节点， 建立开发当前路径与探索新路径之间的平衡。
2.**信息素全局更新规则**只在属于至今最优路径的边上蒸发和释放信息素。
3.**新增信息素局部更新规则**，蚂蚁每次经过空间内的某条边，他都会去除该边上的一定量的信息素，以增加后续蚂蚁探索其余路径的可能性。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031192851313.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031193258206.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201031193334100.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4MDQ4NzU2,size_16,color_FFFFFF,t_70#pic_center)

上面的介绍链接

[(25条消息) 蚁群算法详解（含例程）_馋学习的身子的博客-CSDN博客_蚁群算法](https://blog.csdn.net/qq_38048756/article/details/109383971)




## 基本蚁群的两个过程:

(1)状态转移



(2)信息素更新

计算方法：历史累计信息素-信息素挥发量+蚂蚁行走释放量

### (1)状态转移

为了避免残留信息素过多而淹没启发信息，在每只蚂蚁走完一步或者完成对所有n个城市的遍历(也即一个循环结束)后，要对残留信息进行更新处理。

由此，t+n时刻在路径(i,j)上的信息量可按如下规则进行调整： 

![img](https://img-blog.csdn.net/20170513194029042?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvenVvY2hhb18yMDEz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



### (2)信息素更新模型

蚁周模型（Ant-Cycle模型）

蚁量模型（Ant-Quantity模型）

蚁密模型（Ant-Density模型）



区别：

1.蚁周模型利用的是全局信息，即蚂蚁完成一个循环后更新所有路径上的信息素；

2.蚁量和蚁密模型利用的是局部信息，即蚂蚁完成一步后更新路径上的信息素。

![img](https://img-blog.csdn.net/20170513194317235?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvenVvY2hhb18yMDEz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



流程图

![img](https://img-blog.csdn.net/20170513194429440?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvenVvY2hhb18yMDEz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



蚁群算法中主要参数的选择:

![img](https://img-blog.csdn.net/20170513194531223?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvenVvY2hhb18yMDEz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



蚁群算法中主要参数的理想选择如下:

![img](https://img-blog.csdn.net/20170513194725240?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvenVvY2hhb18yMDEz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



国内外，对于离散域蚁群算法的改进研究成果很多，例如自适应蚁群算法、基于信息素扩散的蚁群算法等，这里仅介绍离散域优化问题的自适应蚁群算法。

自适应蚁群算法：对蚁群算法的状态转移概率、信息素挥发因子、信息量等因素采用自适应调节策略为一种基本改进思路的蚁群算法。

自适应蚁群算法中两个最经典的方法：蚁群系统(AntColony System, ACS)和最大-最小蚁群系统(MAX-MINAnt System, MMAS)。



蚁群系统对基本蚁群算法改进:

①蚂蚁的状态转移规则不同；

![img](https://img-blog.csdn.net/20170513195325373?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvenVvY2hhb18yMDEz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

②全局更新规则不同；

![img](https://img-blog.csdn.net/20170513195352937?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvenVvY2hhb18yMDEz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

③新增了对各条路径信息量调整的局部更新规则

![img](https://img-blog.csdn.net/20170513195445407?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvenVvY2hhb18yMDEz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



[(25条消息) 蚁群算法原理及其实现(python)_fanxin_i的博客-CSDN博客_蚁群算法的基本原理](https://blog.csdn.net/fanxin_i/article/details/80380733?ops_request_misc=%7B%22request%5Fid%22%3A%22165250880516782425128431%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=165250880516782425128431&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-80380733-null-null.142^v9^pc_search_result_control_group,157^v4^control&utm_term=蚁群算法&spm=1018.2226.3001.4187)



# DE

[差分进化算法](https://so.csdn.net/so/search?q=差分进化算法&spm=1001.2101.3001.7020)（Differential Evolution，DE)于1997年由Rainer Storn和Kenneth Price在遗传算法等进化思想的基础上提出的，本质是一种多目标（连续变量）优化算法（MOEAs），用于求解多维空间中整体最优解。

差分进化算法相对于遗传算法不同之处在于遗传算法是根据适应度值来控制父代杂交，变异后产生的子代被选择的概率值，在最大化问题中适应值大的个体被选择的概率相应也会大一些。而差分进化算法变异向量是由父代差分向量生成，并与父代个体向量交叉生成新个体向量，直接与其父代个体进行选择。显然差分进化算法相对遗传算法的逼近效果更加显著。



## 差分进化算法的流程

### 1.种群的初始化

在解空间中随机均匀产生M个个体，每个个体由n维向量组成            ![img](https://img-blog.csdnimg.cn/20190711160432260.png) 

​           

第i个个体的第j维值取值方式如下：                         ![img](https://img-blog.csdnimg.cn/20190711160507945.png)                

对于群体规模参数M，一般介于5×n与10×n之间，但不能少于4×n。



### 变异

在第g次迭代中，从种群中随机选择3个个体Xp1(g),Xp2(g),Xp3(g),并且所选择的个体不一样，那么这三个个体生成的变异向量为：

![img](https://img-blog.csdnimg.cn/20190711160821915.png)

其中，Δp2,p3(g)=Xp2(g)−Xp3(g)是一个差分向量，F是缩放因子。

对于缩放因子F来说，一般取值为[0,2]之间进行选择，通常情况下去0.5.



在这里说一下参数F的自适应调整：

将变异算子中随机选择的三个个体进行从优到劣进行排序，得到Xb,Xm,Xw，他们对于的适应度是fb,fm,fw，变异算子改为：
![img](https://img-blog.csdnimg.cn/20190711161300408.png)

同时，F的取值，根据生成差分向量的两个个体自适应变化：

![img](https://img-blog.csdnimg.cn/20190711161425479.png)



在这里附上一些变异策略：

![img](https://img-blog.csdnimg.cn/20190711161506685.png)



### 交叉

![img](https://img-blog.csdnimg.cn/20190711161537907.png)

其中cr是属于[0,1]之间的值，为交叉概率。

其中参数cr的自适应调整如下：

![img](https://img-blog.csdn.net/20170905234317224?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzc0MjMxOTg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

其中fi是个体xi的适应度值，fmin和fmax分别是当前种群中最差和最优个体的适应度值，f~是当前种群的适应度的平均值，crl和cru分别是cr的下限与上限，一般crl = 0.1,cru = 0.6。



### 选择

![img](https://img-blog.csdnimg.cn/20190711162239754.png)

对于每一个个体来说，得到的解要好于或者持平于个体通过变异，交叉，选择达到全部最优。







# GA

遗传算法将一个问题的解空间编码，每一个编码代表一个个体，建立一个包含潜在的解的群体作为种群。其中，编码中的每一位代表一个基因，环境作用由适应度函数模拟，适应度函数是判断某个解的优劣程度的函数，通常是目标函数本身或其修改形式。

选择又称为选择算子，是指参照适应值函数，按照预先选定的策略随机从父代中挑选一些个体生存下来，剩下的个体则被淘汰。

交叉是指仿照自然界基因传递的过程交配，对存活下来的父代个体的某些基因进行优化组合，办法是将两个父代个体某些对应位置的基因互换，以产生新的个体。

变异是指对编码的某些位置上的基因按一定概率进行的改变。



## *2.选择策略*

### *2.1轮盘赌选择法*

轮盘赌选择法是依据个体的适应度值计算每个个体在子代中出现的概率，并按照此概率随机选择个体构成子代种群。轮盘赌选择策略的出发点是适应度值越好的个体被选择的概率越大。因此，在求解最大化问题的时候，我们可以直接采用适应度值来进行选择。但是在求解最小化问题的时候，我们必须首先将问题的适应度函数进行转换，以将问题转化为最大化问题。下面给出最大化问题求解中遗传算法轮盘赌选择策略的一般步骤：

(1) 将种群中个体的适应度值叠加，得到总适应度值==1 ，其中 为种群中个体个数。

(2) 每个个体的适应度值除以总适应度值得到个体被选择的概率

(3) 计算个体的累积概率以构造一个轮盘。

(4) 轮盘选择：产生一个[0,1]区间内的随机数，若该随机数小于或等于个体的累积概率且大于个体1的累积概率，选择个体进入子代种群。

*重复步骤(4)次，得到的个体构成新一代种群。*

### *2.2 随机遍历抽样法*

像轮盘赌一样计算选择概率，只是在随机遍历选择法中等距离的选择体，设npoint为需要选择的个体数目，等距离的选择个体，选择指针的距离是1/npoint，第一个指针的位置由[0，1/npoint]的均匀随机数决定。

### 2.3 锦标赛选择法

锦标赛方法选择策略每次从种群中取出一定数量个体，然后选择其中最好的一个进入子代种群。重复该操作，直到新的种群规模达到原来的种群规模。具体的操作步骤如下：

(1) 确定每次选择的个体数量(本文以占种群中个体个数的百分比表示)。

(2) 从种群中随机选择个个体(每个个体入选概率相同) 构成组，根据每个个体的适应度值，选择其中适应度值最好的个体进入子代种群。

(3) 重复步骤(2)次，得到的个体构成新一代种群。

需要注意的是，锦标赛选择策略每次是从个个体中选择最好的个体进入子代种群，因此可以通用于最大化问题和最小化问题，不像轮盘赌选择策略那样，在求解最小化问题的时候还需要将适应度值进行转换。



## 交叉

一些交叉算子的总结，了解一下

[(25条消息) 遗传算法中几种交叉算子小结_Ladd7的博客-CSDN博客_交叉算子](https://blog.csdn.net/u012750702/article/details/54563515)



## 变异

在交叉操作过后形成的新个体，有一定的概率会发生基因变异，与选择操作一样，这个概率成为变异概率pm，一般来说变异概率设置得很小，一般pm ≤ 0.05。

适用于二进制编码和实数编码的变异算子：
最简单的做法就是：对交叉后代集中每个后代的每一位，产生一个随机数r∈[0,1]，若r≤Pm 则将该位取反，否者该位不变。

- 基本位变异（Simple Mutation）：对个体编码串中以变异概率、随机指定的某一位或某几位仅因座上的值做变异运算。
- 均匀变异（Uniform Mutation）：分别用符合某一范围内均匀分布的随机数，以某一较小的概率来替换个体编码串中各个基因座上的原有基因值。（特别适用于在算法的初级运行阶段）
- 边界变异（Boundary Mutation）：随机的取基因座上的两个对应边界基因值之一去替代原有基因值。特别适用于最优点位于或接近于可行解的边界时的一类问题。
- 非均匀变异：对原有的基因值做一随机扰动，以扰动后的结果作为变异后的新基因值。对每个基因座都以相同的概率进行变异运算之后，相当于整个解向量在解空间中作了一次轻微的变动。
- 高斯近似变异：进行变异操作时用符号均值为Ｐ的平均值，方差为P**2的正态分布的一个随机数来替换原有的基因值。

另外：
如果变异概率很大，那么整个搜索过程就退化为一个随机搜索过程。所以，比较稳妥的做法是，进化过程刚刚开始的时候，取p为一个比较大的概率，随着搜索过程的进行，p逐渐缩小到0附近。



## 替换策略

除了父子个体的选择、重组、变异之外，还需要考虑的是替换策略——生成的新个体如何取代旧的个体。
有如下几种：

- 如果生成一个子个体，可以用来替换掉最差的父个体。这个替换规则可以（或必须）基于子个体的适应度优于父个体。
- 玻尔兹曼（Boltzmann）选择可以用来确定子个体是否替换父个体。
- 新的子个体替换掉种群中的劣势个体。

 [GA_CVRP.py](pythoncode\GA_CVRP\GA_CVRP.py) 



# -----------------



# 						GA-geatpy

![image-20220426213022324](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220426213022324.png)



![image-20220426213109004](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220426213109004.png)

![image-20220426213209306](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220426213209306.png)



 [精英保存的遗传算法测试.py](pythoncode\精英保存的遗传算法测试.py) 这篇geatpy的版本为2.3.0

文章最后有个遗传算法的框架



# 			    GA丨geatpy库|TSP

本文是利用geatpy库书写的一个TSP问题求解过程。geatpy库是仿照matlab中的gatbx工具箱对应开发的，是华南农业大学、暨南大学、华南理工大学等优秀硕士以及一批优秀校友、优秀本科生组成的团队自研开发的国产python库。

![image-20220426205126301](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220426205126301.png)



https://mp.weixin.qq.com/s/F1NAYPphDR7B7grSGmCpfQ

 [geatpy解决TSP.py](pythoncode\geatpy解决TSP.py) 



# 	 	GA|TSP

20个城市

桌面有geatpy教程

 [遗传算法解决TSP.py](pythoncode\遗传算法解决TSP\遗传算法解决TSP.py) 

https://mp.weixin.qq.com/s/FX_KGdGuEa3VglN74MDIrA



# 							TS|TSP

本算法在求解时主要设置两个禁忌表：全局禁忌表和局部禁忌表，在邻域搜索时采用随机成对交换两城市位置获取新的路径。其中全局禁忌表存储迭代过程中最近n代的结果（n为禁忌长度），局部禁忌表存储每一代领域搜索时遍历到的新路径，以实现全局禁忌（全局不重复搜索）和局部禁忌（邻域遍历不重复搜索）。

禁忌搜索算法对初始解比较敏感，即表现较好的初始解有利于提高算法收敛速度和求解质量。本算法尝试了两种初始化方法：

​		一是随机构造法，这是最常见的初始解构造方法。

​	二是引入贪婪算法构造初始解，过程如下：（1）随机生成出发城市，（2）选择距离当前城市最近的城市作为下一个城市，（3）不断重复步骤（2）直到经过所有的城市。



可以看出贪婪算法构造的初始解表现较佳，且后续优化过程能够较快收敛于最优解。

本算法用的贪婪算法 [禁忌搜索算法解决TSP.py](pythoncode\禁忌搜索算法解决TSP\禁忌搜索算法解决TSP.py) 

https://mp.weixin.qq.com/s/5lUSDsiMfKeiYlwSTJyFfA



# 							ACO|TSP



流程图

![image-20220515092036137](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220515092036137.png)



https://mp.weixin.qq.com/s/pnSeZmG6Y6UoMPOKeTrV-Q

 [蚁群算法解决TSP.py](pythoncode\蚁群算法解决TSP\蚁群算法解决TSP.py) 此算法迭代很慢



算法介绍里的例子-50个城市TSP

 [AC0-TSP50.py](pythoncode\ACO-TSP\AC0-TSP50.py) 





# 									SA|TSP

思路一求解流程：
（1） 贪婪算法构造初始解作为当前解（SA对初始解不敏感，但贪婪算法构造的初始解能够提高求解效率）；
（2） 当前解随机交换生成100个个体，选择最优解作为当代最优解；
（3） 当代最优解优于历史最优解，则更新历史最优解和当前解，否则以一定概率接受当代最优解作为当前解；
（4） 执行退火操作，判断是否满足结束条件，满足退出迭代，否则继续执行步骤(2)-(3)。

 [思路一.py](pythoncode\模拟退火算法求解TSP-思路一\思路一.py) 

思路二：模拟退火算法擅长改造现有求解规则，本算法引入模拟退火思想对贪婪构造过程进行扰动，以扩大搜索空间。首先认为贪婪构造中选择距当前城市最近的城市作为下一个城市是最优的，在此基础上引入概率，以p概率接受最近城市作为下一个城市，以p2概率接受第二近的城市，以p3概率接受第三个城市，以此类推，当前面的最近城市都不接受作为下一个城市时，最后一个城市则必须接受。概率p尝试采用S型函数p = 1/(1+math.exp(-0.03*T))，概率随T变小而变小，p的值域为（0.5,1）。

 [思路二.py](pythoncode\模拟退火算法求解TSP-思路二\思路二.py) 

https://mp.weixin.qq.com/s/gItXl4-iLr4Id7BYDr9kvw



# ALNS|TSP

在本代码中的设计中，权重更新并不是每次选择destroy和repair算子都会更新，而是需要**每隔一定迭代次数后（内循环结束后）**，权重才会更新。因为在本代码中只有两组破坏修复算子组合可以选，因此ALNS需要抉择出好的那组。如是每次迭代都更新权重，那么对于本身随机性较强的ALNS，每次更新这种操作会放大这种随机性，就可能会出现差的那组算子组合的选择概率会一直被提高，好的算子组合概率迟迟不被提高。因此，本文在迭代中设置有一个内循环和外循环，只有内循环结束，达到j_max时，才会进行权重更新，既降低了这种随机性，也综合多次的选择结果，不会降低算法的总体性能。



其权重更新公式如下所示：

![图片](https://mmbiz.qpic.cn/mmbiz_png/VN4lDs1jJOrETYB9K6kVicrLAciccVOH2iasILoicvXwxOkQctDVhdqPP269bX5qDzibHZ073tYXpliaOSVw2tdgqL0w/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

式中，Wd为算子权重，Sd为算子分数，Ud为算子的使用次数。



在官方给的ALNS算法伪代码中，没有涉及权重更新这个步骤。因此小编重新写了一个更容易理解的算法流程，以下是本文中ALNS算法的求解流程图:

![图片](https://mmbiz.qpic.cn/mmbiz_png/VN4lDs1jJOokwKyyibQGBDp4hN2r4tkBJDjSry7YXjFTmhRaibCU2bAJnP8yKbPDU2reaOaBcEHh7gvLrEGrL7zg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



运行程序，迭代1000次后，最终的路线图和迭代过程图都如下所示，可以看到ALNS算法在迭代60次左右的时候就已经收敛了。经过多次的实验可以发现，在求解tsp问题中，ALNS收敛速度较快，且收敛的适应度值相比于遗传算法、禁忌搜索算法会更加稳定。

 [ALNS-TSP.py](pythoncode\ALNS-TSP\ALNS-TSP.py) 



# PSO|TSP

 [PSO-TSP.py](pythoncode\PSO-TSP\PSO-TSP.py) 



# elkai-TSP

elkai是python的第三方库，专门用于解决TSP问题，目前已知能够在规模达到315个节点的问题中求解出最优方案。elkai本身的实现基于大名鼎鼎的LKH算法，该算法被认为是目前解决TSP问题最有效的算法。

  elkai可在Windows、Linux、OS X等系统的python3.5及以上版本中使用

elkai官方网址传送门：https://pypi.org/project/elkai/



返回的解方案solutionbest没有给出完整解方案(少一个终点0),故我们在代码中在解方案最末尾加上了编号0。官方网站上还有种解决方式是使用函数elkai.solve_float_matrix(),但这种方法精度不够，计算出的结果不稳定，不建议使用。

调用elkai解TSP不仅能够得到非常好的解方案，同样用时也很少(其中一个原因是该库用的C语言编译封装实现的)，在我的台式机上运行100次上述代码，规模31的问题平均用时只有0.05秒。



对于调用elkai其实存在一个坑，常出现在距离矩阵内部存在较多浮点数的算例中，这会导致哪怕在很多该类小规模的问题中都算不出最优解

原因是elkai库对浮点数的小数部分不敏感，所以想要避开这个坑的话，我们就需要将输入elkai的距离矩阵乘上一个较大的倍数，例如10000，迫使减小传入的数据小数部分对整个解方案的影响

 [调用elkai(lkh算法)解TSP.py](pythoncode\调用elkai-TSP\调用elkai(lkh算法)解TSP.py) 



# GA&TS|TSP

## 改进思想

禁忌搜索算法的核心思想是不重复已经搜索过的解，以提高搜索的效率。在改进遗传算法上，可以对交叉算子和变异算法做出禁忌操作，具体改进想法如下：
①建立全局禁忌表，存储每一代中的最优解，在所有交叉和变异操作中作为禁忌对象；
②建立局部禁忌表，在交叉算子中，局部禁忌表为上一代种群和当前已交叉生成的子代种群，在变异算子中，局部禁忌表为上一代种群、交叉后待变异的种群和当前已变异生成子代种群；



经过测试，多次求解质量的平均水平相比经典遗传算法有了一定的改进。

 [GA&TS-TSP.py](pythoncode\GA&TS-TSP\GA&TS-TSP.py) 



# GA&PSO|TSP

粒子群算法的核心思想是不断向当前最优解和历史最优解的方向移动，以达到最优解，在改进遗传算法上的思路是在交叉操作上，以一定的概率（轮盘赌操作）与当前最优解或历史最优解进行交叉操作。



结果粒子群算法改进遗传算法的表现与经典遗传算法相比，表现更差了，深入分析后也可以理解这一结果：在我设计的算法中，只按一定的概率与当前最优解或者历史最优解进行交叉，一定程度上使得种群极易陷入局部解，且缺乏多样性的种群很难跳出局部最优解。尝试引入“灾变”技术进行改进（代码中有保留，灾变系数n注释掉的部分），但是效果也不佳，归根到底，种群多样性少会使得种群搜索效率变得较低。
 [GA&PSO-TSP.py](pythoncode\GA&PSO-TSP\GA&PSO-TSP.py) 



# GA&TS&PSO|TSP

粒子群算法改进的效果不佳，一定部分原因是收敛太快导致使得搜索效率低下，为了提高搜索效率，这里再引入禁忌搜索算法的思想尝试进行改进，改进思路是前面两种算法的结合，即在交叉算子和变异算子中引入禁忌搜索的思想，在交叉算子中叠加粒子群算法的思想。



结果：在粒子群改进的基础上加入禁忌搜索算法思想，整体效果比遗传-粒子群算法和经典遗传算法效果稍微稳定一些，但稍微逊色于遗传-禁忌搜索算法。



上面三种分别尝试了使用禁忌搜索算法、粒子群算法对遗传算法进行改进，并尝试结合三种算法的思想构建遗传-禁忌搜索-粒子群算法，在以上三种改进中，遗传-禁忌搜索算法的求解效果是比较好的，当然这也与算法设计有很大的关系，不同的算子设计可能带来不同的效果，而粒子群算法改进遗传算法的效果适得其反，对比单纯用【粒子群算法求解TSP问题】的设计，在遗传算法上可能是选择操作使得进行交叉的种群多样性较少（粒子群和遗传算法都有交叉操作），进而降低了对解空间的搜索效率。

学习完TSP问题求解系列，最大的感受就是初始解的构造上对求解效率的影响非常大，良好的初始解可以减少大量冗余的遍历过程，且极易获得较优解。但是，也存在某些问题上可能很难设计出较好的初始解（比如多约束的VRP问题），这种情况下问题的求解对算法依赖比较大，需要选中更加合适的算法或者结合多种算法思想进行求解。


# -----------------







# 				  	改进PSO|VRPTW



## PSO改进

传统PSO便于处理连续性问题，我们在参考PSO处理VRTTW问题paper的基础上提出了我们的改进算法。主要有以下优势：

1. **改进了位置的更新公式。**使用轮盘赌算法，使得粒子在保持该粒子，粒子最优解，粒子群最优解之间选择，并且有变异因子防止陷入局部最优。与其他的使用粒子速度来进行优化相比，本算法具有更好的收敛性。
2. 结合了两篇paper的长处：**加入了早到的时间惩罚、晚到的罚金惩罚， 将输入订单提前排序，显著提高收敛速度。**
3. **能同时处理软、硬时间窗的VRPTW问题。**
4. **车辆载重量不同的情况下依然能求解，且能产生很好的解。**
5. **按路线、时间线画图等形式直观的展示了解决方案。**



## 数据集

我们采用经典的Solomon数据集作为输入："http://w.cba.neu.edu/~msolomon/problems.htm "
我们在R101、C101、RC101数据集上进行了实验，可见效果非常好：



[改进版粒子群算法求解VRPTW (qq.com)](https://mp.weixin.qq.com/s/jPVKMxsmWWf7cqxBKUEKTg)

 [PSO_plus_plus.py](pythoncode\PSO-for-VRPTW-master\PSO_plus_plus.py) 



# 				PSO|VRPTW

这个算法比较巧妙的地方在于我们将每一个可能的路线规划图作为一个粒子（注意是将整个规划图作为一个粒子而不是规划图中的节点，我们让粒子进行运动并且不断修改它的运动方向，其实就是调整规划图中连接节点各条边的指向，使得粒子位置被不断进行更新，新的粒子对应着各条边被调整后的整幅规划图），我们算法的最终目的就是得到最优的规划图(也就是最好的粒子)，该粒子的适应值(车辆的travelTime和route number)达到最优



data.txt格式说明：首行代表仓库，后面100行代表顾客，分别是顾客编号，X坐标，Y坐标，货物需求，最早服务开始时间，最迟服务开始时间，和服务所需时间。

输出：迭代次数+全局最优适应值（规划图的闭环数量），随着迭代次数增加，适应值不断得到优化，目前是迭代100次。

本程序可以將路线数量从17左右优化到5目前最优解是4圈。



https://mp.weixin.qq.com/s/5iHIgdRPkr33KsV2PNIDHg

用的是matlab，文件在桌面，里面有两篇论文

# 

# 扫描、节约算法| VRP

1.1节约算法(Savings Algorithm)是 Carke 和 Wight 在 1964 年提出的。它是目前用来解决 VRP 模型最有名的启发式算法。

节约算法是用来解决运输车辆数目不确定(运输车辆数目在 VRP 问题中是一个决策变量) 的 VRP 问题，这个算法对有向和无向问题同样有效。

![image-20220426200216009](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220426200216009.png)



1.2扫描算法(Sweep Algorithm)是 Gillett 和 Miller 在 1974 年首先提出的、它也是用于求解车辆数目不限制的 CVRP 问题。

扫描算法分 4 个步骤完成：

①初始扫描点的确定

②以起始点 vo  作为极坐标系的原点，并以连通图中的任意一顾客点和原点的连线定义为角度零，建立极坐标系。然后对所有的顾客所在的位置，进行坐标系的变换， 全部都转换为极坐标系。

③分组。从最小角度的顾客开始，建立一个组，按逆时针方向，将顾客逐个加入到组中，直到顾客的需求总量超出了负载限制。然后建立一个新的组，继续按逆时针方向，将顾客继续加入到组中。

④重复②的过程，直到所有的顾客都被分类为止。

⑤路径优化。对各个分组内的顾客点，就是一个个单独的 TSP 模型的线路优化问题，可以用 TSP 模型的方法对结果进行优化，选择一个合理的路线。



1.3  最近插入法

最近插入法是由 Rosenkrantz 和 Stearns 等人在 1977 提出的一种用于解决 TSP

问题的算法，它比最近邻点法复杂，但是可以得到相对比较满意的结果。其步骤是：

  (1)找到距离 Clk 最小的节点，形成一个子回路(v1, vk)

  (2)在剩下的节点中，寻找一个距离子回路中某一个节点最近的节点。

  (3)在子回路中找到一条弧(i, j)，使得C𝑖𝑖𝑖𝑖 + C𝑖𝑖𝑖𝑖 − C𝑖𝑖𝑖𝑖，最小，然后将节点 vk 加人到子回路中，插人到节点 vi 和 vj 之间；用两条新弧(i, k)(k, j)代替原来的弧(i, j)。

  (4)重复(2)、(3)步骤，直到所有的节点都加人到子回路中。



https://mp.weixin.qq.com/s/Lvnqw7PDKJtUB4H9eBeG_A

代码为文件夹“节约算法解决CVRP”，只有节约算法的代码 [vrp.py](pythoncode\节约算法解决CVRP\vrp.py)  [main.py](pythoncode\节约算法解决CVRP\main.py)  [data1.csv](pythoncode\节约算法解决CVRP\data1.csv)  [data2.csv](pythoncode\节约算法解决CVRP\data2.csv)  [扫描算法介绍及题目.pdf](pythoncode\节约算法解决CVRP\扫描算法介绍及题目.pdf) 





# 			 			改进SA|CVRP

模拟退火算法(Simulated Annealing, SA)的思想借鉴于固体的退火原理，当固体的温度很高的时候，内能比较大，固体的内部粒子处于快速无序运动，当温度慢慢降低的过程中，固体的内能减小，粒子的慢慢趋于有序，最终，当固体处于常温时，内能达到最小，此时，粒子最为稳定。

创新点：**采用2-opt算法与模拟退火算法相融合**，使算法性能提高。



问题规模100个商店，8辆车从仓库出发完成配送后返回仓库，车辆容量约束为200，仓库和商店坐标和需求如Data.xlsx。求解最短的行驶距离

result.txt文件中给出一条最优路径，在当前坐标和容量的条件下求解出最短距离为833，且已知最优距离为826.x，说明该程序求解的结果较为靠近最优解

将CVRP问题转换为TSP问题进行求解，得到TSP问题的优化解后再考虑车辆容量约束进行路径切割，得到CVRP问题的解。这样的处理方式可能会影响CVRP问题解的质量，但简化了问题的求解难度。

![image-20220427102209917](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220427102209917.png)

[(25条消息) Python实现VRP常见求解算法——模拟退火（SA）_Better.C的博客-CSDN博客](https://blog.csdn.net/python_n/article/details/114416957?spm=1001.2014.3001.5502)

 [SA_CVRP.py](pythoncode\SA_CVRP\SA_CVRP.py) 





# 				  		DPSO|CVRP

 [DPSO_CVRP.py](pythoncode\DPSO_CVRP\DPSO_CVRP.py) 

[(25条消息) Python实现VRP常见求解算法——离散粒子群（DPSO）_Better.C的博客-CSDN博客_python求解vrp](https://blog.csdn.net/python_n/article/details/113811576?spm=1001.2014.3001.5502)



# 							DQPSO|CVRP

离散量子行为粒子群算法

 [DQPSO_CVRP.py](pythoncode\DQPSO_CVRP\DQPSO_CVRP.py) 

[(25条消息) Python实现VRP常见求解算法——离散量子行为粒子群算法（DQPSO）_Better.C的博客-CSDN博客_量子行为粒子群算法](https://blog.csdn.net/python_n/article/details/114434563?spm=1001.2014.3001.5502)



[(25条消息) 【超参数寻优】量子粒子群算法（QPSO） 超参数寻优的python实现_Luqiang_Shi的博客-CSDN博客_量子粒子群](https://blog.csdn.net/Luqiang_Shi/article/details/84757727)

上面讲的是量子粒子群



PSO算法的缺点：
1、需要设定的参数（惯性因子w ，局部学习因子c 1 和全局学习因子c 2）太多，不利于找到待优化模型的最优参数。
2、粒子位置变化缺少随机性，容易陷入局部最优的陷阱。



量子粒子群优化（Quantum Particle Swarm Optimization，QPSO）算法取消了粒子的移动方向属性，粒子位置的更新跟该粒子之前的运动没有任何关系，这样就增加了粒子位置的随机性



量子粒子群算法中引入的新名词：
mbest：表示pbest的平均值，即平均的粒子历史最好位置。



![image-20220511155722713](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220511155722713.png)



# 				  			ALNS|CVRP

自适应大邻域算法

#### repair算子中greedy repair

根据被移除的需求节点插入已分配节点序列中每个可能位置的目标函数增量大小，依次选择目标函数增量最小的需求节点与插入位置组合，直到所有被移除的需求节点都重新插入为止（可简单理解为，依次选择使目标函数增量最小的需求节点与其最优的插入位置）；

![image-20220513162858469](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220513162858469.png)



#### repair算子中regret repair

计算被移除节点插回到已分配节点序列中n个次优位置时其目标函数值与最优位置的目标函数值的差之和，然后选择差之和最大的需求节点及其最优位置。（可简单理解为，优先选择n个次优位置与最优位置距离较远的需求节点及其最优位置。）；

![image-20220513163327401](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220513163327401.png)



#### 更新算子权重

对于算子权重的更新有两种策略，一种是每执行依次destory和repair更新一次，另一种是每执行pu次destory和repair更新一次权重。前者，能够保证权重及时得到更新，但却需要更多的计算时间；后者，通过合理设置pu参数，节省了计算时间，同时又不至于权重更新太滞后。这里采用后者更新策略。

![image-20220513165645250](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220513165645250.png)

 [ALNS_CVRP.py](pythoncode\ALNS_CVRP\ALNS_CVRP.py) 



# 				   TS|CVRP

禁忌搜索(Tabu Search)

 [TS_CVRP.py](pythoncode\TS_CVRP\TS_CVRP.py) 



# 					 ACO|CVRP



# 				   DE|CVRP



# 					 GA|CVRP

 [GA_CVRP.py](pythoncode\GA_CVRP\GA_CVRP.py) 

# ALNS原|CVRP



## **生成算例**

我们在指定大小的地图上随机生成算例,其中，客户规模设为20，客户需求随机范围为5~20，横纵坐标均在0~100的范围内均匀随机生成，车辆最大容量为80。



## **生成初始解**

  关于CVRP问题的初始解，我们选用扫描算法来生成，通常扫描算法得到的各路径中客户点顺序是按角度大小排列的，这里我们得到各路径后再利用LK算子进行局部优化，然后输出最终的初始解



## **破坏算子**

​    我们选用了三种破坏算子:

​      1.随机移除客户点；

​      2.最坏距离移除客户点+小扰动；

​      3.移除总里程最大路径+小扰动。

所谓小扰动，即在计算出各客户点/路径移除后的里程减少量的基础上，乘上一个随机范围接近且不大于1的随机数，再根据移除规则选择待移除客户点，例如本文选择的扰动随机数的范围为[0.7,1]。



## **修复算子**

​    我们选用两种修复算子：

​      1.贪婪插入；

​      2.贪婪插入+小扰动

注意，在插入过程中，我们通过循环遍历将当前待插入点插入到各弧中比较最好的插入位置时，需要计算插入后的里程增量，此时的里程计算只需要涉及当前待插入点和弧两端点，而不需要计算整条路径长，这样可以大大减少计算量。



## **其他参数设置**

​    本文的接受准则采用模拟退火接受准则，初始温度为30，衰减率为0.94，并采用再升温策略，即当温度衰减到低于0.1时，当前温度恢复到初始温度30。

​    因为算子数量较少，本文设置每隔20次迭代更新一次算子权值。

​    最大迭代次数设为10000(主要靠达到最大未改进次数终止算法，所以最大迭代次数可以设的大一点)，最大未改进次数为200。



## **utils代码中部分组件介绍**

​    1.距离矩阵的计算：我们计算各节点间距离矩阵，可能首先想到的方法是通过两层节点的循环，计算一个个节点对的距离，然后填入建好的空距离矩阵中，但python中循环较为耗时，这里推荐使用scipy库来完成距离矩阵的计算:

```
"计算距离矩阵，分别传入x,y的坐标array，返回距离矩阵"def genDistanceMat(x,y):    X=np.array([x,y])    distMat=ssp.distance.pdist(X.T)    sqdist=ssp.distance.squareform(distMat)    return sqdist
```

​    2.路径总里程的计算:对于一条路径总里程的计算，通常会采用循环的方式累加计算，例如对于一条route的计算如下:

```
dis = 0#route里程长度for i in range(1,len(route)):    dis += distance_matrix[route[i-1]][[route[i]]#其中distance_matrix为距离矩阵
```

这里我们可以利用numpy快速处理批量数组的特性，减少代码量的同时加快计算速度:

```
dis = np.sum(distance_matrix[route[:-1],route[1:]])
```

 [ALNS原_CVRP.py](ALNS-CVRP\ALNS原_CVRP.py) 



# CW|CVRP

节约里程法（CW算法）是针对CVRP问题开发的一个[贪婪算法](https://so.csdn.net/so/search?q=贪婪算法&spm=1001.2101.3001.7020)，基本思想是不断优先将**合并后距离节约最大的线路**进行合并，节约里程法分为两种：序贯法和并列法，两者基本思想一样，区别在于计算过程中处理线路的顺序，序贯法是一辆车一辆车的装，而并列法是允许并行装车。两种方法很难评价优劣，在不同的[数据集](https://so.csdn.net/so/search?q=数据集&spm=1001.2101.3001.7020)上存在不同的优劣表现。



序贯的算法写的比较冗余，在写并列算法的时候进行了一点改进。总体来说序贯法“倾向于”使用较少的车辆，相应的总里程可能长一点，而并列算法倾向于使用更多的车辆，其总里程可能稍微短一些。节约里程法可以获得一个比较优的解，但相比启发式算法（遗传算法啊、粒子群算法等），其解的质量还是差一些，不过CW算法是确定性算法，计算时间比启发式算法少的多。


 [CW序贯法.py](pythoncode\CW-CVRP\CW序贯法.py) 

 [CW并列法.py](pythoncode\CW-CVRP\CW并列法.py) 



# 			-----------------



# 		GA|MDCVRP

在CVRP作了如下修改

- Model数据结构类中采用demand_dict属性存储需求节点集合，depot_dict属性储存车场节点集合，demand_id_list属性储存需求节点id集合，distance_matrix属性储存任意节点间的欧式距离；
- Node数据结构类中去掉node_seq属性，增加depot_capacity属性记录车场的车队限制
- splitRoutes函数中分割车辆路径时分配最近车场作为其车辆提供基地（满足车场车队数量限制）

 [GA_MDCVRP.py](pythoncode\GA_MDCVRP\GA_MDCVRP.py) 







# 		TS|MDCVRP





# 		ACO|MDCVRP



# 				DPSO|MDCVRP	





# 							SA|MDCVRP



# 						ALNS|MDCVRP



# ==========



# GA|MDVRPTW

进行了改进（"splitRoutes"函数），在分割车辆路径时不仅考虑了车辆容量限制，还考虑了节点的时间窗约束，以此使得分割后的路径可行。

 [GA_MDVRPTW.py](pythoncode\GA_MDVRPTW\GA_MDVRPTW.py) 



# ==========



# GA|MDHFVRPTW

heterogeneous fixed 异构固定

也可求解HFVRPTW

本算法继续采用将所有需求节点构造为一个有序列表的编码方式，并运用在交叉、变异等寻优过程中。当需要评估染色体质量时需采用split方法，在考虑车场、异构固定车队、服务时间窗等约束条件下，将有序列表分割为多个可行的车辆路径。并做了适当微调。整个算法的函数调用关系如下图（采用PyCallGraph绘制）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2ec3d9956e8c47209c65767c0b15617b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQmV0dGVyLkM=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

 [GA_MDHVRPTW.py](pythoncode\GA_MDHFVRPTW\GA_MDHVRPTW.py) 





# ==========



# 谷歌优化工具(OR-Tools - Google Optimization Tools)

google提供的一套运筹规划的运算工具，是一个开源、快速、可移植的软件模块，它针对不同的规划场景，提供不同的求解方式，用以解决组合优化问题法。



## 示例代码-线性优化

### 需要求解的问题

```python
max Z = 2*x + 2*y + 3*z
约束：2*x + 7*y + 3*z <= 50
      3*x - 5*y + 7*z <= 45
      5*x + 2*y - 6*z <= 37
      x,y,z integers
```

### 结果

```
Maximum of objective function: 35

x value:  7
y value:  3
z value:  5
```

 [test.py](pythoncode\test.py) 



#   强化学习

回顾并总结了强化学习在库存控制、路径优化、装箱配载和车间作业调度等方面的研究成果，,并将最新的深度强化学习以及传统方法在运筹学领域的应用研究进行了对比分析

强化学习(Reinforcement Learning,以下简称RL），是机器学习的一个重要分支，它是基于Agent与环境进行交互

## **1 强化学习简介**

RL基本结构如图1所示,在每个时间步长内,Agent感知环境状态st,并根据既定的策略采取行动at,得到执行at所获得的立即奖赏rt,同时使环境由状态st转换为st+1。RL的目的是让Agent学习到一种策略,实现状态到动作的映射,Agent在该策略的指导下行动,获得最大的奖赏。

![image-20220511141941654](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220511141941654.png)

![image-20220511142441272](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220511142441272.png)

RL能较好地克服传统运筹学建模方法的缺点:(1)在建模难、建模不准确的问题方面,RL可以通过Agent与环境的不断交互,学习到最优策略;(2)在传统方法难以解决高维度的问题方面,RL提供了包括值函数近似以及直接策略搜索等近似算法;(3)在难以求解动态与随机型问题方面,RL可在Agent与环境之间的交互以及状态转移过程中加入随机因素。RL的这些优点使得其适合求解运筹学领域的大规模动态、随机决策问题,如库存控制、路径优化、装箱配载以及车间作业调度等问题,为运筹优化的研究提供一个新视角。



## **路径优化**

路径优化问题的“不确定性”主要体现在信息演变和信息质量变化这两个方面。

信息演变是指决策者掌握的某些信息有可能 会在实际中随时间发生变化，比如车辆旅行时间受 实时交通路况影响随时发生变化以及在配送服务 时可能有新的顾客产生新的需求等; 

而信息质量变化是指某些信息存在不确定性，比如决策者只能得 知顾客的实际需求是按照某种概率分布存在但无法明确预知客户的需求。学者们将前者称为动态路径优化问题，后者称为随机路径优化问题



#### RL应用于TSP问题文献汇总

![图片](https://mmbiz.qpic.cn/mmbiz_png/NsRTyVKXLxFXdJBwCwYENUZSeG2gvFB8icaXhN9xgiaUEGQNWBuqTNfX6ibyK5YcePsqT1jBQ9AIkuvcekdKS5PXA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



#### RL应用于VRP问题文献汇总

![图片](https://mmbiz.qpic.cn/mmbiz_png/NsRTyVKXLxFXdJBwCwYENUZSeG2gvFB8TvCNBdP4miaTXVGWz8nTH88vnY8YcZhlibcvrgKqjyLmx4344bu0hSbg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



https://mp.weixin.qq.com/s/rDF6a-VjALFD9mCd_S8IWA



#         							A* 算法

A*（念做：A Star）算法是一种很常用的路径查找和图形遍历算法。它有较好的性能和准确度。



## 算法介绍

A*算法最初发表于1968年，由Stanford研究院的Peter Hart, Nils Nilsson以及Bertram Raphael发表。它可以被认为是Dijkstra算法的扩展。

由于借助启发函数的引导，A*算法通常拥有更好的性能。

## 广度优先搜索

为了更好的理解A*算法，我们首先从广度优先（Breadth First）算法讲起。

正如其名称所示，广度优先搜索以广度做为优先级进行搜索。

从起点开始，首先遍历起点周围邻近的点，然后再遍历已经遍历过的点邻近的点，逐步的向外扩散，直到找到终点。

这种算法就像洪水（Flood fill）一样向外扩张，算法的过程如下图所示：



在上面这幅动态图中，算法遍历了图中所有的点，这通常没有必要。对于有明确终点的问题来说，一旦到达终点便可以提前终止算法，下面这幅图对比了这种情况：

![image-20220427113127597](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220427113127597.png)

在执行算法的过程中，每个点需要记录达到该点的前一个点的位置 – 可以称之为父节点。这样做之后，一旦到达终点，便可以从终点开始，反过来顺着父节点的顺序找到起点，由此就构成了一条路径。

## Dijkstra算法

Dijkstra算法是由计算机科学家**Edsger W. Dijkstra**[1]在1956年提出的。

Dijkstra算法用来寻找图形中节点之间的最短路径。

在Dijkstra算法中，需要计算每一个节点距离起点的总移动代价。同时，还需要一个优先队列结构。对于所有待遍历的节点，放入优先队列中会按照代价进行排序。

在算法运行的过程中，每次都从优先队列中选出代价最小的作为下一个遍历的节点。直到到达终点为止。下面对比了不考虑节点移动代价差异的广度优先搜索与考虑移动代价的Dijkstra算法的运算结果：![图片](https://mmbiz.qpic.cn/mmbiz_gif/QtPIxk7nOVccsFicib3Fnh48dkGSlOAf6yGicibGFqThzgD22ghGx9BmuX42Fz1PLriayqOPbKaf7WA9oJM8G0pwNWg/640?wx_fmt=gif&wxfrom=5&wx_lazy=1)

## 最佳优先搜索

在一些情况下，如果我们可以预先计算出每个节点到终点的距离，则我们可以利用这个信息更快的到达终点。

其原理也很简单。与Dijkstra算法类似，我们也使用一个优先队列，但此时以每个节点到达终点的距离作为优先级，每次始终选取到终点移动代价最小（离终点最近）的节点作为下一个遍历的节点。这种算法称之为最佳优先（Best First）算法。

这样做可以大大加快路径的搜索速度，如下图所示：

![图片](https://mmbiz.qpic.cn/mmbiz_gif/QtPIxk7nOVccsFicib3Fnh48dkGSlOAf6yWrVNMnh73Z3rlAbt8VIdicmpOWxrjaqUSIZotIk8ib8zl5hyuiaRQcvOg/640?wx_fmt=gif&wxfrom=5&wx_lazy=1)

但这种算法会不会有什么缺点呢？答案是肯定的。

因为，如果起点和终点之间存在障碍物，则最佳优先算法找到的很可能不是最短路径，下图描述了这种情况。

![图片](https://mmbiz.qpic.cn/mmbiz_gif/QtPIxk7nOVccsFicib3Fnh48dkGSlOAf6yZ2dUJgXyqIbwHl6ibRTSaIcXD6HKKt3bT5ImNmGPWmkpOXdlnnyMX7w/640?wx_fmt=gif&wxfrom=5&wx_lazy=1)

## A*算法

对比了上面几种算法，最后终于可以讲解本文的重点：A*算法了。

下面的描述我们将看到，A*算法实际上是综合上面这些算法的特点于一身的。

A*算法通过下面这个函数来计算每个节点的优先级。

> f(n)=g(n)+h(n)

其中：

- f(n) 是节点n的综合优先级。当我们选择下一个要遍历的节点时，我们总会选取综合优先级最高（值最小）的节点。
- g(n) 是节点n距离起点的代价。
- h(n)是节点n距离终点的预计代价，这也就是A*算法的启发函数。关于启发函数我们在下面详细讲解。

另外，A*算法使用两个集合来表示待遍历的节点，与已经遍历过的节点，这通常称之为`open_set`和`close_set`。

完整的A*算法描述如下：

![image-20220427113938770](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220427113938770.png)

### 启发函数

上面已经提到，启发函数会影响A*算法的行为。

- 在极端情况下，当启发函数h(n)始终为0，则将由g(n)决定节点的优先级，此时算法就退化成了Dijkstra算法。
- 如果h(n)始终小于等于节点n到终点的代价，则A*算法保证一定能够找到最短路径。但是当h(n)h(n)的值越小，算法将遍历越多的节点，也就导致算法越慢。
- 如果h(n)完全等于节点n到终点的代价，则A*算法将找到最佳路径，并且速度很快。可惜的是，并非所有场景下都能做到这一点。因为在没有达到终点之前，我们很难确切算出距离终点还有多远。
- 如果h(n)的值比节点n到终点的代价要大，则A*算法不能保证找到最短路径，不过此时会很快。
- 在另外一个极端情况下，如果h(n)相较于g(n)大很多，则此时只有h(n)产生效果，这也就变成了最佳优先搜索。

由上面这些信息我们可以知道，通过调节启发函数我们可以控制算法的速度和精确度。因为在一些情况，我们可能未必需要最短路径，而是希望能够尽快找到一个路径即可。这也是A*算法比较灵活的地方。



对于网格形式的图，有以下这些启发函数可以使用：

- 如果图形中只允许朝上下左右四个方向移动，则可以使用曼哈顿距离（Manhattan distance）。
- 如果图形中允许朝八个方向移动，则可以使用对角距离。
- 如果图形中允许朝任何方向移动，则可以使用欧几里得距离（Euclidean distance）。



## 关于距离

### 曼哈顿距离

如果图形中只允许朝上下左右四个方向移动，则启发函数可以使用曼哈顿距离，它的计算方法如下图所示：

![图片](https://mmbiz.qpic.cn/mmbiz_png/QtPIxk7nOVccsFicib3Fnh48dkGSlOAf6yvk2SyqtSqH9K2HfGaEElluxJWr3h50xIGX3Qia262Y7rZ92ibZSGYJgw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

计算曼哈顿距离的函数如下，这里的D是指两个相邻节点之间的移动代价，通常是一个固定的常数。

![image-20220427115246187](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220427115246187.png)



### 对角距离

如果图形中允许斜着朝邻近的节点移动，则启发函数可以使用对角距离。它的计算方法如下：

![图片](https://mmbiz.qpic.cn/mmbiz_png/QtPIxk7nOVccsFicib3Fnh48dkGSlOAf6ygAZZYtaIFaHwSficxxKEibcSUPa3Rzs6uGuqf9Txhtib3FrOsYXfwv14w/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

计算对角距离的函数如下，这里的D2指的是两个斜着相邻节点之间的移动代价。如果所有节点都正方形，则其值就是√2*D。

![image-20220427115400080](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220427115400080.png)



### 欧几里得距离

如果图形中允许朝任意方向移动，则可以使用欧几里得距离。

欧几里得距离是指两个节点之间的直线距离，因此其计算方法也是我们比较熟悉的：![图片](https://mmbiz.qpic.cn/mmbiz_png/QtPIxk7nOVccsFicib3Fnh48dkGSlOAf6yqo4mia4rgx2LjWA93jda35O7gl9j0fQOrX5nFvnzR6yu5TIHjWQu7mg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

其函数表示如下：

![image-20220427115429920](C:\Users\35405\AppData\Roaming\Typora\typora-user-images\image-20220427115429920.png)

 [a_star.py](pythoncode\a-star-algorithm-master\a_star.py) 

https://mp.weixin.qq.com/s/xS8apO0fxDCb6Dz6SEjHCg

链接里有代码的详细解释



# 	   英雄联盟的路径规划

这个算法的应用场景非常广泛，例如：地图上的导航路线设计、机器人的行走路线设计、游戏里人物的行走路线规划等等，是最贴切我们现实生活的一种问题场景的求解算法了。

当我们利用技术，实时修改数据文件里的数据内容，是不是就实现了游戏里的动态路径规划了？

用的matlab，文件名为er_wei

https://mp.weixin.qq.com/s/kN4SNApNUe3c9P0-nlvHCw

