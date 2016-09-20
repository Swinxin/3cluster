# cluster Hierarchical clustering and kmeans
对文本数据集分别Hierarchical clustering and kmeans方法
几点值得注意技术
----------
###数据归一化###
rows 存放数据集，找到每一维度的最大值与最小值；得到长度为n的list，每一维度存放元组(min,max)

	ranges = [(min([row[i] for row in rows]),max([row[i] for row in rows]))
                for i in range(len(rows[0]))]
###二维形式展示数据###
为了查看数据的分布情况，往往需要将数据给绘制出来，二维数据可以将数据分布到平面坐标系中，但是三维..n维怎么做呢，可以通过多维度缩放，即样本与样本的相对位置，找到样本之间的距离（欧式，皮尔逊等）在平面中画出相对位置

	def scaleDown(data,distance = pearson,rate = 0.01):
	    n = len(data)
	    #对每一个数据样本计算真是距离  n*n 的矩阵
	    realDist = [[distance(data[i],data[j]) for j in range(n)] 
	                for i in range(0,n)]
	    #随机初始化节点在二维空间中的起始位置  n*2 的矩阵
	    location = [[random.random(),random.random()] for i in range(n)]
	    fakeDist = [[0.0] * n for i in range(n)]
	    lastError = None
	    for m in range(100):
	        #投影后的距离
	        for i in range(n):
	            for j in range(n):
	                #sum([d,d,d]) //
	                fakeDist[i][j] = sqrt( sum([pow(location[i][x] - location[j][x],2)
	                                    for x in range(len(location[i])) ]))
	        #移动节点
	        grad = [[0.0]*2 for i in range(n)]
	        totalError = 0
	        for k in range(n):
	            for j in range(n):
	                if j == k:
	                    continue
	                errorTerm = (fakeDist[j][k]-realDist[j][k]) / realDist[j][k]
	                #误差值等于目标距离与当前距离之间差值的百分比
	                grad[k][0] += ((location[k][0] - location[j][0])/fakeDist[j][k])*errorTerm
	                grad[k][1] += ((location[k][1] - location[j][1])/fakeDist[j][k])*errorTerm
	                
	                #记录总的误差值
	                totalError += abs(errorTerm)
	        print totalError
	        #如果节点移动之后，误差值变大，则程序结束
	        if lastError and lastError < totalError:
	            break
	        lastError = totalError
	        
	        #根据rate参数与grad相乘的结果，移动每一个节点
	        for k in range(k):
	            location[k][0] -= rate*grad[k][0]
	            location[k][1] -= rate*grad[k][1]
	    return location 
