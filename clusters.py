# -*- coding: utf-8 -*-
"""
@author: swx
"""
#整理数据集，返回列、行名和数据
def readFile(path="data.txt"):
    lines = [line for line in open(path)]
    #列的名称，这里指单词
    colNames = lines[0].split("\t")[1:] 
    rowNames = []
    data = []
    for line in lines[1:]:
        blog = line.strip().split("\t")
        rowNames.append(blog[0])
        data.append([float(x) for x in blog[1:]])
    return rowNames,colNames,data
from math import sqrt
#定义距离
def pearson(v1,v2):
    #求和
    sum1 = sum(v1)
    sum2 = sum(v2)
    #求平方和
    sum1Sq = sum([pow(x,2) for x in v1])
    sum2Sq = sum([pow(y,2) for y in v2])
    #求乘积和∑xy
    pSum = sum([v1[i] * v2[i] for i in range(len(v1))])
    #计算pearson
    numerator = pSum - (sum1 * sum2 /len(v1))
    denominator = sqrt((sum1Sq - pow(sum1,2) / len(v1)) * (sum2Sq - pow(sum2,2) / len(v1)))
    if denominator == 0:
        return 0
    #v1和v2 越相似pearson越大
    return 1.0 - numerator / denominator #1-pearson，越相似，值越小
 
#分层聚类，每一个节点都可以看做是二叉树中的一个节点
class Bicluster:
    def __init__(self,vec,left=None,right=None,distance=0.0,id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance
#hierarchical clustering
#缺点：在没有额外的投入的情况下，树形视图并不是拆分数据集，必须计算每两个数据项的关系，
        #并且再合并之后，还需要参与新的计算，计算量也巨大
def hcluster(rows,distance = pearson):
    distances = {}
    currentClustId = -1
    #开始聚类,也就是开始建立树，每一条数据就是一个节点，全是叶子节点
    clust = [Bicluster(rows[i],id = i) for i in range(len(rows))]
    while len(clust) > 1:
        lowestPair = (0,1)
        closest = distance(clust[0].vec,clust[1].vec)
        
        for i in range(len(clust)):
            for j in range(i + 1,len(clust)):
                #(a,b)是否在distances中
                if (clust[i].id,clust[j].id) not in distances:
                    distances[(clust[i].id,clust[j].id)] = distance(clust[i].vec,clust[j].vec)
                d = distances[(clust[i].id,clust[j].id)]
                if d < closest:
                    closest = d
                    lowestPair = (i,j)
                #计算两个聚类的平均值
        avgVec = [(clust[lowestPair[0]].vec[i] + clust[lowestPair[1]].vec[j])/2.0 for i in range(len(clust[0].vec))]
        newCluster = Bicluster(avgVec,left = clust[lowestPair[0]],right=clust[lowestPair[1]],
                               distance=closest,id=currentClustId)  
        #不在原始数据集上聚类，其id为负                       
        currentClustId -= 1
        del clust[lowestPair[1]] #删除的时候需要逆序
        del clust[lowestPair[0]]
        clust.append(newCluster)
    return clust[0]
def printClust(clust,labels = None,n = 0):
    for i in range(n):print " "
    if clust.id < 0:
        #负数标记这是一个分支
        print "-"
    else:
        if labels == None:
            print clust.id
        else :
            print labels[clust.id]
    if clust.left != None:
        printClust(clust.left,labels=labels,n=n+1)
        printClust(clust.right,labels=labels,n=n+1)
#将data转置
def transposeData(data):
    newdata = []
    for i in range(len(data[0])):
        row = []
        for j in range(len(data)):
            row.append(data[j][i])
        newdata.append(row)
    return newdata
  
import random  
#Kmeans
def kmeans(rows,distance = pearson,k=4):
    #确定每个维度取值范围(min,max)*n维
    ranges = [(min([row[i] for row in rows]),max([row[i] for row in rows]))
                for i in range(len(rows[0]))]
    #随机创建K个中心点           random(max - min) + min
    clusters = [[random.random()*(ranges[i][1] - ranges[i][0])+ranges[i][0] 
                for i in range(len(rows[0]))]
                        for j in range(k)]
    lastmatches = None
    for t in range(100):
        print "%d times iteration " %t
        bestmatches = [[]for i in range(k)]
        #遍历整个数据集
        for j in range(len(rows)):
            row = rows[j]
            bestmatch = 0#中心点
            for i in range(k): #遍历每个中心点
                d = distance(clusters[i],row)
                if d < distance(clusters[bestmatch],row):
                    bestmatch = i #
            bestmatches[bestmatch].append(j)#放入样本的编号
        #如果聚类不再发生变化，提前结束
        if bestmatches == lastmatches:
            break
        lastmatches = bestmatches
        #移动中心点
        for i in range(k):#
            avgs = [0.0]*len(row[0])#新的中心点
            if len(bestmatches) > 0:#确保每个中心点list中有样本点
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):#同一组的向量相加
                        avgs[m] += rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i]) #avg/点的个数
                clusters[i] = avgs
    return bestmatches
 
#数据集中 值只有1和0，代表有和没有，Tanimoto 系数   
def tainmoto(v1,v2):
    c1,c2,share = 0,0,0
    for i in range(len(v1)):
        if v1[i] != 0:
            c1+=1
        if v2[i] != 0:
            c2 += 1
        if v1[i] != 0 and v2[i] != 0:
            share += 1
    return 1.0 - (float(share) / (c1 + c2 - share))
#对数据进行缩放调整,二维数据（x，y）可以在坐标系中展示出来，但是多维数据则不能，思路：使用欧氏距离
#pearson等距离，在平面展示其相对距离
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
from PIL import Image,ImageDraw
def draw2d(data,labels,jpeg = "a.jpg"):
    img = Image.new('RGB',(2000,2000),(255,255,255))
    draw = ImageDraw.Draw(img)
    for i in range(len(data)):
        x = (data[i][0] + 0.5) *1000
        y = (data[i][1] + 0.5) *1000
        draw.text((x,y),labels[i],(0,0,0))
    img.save(jpeg,'JPEG')
if __name__ == "__main__":
    blogName ,words,data = readFile()
    coordinated = scaleDown(data)
    draw2d(coordinated,blogName)