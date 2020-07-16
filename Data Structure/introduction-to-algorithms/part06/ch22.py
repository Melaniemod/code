# -*-coding:utf-8-*-

from  collections import deque
import math


# 节点类
class Vertex:
    def __init__(self,x):
        self.key = x
        self.color = 'white'
        self.d = 10000
        self.f = 10000
        self.pi = None
        self.adj = []


class Graph:
    def __init__(self,V = []):
        self.V = V


    def Bfs(self,s):
        "广度优先遍历"
        for u in self.V:
            u.color = 'white'
            u.pi = None
            u.d = 0
        s.color = 'gray'
        s.pi = None
        s.d = 0
        Q = [s]
        while Q:
            u = Q.pop()
            print(u.key)
            for v in u.adj:
                if v.color == 'white':
                    v.color = 'gray'
                    v.d = u.d+1
                    v.pi = u
                    Q.append(v)
            u.color = 'black'

    def Dfs(self):
        "深度优先遍历"
        for u in self.V:
            u.color = 'white'
            u.pi = None
        global time
        time = 0
        for u in self.V:
            if u.color == 'white':
                list = [u]
                self.DfsVisit(u,list)
                print("".join([i.key for i in list]))

    def DfsVisit(self,u,list):
        global time
        time += 1
        u.d = time
        u.color = 'gray'
        for v in u.adj:
            if v.color == 'white':
                list.append(v)
                v.pi = u
                self.DfsVisit(v,list)
        u.color = 'black'
        time += 1
        u.f = time

    def GraphTrans(self):
        for u in self.V:
            u.adj = (u.adj,[])

        for u in self.V:
            for v in u.adj[0]:
                v.adj[1].append(u)

        for u in self.V:
            u.adj = u.adj[1]

    def StronglyConnComponents(self):
        self.Dfs()
        self.GraphTrans()
        self.V.sort(key=lambda v:v.f,reverse=True)
        self.Dfs()


class MinTree:
    # todo 最小生成树
    def __init__(self,maps):
        "输入邻接矩阵"
        self.maps = maps
        self.nodenum = self.get_nodenum()
        self.edgenum = self.get_edgenum()

    def get_nodenum(self):
        return len(self.maps)

    def get_edgenum(self):
        cnt = 0
        for i in range(len(self.maps)):
            for j in  range(i):
                if self.maps[i][j] > 0 and (self.maps[i][j]<9999):
                    cnt += 1
        return cnt

    def kruskal(self):
        res = []
        if self.nodenum <=0 or self.edgenum < self.nodenum -1:
            return res
        ls_edge = []
        for i in range(self.nodenum):
            for j in range(i,self.nodenum):
                if self.maps[i][j] < 9999:
                    ls_edge.append([i,j,self.maps[i][j]])
        ls_edge.sort(key=lambda x :x[2])
        group = [[i] for i in range(self.nodenum)]
        # 将在同一个连通分支上的集合合并，i是集合代表
        for edge in ls_edge:
            for i in range(len(group)):
                if edge[0] in group[i]:
                    m = i
                if edge[1] in group[i]:
                    n = i
            if m != n:
                group[m] = group[m]+group[n]
                group[n] = []
                res.append(edge)
        return res

    def prim(self,select_node=[0]):
        res = []
        if self.nodenum <= 0 or self.edgenum < self.nodenum-1:
            return res
        candidate_node = [i for i in range(1,self.nodenum)]
        while candidate_node:
            begin,end,minweight = 0,0,9999
            for i in select_node:
                for j in candidate_node:
                    if self.maps[i][j]<minweight:
                        begin = i
                        end = j
                        minweight = self.maps[i][j]
            res.append([begin,end,minweight])
            candidate_node.remove(end)
            select_node.append(end)
        return res




#
# print(inf>0)
#print(None >2)

class BellmanFordSP(object):
    def __init__(self,Graph,s):
        '''
        :param Graph: 有向图的邻接矩阵
        :param s:  起点Start
        '''
        self.Graph = Graph
        self.edgeTo = []   #用来存储路径结束的横切边（即最短路径的最后一条边的两个顶点）
        self.distTo = []   #用来存储到每个顶点的最短路径
        self.s = s         #起点start
        self.inf = math.inf

    #打印顶点S到某一点的最短路径
    def PrintPath(self,end):
        path = [end]
        while self.edgeTo[end] != None:
            path.insert(0,self.edgeTo[end])    #倒排序
            end = self.edgeTo[end]
        return path

    # 路径中含有正（负）权重环判定，即是判断当前顶点是否存在于一个环中。
    def cycle_assert(self, vote):
        '''
        思路：利用顶点出度、入度，当前顶点满足环的“必要条件”是至少1出度、1入度。
        再判断进行看是否起点能否回到起点的路径判断。两项满足则为环。
        '''
        path = [vote]
        while self.edgeTo[vote] != None:
            path.insert(0,self.edgeTo[vote])
            vote = self.edgeTo[vote]
            if path[0] == path[-1]:
                break

        print(path)
        if path[0] == path[-1]:
            return True
        else:
            return False

    #主程序
    def bellmanford(self):
        d = deque()        #导入优先队列（队列性质：先入先出）
        for i in range(len(self.Graph[0])):  #初始化横切边与最短路径-“树”
            self.distTo.append(self.inf)
            self.edgeTo.append(None)
        self.distTo[self.s] = 0             #将顶点s加入distTo中
        #print(self.edgeTo,self.distTo)
        count  = 0          #计数标志
        d.append(self.Graph[self.s].index(min(self.Graph[self.s])))  #将直接距离顶点S最近的点加入队列
        for i in self.Graph[self.s]:         #将除直接距离顶点S的点外的其他顶点加入队列
            if i != self.inf and count not in d:
                d.append(count)
            count += 1
        for j in d:       #处理刚加入队列的顶点
            self.edgeTo[j] = self.s
            self.distTo[j] = self.Graph[self.s][j]
        #print(d,self.edgeTo,self.distTo)
        #print(d)
        while d:
            count = 0
            vote = d.popleft()        #弹出将该点作为顶点S，重复操作，直到队列为空
            for i in self.Graph[vote]:   #进行边的松弛技术
                if i != self.inf and i > 0 and self.distTo[vote] + i < self.distTo[count]:
                    self.edgeTo[count] = vote
                    self.distTo[count] = self.distTo[vote] + i
                    self.distTo[count] = round(self.distTo[count], 2)
                    if count not in d:
                        d.append(count)

                #处理满足条件且含有正（负）权重环的路径情况
                elif i != self.inf and i < 0 and self.distTo[vote] + i < self.distTo[count]:
                    temp  = self.edgeTo[count]    #建立临时空间存储原横切边
                    #print(vote,count)
                    self.edgeTo[count]  = vote
                    flage = self.cycle_assert(count)   #判读若该点构成环切该点即是起点有事终点，则存在环
                    if flage:                     #有环，消除该环
                        self.edgeTo[count] = temp
                        self.Graph[vote][count] = self.inf
                    else:                       #无环，与第一个if相同处理
                        self.distTo[count] = self.distTo[vote] + i
                        self.distTo[count] = round(self.distTo[count], 2)
                        if count not in d:
                            d.append(count)

                elif i != self.inf and  self.distTo[vote] + i >= self.distTo[count]:
                    self.Graph[vote][count] = self.inf   #删除该无用边
                    #if count not in d:
                        #d.append(count)
                count += 1
            #print(d)

        #print(self.edgeTo,self.distTo)
        for i in range(len(self.Graph[0])):
            path = self.PrintPath(i)
            print("%d to %d(%.2f)：" %(path[0],i,self.distTo[i]),end="")
            if len(path) == 1 and path[0] == self.s:
                print("")
            else:
                for i in path[:-1]:
                        print('%d->' %(i),end = "")
                print(path[-1])




if __name__ == '__main__':
    a,b,c,d,e,f,g,h = [Vertex(i) for i in ['a','b','c','d','e','f','g','h']]

    a.adj = [b]
    b.adj = [c,e,f]
    c.adj = [d,g]
    d.adj = [c,h]
    e.adj = [a,f]
    f.adj = [g]
    g.adj = [f,h]
    h.adj = [h]

    # G = Graph([a,b,c,d,e,f,g,h])
    # G.Dfs()
    # G.Bfs(a)
    # print("===")
    # G.StronglyConnComponents()


    # todo 最小生成树算法
    # max_value = 9999
    # row0 = [0, 7, max_value, max_value, max_value, 5]
    # row1 = [7, 0, 9, max_value, 3, max_value]
    # row2 = [max_value, 9, 0, 6, max_value, max_value]
    # row3 = [max_value, max_value, 6, 0, 8, 10]
    # row4 = [max_value, 3, max_value, 8, 0, 4]
    # row5 = [5, max_value, max_value, 10, 4, 0]
    # maps = [row0, row1, row2, row3, row4, row5]
    # graph = MinTree(maps)
    #
    # print('邻接矩阵为\n%s' % graph.maps)
    # print('节点数量为%d，边数为%d\n' % (graph.nodenum, graph.edgenum))
    # print('------最小生成树kruskal算法------')
    # print(graph.kruskal())
    # print('------最小生成树prim算法')
    # print(graph.prim())



    # todo Bellman-ford算法
    #含有负权重值的图
    inf = math.inf
    Graph = [[inf,inf,0.26,inf,0.38,inf,inf,inf],
             [inf,inf,inf,0.29,inf,inf,inf,inf],
             [inf,inf,inf,inf,inf,inf,inf,0.34],
             [inf,inf,inf,inf,inf,inf,0.52,inf],
             [inf,inf,inf,inf,inf,0.35,inf,0.37],
             [inf,0.32,inf,inf,0.35,inf,inf,0.28],
             #[0.58,inf,0.40,inf,0.93,inf,inf,inf],
             [-1.40,inf,-1.20,inf,-1.25,inf,inf,inf],
             [inf,inf,inf,0.39,inf,0.28,inf,inf],
            ]
      #路径之中含有负权重环图
    Graph1 = [[inf,inf,0.26,inf,0.38,inf,inf,inf],
             [inf,inf,inf,0.29,inf,inf,inf,inf],
             [inf,inf,inf,inf,inf,inf,inf,0.34],
             [inf,inf,inf,inf,inf,inf,0.52,inf],
             [inf,inf,inf,inf,inf,0.35,inf,0.37],
             [inf,0.32,inf,inf,-0.66,inf,inf,0.28],
             [0.58,inf,0.40,inf,0.93,inf,inf,inf],
             [inf,inf,inf,0.39,inf,0.28,inf,inf],
            ]

    Graph2 = [[inf,0,5,inf,inf,inf],
              [inf,inf,inf,30,35,inf],
              [inf,inf,inf,15,20,inf],
              [inf,inf,inf,inf,inf,20],
              [inf,inf,inf,inf,inf,10],
              [inf,inf,inf,inf,inf,inf],
              ]

    Graph3 = [[inf,0,5,inf],
              [inf,inf,inf,35],
              [inf,-7,inf,inf],
              [inf,inf,inf,inf]]

    F = BellmanFordSP(Graph,0)
    F.bellmanford()