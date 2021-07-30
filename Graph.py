

class graph:
    def __init__(self):
        self.graph = {}
        self.list = []

    def add_node(self, node):
        for i in node:
            if i not in self.list:
                self.list.append(i)
            else:
                print('node ',i,' already exist')

    def addEdge(self, u, v):
        temp = []
        list = self.list
        graph = self.graph
        if u in list and v in list:
            if u not in graph:
                temp.append(v)
                graph[u] = temp

            elif u in graph:
                temp.extend(graph[u])
                temp.append(v)
                graph[u] = temp

        else:
            print("Nodes don't exist!")

        return graph

    # adjacent matrix
    def BFS_1(self, v, n, start):
        q = []
        bfs = []
        m = n+1
        vis = [0] *m
        bfs.append(start)
        q.append(start)
        vis[start] = 1
        while q:
            i = q.pop(0)
            for j in range(1, n):
                if v[i][j] == 1 and vis[j] == 0:
                    vis[j] = 1
                    bfs.append(j)
                    q.append(j)
        return bfs

    #adjacent list
    def BFS_2(self, v, n, start):
        q = []
        bfs = []
        m= n+1
        vis = [0] * m
        for i in range(start, n):
            if vis[i] == 0:
                q.append(i)
                vis[i] = 1
                while q:
                    node = q.pop(0)
                    bfs.append(node)
                    for j in v[node]:
                        if vis[j] == 0:
                            vis[j] = 1
                            q.append(j)

        return bfs

    # adjacent matrix in recursion
    def Rdfs_1(self,node, vis, v, dfs, n):
        dfs.append(node)
        vis[node] = 1
        for j in range(node, n):
            if v[node][j] == 1 and vis[j] == 0:
                self.Rdfs_1(j,vis,v,dfs, n)
        return dfs

    def DFS_1(self, v, n, i):
        dfs = []
        vis = [0] * (n+1)
        for j in range(1, n):
            if vis[j] == 0:
                self.Rdfs_1(j,vis,v,dfs, n)
        return dfs
        

    def dfs(self,i,vis,v,dfs):
        vis[i] = 1
        dfs.append(i)
        for j in v[i]:
            if vis[j] == 0:
                self.dfs(j,vis,v,dfs)
        return dfs

    def DFS_2(self,v,n,i):
        vis = [0] * (n+1)
        dfs = []
        for j in range(i, n):
            if vis[j] == 0:
                self.dfs(j,vis,v,dfs)
        return dfs
       

    

        
        

        



import sys
sys.setrecursionlimit(5000)
g = graph()
g.add_node([0,1,2,3])
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
v = g.addEdge(3, 3)
print('edges: ',v)

B = g.BFS_2(v, 4,2)

A = g.BFS_1([[0,0,0,0,0,0,0],
 [0,0,1,1,0,0,0],
 [0,1,0,0,1,0,0],
 [0,1,0,0,1,0,0],
 [0,0,1,1,0,1,1],
 [0,0,0,0,1,0,0],
 [0,0,0,0,1,0,0]], 7, 3)

C = g.DFS_1([[0,0,0,0,0,0,0],
 [0,0,1,1,0,0,0],
 [0,1,0,0,1,0,0],
 [0,1,0,0,1,0,0],
 [0,0,1,1,0,1,1],
 [0,0,0,0,1,0,0],
 [0,0,0,0,1,0,0]], 7, 3)



D = g.DFS_2(v, 4, 1)

for i in A:
    print(i, end= ' ')
print()

for i in B:
    print(i, end= ' ')
print()
for i in C:
    print(i, end= ' ')
print()
for i in D:
    print(i, end= ' ')