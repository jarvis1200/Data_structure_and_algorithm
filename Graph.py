

class graph_1:
    def __init__(self, n):
        self.list = []
        self.graph = [[] for i in range(n+1)]

    def add_node(self, node):
        for i in node:
            if i not in self.list:
                self.list.append(i)
            else:
                print('node ',i,' already exist')

    def addEdge(self, u, v, is_directed = False):
        
        list = self.list
        graph = self.graph
        if u in list and v in list:
            graph[u].append(v)
            if is_directed == False:
                graph[v].append(u)
        else:
            print('node not exist')
            
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
'''g = graph_1()
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
    print(i, end= ' ')'''



# Create a cycle detection in undirected graph using DFS and BFS

class Node:
    def __init__(self, first, second):
        self.first = first
        self.second = second


# BFS for cycle detection
class Solution:
    def checkforCycle(self, adj , s, vis):
        q = []
        n = Node(s, -1)
        q.append(n)
        vis[s] = 1
        while q != []:
            w = q.pop(0)
            node = w.first
            prev = w.second
            for it in adj[node]:
                if vis[it] == 0:
                    q.append(Node(it, node))
                    vis[it] = 1
                elif prev != it:
                    return True
        return False

    def is_cycle(self, adj, n):
        vis = [0] *(n+1)
        for i in range(1, n):
            if vis[i] == 0:
                if self.checkforCycle(adj, i, vis):
                    return True

        return False


    
    # DFS for cycle detection
    def check_Cycle_DFS(self, node, parent, v, vis):
        vis[node] = 1
        for it in v[node]:
            if vis[it] == 0:
                if self.check_Cycle_DFS(it, node, v, vis):
                    return True
            elif parent != Node:
                return True
        return False

    def isCycle_dfs(self, s, v):
        vis = [0] *(s+1)
        for i in range(1, s):
            if vis[i] ==0:
                if self.check_Cycle_DFS(i, -1, v, vis):
                    return True

        return False


'''C = Solution()
g = graph_1()
g.add_node([0,1,2,3,4,5,6])
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 2)
g.addEdge(3, 4)
g.addEdge(4, 3)
g.addEdge(4, 5)
g.addEdge(5, 4)
g.addEdge(5, 6)
g.addEdge(6, 5)
v = g.addEdge(6, 6)'''

'''A = C.is_cycle(v, 7)
print(A)'''

'''B = C.isCycle_dfs(7, v)
print(B)
'''

#bipartitie graph
class bipartite:
    def check_bipartite(self, node, v, color):
        q = []
        q.append(node)
        color[node] = 1
        while q != []:
            nde = q.pop(0)
            for it in v[nde]:
                if color[it] == -1:
                    q.append(it)
                    color[it] = 1 -color[nde]
                elif color[it] == color[nde]:
                    return False
        return True

    def is_bipartite(self, v, n):
        color = [-1] *(n+1)
        for i in range(1, n):
            if color[i] == -1:
                if self.check_bipartite(i, v, color):
                    return False

        return True



    # bipartite graph using dfs
    def check_bipartite_dfs(self, node, v, color):
        if color[node] == -1:
            color[node] = 1
        for it in v[node]:
            if color[it] == -1:
                color[it] = 1- color[node]
                if self.check_bipartite_dfs(it, v, color) != True:
                    return False
            elif color[it] == color[node]:
                return False
        return True


    def bipartite_dfs(self, n, v):
        color = [-1] * (n+1)
        for i in range(1, n):
            if color[i] == -1:
                if self.check_bipartite_dfs(i, v, color):
                    return False
        return True





'''B = bipartite()
g = graph_1()
g.add_node([1,2,3,4,5,6,7,8])
g.addEdge(1, 2)
g.addEdge(2,3)
g.addEdge(3,4)
g.addEdge(4,5)
g.addEdge(5,6)
g.addEdge(6,7)
g.addEdge(7,2)
g.addEdge(5,8)
v = g.addEdge(8,8)
print(B.is_bipartite(v, 8))
print(B.bipartite_dfs(8, v))'''


# check cycle in directed graph using dfs

class cycle:
    def check_cycle_dfs(self, node, v, vis, dfsvis):
        vis[node] = 1
        dfsvis[node] = 1
        for it in v[node]:
            if vis[it] == -1:
                if self.check_cycle_dfs(it, v, vis, dfsvis) == True:
                    return True
            elif dfsvis[it] == 1:
                return True

        dfsvis[node] = 0
        return False

    def is_cycle_dfs(self, n, v):
        vis = [-1] * (n+1)
        dfsvis = [-1] * (n+1)
        for i in range(1, n+1):
            if vis[i] == -1:
                if self.check_cycle_dfs(i, v, vis, dfsvis):
                    return True
        return False


'''S = cycle()
g = graph_1()
g.add_node([1,2,3,4,5,6,7,8,9])
g.addEdge(1, 2)
g.addEdge(2, 3)
g.addEdge(3, 4)
g.addEdge(3,6)
g.addEdge(4, 5)
g.addEdge(5, 5)
g.addEdge(6, 5)
g.addEdge(7,2)
g.addEdge(7,8)
g.addEdge(8,9)
v = g.addEdge(9,7)'''
'''print(v)


print(S.is_cycle_dfs(9, v))'''



# Topological Sorting using dfs
class topo:
    def findTopoSort(self, node, vis, st, adj):
        vis[node] = 1
        for it in adj[node]:
            if vis[it] == 0:
                self.findTopoSort(it, vis, st, adj)
        st.append(node)
        return st

    def TopoSort(self, n, adj):
        vis = [0] * (n)
        stack = []
        for i in range(1, n):
            if vis[i] == 0:
                self.findTopoSort(i, vis, stack, adj)

        topo = []
        while stack != []:
            topo.append(stack.pop())
        return topo


    #topological sort using kahns algo in bfs

    def findTopoSort_bfs(self, n, adj):
        in_degree = [0]*(n)
         
        # Traverse adjacency lists to fill indegrees of
           # vertices.  This step takes O(V + E) time
        for i in adj:
            for j in adj[i]:
                in_degree[j] += 1
 
        # Create an queue and enqueue all vertices with
        # indegree 0
        queue = []
        for i in range(n):
            if in_degree[i] == 0:
                queue.append(i)
 
        # Initialize count of visited vertices
        cnt = 0
 
        # Create a vector to store result (A topological
        # ordering of the vertices)
        top_order = []
 
        # One by one dequeue vertices from queue and enqueue
        # adjacents if indegree of adjacent becomes 0
        while queue:
 
            # Extract front of queue (or perform dequeue)
            # and add it to topological order
            u = queue.pop(0)
            top_order.append(u)
 
            # Iterate through all neighbouring nodes
            # of dequeued node u and decrease their in-degree
            # by 1
            for i in adj[u]:
                in_degree[i] -= 1
                # If in-degree becomes zero, add it to queue
                if in_degree[i] == 0:
                    queue.append(i)
 
            cnt += 1
 
        # Check if there was a cycle
        
            # Print topological order
        print( top_order)
'''
T  = topo()
g = graph_1()
g.add_node([0,1,2,3,4,5])
g.addEdge(5, 2)
g.addEdge(5, 0)
g.addEdge(4, 0)
g.addEdge(4, 1)
g.addEdge(2, 3)
g.addEdge(1, 1)
g.addEdge(0,0)
v = g.addEdge(3, 1)'''
'''
print(v)
print(T.TopoSort(6, v))
T.findTopoSort_bfs(6, v)
'''

class kahn:
    def is_cycle_kahn(self, n, adj):
        topo = [0] * (n)
        deg = [0] * (n)
        for i in adj:
            for j in adj[i]:
                deg[j] += 1

        q = []
        for i in range(n):
            if deg[i] == 0:
                q.append(i)
        ind = 0
        while q != []:
            node = q.pop(0)
            topo[ind] = node
            ind += 1
            for i in adj[node]:
                deg[i] -= 1
                if deg[i] == 0:
                    q.append(i)
        if ind != n:
            return True
        return False


'''g = graph_1()
g.add_node([0,1,2,3,4,5])
g.addEdge(5, 2)
g.addEdge(5, 0)
g.addEdge(4, 0)
g.addEdge(4, 1)
g.addEdge(2, 3)
g.addEdge(1, 1)
g.addEdge(0,0)
v = g.addEdge(3, 1)'''

'''k = kahn()
print(k.is_cycle_kahn(6, v))'''

# Shortest distance in undirected graph with unit weights

class dis_find:
    def dis_find_und(self,adj, n,s):
        dist = [float('inf')] * n
        q = []
        dist[s] = 0
        q.append(s)
        while q != []:
            node = q.pop(0)
            for it in adj[node]:
                if dist[node] + 1 < dist[it]:
                    dist[it] = dist[node] + 1
                    q.append(it)
        return dist

'''
D = dis_find()  
adj = {0:[1,2],1:[2],2:[3],3:[4],4:[5],5:[0]}
print(D.dis_find_und(adj, 6, 0))'''

#Adjacent list with weight and edge list
class node:
    def __init__(self,):
        self.graph = {}
        self.list = []
        self.u = 0
        self.v = 0
        self.w = 0

    
    # add node , edges , weight
    def add_node(self, node):
        for i in node:
            if i is not self.list:
                self.list.append(i)
                self.graph[i] = []
            else:
                print("Node already exists")
    
    # add edge
    def add_edge(self, u, v, w, is_directed = False):
        if u in self.list and v in self.list:
            self.graph[u].append([v, w])
            if is_directed is False:
                self.graph[v].append([u, w])
        else:
            print("Node doesn't exist")

        return self.graph

class dag:
    def toposort(self,node, vis, st, adj):
        vis[node] = 1
        for it in adj[node]:
            if vis[it[0]] == 0:
                self.toposort(it[0], vis, st, adj)
        st.append(node)
        return st

    def is_dag(self, s, adj, n):
        st = []
        vis = [0] * n
        dist = [float('inf')] * n
        for i in range(n):
            if vis[i] == 0:
                self.toposort(i, vis, st, adj)
        
        dist[s] = 0
        while st:
            node = st.pop()
            if dist[node] != float('inf'):
                for it in adj[node]:
                    if dist[it[0]] > dist[node] + it[1]:
                        dist[it[0]] = dist[node] + it[1]
        return dist

'''n = node()
n.add_node([0,1,2,3,4,5])
n.add_edge(0,1,1,True)
n.add_edge(0,4,1,True)
n.add_edge(1,2,3,True)
n.add_edge(2,3,6, True)
n.add_edge(3,3,0, True)
n.add_edge(4,2,2,True)
n.add_edge(4,5,4,True)
k = n.add_edge(5,3,1,True)'''
'''print(k)
d = dag()
print(d.is_dag(0, k, 6))'''


from queue import PriorityQueue
# dijksta algorithm
class dijksta:
    # priority deque
    def priority_deque(self, q):
        try :
            max = 0
            for i in range(len(q)):
                if q[i] > q[max]:
                    max = i
            
            return q.pop(max)

        except IndexError:
            print()
            exit()

    def dijkstra_path(self, adj, s, n):
        q = []
        dist = [float('inf')] * (n+1)
        dist[s] = 0
        q.append([0,s])
        while q:
            node = self.priority_deque(q)
            u = node[1]
            for v in adj[u]:
                e = v[0]
                w = v[1]
                i = 1
                if i < n and dist[e] > dist[u] + w:
                    dist[e] = dist[u] +w
                    q.append([dist[e], e])
                    i += 1
        return dist[1:]

'''n = node()
n.add_node([1,2,3,4,5])
n.add_edge(1,2,2)
n.add_edge(1,4,1)
n.add_edge(2,1,2)
n.add_edge(2,5,5)
n.add_edge(2,3,4)
n.add_edge(3,2,4)
n.add_edge(3,4,3)
n.add_edge(3,5,1)
n.add_edge(4,1,1)
n.add_edge(4,3,3)
n.add_edge(5,2,5)
v = n.add_edge(5,3,1)'''
'''print(v)
d = dijksta()
print(d.dijkstra_path(v, 1, 5))'''

# Prims algorithm

class MST:
    def priority_deque(self, q):
        try :
            max = 0
            for i in range(len(q)):
                if q[i] > q[max]:
                    max = i
            
            return q.pop(max)

        except IndexError:
            print()
            exit()

    def prim(self, n, adj):
        parent = [None] * (n)
        key = [float('inf')] * (n+1)
        mstset = [False] * (n+1)
        q = []

        key[0] = 0
        parent[0] = -1
        q.append([0,0])
        while q:
            node = self.priority_deque(q)
            u = node[0]
            print(u)
            mstset[u] = True

            for v in adj[u]:
                if( mstset[v[0]] == False and v[1] < key[v[0]]):
                    parent[v[0]] = u
                    key[v[0]] = v[1]
                    q.append([v[0], key[v[0]]])
        return parent

'''g = node()
g.add_node([0,1,2,3,4])
g.add_edge(0,1,2)
g.add_edge(0,3,6)
g.add_edge(1,0,2)
g.add_edge(1,3,8)
g.add_edge(1,4,5)
g.add_edge(1,2,3)
g.add_edge(2,1,3)
g.add_edge(2,4,7)
g.add_edge(3,0,6)
g.add_edge(3,1,8)
g.add_edge(4,1,5)
v = g.add_edge(4,2,7)'''
'''print(v)

p = MST()
print(p.prim(5, v))'''

class union:
    def __init__(self):
        self.parent = [None] * 10
        self.rank = [None] * 10
    
    def make_set(self, n):
        for i in range(n):
            self.parent[i] = i
            self.rank[i] = 0

    def find(self, i):
        if self.parent[i] == i:
             return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        u = self.find(i)
        v = self.find(j)
        rank = self.rank
        if rank[u] < rank[v]:
            self.parent[u] = v

        elif rank[v] < rank[u]:
            self.parent[v] = u

        else:
            self.parent[v] = u
            self.rank[u] += 1

c = union()
c.make_set(10)
c.union(1,5)
'''print(c.parent)
print(c.rank)
print(c.find(1))'''

#kruksal algorithm

class Graph2:
 
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []  # default dictionary
        # to store graph
 
    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
 
    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
       
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])
 
    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
 
        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
 
        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
 
    # The main function to construct MST using Kruskal's
        # algorithm
    def KruskalMST(self):
 
        result = []  # This will store the resultant MST
         
        # An index variable, used for sorted edges
        i = 0
         
        # An index variable, used for result[]
        e = 0
 
        # Step 1:  Sort all the edges in
        # non-decreasing order of their
        # weight.  If we are not allowed to change the
        # given graph, we can create a copy of graph
        self.graph = sorted(self.graph,
                            key=lambda item: item[2])
 
        parent = []
        rank = []
 
        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
 
        # Number of edges to be taken is equal to V-1
        while e < self.V - 1 and i < len(self.graph):
 
            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
 
            # If including this edge does't
            #  cause cycle, include it in result
            #  and increment the indexof result
            # for next edge
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
            # Else discard the edge
 
        minimumCost = 0
        print ("Edges in the constructed MST")
        for u, v, weight in result:
            minimumCost += weight
            print("%d -- %d == %d" % (u, v, weight))
        print("Minimum Spanning Tree" , minimumCost)
 
# Driver code
'''g = Graph2(6)
g.addEdge(1,4,1)
g.addEdge(1,2,2)
g.addEdge(2,3,3)
g.addEdge(2,4,3)
g.addEdge(1,5,4)
g.addEdge(3,4,5)
g.addEdge(2,6,7)
g.addEdge(3,6,8)
vl= g.addEdge(4,5,9)
print(vl)
'''
 
# Function call
'''g.KruskalMST()'''


class bridge:

    def dfsBridge(self, node , parent, vis, tin, low, timer, adj):
        vis[node] = True
        timer += 1
        tin[node] = timer
        low[node] = timer
        for it in adj[node]:
            if it == parent:
                continue
            if vis[it] == False:
                self.dfsBridge(it, node, vis, tin, low, timer, adj)
                low[node] = min(low[node], low[it])
                if (low[it] > tin[node]):
                    print(str(node) + " <--> " + str(it))
            else:
                low[node]= min(low[node], tin[it])
        return

    def printBridge(self, adj, n):
        vis = [False] * n
        tin = [0] * n
        low = [0] * n
        timer = 0
        for i in range(1,n-1):
            if vis[i] == False:
                self.dfsBridge(i, -1, vis, tin, low, timer, adj)


    # Articulation Point:
    def dfsArti(self, node, parent, vis, tin, low, adj, timer, isarti):
        vis[node] = 1
        timer += 1
        tin[node] = timer
        low[node] = timer
        child = 0
        for it in adj[node]:
            if it == parent:
                continue
            if vis[it] == 0:
                self.dfsArti(it, node, vis, tin, low, adj, timer, isarti)
                low[node] = min(low[node], low[it])
                if low[it] >= tin[node] and parent != -1:
                    isarti[node] = 1
                child += 1
            else:
                low[node] = min(low[node], tin[it])
        if parent != -1 and child > 1:
            isarti[node] = 1
        return 


    def findArticulationPoints(self, adj, n):
        vis = [0] * n
        tin = [0] * n
        low = [0] * n
        isarti = [0] * n
        timer = 0
        for i in range(1, n):
            if vis[i] == 0:
                self.dfsArti(i, -1, vis, tin, low, adj, timer, isarti)
        
        for i in range(1, n):
            if isarti[i] == 1:
                print(str(i) + " is an Articulation Point")

'''g = graph_1(12)
g.add_node([1,2,3,4,5,6,7,8,9,10,11,12])
g.addEdge(1,2)
g.addEdge(1,4)
g.addEdge(2,3)
g.addEdge(4,3)
g.addEdge(4,5)
g.addEdge(5,6)
g.addEdge(6,7)
g.addEdge(6,9)
g.addEdge(7,8)
g.addEdge(8,9)
g.addEdge(8,10)
g.addEdge(10,11)
g.addEdge(11,12)
v = g.addEdge(10,12)'''
    
B = bridge()
#B.printBridge(v, 13)

'''8 <--> 10
5 <--> 6
4 <--> 5'''

#B.findArticulationPoints(v, 13)

'''4 is an Articulation Point
5 is an Articulation Point
6 is an Articulation Point
8 is an Articulation Point
10 is an Articulation Point'''

# kosaraju Algorithm

class kosaraju:
    def K_dfs(self, node, st, adj, vis):
        vis[node] = 1
        for it in adj[node]:
            if vis[it] == 0:
                self.K_dfs(it, st, adj, vis)
        st.append(node)
        return

    def revDFS(self, node, trans, vis):
        vis[node] = 1
        print(str(node)+ ' ')
        for it in trans[node]:
            if vis[it] == 0:
                self.revDFS(it, trans, vis)
        return

    def kosaraju_algo(self, adj, n):
        vis = [0] * n
        st = []
        for i in range(n):
            if vis[i] == 0:
                self.K_dfs(i, st, adj, vis)

        trans = [[] for _ in range(n)] 

        for i in range(n):
            vis[i] = 0
            for it in adj[i]:
                trans[it].append(i)

        while st:
            node = st.pop()
            if vis[node] == 0:
                print('Strong_Connected_Graph')
                self.revDFS(node, trans, vis)
                print()


'''g = graph_1(6)
g.add_node([1,2,3,4,5,6])
g.addEdge(1,2,True)
g.addEdge(2,3,True)
g.addEdge(3,1,True)
g.addEdge(2,4,True)
g.addEdge(4,5,True)
g.addEdge(5,6,True)
v = g.addEdge(6,4,True)

K = kosaraju()
K.kosaraju_algo(v, 7)'''

'''Strong_Connected_Graph
1
3
2

Strong_Connected_Graph
4
6
5'''

# bellman ford algorithm
class bellman:
    def bellman_algo(self, adj, s, n):
        dist = [float('inf')] * n
        dist[s] = 0
        for i in range(1, n-1):
            for u,v,w in adj:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
        
        f = 0
        for u,v,w in adj:
            if dist[u] + w < dist[v]:
                print('Negative Cycle')
                f = 1
                break

        if f == 0:
            for i in range(n):
                print(str(i) + ' -> ' + str(dist[i]))
            

m = Graph2(6)
m.addEdge(3,2,6)
m.addEdge(5,3,1)
m.addEdge(0,1,5)
m.addEdge(1,5,-3)
m.addEdge(1,2,-2)
m.addEdge(3,4,-2)
m.addEdge(2,4,3)

print('Graph:' + str(m.graph))
F = bellman()
F.bellman_algo(m.graph, 0, 6)

'''Graph:[[3, 2, 6], [5, 3, 1], [0, 1, 5], [1, 5, -3], [1, 2, -2], [3, 4, -2], [2, 4, 3]]
0 -> 0
1 -> 5
2 -> 3
3 -> 3
4 -> 1
5 -> 2'''



