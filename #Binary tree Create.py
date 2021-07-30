#Binary tree Create using queue
from queue import Queue
import queue

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class tree:

    def __init__(self):
        self.root = None

    def create_tree(self, dataset):
        for i in range(len(dataset)):
            if dataset[i] == None:
                continue
            if self.root is None:
                self.root = Node(dataset[i])
            else:
                current = self.root
                while True:
                    if dataset[i] < current.data:
                        if current.left is None:
                            current.left = Node(dataset[i])
                            break
                        else:
                            current = current.left
                    else:
                        if current.right is None:
                            current.right = Node(dataset[i])
                            break
                        else:
                            current = current.right
        return self.root
        
        


        
    def preorder(self, root):
        if root is None:
            return
        print(root.data, end=' ')
        self.preorder(root.left)
        self.preorder(root.right)

    def inorder(self, root):
        if root is None:
            return
        self.inorder(root.left)
        print(root.data, end=' ')
        self.inorder(root.right)

    def postorder(self, root):
        if root is None:
            return
        self.postorder(root.left)
        self.postorder(root.right)
        print(root.data, end=' ')

    def levelorder(self, root):
        if root is None:
            return
        queue = Queue()
        queue.put(root)
        while not queue.empty():
            current = queue.get()
            print(current.data, end=' ')
            if current.left is not None:
                queue.put(current.left)
            if current.right is not None:
                queue.put(current.right)

    
    #iterative way of preorder tree traversal
    def iterative_preorder(self, root):
        if root is None:
            return
        stack = []
        stack.append(root)
        while len(stack) > 0:
            current = stack.pop()
            print(current.data, end=' ')
            if current.right is not None:
                stack.append(current.right)
            if current.left is not None:
                stack.append(current.left)

    def iterative_inorder(self,root):
        if root is None:
            return
        stack = []
        current = root
        while True:
            if current is not None:
                stack.append(current)
                current = current.left
            else:
                if len(stack) > 0:
                    current = stack.pop()
                    print(current.data, end=' ')
                    current = current.right
                else:
                    break

    def iterative_postorder(self,root):
        if root is None:
            return
        stack1 = []
        stack2 = []
        current = root
        stack1.append(current)
        while len(stack1) > 0:
            current = stack1.pop()
            stack2.append(current)
            if current.left is not None:
                stack1.append(current.left)
            if current.right is not None:
                stack1.append(current.right)
        while len(stack2) > 0:
            print(stack2.pop().data, end=' ')

    
    def iterativr2_preorder(self,root):
        stack = []
        while root is not None or stack != []:
            if root != None:
                print(root.data, end=' ')
                stack.append(root)
                root = root.left
            else:
                root = stack.pop()
                root = root.right



        
    def iterative2_inorder(self,root):
        stack = []
        while root is not None or stack != []:
            if root != None:
                stack.append(root)
                root = root.left 
            else:
                root = stack.pop()
                print(root.data, end=' ')
                root = root.right
    

    def iterative_levelorder(self, root):
        queue = []
        print(root.data, end=' ')
        queue.append(root)
        while queue != []:
            p = queue.pop(0)
            if p.left is not None:
                print(p.left.data, end=' ')
                queue.append(p.left)
            if p.right is not None:
                print(p.right.data, end=' ')
                queue.append(p.right)
            

    #generate traversal tree from preorder and inorder traversal
    def generate_tree(self, preorder, inorder):
        if len(preorder) == 0:
            return None
        root = Node(preorder[0])
        if len(preorder) == 1:
            return root
        index = inorder.index(preorder[0])
        root.left = self.generate_tree(preorder[1:index+1], inorder[0:index])
        root.right = self.generate_tree(preorder[index+1:], inorder[index+1:])
        return root

    def count(self, root):
        x = 0
        y = 0
        if root is None:
            return 0
        
        if root != None:
            x = self.count(root.left)
            y = self.count(root.right)
            return x + y + 1

        
    def height(self, root):
        if root is None:
            return 0
        else:
            x = self.height(root.left)
            y = self.height(root.right)
            if x > y:
                return x + 1
            else:
                return y + 1
            


    def print_tree(self, root):
        if root is None:
            return
        print(root.data, end=' ')
        self.print_tree(root.left)
        self.print_tree(root.right)

    def leaf_node(self,root):
        if root is None:
            return 0
        else:
            if root != None:
                x = self.leaf_node(root.left)
                y = self.leaf_node(root.right)
                if root.left == None and root.right == None:
                    return x + y +1
                else:
                    return x + y
        

    def degree_2(self,root):
        if root is None:
            return 0
        else:
            if root != None:
                x = self.leaf_node(root.left)
                y = self.leaf_node(root.right)
                if root.left != None and root.right != None:
                    return x + y +1
                else:
                    return x + y



    def degree_1_or_2(self,root):
        if root is None:
            return 0
        else:
            if root != None:
                x = self.leaf_node(root.left)
                y = self.leaf_node(root.right)
                if root.left != None or root.right != None:
                    return x + y +1
                else:
                    return x + y

    def degree_1(self,root):
        if root is None:
            return 0
        else:
            if root != None:
                x = self.leaf_node(root.left)
                y = self.leaf_node(root.right)
                if (root.left != None and root.right == None) or (root.left == None and root.right != None):
                    return x + y +1
                else:
                    return x + y

                    
        
        
        
        

                
            
                
                




T = tree()
S = T.create_tree([5,2,7,1,3,6,8,9,4])
#T.preorder(S)
#T.inorder(S)
#T.postorder(S)
#T.levelorder(S)
#T.iterative_preorder(S)
#T.iterative_inorder(S)
#T.iterative_postorder(S)
#T.iterativr2_preorder(S)
#T.iterative2_inorder(S)
#T.iterative_levelorder(S)
p = T.generate_tree([5, 2, 1, 3, 4, 7, 6, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9])
T.print_tree(p)
print(T.count(p))
print(T.height(p))
print(T.leaf_node(p))
print(T.degree_2(p))
print(T.degree_1_or_2(p))
print(T.degree_1(p))
