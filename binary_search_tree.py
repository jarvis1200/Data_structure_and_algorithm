# create an binary search tree
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None
    
    #create binaryy search tree in iteration with list of items
    def create_bst_with_list(self, dataset):
        for data in dataset:
            if data is None:
                continue
            if self.root is None:
                self.root = Node(data)

            else:
                current = self.root
                while True:
                    if data < current.data:
                        if current.left is None:
                            current.left = Node(data)
                            break
                        else:
                            current = current.left
                    else:
                        if current.right is None:
                            current.right = Node(data)
                            break
                        else:
                            current = current.right

        return self.root
    

    def searching(self, key,root):

        if root is None:
            return None
        
        while root != None:
            if key == root.data:
                return True

            elif key < root.data:
                root = root.left

            else:
                root = root.right

        return 0
        
      


        

  


    def inorder(self, root):
        if root is None:
            return None

        stack = []
        while root is not None or stack != []:
            if root != None:
                stack.append(root)
                root = root.left 
            else:
                root = stack.pop()
                print(root.data, end=' ')
                root = root.right


    def findmin(self,root):
        if root is None:
            return None

        while root.left:
            root = root.left

        return root

    def findmax(self,root):
        if root is None:
            return None

        while root.right:
            root = root.right

        return root

    def calculateSum(self,root):
        left_sum = self.calculateSum(root.left) if root.left else 0
        right_sum = self.calculateSum(root.right) if root.right else 0
        return left_sum + right_sum + root.data

    def iterative_calculateSum(self,root):
        if root is None:
            return 0

        s = 0
        stack = []
        while root or stack:
            if root:
                stack.append(root)
                root = root.left

            else:
                root = stack.pop()
                s += root.data
                root = root.right

        return s
    #iterativly insert node into binary search tree
    def insert_node(self, root, data):
        first = root
        while root != None:
            curr = root
            if data == root.data:
                return 
            elif data < root.data:
                root = root.left
            else:
                root = root.right
            
        p = Node(data)
        if p.data < curr.data:
            curr.left = p
        else:
            curr.right = p

        return first

    def insert_node_recursive(self, root, data):
        if root == None:
            p = Node(data)
            return p

        if data < root.data:
            root.left = self.insert_node_recursive(root.left, data)
        else:
            root.right = self.insert_node_recursive(root.right, data)

        return root



    def delete_node(self, root, data):
        if root is None:
            return root

        elif data < root.data:
            root.left = self.delete_node(root.left, data)

        elif data > root.data:
            root.right = self.delete_node(root.right, data)

        else:
            # 0 child
            if root.left == None and root.right == None:
                del root
                root = None
            
            # 1 child
            elif root.left == None:
                temp = root
                root = root.right
                del temp

            elif root.right == None:
                temp = root
                root = root.left
                del temp

            # 2 child

            else:
                temp = self.findmax(root.left)
                root.data = temp.data
                root.left = self.delete_node(root.left, temp.data)

        return root


    #generate BST using preorder
    def generate_preorder(self, pre):

        stack = []
        
        root = Node(pre[0])
        stack.append(root)
        for val in pre[1:]:
            if val < stack[-1].data:
                t = Node(val)
                stack[-1].left = t
                stack.append(t)

            else:
                while stack and stack[-1].data < val:
                    temp = stack.pop()
                temp.right = Node(val)
                stack.append(temp.right)
        return root
                
            
                
               
                





        
        






bst = BST()
A = bst.create_bst_with_list([5,4,6,3,2,1,8,9])
bst.inorder(A)
B = bst.searching(9, A)
if B == True:
    print('key is Found!')
else:
    print('key not found')

bst.findmin(A)
bst.findmax(A)
print(bst.calculateSum(A))
print(bst.iterative_calculateSum(A))

C = bst.insert_node(A, 7)
bst.inorder(C)
print( )
D = bst.insert_node_recursive(A, 10)
bst.inorder(D)

print( )
E = bst.delete_node(D, 5)
bst.inorder(E)

print()
F = bst.generate_preorder([30,20,10,15,25,40,50,45])
bst.inorder(F)