class Node:
    def __init__(self, Data):
        self.data = Data
        self.left = None
        self.right = None
        self.height = 1

class AVL:
    def __init__(self):
        self.root = None

    def nodeHeight(self, root):
        if root is None:
            return 0
        hl = self.nodeHeight(root.left) if root.left is not None else 0
        hr = self.nodeHeight(root.right) if root.right is not None else 0
        return max(hl, hr) + 1
        

    def balance(self, root):
        if root is None:
            return 0

        hl = self.nodeHeight(root.left) if root.left is not None else 0
        hr = self.nodeHeight(root.right) if root.right is not None else 0

        return hl - hr

    def LLRotate(self, root):
        p = root
        pl = p.left
        plr = pl.right

        pl.right = p
        p.left = plr

        p.height = self.nodeHeight(p)
        pl.height = self.nodeHeight(pl)

        if root == p:
            root = pl

        return pl

    def RRRotate(self, root):
        p = root
        pl = p.right
        plr = pl.left

        pl.left = p
        p.right = plr

        p.height = self.nodeHeight(p)
        pl.height = self.nodeHeight(pl)

        if root == p:
            root = pl

        return pl


    def LRRotate(self, root):
        p = root
        pl = p.left
        plr = pl.right

        pl.right = plr.left
        p.left = plr.right

        plr.left = p
        plr.right = pl

        p.height = self.nodeHeight(p)
        pl.height = self.nodeHeight(pl)
        plr.height = self.nodeHeight(plr)

        if root == p:
            root = plr

        return plr

    def RLRotate(self, root):
        p = root
        pr = p.right
        prl = pr.left

        pr.left = prl.right
        p.right = prl.left

        prl.left = p
        prl.right = pr

        p.height = self.nodeHeight(p)
        pr.height = self.nodeHeight(pr)
        prl.height = self.nodeHeight(prl)

        

        if root == p:
            root = prl
        
        return prl


        

    def insert(self, root, items):
        for i in items:
            if root is None:
                root = Node(i)
            else:
                curr = root
                while True:
                    
                    if i < curr.data and i != curr.left:
                        t = Node(i)
                        if curr.left is None:
                            curr.left = t
                            break
                            
                        else:
                            curr = curr.left

                    elif i > curr.data and i != curr.right:
                        t = Node(i)
                        if curr.right is None:
                            curr.right = t
                            break
                            
                        else:
                            curr = curr.right
                    else:
                        break

        

        root.height = self.nodeHeight(root)
        if (self.balance(root) > 1 and self.balance(root.left) >= 0):
            return self.LLRotate(root)

        elif(self.balance(root) > 1 and self.balance(root.right) <= 0):
            return self.LRRotate(root)

        elif(self.balance(root) < -1 and self.balance(root.right) <= 0):
            return self.RRRotate(root)

        elif(self.balance(root) < -1 and self.balance(root.left) >= 0):
            return self.RLRotate(root)

        return root

    def findMax(self, root):
        if root is None:
            return None

        while root.right:
            root = root.right

        return root


    def findMin(self, root):
        if root is None:
            return None

        while root.left:
            root =root.left

        return root


    def delete(self, root, data):
        if root is None:
            return root

        elif data < root.data:
            root.left = self.delete(root.left, data)

        elif data > root.data:
            root.right = self.delete(root.right, data)

        else:
            
            # 1 child
            if root.left == None:
                temp = root
                root = root.right
                del temp
                return root

            elif root.right == None:
                temp = root
                root = root.left
                del temp
                return root

            # 2 child

            
            temp = self.findMin(root.right)
            root.data = temp.data
            root.right = self.delete(root.right, temp.data)


        if root is None:
            return root

        root.height = self.nodeHeight(root)
        balance = self.balance(root)
        Lbal = self.balance(root.left)
        Rbal = self.balance(root.right)

        if balance >= 1 and Lbal >= 0:
            return self.LLRotate(root)

        elif balance >= 1 and Rbal <= 0:
            return self.LRRotate(root)

        elif balance <= 1 and Rbal <= 0:
            return self.RRRotate(root)

        elif balance <= 1 and Lbal >= 0:
            return self.RLRotate(root)

        else:
            return root

                
                

        

    
    def inorder(self, root):
        
        if root is not None:
            self.inorder(root.left)
            print(root.data, end = ' ')
            self.inorder(root.right)


    def print_node(self, root):
        if root is None:
            return
        print(root.data, end= ' ')
        self.print_node(root.left) 
        self.print_node(root.right)

            

            







avl = AVL()
A = avl.insert(None, [10, 5, 15, 3, 7, 18, 2, 4, 6, 11, 16, 19, 1, 8, 12, 17, 20, 13,9, 14, 16, 17, 18, 19, 20])
avl.print_node(A)
print()
avl.inorder(A)
print()
print(avl.nodeHeight(A))
print()

B = avl.delete(A,10)
avl.inorder(B)
print()
avl.print_node(B)
print()
print(avl.nodeHeight(A))


                