# create a tree using the given data method with queue

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class tree:

    def create_node(self, data):
        return Node(data)
# create tree with the list of items
    def insert_node(self,root,dataset):
        first = dataset[0]
        if root is None:
            root = self.create_node(first)

        if len(dataset) > 1:
            second = dataset[1]
            if first < root.data:
                root.left = self.insert_node(root.left,dataset[1:])
            else:
                root.right = self.insert_node(root.right,dataset[1:])
        return root
        
# create function to get the level of the tree
    def get_level(self, root):
        if root is None:
            return 0
        else:
            l = self.get_level(root.left)
            r = self.get_level(root.right)
            if l > r:
                return l + 1
            else:
                return r + 1

        
# create function to create a tree with queue
    def create_tree(self, root, dataset):
        if len(dataset) == 0:
            return
        else:
            root = self.insert_node(root,dataset[0])
            self.create_tree(root,dataset[1:])
        


    def print_node(self, root):
        if root is None:
            return
        print(root.data)
        self.print_node(root.left) 
        self.print_node(root.right)


T = tree()
root = T.insert_node(None, [10, 5, 15, 3, 7, 20])
T.print_node(root)
print(T.get_level(root))
