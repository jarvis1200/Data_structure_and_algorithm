class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = None


    def add_child(self, child):
        child.parent = self
        self.children.append(child)

def print_tree(root):
    spaces = ' ' * get_level(root) *1
    prefix = spaces + '|__' if root.parent else ''
    print(prefix + str(root.data))

    if root.children:
        for child in root.children:
            print_tree(child)

def get_tree():
    root = TreeNode(1)

    child1 = TreeNode(2)
    child1.add_child(TreeNode(4))
    child1.add_child(TreeNode(5))

    child2 = TreeNode(3)
    child2.add_child(TreeNode(6))
    child2.add_child(TreeNode(7))

    root.add_child(child1)
    root.add_child(child2)

    return root

def get_level(root):
    
    level = 0
    while root.parent:
        root = root.parent
        level += 1

    return level

if __name__ == '__main__':
    root = get_tree()
    print_tree(root)
    pass
