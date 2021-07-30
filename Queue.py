class queue:
    def __init__(self):
        self.size = 0
        self.front = 0
        self.rear = 0
        self.queue = [None]

class Q:
    def create_queue(self, size):
        self.size = size
        self.queue = [None] * self.size
        self.front = -1
        self.rear = -1

    def enqueue(self, dataset):
        if self.size == 0:
            self.create_queue(self.size)
        else:
            for i in dataset:
                self.rear += 1
                if self.rear != self.size:
                    self.queue[self.rear] = i
                else:
                    print("Queue is full")
            
        return self.queue

    def dequeue(self):
        x = -1
        queue = self.queue
        if self.size == 0:
            print("Queue is empty")
        
        else:
            self.front += 1
            x = queue[self.front]
            self.queue[self.front] = None
            return x
            
            

    def peek(self):
        if self.size == 0:
            print("Queue is empty")
        else:
            return self.queue[self.front]

    def is_empty(self):
        if self.size == 0:
            return True
        else:
            return False

    def is_full(self):
        if self.size == self.size:
            return True
        else:
            return False

    def size_of_Q(self):
        print(self.size)

    def print_queue(self):
        print(self.queue)


    def circular_enqueue(self,data):
        q = self.queue
        if self.size == 0:
            self.create_queue(self.size)
        else:
            self.rear = (self.rear+1) % self.size
            q[self.rear] = data
            self.size += 1
            return self.queue
        

    def circular_dequeue(self):
        x = -1
        q = self.queue
        if self.size == 0:
            print("Queue is empty")

        else:
            self.front = (self.front+1) % self.size
            x = q[self.front]
            q[self.front] = None
            self.size -= 1
            return x


    # double ended queue DEQUE add value using front
    def DEqueue(self, data):
        q = self.queue
        if self.size == 0:
            print("Queue is empty")
        else:
            
            if q[self.front] == None:
                q[self.front] = data
                self.front -= 1
                self.size += 1
            else:
                self.front += 1
                q[self.front] = None
                self.size -= 1
            return self.queue

    # priority queue
    def priority_enqueue(self):
        try:
            max = 0
            for i in range(len(self.queue)):
                if self.queue[i] > self.queue[max]:
                    max = i
            item = self.queue[max]
            del self.queue[max]
            return item
        except IndexError:
            print()
            exit()

    #queue using two stacks
    def stack_enqueue(self,data):
        s1 = []
        s2 = []

        while len(s1) != 0:
            s2.append(s1[-1])
            s1.pop()

        s1.append(data)

        while len(s2) != 0:
            s1.append(s2[-1])
            s2.pop()

        return s1

    def stack_dequeue(self,s1):
        if len(s1) == 0:
            print("Queue is empty")

        else:
            x = s1[-1]
            s1.pop()
            return x
        
            


# queue using linked list
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

    

class linkedlist:
    def linked_enqueue(self, root,data):
        
        if root is None:
            root = Node(data)
            return root
        else:
            root.next = self.linked_enqueue(root.next,data)
            return root

    def linked_dequeue(self,root):
        front = root
        t = Node(None)
        x = -1
        if front is None:
            print("Linked list is empty")

        else:
            front = front.next
            x = front.data
            t.data = x
            root.data = None
            return t

    def display_linked(self,head):
        front = head
        if front is None:
            print("Linked list is empty")
        else:
            itr = ''
            while front is not None:
                itr += str(front.data) + " -> "
                front = front.next
            print(itr)


    

    

q = Q()
q.create_queue(5)
q.enqueue([1,2,3,4,5])
q.print_queue()
print(q.dequeue())
q.print_queue()
print(q.peek())
print(q.is_empty())
print(q.is_full())
q.size_of_Q()
q.circular_enqueue(6)
q.print_queue()
q.circular_dequeue()
q.print_queue()
q.circular_enqueue(7)
q.print_queue()
q.circular_dequeue()
q.print_queue()
q.size_of_Q()
q.DEqueue(8)
q.print_queue()

print(q.priority_enqueue())
q.print_queue()

M = q.stack_enqueue(5)
M= q.stack_enqueue(6)

print(q.stack_dequeue(M))


LL = linkedlist()
W = LL.linked_enqueue(None,1)
Q = LL.linked_enqueue(W,2)
LL.display_linked(Q)
LL.linked_enqueue(Q,3)
LL.display_linked(Q)
LL.linked_dequeue(Q)
LL.display_linked(Q)