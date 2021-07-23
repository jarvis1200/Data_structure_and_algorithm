class Node:
    def __init__(self, data = None, next = None):
        self.data = data
        self.next = next
        
class Linkedlist:
    def __init__(self):
        self.head = None
        
    def insert_at_beginning(self, data):
        node = Node(data, self.head)
        self.head = node
        
    def print(self,head):
        if head is None:
            print('Empty list')
            return
        
        it = head
        itstr = ''
        while it:
            itstr += str(it.data) + ' --> '
            it = it.next
            
        print(itstr)
        
    def insert_at_end(self,data):
        if self.head is None:
            self.head = Node(data, None)
            return
        
        it = self.head
        while it.next:
            it = it.next
            
        it.next = Node(data, None)
        
    def insert_values(self,Dataset):
        self.head = None
        for data in Dataset:
            self.insert_at_end(data)
            
    def len_list(self):
        it = self.head
        
        cnt = 0
        while it:
            cnt+=1
            it = it.next
            
        return cnt
        
    def remove_at(self, ind):
        if ind < 0 or ind > self.len_list():
            raise Exception('invalid index')
            
        if ind == 0:
            self.head = self.head.next
            return
        
        cnt = 0
        it = self.head
        
        while it:
            if cnt == ind-1:
                it.next = it.next.next
                break
            it = it.next
            cnt+=1
            
            
    def insert_at(self, data, ind):
        if ind < 0 or ind > self.len_list():
            raise Exception('invalid index')
            
        if ind == 0:
            node = Node(data, self.head)
            self.head = node
            return
        
        cnt = 0
        it = self.head
        while it:
            if cnt == ind-1:
                node = Node(data, it.next)
                it.next = node
                break
                
            it = it.next
            cnt +=1
            
    def insert_after_val(self, val, data):
        it = self.head
        while it:
            if it.data == val:
                it.next = Node(data, it.next)
                it.next = it.next
                break
            it = it.next
        return False
        
    def remove_by_val(self, data):
        
        
        if self.head.data == data:
            self.head = self.head.next
            return
        
        it = self.head
        while it is not None:
            if it.next is not None and it.next.data == data:
                it.next = it.next.next
            else:
                it = it.next
    
        return False
        
    def reverseList(self):
        
        prev = None
        curr = self.head
        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        
        return prev
        
        
    def splitList(self,  head1, head2):
        #code here
        slow = self.head
        fast = self.head.next
        
        while fast != self.head and fast.next != self.head:
            slow = slow.next
            fast = fast.next.next
            
        head1 = self.head
        head2 = slow.next
        slow.next = self.head1
        
        curr = head2
        while curr.next != self.head:
            curr = curr.next
            
        curr.next = head2
        
        
        #this is to emulate pass by reference in python please don't delete below line.
        return head1,head2
        
    
        
    

    def sortedInsert(self,  data):
            if self.head is None:
                return Node(data)
            if self.head.next is None:
                if data < self.head.data:
                    self.head.next = Node(data)
                    self.head.next.next = self.head
                else:
                    self.head.next = Node(data)
            else:
                if data < self.head.data:
                    new_node = Node(data)
                    new_node.next = self.head
                    self.head = new_node
                else:
                    curr = self.head
                    while curr.next is not None and curr.next.data < data:
                        curr = curr.next
                    new_node = Node(data)
                    new_node.next = curr.next
                    curr.next = new_node
            return self.head
            
            
    def detectLoop(self):
    
        if self.head == None:
            return False
        slow = self.head
        fast = self.head
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
        
    def deleteMid(head):
    
        if head == None or head.next == None:
            return None
        curr = head
        while curr.next != None and curr.next.next != None:
            curr = curr.next
        curr.next = curr.next.next
        return head
    
    
    def deleteAtPosition(head,pos):
        if pos == 0:
            return head.next
        current = head
        for i in range(pos-1):
            current = current.next
        current.next = current.next.next
        return head
        
    def deleteNode(self,curr_node):
        #code here
        if curr_node is not None:
            curr_node.data = curr_node.next.data
            curr_node.next = curr_node.next.next
        return curr_node
        
        
    #Intersection Point in Y Shapped Linked Lists
    def intersetPoint(head1,head2):
    
        L1 = get_len(head1)
        L2 = get_len(head2)

        ab = abs(L1-L2)

        ptr1 = head1
        ptr2 = head2

        if L1 > L2:
            for i in range(ab):
                ptr1 = ptr1.next
        elif L2 > L1:
            for i in range(ab):
                ptr2 = ptr2.next

        while ptr1 != ptr2:
            ptr1 = ptr1.next
            ptr2 = ptr2.next

        if ptr1:
            return ptr1.data

        return -1

    
    def merge_LL(self,first, second):
        last = Node(None, None)
        
        if first.data < second.data:
            third = last = first
            first = first.next
            third.next = None
            
        else: 
            third = last = second
            second = second.next
            third.next = None
            
        
        while (first and second):
            if first.data < second.data:
                last.next = first
                last = first
                first = first.next
                last.next = None
            else:
                last.next = second
                last = second
                second = second.next
                last.next = None
        
        if first:
            last.next = first
        if second:
            last.next = second
            
        return third
        
        

     def find_loop(self, head):
            fast = head
            slow = head
            while fast is not None and fast.next is not None:
                fast = fast.next.next
                slow = slow.next
                if fast is slow:
                    print('True')
            print('False')
            
            
    def circulateLinkedList(self,items):
        if len(items) == 0:
            return None
        head = Node(items[0])
        tail = head
        for i in range(1, len(items)):
            tail.next = Node(items[i])
            tail = tail.next
        tail.next = head
        return head
        
    def display_circulated(self,head):
        if head is None:
            print("Empty Linked List")
            return
        flag = 0
        p = head
        t = ''
        while p != head or flag == 0:
            flag = 1
            t += str(p.data) + '==>'
            p = p.next
        flag = 0
        print(t)
        
        
        
        
ll = Linkedlist()
ll.insert_values(["banana","mango","grapes","orange"])
ll.print()
ll.insert_after_val("mango","apple") # insert apple after mango
ll.print()
ll.remove_by_val("orange") # remove orange from linked list
ll.print()
ll.remove_by_val("figs")
ll.print()
ll.remove_by_val("banana")
ll.remove_by_val("mango")
ll.remove_by_val("apple")
ll.remove_by_val("grapes")
ll.print()
ll.insert_values([45,7,12,567,99])
ll.insert_at_end(67)
ll.print()
first = l.insert_values([1,3,5,7,9])
second = l.insert_values([2,4,6,8,10])
l.print(first)
l.print(second)
A= l.merge_LL(first, second)
l.print(A)
