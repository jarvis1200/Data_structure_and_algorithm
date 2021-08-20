'''Place k elements such that minimum distance is maximized
Difficulty Level : Medium
Last Updated : 05 Jul, 2021
Given an array representing n positions along a straight line. Find k (where k <= n) elements from the array such that the minimum distance between any two (consecutive points among the k points) is maximized.

Examples :  

Input : arr[] = {1, 2, 8, 4, 9}
            k = 3
Output : 3
Largest minimum distance = 3
3 elements arranged at positions 1, 4 and 8, 
Resulting in a minimum distance of 3

Input  : arr[] = {1, 2, 7, 5, 11, 12}
             k = 3
Output : 5
Largest minimum distance = 5
3 elements arranged at positions 1, 7 and 12, 
resulting in a minimum distance of 5 (between
7 and 12)
Recommended: Please try your approach on {IDE} first, before moving on to the solution.
A Naive Solution is to consider all subsets of size 3 and find the minimum distance for every subset. Finally, return the largest of all minimum distances.

An Efficient Solution is based on Binary Search. We first sort the array. Now we know maximum possible value result is arr[n-1] – arr[0] 
(for k = 2). We do a binary search for maximum result for given k. We start with the middle of the maximum possible result. If the middle is a feasible solution, we search on the right half of mid.
 Else we search is left half. To check feasibility, we place k elements under given mid-distance.'''

 # Python 3 program to find largest minimum
# distance among k points.
 
# Returns true if it is possible to arrange
# k elements of arr[0..n-1] with minimum
# distance given as mid.
 
 
def isFeasible(mid, arr, n, k):
 
    # Place first element at arr[0] position
    pos = arr[0]
 
    # Initialize count of elements placed.
    elements = 1
 
    # Try placing k elements with minimum
    # distance mid.
    for i in range(1, n, 1):
        if (arr[i] - pos >= mid):
 
            # Place next element if its distance
            # from the previously placed element
            # is greater than current mid
            pos = arr[i]
            elements += 1
 
            # Return if all elements are placed
            # successfully
            if (elements == k):
                return True
    return 0
 
# Returns largest minimum distance for k elements
# in arr[0..n-1]. If elements can't be placed,
# returns -1.
 
 
def largestMinDist(arr, n, k):
 
    # Sort the positions
    arr.sort(reverse=False)
 
    # Initialize result.
    res = -1
 
    # Consider the maximum possible distance
    left = arr[0]
    right = arr[n - 1] - arr[0]
 
    # left is initialized with 1 and not with arr[0]
    # because, minimum distance between each element
    # can be one and not arr[0]. consider this example:
    # arr[] = {9,12} and you have to place 2 element
    # then left = arr[0] will force the function to
    # look the answer between range arr[0] to arr[n-1],
    # i.e 9 to 12, but the answer is 3 so It is required
    # that you initialize the left with 1
 
    # Do binary search for largest
    # minimum distance
    while (left < right):
        mid = (left + right) // 2
 
        # If it is possible to place k elements
        # with minimum distance mid, search for
        # higher distance.
        if (isFeasible(mid, arr, n, k)):
 
            # Change value of variable max to mid iff
            # all elements can be successfully placed
            res = max(res, mid)
            left = mid + 1
 
        # If not possible to place k elements,
        # search for lower distance
        else:
            right = mid
 
    return abs(res)
 
 
# Driver code
if __name__ == '__main__':
    arr = [1, 2, 8, 4, 9]
    n = len(arr)
    k = 3
    print(largestMinDist(arr, n, k))
 
# 3

'''How to validate HTML tag using Regular Expression
Last Updated : 04 Feb, 2021
Given string str, the task is to check whether it is a valid HTML tag or not by using Regular Expression.
The valid HTML tag must satisfy the following conditions: 

It should start with an opening tag (<).
It should be followed by a double quotes string or single quotes string.
It should not allow one double quotes string, one single quotes string or a closing tag (>) without single or double quotes enclosed.
It should end with a closing tag (>).
Examples: 

Input: str = “<input value = ‘>’>”; 
Output: true 
Explanation: The given string satisfies all the above mentioned conditions.
Input: str = “<br/>”; 
Output: true 
Explanation: The given string satisfies all the above mentioned conditions.
Input: str = “br/>”; 
Output: false 
Explanation: The given string doesn’t starts with an opening tag “<“. Therefore, it is not a valid HTML tag.
Input: str = “<‘br/>”; 
Output: false 
Explanation: The given string has one single quotes string that is not allowed. Therefore, it is not a valid HTML tag.
Input: str = “<input value => >”; 
Output: false 
Explanation: The given string has a closing tag (>) without single or double quotes enclosed that is not allowed. Therefore, it is not a valid HTML tag.

Approach: The idea is to use Regular Expression to solve this problem. The following steps can be followed to compute the answer. 

Get the String.
Create a regular expression to check valid HTML tag as mentioned below: 
 
regex = “<(“[^”]*”|'[^’]*’|[^'”>])*>”; 



Where: 
< represents the string should start with an opening tag (<).
( represents the starting of the group.
“[^”]*” represents the string should allow double quotes enclosed string.
| represents or.
‘[^’]*‘ represents the string should allow single quotes enclosed string.
| represents or.
[^'”>] represents the string should not contain one single quote, double quotes, and “>”.
) represents the ending of the group.
* represents 0 or more.
> represents the string should end with a closing tag (>).
Match the given string with the regular expression. In Java, this can be done by using Pattern.matcher().
Return true if the string matches with the given regular expression, else return false.
Below is the implementation of the above approach:'''

# Python3 program to validate
# HTML tag using regex.  
# using regular expression
import re
 
# Function to validate
# HTML tag using regex.
def isValidHTMLTag(str):
 
    # Regex to check valid
    # HTML tag using regex.
    regex = "<(\"[^\"]*\"|'[^']*'|[^'\">])*>"
     
    # Compile the ReGex
    p = re.compile(regex)
 
    # If the string is empty
    # return false
    if (str == None):
        return False
 
    # Return if the string
    # matched the ReGex
    if(re.search(p, str)):
        return True
    else:
        return False
 
# Driver code
 
# Test Case 1:
str1 = "<input value = '>'>"
print(isValidHTMLTag(str1))
 
# Test Case 2:
str2 = "<br/>"
print(isValidHTMLTag(str2))
 
# Test Case 3:
str3 = "br/>"
print(isValidHTMLTag(str3))
 
# Test Case 4:
str4 = "<'br/>"
print(isValidHTMLTag(str4))
 

''' Search in Rotated Sorted Array
Medium

9277

731

Add to List

Share
There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
Example 3:

Input: nums = [1], target = 0
Output: -1
 

Constraints:

1 <= nums.length <= 5000
-104 <= nums[i] <= 104
All values of nums are unique.
nums is guaranteed to be rotated at some pivot.
-104 <= target <= 104'''

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        low = 0
        high = n-1
        while low <=high:
            mid = (low+high)//2
            
            if nums[mid] == target:
                return mid
            
            if nums[low] <= nums[mid]:
                if target >= nums[low] and target <= nums[mid]:
                    high = mid-1
                else:
                    low = mid+1
                    
            else:
                if target >= nums[mid] and target <= nums[high]:
                    low = mid+1
                else:
                    high = mid-1
        return -1

'''Search a 2D Matrix
Medium

4110

213

Add to List

Share
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted from left to right.
The first integer of each row is greater than the last integer of the previous row.
 

Example 1:


Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true
Example 2:


Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
Output: false
 

Constraints:

m == matrix.length
n == matrix[i].length
1 <= m, n <= 100
-104 <= matrix[i][j], target <= 104'''

def searchMatrix(self, matrix, target):
        if not matrix or target is None:
            return False

        rows, cols = len(matrix), len(matrix[0])
        low, high = 0, rows * cols - 1
        
        while low <= high:
            mid = (low + high) / 2
            num = matrix[mid / cols][mid % cols]

            if num == target:
                return True
            elif num < target:
                low = mid + 1
            else:
                high = mid - 1
        
        return False


'''Minimize Cash Flow among a given set of friends who have borrowed money from each other
Difficulty Level : Hard
Last Updated : 15 Jul, 2021
Given a number of friends who have to give or take some amount of money from one another. Design an algorithm by which the total cash flow among all the friends is minimized. 
Example: 
Following diagram shows input debts to be settled. 
 

cashFlow

Above debts can be settled in following optimized way 
 

cashFlow

 



Recommended: Please try your approach on {IDE} first, before moving on to the solution.
The idea is to use Greedy algorithm where at every step, settle all amounts of one person and recur for remaining n-1 persons. 
How to pick the first person? To pick the first person, calculate the net amount for every person where net amount is obtained by subtracting all debts (amounts to pay) from all credits (amounts to be paid). Once net amount for every person is evaluated, find two persons with maximum and minimum net amounts. These two persons are the most creditors and debtors. The person with minimum of two is our first person to be settled and removed from list. Let the minimum of two amounts be x. We pay ‘x’ amount from the maximum debtor to maximum creditor and settle one person. If x is equal to the maximum debit, then maximum debtor is settled, else maximum creditor is settled.
The following is detailed algorithm.
Do following for every person Pi where i is from 0 to n-1. 

Compute the net amount for every person. The net amount for person ‘i’ can be computed by subtracting sum of all debts from sum of all credits.
Find the two persons that are maximum creditor and maximum debtor. Let the maximum amount to be credited maximum creditor be maxCredit and maximum amount to be debited from maximum debtor be maxDebit. Let the maximum debtor be Pd and maximum creditor be Pc.
Find the minimum of maxDebit and maxCredit. Let minimum of two be x. Debit ‘x’ from Pd and credit this amount to Pc
If x is equal to maxCredit, then remove Pc from set of persons and recur for remaining (n-1) persons.
If x is equal to maxDebit, then remove Pd from set of persons and recur for remaining (n-1) persons.'''

# Python3 program to fin maximum
# cash flow among a set of persons
  
# Number of persons(or vertices in graph)
N = 3
  
# A utility function that returns
# index of minimum value in arr[]
def getMin(arr):
      
    minInd = 0
    for i in range(1, N):
        if (arr[i] < arr[minInd]):
            minInd = i
    return minInd
  
# A utility function that returns
# index of maximum value in arr[]
def getMax(arr):
  
    maxInd = 0
    for i in range(1, N):
        if (arr[i] > arr[maxInd]):
            maxInd = i
    return maxInd
  
# A utility function to
# return minimum of 2 values
def minOf2(x, y):
  
    return x if x < y else y
  
# amount[p] indicates the net amount to
# be credited/debited to/from person 'p'
# If amount[p] is positive, then i'th 
# person will amount[i]
# If amount[p] is negative, then i'th
# person will give -amount[i]
def minCashFlowRec(amount):
  
    # Find the indexes of minimum
    # and maximum values in amount[]
    # amount[mxCredit] indicates the maximum
    # amount to be given(or credited) to any person.
    # And amount[mxDebit] indicates the maximum amount
    # to be taken (or debited) from any person.
    # So if there is a positive value in amount[], 
    # then there must be a negative value
    mxCredit = getMax(amount)
    mxDebit = getMin(amount)
  
    # If both amounts are 0, 
    # then all amounts are settled
    if (amount[mxCredit] == 0 and amount[mxDebit] == 0):
        return 0
  
    # Find the minimum of two amounts
    min = minOf2(-amount[mxDebit], amount[mxCredit])
    amount[mxCredit] -=min
    amount[mxDebit] += min
  
    # If minimum is the maximum amount to be
    print("Person " , mxDebit , " pays " , min
        , " to " , "Person " , mxCredit)
  
    # Recur for the amount array. Note that
    # it is guaranteed that the recursion
    # would terminate as either amount[mxCredit] 
    # or amount[mxDebit] becomes 0
    minCashFlowRec(amount)
  
# Given a set of persons as graph[] where
# graph[i][j] indicates the amount that
# person i needs to pay person j, this
# function finds and prints the minimum 
# cash flow to settle all debts.
def minCashFlow(graph):
  
    # Create an array amount[],
    # initialize all value in it as 0.
    amount = [0 for i in range(N)]
  
    # Calculate the net amount to be paid
    # to person 'p', and stores it in amount[p].
    # The value of amount[p] can be calculated by
    # subtracting debts of 'p' from credits of 'p'
    for p in range(N):
        for i in range(N):
            amount[p] += (graph[i][p] - graph[p][i])
  
    minCashFlowRec(amount)
      
# Driver code
  
# graph[i][j] indicates the amount
# that person i needs to pay person j
graph = [ [0, 1000, 2000],
          [0, 0, 5000],
          [0, 0, 0] ]
  
minCashFlow(graph)
  
# This code is contributed by Anant Agarwal.

'''Output: 
Person 1 pays 4000 to Person 2
Person 0 pays 3000 to Person 2'''


'''Add Two Numbers
Medium

13423

3019

Add to List

Share
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

 

Example 1:


Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.
Example 2:

Input: l1 = [0], l2 = [0]
Output: [0]
Example 3:

Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]
 

Constraints:

The number of nodes in each linked list is in the range [1, 100].
0 <= Node.val <= 9
It is guaranteed that the list represents a number that does not have leading zeros.'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        
        """
        carry = 0
        root = n = ListNode(0)
        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1+v2+carry, 10)
            n.next = ListNode(val)
            n = n.next
        return root.next
        
'''Accepted
Runtime: 27 ms
Your input
[2,4,3]
[5,6,4]
Output
[7,0,8]
Expected
[7,0,8]'''

'''Vertical Order Traversal of a Binary Tree
Hard

1914

2705

Add to List

Share
Given the root of a binary tree, calculate the vertical order traversal of the binary tree.

For each node at position (row, col), its left and right children will be at positions (row + 1, col - 1) and (row + 1, col + 1) respectively. The root of the tree is at (0, 0).

The vertical order traversal of a binary tree is a list of top-to-bottom orderings for each column index starting from the leftmost column and ending on the rightmost column. There may be multiple nodes in the same row and same column. In such a case, sort these nodes by their values.

Return the vertical order traversal of the binary tree.

 

Example 1:


Input: root = [3,9,20,null,null,15,7]
Output: [[9],[3,15],[20],[7]]
Explanation:
Column -1: Only node 9 is in this column.
Column 0: Nodes 3 and 15 are in this column in that order from top to bottom.
Column 1: Only node 20 is in this column.
Column 2: Only node 7 is in this column.
Example 2:


Input: root = [1,2,3,4,5,6,7]
Output: [[4],[2],[1,5,6],[3],[7]]
Explanation:
Column -2: Only node 4 is in this column.
Column -1: Only node 2 is in this column.
Column 0: Nodes 1, 5, and 6 are in this column.
          1 is at the top, so it comes first.
          5 and 6 are at the same position (2, 0), so we order them by their value, 5 before 6.
Column 1: Only node 3 is in this column.
Column 2: Only node 7 is in this column.
Example 3:


Input: root = [1,2,3,4,6,5,7]
Output: [[4],[2],[1,5,6],[3],[7]]
Explanation:
This case is the exact same as example 2, but with nodes 5 and 6 swapped.
Note that the solution remains the same since 5 and 6 are in the same location and should be ordered by their values.
 

Constraints:

The number of nodes in the tree is in the range [1, 1000].
0 <= Node.val <= 1000'''

class Solution:
    def __init__(self):
        self.point_list = []
        
    def dfs(self, root: TreeNode, x:int, y:int):
        self.point_list.append([x,y,root.val]) # keep in list all points
        if root.left:
            self.dfs(root.left, x-1, y-1)
        if root.right:
            self.dfs(root.right, x+1, y-1)
        
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        self.dfs(root, 0, 0)
        self.point_list.sort(key = lambda a:a[2]) # sort by value
        self.point_list.sort(key = lambda a:a[1], reverse = True) # sort by y
        self.point_list.sort(key = lambda a:a[0]) # sort by x
		
        res = []
        left = abs(self.point_list[0][0])
        right = self.point_list[-1][0]
		
        for i in range(left+right+1):
            res.append([j[2] for j in self.point_list if j[0]==i-left])
			
        return res 

'''Your input
[3,9,20,null,null,15,7]
Output
[[9],[3,15],[20],[7]]
Expected
[[9],[3,15],[20],[7]]'''

#or

class Solution:
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        def x():
            return defaultdict(list)
        yval = defaultdict(x)
        
        def traverse(root,x,y):
            if root==None:
                return
            yval[x][y].append(root.val)
            traverse(root.left, x-1, y+1)
            traverse(root.right, x+1, y+1)
            
        traverse(root,0,0)
        
        x =  [[yval[key][ins] for ins in sorted(yval[key])] for key in sorted(yval)]
    
        new = []
        for i in x:
            temp=[]
            for j in i:
                temp+=sorted(j)
            new.append(temp)
        return new

'''Swapping Nodes in a Linked List
Medium

886

53

Add to List

Share
You are given the head of a linked list, and an integer k.

Return the head of the linked list after swapping the values of the kth node from the beginning and the kth node from the end (the list is 1-indexed).

 

Example 1:


Input: head = [1,2,3,4,5], k = 2
Output: [1,4,3,2,5]
Example 2:

Input: head = [7,9,6,6,7,8,3,0,9,5], k = 5
Output: [7,9,6,6,8,7,3,0,9,5]
Example 3:

Input: head = [1], k = 1
Output: [1]
Example 4:

Input: head = [1,2], k = 1
Output: [2,1]
Example 5:

Input: head = [1,2,3], k = 2
Output: [1,2,3]
 

Constraints:

The number of nodes in the list is n.
1 <= k <= n <= 105
0 <= Node.val <= 100'''

def swapNodes(self, head: ListNode, k: int) -> ListNode:
    dummy = pre_right = pre_left = ListNode(next=head)
    right = left = head
    for i in range(k-1):
        pre_left = left
        left = left.next
    
    null_checker = left
    
    while null_checker.next:
        pre_right = right
        right = right.next
        null_checker = null_checker.next
        
    if left == right:
        return head
    
    pre_left.next, pre_right.next = right, left
    left.next, right.next = right.next, left.next
    return dummy.next  

'''Your input
[1,2,3,4,5]
2
Output
[1,4,3,2,5]
Expected
[1,4,3,2,5]'''

#or

def swapNodes(self, head: ListNode, k: int) -> ListNode:
	
	    # Initial State
        slow, fast = head, head
		
		# Phase 1
        for _ in range(k - 1):
            fast = fast.next
        first = fast

        # Phase 2
        while fast.next:
            slow, fast = slow.next, fast.next
		
		# Last
        first.val, slow.val = slow.val, first.val

        return head


'''Cousins in Binary Tree
Easy

1806

104

Add to List

Share
Given the root of a binary tree with unique values and the values of two different nodes of the tree x and y, return true if the nodes corresponding to the values x and y in the tree are cousins, or false otherwise.

Two nodes of a binary tree are cousins if they have the same depth with different parents.

Note that in a binary tree, the root node is at the depth 0, and children of each depth k node are at the depth k + 1.

 

Example 1:


Input: root = [1,2,3,4], x = 4, y = 3
Output: false
Example 2:


Input: root = [1,2,3,null,4,null,5], x = 5, y = 4
Output: true
Example 3:


Input: root = [1,2,3,null,4], x = 2, y = 3
Output: false
 

Constraints:

The number of nodes in the tree is in the range [2, 100].
1 <= Node.val <= 100
Each node has a unique value.
x != y
x and y are exist in the tree.'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
		# store (parent, depth) tuple
        res = []
		
		# bfs
        queue = deque([(root, None, 0)])        
        while queue:
			# minor optimization to stop early if both targets found
            if len(res) == 2:
                break
            node, parent, depth = queue.popleft()
            # if target found
            if node.val == x or node.val == y:
                res.append((parent, depth))
            if node.left:
                queue.append((node.left, node, depth + 1))
            if node.right:
                queue.append((node.right, node, depth + 1))

		# unpack two nodes
        node_x, node_y = res
		
		# compare and decide whether two nodes are cousins		
        return node_x[0] != node_y[0] and node_x[1] == node_y[1]

'''Your input
[1,2,3,4]
4
3
Output
false
Expected
false'''

#or

class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
		# store (parent, depth) tuple
            res = [] 
            
            # dfs
            self.dfs(root, None, 0)

            # unpack two nodes found
            node_x, node_y = res  
            
            # compare and decide whether two nodes are cousins
            return node_x[0] != node_y[0] and node_x[1] == node_y[1]

    def dfs(self,node, parent, depth):
            if not node:
                return
            if node.val == x or node.val == y:
                res.append((parent, depth))
            dfs(node.left, node, depth + 1)
            dfs(node.right, node, depth + 1)

            return res

#or

class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        
        if not root:
            return false
        
        stack = [[root, 0, -1]]
        
        cousin_1 = []
        cousin_2 = []
        
        while stack:
            node, h, parent = stack.pop()
            
            # find cousins
            if node.val == x: cousin_1 = [h, parent]
            if node.val == y: cousin_2 = [h, parent]
            
            if node.right: stack.append([node.right, h+1, node.val])
            if node.left: stack.append([node.left, h+1, node.val])
        
        if cousin_1[0] == cousin_2[0] and cousin_1[1] != cousin_2[1]:
            return True
        return False

'''Check if a Binary Tree is an Even-Odd Tree or not
Difficulty Level : Medium
Last Updated : 03 Aug, 2021
Given a Binary Tree, the task is to check if the binary tree is an Even-Odd binary tree or not. 

A Binary Tree is called an Even-Odd Tree when all the nodes which are at even levels have even values (assuming root to be at level 0) and all the nodes which are at odd levels have odd values.

 Examples:

Input: 

             2
            / \
           3   9
          / \   \
         4   10  6
Output: YES 
Explanation: 
Only node on level 0 (even) is 2 (even). 
Nodes present in level 1 are 3 and 9 (both odd). 
Nodes present in level 2 are 4, 10 and 6 (all even). 
Therefore, the Binary tree is an odd-even binary tree.



Input:  

             4
            / \
           3   7
          / \   \
         4   10  5
Output: NO 
 

Recommended: Please try your approach on {IDE} first, before moving on to the solution.
Approach: Follow the steps below to solve the problem: 

The idea is to perform level-order traversal and check if the nodes present on even levels are even valued or not and nodes present on odd levels are odd valued or not.
If any node at an odd level is found to have odd value or vice-versa, then print “NO“.
Otherwise, after complete traversal of the tree, print “YES“.'''

# Tree node
class Node:
     
    def __init__(self, data):
         
        self.left = None
        self.right = None
        self.val = data
         
# Function to return new tree node
def newNode(data):
 
    temp = Node(data)
     
    return temp
 
# Function to check if the
# tree is even-odd tree
def isEvenOddBinaryTree(root):
     
    if (root == None):
        return True
  
    q = []
     
    # Stores nodes of each level
    q.append(root)
  
    # Store the current level
    # of the binary tree
    level = 0
  
    # Traverse until the
    # queue is empty
    while (len(q) != 0):
  
        # Stores the number of nodes
        # present in the current level
        size = len(q)
         
        for i in range(size):
            node = q[0]
            q.pop(0)
  
            # Check if the level
            # is even or odd
            if (level % 2 == 0):
  
                if (node.val % 2 == 1):
                    return False
                 
                elif (level % 2 == 1):
                    if (node.val % 2 == 0):
                        return False
                 
                # Add the nodes of the next
                # level into the queue
                if (node.left != None):
                    q.append(node.left)
                 
                if (node.right != None):
                    q.append(node.right)
                 
            # Increment the level count
            level += 1
         
        return True
     
# Driver code
if __name__=="__main__":
     
    # Construct a Binary Tree
    root = None
    root = newNode(2)
    root.left = newNode(3)
    root.right = newNode(9)
    root.left.left = newNode(4)
    root.left.right = newNode(10)
    root.right.right = newNode(6)
  
    # Check if the binary tree
    # is even-odd tree or not
    if (isEvenOddBinaryTree(root)):
        print("YES")
    else:
        print("NO")

#or


#include <bits/stdc++.h>
using namespace std;
   
// tree node
struct Node
{
    int data;
    Node *left, *right;
};
   
// returns a new
// tree Node
Node* newNode(int data)
{
    Node* temp = new Node();
    temp->data = data;
    temp->left = temp->right = NULL;
    return temp;
}
   
// Utility function to recursively traverse tree and check the diff between child nodes
bool BSTUtil(Node * root){
    if(root==NULL)
        return true;
     
    //if left nodes exist and absolute difference between left child and parent is divisible by 2, then return false       
    if(root->left!=NULL && abs(root->data - root->left->data)%2==0)
        return false;
     //if right nodes exist and absolute difference between right child and parent is divisible by 2, then return false
    if(root->right!=NULL && abs(root->data - root->right->data)%2==0)
        return false;
     
    //recursively traverse left and right subtree
    return BSTUtil(root->left) && BSTUtil(root->right);
}
 
// Utility function to check if binary tree is even-odd binary tree
bool isEvenOddBinaryTree(Node * root){
    if(root==NULL)
        return true;
     
    // if root node is odd, return false
    if(root->data%2 != 0)
        return false;
     
    return BSTUtil(root);  
}
   
// driver program
int main()
{
    // construct a tree
    Node* root = newNode(5);
    root->left = newNode(2);
    root->right = newNode(6);
    root->left->left = newNode(1);
    root->left->right = newNode(5);
    root->right->right = newNode(7);
    root->left->right->left = newNode(12);
 
    root->right->right->right = newNode(14);
    root->right->right->left = newNode(16);
     
    if(BSTUtil(root))
      cout<<"YES";
    else
      cout<<"NO";
    return 0;
}


'''Sum of Left Leaves
Easy

2170

195

Add to List

Share
Given the root of a binary tree, return the sum of all left leaves.

 

Example 1:


Input: root = [3,9,20,null,null,15,7]
Output: 24
Explanation: There are two left leaves in the binary tree, with values 9 and 15 respectively.
Example 2:

Input: root = [1]
Output: 0
 

Constraints:

The number of nodes in the tree is in the range [1, 1000].
-1000 <= Node.val <= 1000'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        
        def dfs(node, isleft):
            if not node:
                return 0
            
            if not node.left and not node.right and isleft:
                return node.val
            
            return dfs(node.left, True) +  dfs(node.right, False)
        
        return dfs(root, False)

'''Your input
[3,9,20,null,null,15,7]
Output
24
Expected
24'''

#or

class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        result = 0
        stack = [(root, False)]
        while stack:
            curr, is_left = stack.pop()
            if not curr:
                continue
            if not curr.left and not curr.right:
                if is_left:
                    result += curr.val
            else:
                stack.append((curr.left, True))
                stack.append((curr.right, False))
        return result

'''Expression Evaluation
Difficulty Level : Medium
Last Updated : 16 Oct, 2020
 
Evaluate an expression represented by a String. The expression can contain parentheses, you can assume parentheses are well-matched. For simplicity, you can assume only binary operations allowed are +, -, *, and /. Arithmetic Expressions can be written in one of three forms:
Infix Notation: Operators are written between the operands they operate on, e.g. 3 + 4.
Prefix Notation: Operators are written before the operands, e.g + 3 4
Postfix Notation: Operators are written after operands.
Infix Expressions are harder for Computers to evaluate because of the additional work needed to decide precedence. Infix notation is how expressions are written and recognized by humans and, generally, input to programs. Given that they are harder to evaluate, they are generally converted to one of the two remaining forms. A very well known algorithm for converting an infix notation to a postfix notation is Shunting Yard Algorithm by Edgar Dijkstra. This algorithm takes as input an Infix Expression and produces a queue that has this expression converted to postfix notation. The same algorithm can be modified so that it outputs the result of the evaluation of expression instead of a queue. The trick is using two stacks instead of one, one for operands, and one for operators. The algorithm was described succinctly on http://www.cis.upenn.edu/ matuszek/cit594-2002/Assignments/5-expressions.htm, and is reproduced here. (Note that credit for succinctness goes to the author of said page) 

 
1. While there are still tokens to be read in,
   1.1 Get the next token.
   1.2 If the token is:
       1.2.1 A number: push it onto the value stack.
       1.2.2 A variable: get its value, and push onto the value stack.
       1.2.3 A left parenthesis: push it onto the operator stack.
       1.2.4 A right parenthesis:
         1 While the thing on top of the operator stack is not a 
           left parenthesis,
             1 Pop the operator from the operator stack.
             2 Pop the value stack twice, getting two operands.
             3 Apply the operator to the operands, in the correct order.
             4 Push the result onto the value stack.
         2 Pop the left parenthesis from the operator stack, and discard it.
       1.2.5 An operator (call it thisOp):
         1 While the operator stack is not empty, and the top thing on the
           operator stack has the same or greater precedence as thisOp,
           1 Pop the operator from the operator stack.
           2 Pop the value stack twice, getting two operands.
           3 Apply the operator to the operands, in the correct order.
           4 Push the result onto the value stack.
         2 Push thisOp onto the operator stack.
2. While the operator stack is not empty,
    1 Pop the operator from the operator stack.
    2 Pop the value stack twice, getting two operands.
    3 Apply the operator to the operands, in the correct order.
    4 Push the result onto the value stack.
3. At this point the operator stack should be empty, and the value
   stack should have only one value in it, which is the final result.'''


# Python3 program to evaluate a given
# expression where tokens are
# separated by space.
 
# Function to find precedence
# of operators.
def precedence(op):
     
    if op == '+' or op == '-':
        return 1
    if op == '*' or op == '/':
        return 2
    return 0
 
# Function to perform arithmetic
# operations.
def applyOp(a, b, op):
     
    if op == '+': return a + b
    if op == '-': return a - b
    if op == '*': return a * b
    if op == '/': return a // b
 
# Function that returns value of
# expression after evaluation.
def evaluate(tokens):
     
    # stack to store integer values.
    values = []
     
    # stack to store operators.
    ops = []
    i = 0
     
    while i < len(tokens):
         
        # Current token is a whitespace,
        # skip it.
        if tokens[i] == ' ':
            i += 1
            continue
         
        # Current token is an opening
        # brace, push it to 'ops'
        elif tokens[i] == '(':
            ops.append(tokens[i])
         
        # Current token is a number, push
        # it to stack for numbers.
        elif tokens[i].isdigit():
            val = 0
             
            # There may be more than one
            # digits in the number.
            while (i < len(tokens) and
                tokens[i].isdigit()):
             
                val = (val * 10) + int(tokens[i])
                i += 1
             
            values.append(val)
             
            # right now the i points to
            # the character next to the digit,
            # since the for loop also increases
            # the i, we would skip one
            #  token position; we need to
            # decrease the value of i by 1 to
            # correct the offset.
            i-=1
         
        # Closing brace encountered,
        # solve entire brace.
        elif tokens[i] == ')':
         
            while len(ops) != 0 and ops[-1] != '(':
             
                val2 = values.pop()
                val1 = values.pop()
                op = ops.pop()
                 
                values.append(applyOp(val1, val2, op))
             
            # pop opening brace.
            ops.pop()
         
        # Current token is an operator.
        else:
         
            # While top of 'ops' has same or
            # greater precedence to current
            # token, which is an operator.
            # Apply operator on top of 'ops'
            # to top two elements in values stack.
            while (len(ops) != 0 and
                precedence(ops[-1]) >=
                   precedence(tokens[i])):
                         
                val2 = values.pop()
                val1 = values.pop()
                op = ops.pop()
                 
                values.append(applyOp(val1, val2, op))
             
            # Push current token to 'ops'.
            ops.append(tokens[i])
         
        i += 1
     
    # Entire expression has been parsed
    # at this point, apply remaining ops
    # to remaining values.
    while len(ops) != 0:
         
        val2 = values.pop()
        val1 = values.pop()
        op = ops.pop()
                 
        values.append(applyOp(val1, val2, op))
     
    # Top of 'values' contains result,
    # return it.
    return values[-1]
 
# Driver Code
if __name__ == "__main__":
     
    print(evaluate("10 + 2 * 6"))
    print(evaluate("100 * 2 + 12"))
    print(evaluate("100 * ( 2 + 12 )"))
    print(evaluate("100 * ( 2 + 12 ) / 14"))


'''Output:
22
212
1400
100
Time Complexity: O(n) 
Space Complexity: O(n)'''

'''Minimum number of jumps to reach end
Difficulty Level : Medium
Last Updated : 17 Aug, 2021
 
Given an array of integers where each element represents the max number of steps that can be made forward from that element. Write a function to return the minimum number of jumps to reach the end of the array (starting from the first element). If an element is 0, they cannot move through that element. If the end isn’t reachable, return -1.

Examples: 

Input: arr[] = {1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9}
Output: 3 (1-> 3 -> 8 -> 9)
Explanation: Jump from 1st element 
to 2nd element as there is only 1 step, 
now there are three options 5, 8 or 9. 
If 8 or 9 is chosen then the end node 9 
can be reached. So 3 jumps are made.

Input:  arr[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
Output: 10
Explanation: In every step a jump 
is needed so the count of jumps is 10.
The first element is 1, so can only go to 3. The second element is 3, so can make at most 3 steps eg to 5 or 8 or 9.

Recommended: Please solve it on “PRACTICE ” first, before moving on to the solution.
 
Method 1: Naive Recursive Approach. 
Approach: A naive approach is to start from the first element and recursively call for all the elements reachable from first element. The minimum number of jumps to reach end from first can be calculated using minimum number of jumps needed to reach end from the elements reachable from first. 

minJumps(start, end) = Min ( minJumps(k, end) ) for all k reachable from start'''

# Python3 program to find Minimum
# number of jumps to reach end
 
# Returns minimum number of jumps
# to reach arr[h] from arr[l]
def minJumps(arr, l, h):
 
    # Base case: when source and
    # destination are same
    if (h == l):
        return 0
 
    # when nothing is reachable
    # from the given source
    if (arr[l] == 0):
        return float('inf')
 
    # Traverse through all the points
    # reachable from arr[l]. Recursively
    # get the minimum number of jumps
    # needed to reach arr[h] from
    # these reachable points.
    min = float('inf')
    for i in range(l + 1, h + 1):
        if (i < l + arr[l] + 1):
            jumps = minJumps(arr, i, h)
            if (jumps != float('inf') and
                       jumps + 1 < min):
                min = jumps + 1
 
    return min
 
# Driver program to test above function
arr = [1, 3, 6, 3, 2, 3, 6, 8, 9, 5]
n = len(arr)
print('Minimum number of jumps to reach',
     'end is', minJumps(arr, 0, n-1))
 

#or

def minJumps(arr, l, h):
 
    # Base case: when source and
    # destination are same
    if (h == l):
        return 0
 
    # when nothing is reachable
    # from the given source
    if (arr[l] == 0):
        return float('inf')
 
    # Traverse through all the points
    # reachable from arr[l]. Recursively
    # get the minimum number of jumps
    # needed to reach arr[h] from
    # these reachable points.
    min = float('inf')
    for i in range(l + 1, h + 1):
        if (i < l + arr[l] + 1):
            jumps = minJumps(arr, i, h)
            if (jumps != float('inf') and
                       jumps + 1 < min):
                min = jumps + 1
 
    return min
 
# Driver program to test above function
arr = [1, 3, 6, 3, 2, 3, 6, 8, 9, 5]
n = len(arr)
print('Minimum number of jumps to reach',
     'end is', minJumps(arr, 0, n-1))


'''Two elements whose sum is closest to zero
Difficulty Level : Easy
Last Updated : 17 Aug, 2021
Question: An Array of integers is given, both +ve and -ve. You need to find the two elements such that their sum is closest to zero.
For the below array, program should print -80 and 85.

Recommended: Please solve it on “PRACTICE” first, before moving on to the solution.
METHOD 1 (Simple) 
For each element, find the sum of it with every other element in the array and compare sums. Finally, return the minimum sum.

Implementation:'''

def minAbsSumPair(arr,arr_size):
    inv_count = 0
 
    # Array should have at least
    # two elements
    if arr_size < 2:
        print("Invalid Input")
        return
 
    # Initialization of values
    min_l = 0
    min_r = 1
    min_sum = arr[0] + arr[1]
    for l in range (0, arr_size - 1):
        for r in range (l + 1, arr_size):
            sum = arr[l] + arr[r]                
            if abs(min_sum) > abs(sum):        
                min_sum = sum
                min_l = l
                min_r = r
 
    print("The two elements whose sum is minimum are",
            arr[min_l], "and ", arr[min_r])
 
# Driver program to test above function
arr = [1, 60, -10, 70, -80, 85]
 
minAbsSumPair(arr, 6)

'''The two elements whose sum is minimum are -80 and 85'''


# Python3 implementation using STL
import sys
 
def findMinSum(arr, n):
     
    for i in range(1, n):
         
        # Modified to sort by absolute values
        if (not abs(arr[i - 1]) < abs(arr[i])):
            arr[i - 1], arr[i] = arr[i], arr[i - 1]
 
    Min = sys.maxsize
    x = 0
    y = 0
   
    for i in range(1, n):
         
        # Absolute value shows how
        # close it is to zero
        if (abs(arr[i - 1] + arr[i]) <= Min):
             
            # If found an even close value
            # update min and store the index
            Min = abs(arr[i - 1] + arr[i])
            x = i - 1
            y = i
 
    print("The two elements whose sum is minimum are",
          arr[x], "and", arr[y])
 
# Driver code
arr = [ 1, 60, -10, 70, -80, 85 ]
n = len(arr)
 
findMinSum(arr, n)
 

'''Intersection of Two Linked Lists
Easy

6468

701

Add to List

Share
Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return null.

For example, the following two linked lists begin to intersect at node c1:


The test cases are generated such that there are no cycles anywhere in the entire linked structure.

Note that the linked lists must retain their original structure after the function returns.

Custom Judge:

The inputs to the judge are given as follows (your program is not given these inputs):

intersectVal - The value of the node where the intersection occurs. This is 0 if there is no intersected node.
listA - The first linked list.
listB - The second linked list.
skipA - The number of nodes to skip ahead in listA (starting from the head) to get to the intersected node.
skipB - The number of nodes to skip ahead in listB (starting from the head) to get to the intersected node.
The judge will then create the linked structure based on these inputs and pass the two heads, headA and headB to your program. If you correctly return the intersected node, then your solution will be accepted.

 

Example 1:


Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
Output: Intersected at '8'
Explanation: The intersected node's value is 8 (note that this must not be 0 if the two lists intersect).
From the head of A, it reads as [4,1,8,4,5]. From the head of B, it reads as [5,6,1,8,4,5]. There are 2 nodes before the intersected node in A; There are 3 nodes before the intersected node in B.
Example 2:


Input: intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
Output: Intersected at '2'
Explanation: The intersected node's value is 2 (note that this must not be 0 if the two lists intersect).
From the head of A, it reads as [1,9,1,2,4]. From the head of B, it reads as [3,2,4]. There are 3 nodes before the intersected node in A; There are 1 node before the intersected node in B.
Example 3:


Input: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
Output: No intersection
Explanation: From the head of A, it reads as [2,6,4]. From the head of B, it reads as [1,5]. Since the two lists do not intersect, intersectVal must be 0, while skipA and skipB can be arbitrary values.
Explanation: The two lists do not intersect, so return null.
 

Constraints:

The number of nodes of listA is in the m.
The number of nodes of listB is in the n.
0 <= m, n <= 3 * 104
1 <= Node.val <= 105
0 <= skipA <= m
0 <= skipB <= n
intersectVal is 0 if listA and listB do not intersect.
intersectVal == listA[skipA] == listB[skipB] if listA and listB intersect.
 

Follow up: Could you write a solution that runs in O(n) time and use only O(1) memory?'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        a = headA
        b = headB
        while a != b:
            if a == None:
                a = headB
            else:
                a = a.next
                
            if b == None:
                b = headA
            else:
                b = b.next
                
        return a
        
#OR

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        A = headA
        B = headB
        if not A or not B: return None

        # Concatenate A and B
        last = A
        while last.next: last = last.next
        last.next = B

        # Find the start of the loop
        fast = slow = A
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow == fast:
                fast = A
                while fast != slow:
                    slow, fast = slow.next, fast.next
                last.next = None
                return slow

        # No loop found
        last.next = None
        return None

'''Your input
8
[4,1,8,4,5]
[5,6,1,8,4,5]
2
3
Output
Intersected at '8'
Expected
Intersected at '8'''

''' Symmetric Tree
Easy

7029

182

Add to List

Share
Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

 

Example 1:


Input: root = [1,2,2,3,4,4,3]
Output: true
Example 2:


Input: root = [1,2,2,null,3,null,3]
Output: false
 

Constraints:

The number of nodes in the tree is in the range [1, 1000].
-100 <= Node.val <= 100
 

Follow up: Could you solve it both recursively and iteratively?'''

def isSymmetric(self, root):
    if not root:
        return True
    return self.dfs(root.left, root.right)
    
def dfs(self, l, r):
    if l and r:
        return l.val == r.val and self.dfs(l.left, r.right) and self.dfs(l.right, r.left)
    return l == r
	
def isSymmetric(self, root):
    if not root:
        return True
    stack = [(root.left, root.right)]
    while stack:
        l, r = stack.pop()
        if not l and not r:
            continue
        if not l or not r or (l.val != r.val):
            return False
        stack.append((l.left, r.right))
        stack.append((l.right, r.left))
    return True

'''Your input
[1,2,2,3,4,4,3]
Output
true
Expected
true'''

#or

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        
        if not root:
            return True
        
        left=root.left
        right=root.right
        
        if not left and not right:
            return True
        
        def check(t1, t2):
            if not t1 and not t2:
                return True
            if t1 and not t2:
                return False
            if not t1 and t2:
                return False
            if t1.val!=t2.val:
                return False
            return check(t1.left, t2.right) and check(t1.right, t2.left)
        
        return check(left, right)

'''Group Anagrams
Medium

6562

252

Add to List

Share
Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

 

Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
Example 2:

Input: strs = [""]
Output: [[""]]
Example 3:

Input: strs = ["a"]
Output: [["a"]]
 

Constraints:

1 <= strs.length <= 104
0 <= strs[i].length <= 100
strs[i] consists of lowercase English letters.'''

'''Algorithm:

We can see that wherever the frequency of letters in a string are same, they can be called anagrams
So, the first thought would be to use Counter to calculate that and then distinguish.
But, for that we would have to use Counter as a key in Dictionary and when we find an equal Counter to that while iterating with strs we will append the list with that element and that cannot happen as Counter object is unhashable.
So, here we first sort every word in that array and make a new array such that they all have the similar index.
Then you use the words in sorted array as key and keep appending the words that are corresponding in the other list.
For this we use dictionary. As anagrams have same letters, their sorted version would also be same, so we keep their sorted version as the key and we keep the elements that are correspondingly in the similar array in the list of their key.
This is possible because whatever operation that we make, the corresponding remains same.
I can't explain better than this please help !!!!!! 
The longer version. Concise version in below this:'''

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        
        sorted_letters_list = []
        for word in strs:
            sorted_letters_list.append(sorted(word))
        
        sorted_letter = []
        for letter_list in sorted_letters_list:
            sorted_letter.append(''.join(letter_list))
        
        out_dict = dict()
        for idx,val in enumerate(sorted_letter):
            if val in out_dict:
                out_dict[val].append(strs[idx])
            else:
                out_dict[val] = [strs[idx]]
        
        return [out_dict[x] for x in out_dict]
#This is the concise version of the above code:

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        out = map(''.join ,map(sorted, strs))
        
        out_dict = defaultdict(list)
        for idx,val in enumerate(out):
            out_dict[val].append(strs[idx])
        
        return (out_dict[x] for x in out_dict)
'''Time: O(N * MLogM)
Space: O(N)
where N is the length of the list and M is the length of largest word.'''


#O(w*n*log(n)) Time | O(wn) Space where w-Number of words
    #n-length of the longest word
anagrams = {}
for word in strs:
    sortedWord = "".join(sorted(word))
    if sortedWord in anagrams:
        anagrams[sortedWord].append(word)
    else:
        anagrams[sortedWord] = [word]
return list(anagrams.values())


'''Find the row with maximum number of 1s
Difficulty Level : Medium
Last Updated : 23 Jul, 2021
Given a boolean 2D array, where each row is sorted. Find the row with the maximum number of 1s. 

Example:  

Input matrix
0 1 1 1
0 0 1 1
1 1 1 1  // this row has maximum 1s
0 0 0 0

Output: 2
Recommended: Please solve it on “PRACTICE ” first, before moving on to the solution. 
 
A simple method is to do a row wise traversal of the matrix, count the number of 1s in each row and compare the count with max. Finally, return the index of row with maximum 1s. The time complexity of this method is O(m*n) where m is number of rows and n is number of columns in matrix.

We can do better. Since each row is sorted, we can use Binary Search to count of 1s in each row. We find the index of first instance of 1 in each row. The count of 1s will be equal to total number of columns minus the index of first 1.'''

# The main function that returns index
# of row with maximum number of 1s.
def rowWithMax1s(mat) : 
 
    # Initialize max using values from first row.
    max_row_index = 0;
    max = first(mat[0], 0, C - 1)
 
    # Traverse for each row and count number of 1s
    # by finding the index of first 1
    for i in range(1, R):
       
        # Count 1s in this row only if this row
        # has more 1s than max so far
 
        # Count 1s in this row only if this row
        # has more 1s than max so far
        if (max != -1 and mat[i][C - max - 1] == 1):
           
            # Note the optimization here also
            index = first (mat[i], 0, C - max)
 
            if (index != -1 and C - index > max):
                max = C - index
                max_row_index = i
        else:
            max = first(mat[i], 0, C - 1)
           
    return max_row_index

#OR

def rowWithMax1s( mat):
     
    # Initialize max values
    R = len(mat)
    C = len(mat[0])
    max_row_index = 0
    index=C-1;
    # Traverse for each row and
    # count number of 1s by finding
    # the index of first 1
    for i in range(0, R):
      flag=False #to check whether a row has more 1's than previous
      while(index >=0 and mat[i][index]==1):
        flag=True #present row has more 1's than previous
        index-=1
        if(flag): #if the present row has more 1's than previous
          max_row_index = i
      if max_row_index==0 and mat[0][C-1]==0:
        return 0;
    return max_row_index
 
# Driver Code
mat = [[0, 0, 0, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 0, 0, 0]]
print ("Index of row with maximum 1s is",
    rowWithMax1s(mat))

#Index of row with maximum 1s is 2

'''
Time Complexity: O(m*n)
Space Complexity: O(1)
'''

''' Minimum Cost to Merge Stones
Hard

1152

68

Add to List

Share
There are n piles of stones arranged in a row. The ith pile has stones[i] stones.

A move consists of merging exactly k consecutive piles into one pile, and the cost of this move is equal to the total number of stones in these k piles.

Return the minimum cost to merge all piles of stones into one pile. If it is impossible, return -1.

 

Example 1:

Input: stones = [3,2,4,1], k = 2
Output: 20
Explanation: We start with [3, 2, 4, 1].
We merge [3, 2] for a cost of 5, and we are left with [5, 4, 1].
We merge [4, 1] for a cost of 5, and we are left with [5, 5].
We merge [5, 5] for a cost of 10, and we are left with [10].
The total cost was 20, and this is the minimum possible.
Example 2:

Input: stones = [3,2,4,1], k = 3
Output: -1
Explanation: After any merge operation, there are 2 piles left, and we can't merge anymore.  So the task is impossible.
Example 3:

Input: stones = [3,5,1,2,6], k = 3
Output: 25
Explanation: We start with [3, 5, 1, 2, 6].
We merge [5, 1, 2] for a cost of 8, and we are left with [3, 8, 6].
We merge [3, 8, 6] for a cost of 17, and we are left with [17].
The total cost was 25, and this is the minimum possible.
 '''

def mergeStones(self, stones: List[int], K: int) -> int:
        def recursive(i, j, piles):
            if i == j and piles == 1:
                return 0
            if (j - i + 1 - piles) % (K - 1) != 0: 
                return float('inf')  # means impossible
            if (i, j, piles) in dp:
                return dp[(i, j, piles)]
            if piles == 1:
                dp[(i,j,piles)] = recursive(i, j, K) + pre_sum[j+1] - pre_sum[i]
                return dp[(i,j,piles)]
            else:
                min_cost = float('inf')
                for k in range(i, j, K - 1):
                    min_cost = min(min_cost, recursive(i, k, 1) + recursive(k + 1, j, piles - 1))
                dp[(i, j, piles)] = min_cost
                return dp[(i, j, piles)]
        
        n = len(stones)
        if (n - 1) % (K - 1) != 0:
            return -1
        pre_sum = [0] * (n + 1)
        for i in range(n):
            pre_sum[i + 1] = pre_sum[i] + stones[i]
        dp = {}
        return recursive(0, n - 1, 1)

'''Your input
[3,2,4,1]
2
Output
20
Expected
20'''

#https://leetcode.com/problems/minimum-cost-to-merge-stones/discuss/446088/Python-DP-with-explaination

'''Find the smallest and second smallest elements in an array
Difficulty Level : Basic
Last Updated : 08 Apr, 2021
Write an efficient C program to find smallest and second smallest element in an array.
 



Example: 
 

Input:  arr[] = {12, 13, 1, 10, 34, 1}
Output: The smallest element is 1 and 
        second Smallest element is 10
 

Recommended: Please solve it on “PRACTICE” first, before moving on to the solution.
A Simple Solution is to sort the array in increasing order. The first two elements in sorted array would be two smallest elements. Time complexity of this solution is O(n Log n).
A Better Solution is to scan the array twice. In first traversal find the minimum element. Let this element be x. In second traversal, find the smallest element greater than x. Time complexity of this solution is O(n).
The above solution requires two traversals of input array. 
An Efficient Solution can find the minimum two elements in one traversal. Below is complete algorithm.
Algorithm: 
 



1) Initialize both first and second smallest as INT_MAX
   first = second = INT_MAX
2) Loop through all the elements.
   a) If the current element is smaller than first, then update first 
       and second. 
   b) Else if the current element is smaller than second then update 
    second'''

# Python program to find smallest and second smallest elements
import sys
 
def print2Smallest(arr):
 
    # There should be atleast two elements
    arr_size = len(arr)
    if arr_size < 2:
        print( "Invalid Input")
        return
 
    first = second = sys.maxint
    for i in range(0, arr_size):
 
        # If current element is smaller than first then
        # update both first and second
        if arr[i] < first:
            second = first
            first = arr[i]
 
        # If arr[i] is in between first and second then
        # update second
        elif (arr[i] < second and arr[i] != first):
            second = arr[i];
 
    if (second == sys.maxint):
        print( "No second smallest element")
    else:
        print ('The smallest element is',first,'and' \
              ' second smallest element is',second)
 
# Driver function to test above function
arr = [12, 13, 1, 10, 34, 1]
print2Smallest(arr)

#The smallest element is 1 and second Smallest element is 10

#https://www.geeksforgeeks.org/to-find-smallest-and-second-smallest-element-in-an-array/

'''Longest Common Prefix
Easy

5136

2429

Add to List

Share
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

 

Example 1:

Input: strs = ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.'''

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs: return ''
        
        m = min(strs)
        M = max(strs)
        i = 0
        for i in range(min(len(m),len(M))):
            if m[i] != M[i]: break
        else: i += 1
        return m[:i]

def longestCommonPrefix(self, strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    if not strs:
        return ""
    shortest = min(strs,key=len)
    for i, ch in enumerate(shortest):
        for other in strs:
            if other[i] != ch:
                return shortest[:i]
    return shortest 

'''Your input
["flower","flow","flight"]
Output
"fl"
Expected
"fl"'''

'''Rat in a Maze | Backtracking-2
Difficulty Level : Medium
Last Updated : 29 Jul, 2021
We have discussed Backtracking and Knight’s tour problem in Set 1. Let us discuss Rat in a Maze as another example problem that can be solved using Backtracking.

A Maze is given as N*N binary matrix of blocks where source block is the upper left most block i.e., maze[0][0] and destination block is lower rightmost block i.e., maze[N-1][N-1]. A rat starts from source and has to reach the destination. The rat can move only in two directions: forward and down. 

In the maze matrix, 0 means the block is a dead end and 1 means the block can be used in the path from source to destination. Note that this is a simple version of the typical Maze problem. For example, a more complex version can be that the rat can move in 4 directions and a more complex version can be with a limited number of moves.

Following is an example maze.  

 Gray blocks are dead ends (value = 0).




Following is a binary matrix representation of the above maze. 

{1, 0, 0, 0}
{1, 1, 0, 1}
{0, 1, 0, 0}
{1, 1, 1, 1}
Following is a maze with highlighted solution path.



Following is the solution matrix (output of program) for the above input matrix. 

{1, 0, 0, 0}
{1, 1, 0, 0}
{0, 1, 0, 0}
{0, 1, 1, 1}
All entries in solution path are marked as 1.
Recommended: Please solve it on “PRACTICE” first, before moving on to the solution.
Backtracking Algorithm: Backtracking is an algorithmic-technique for solving problems recursively by trying to build a solution incrementally. Solving one piece at a time, and removing those solutions that fail to satisfy the constraints of the problem at any point of time (by time, here, is referred to the time elapsed till reaching any level of the search tree) is the process of backtracking.

Approach: Form a recursive function, which will follow a path and check if the path reaches the destination or not. If the path does not reach the destination then backtrack and try other paths. 

Algorithm:  

Create a solution matrix, initially filled with 0’s.
Create a recursive function, which takes initial matrix, output matrix and position of rat (i, j).
if the position is out of the matrix or the position is not valid then return.
Mark the position output[i][j] as 1 and check if the current position is destination or not. If destination is reached print the output matrix and return.
Recursively call for position (i+1, j) and (i, j+1).
Unmark position (i, j), i.e output[i][j] = 0.
'''


# Maze size
N = 4
 
# A utility function to print solution matrix sol
def printSolution( sol ):
     
    for i in sol:
        for j in i:
            print(str(j) + " ", end ="")
        print("")
 
# A utility function to check if x, y is valid
# index for N * N Maze
def isSafe( maze, x, y ):
     
    if x >= 0 and x < N and y >= 0 and y < N and maze[x][y] == 1:
        return True
     
    return False
 
""" This function solves the Maze problem using Backtracking.
    It mainly uses solveMazeUtil() to solve the problem. It
    returns false if no path is possible, otherwise return
    true and prints the path in the form of 1s. Please note
    that there may be more than one solutions, this function
    prints one of the feasable solutions. """
def solveMaze( maze ):
     
    # Creating a 4 * 4 2-D list
    sol = [ [ 0 for j in range(4) ] for i in range(4) ]
     
    if solveMazeUtil(maze, 0, 0, sol) == False:
        print("Solution doesn't exist");
        return False
     
    printSolution(sol)
    return True
     
# A recursive utility function to solve Maze problem
def solveMazeUtil(maze, x, y, sol):
     
    # if (x, y is goal) return True
    if x == N - 1 and y == N - 1 and maze[x][y]== 1:
        sol[x][y] = 1
        return True
         
    # Check if maze[x][y] is valid
    if isSafe(maze, x, y) == True:
        # Check if the current block is already part of solution path.   
        if sol[x][y] == 1:
            return False
           
        # mark x, y as part of solution path
        sol[x][y] = 1
         
        # Move forward in x direction
        if solveMazeUtil(maze, x + 1, y, sol) == True:
            return True
             
        # If moving in x direction doesn't give solution
        # then Move down in y direction
        if solveMazeUtil(maze, x, y + 1, sol) == True:
            return True
           
        # If moving in y direction doesn't give solution then
        # Move back in x direction
        if solveMazeUtil(maze, x - 1, y, sol) == True:
            return True
             
        # If moving in backwards in x direction doesn't give solution
        # then Move upwards in y direction
        if solveMazeUtil(maze, x, y - 1, sol) == True:
            return True
         
        # If none of the above movements work then
        # BACKTRACK: unmark x, y as part of solution path
        sol[x][y] = 0
        return False
 
# Driver program to test above function
if __name__ == "__main__":
    # Initialising the maze
    maze = [ [1, 0, 0, 0],
             [1, 1, 0, 1],
             [0, 1, 0, 0],
             [1, 1, 1, 1] ]
              
    solveMaze(maze)

'''Maximum Tip Calculator
Last Updated : 17 Aug, 2021
Rahul and Ankit are the only two waiters in the Royal Restaurant. Today, the restaurant received N orders. The amount of tips may differ when handled by different waiters and given as arrays A[] and B[] such that if Rahul takes the ith Order, he would be tipped A[i] rupees, and if Ankit takes this order, the tip would be B[i] rupees.

In order to maximize the total tip value, they decided to distribute the order among themselves. One order will be handled by one person only. Also, due to time constraints, Rahul cannot take more than X orders and Ankit cannot take more than Y orders. It is guaranteed that X + Y is greater than or equal to N, which means that all the orders can be handled by either Rahul or Ankit. The task is to find out the maximum possible amount of total tip money after processing all the orders.

Examples:

Input: N = 5, X = 3, Y = 3, A[] = {1, 2, 3, 4, 5}, B[] = {5, 4, 3, 2, 1}
Output: 21
Explanation:
Step 1: 5 is included from Ankit’s array
Step 2: 4 is included from Ankit’s array
Step 3: As both of them has same value 3 then choose any one of them
Step 4: 4 is included from Rahul’s array
Step 4: 5 is included from Rahul’s array
Therefore, the maximum possible amount of total tip money sums up to 21.

Input: N = 7, X = 3, Y = 4, A[] = {8, 7, 15, 19, 16, 16, 18}, B[] = {1, 7, 15, 11, 12, 31, 9}
Output: 110



Recommended: Please try your approach on {IDE} first, before moving on to the solution.
Naive Approach: The simplest approach is to traverse the given arrays and start traversing both the arrays simultaneously and pick the maximum element among them and reduce the count of X if the element is taken from X else the count of Y. If one of the X or Y becomes 0, traverse other non-zero array and add its value to the maximum profit. As in every step, there is a choice to be made, this is similar to the 0-1 Knapsack Problem, in which decisions are made whether to include or exclude an element.'''

def maximumTip(arr1, arr2, n, x, y):
 
    # Base Condition
    if n == 0:
        return 0
 
    # If both have non-zero count then
    # return max element from both array
    if x != 0 and y != 0:
        return max(
            arr1[n-1] + maximumTip(arr1, arr2, n - 1, x-1, y),
            arr2[n-1] + maximumTip(arr1, arr2, n-1, x, y-1)
            )
 
    # Traverse first array, as y
    # count has become 0
    if y == 0:
        return arr1[n-1] + maximumTip(arr1, arr2, n-1, x-1, y)
 
    # Traverse 2nd array, as x
    # count has become 0
    else:
        return arr2[n - 1] + maximumTip(arr1, arr2, n-1, x, y-1)
 
 
# Drive Code
N = 5
X = 3
Y = 3
A = [1, 2, 3, 4, 5]
B = [5, 4, 3, 2, 1]
 
# Function Call
print(maximumTip(A, B, N, X, Y))

'''Output
21
Time Complexity: O(2N)
Auxiliary Space: O(1)

Efficient Approach: The above approach can be optimized by using Dynamic Programming and Memoization. If execution is traced for the values of N, X, Y, it can be seen that are there are Overlapping Subproblems. These overlapping subproblems can be computed once and stored and used when the same subproblem is called in the recursive call. Below are the steps:

Initialize a Map/Dictionary to store the overlapping subproblems result. The keys of the map will be combined values of N, X, and Y.
At each recursive call, check if a given key is present in the map then return the value from the map itself.
Else, call the function recursively and store the value in the map and return the stored value.
If X and Y are non-zero, recursively call function and take the maximum of the value returned when X is used and when Y is used.
If X or Y is zero, recursively call for the non-zero array.
After the above recursive calls end, then print the maximum possible amount of tip calculated.'''

# Python program for the above approach
 
 
# Function that finds the maximum tips
# from the given arrays as per the
# given conditions
def maximumTip(arr1, arr2, n, x, y, dd):
 
    # Create key of N, X, Y
    key = str(n) + '_' + str(x) + '_' + str(y)
 
    # Return if the current state is
    # already calculated
    if key in dd:
        return dd[key]
 
    # Base Condition
    if n == 0:
        return 0
 
    # Store and return
    if x != 0 and y != 0:
        dd[key] = max(
            arr1[n-1] + maximumTip(arr1, arr2, n-1, x-1, y, dd),
            arr2[n-1] + maximumTip(arr1, arr2, n-1, x, y-1, dd)
        )
 
        # Return the current state result
        return dd[key]
 
    # If y is zero, only x value
    # can be used
    if y == 0:
        dd[key] = arr1[n-1] + maximumTip(arr1, arr2, n-1, x-1, y, dd)
 
        # Return the current state result
        return dd[key]
 
    # If x is zero, only y value
    # can be used
    else:
 
        dd[key] = arr2[n-1] + maximumTip(arr1, arr2, n-1, x, y-1, dd)
 
        # Return the current state result
        return dd[key]
 
 
# Drive Code
N = 5
X = 3
Y = 3
A = [1, 2, 3, 4, 5]
B = [5, 4, 3, 2, 1]
 
# Stores the results of the
# overlapping state
dd = {}
 
# Function Call
print(maximumTip(A, B, N, X, Y, dd))

'''Maximum 0’s between two immediate 1’s in binary representation
Difficulty Level : Hard
Last Updated : 16 Jul, 2021
Given a number n, the task is to find the maximum 0’s between two immediate 1’s in binary representation of given n. Return -1 if binary representation contains less than two 1’s.

Examples : 

Input : n = 47
Output: 1
// binary of n = 47 is 101111

Input : n = 549
Output: 3
// binary of n = 549 is 1000100101

Input : n = 1030
Output: 7
// binary of n = 1030 is 10000000110

Input : n = 8
Output: -1
// There is only one 1 in binary representation
// of 8.
Recommended: Please solve it on “PRACTICE” first, before moving on to the solution.
The idea to solve this problem is to use shift operator. We just need to find the position of two immediate 1’s in binary representation of n and maximize the difference of these position. 

Return -1 if number is 0 or is a power of 2. In these cases there are less than two 1’s in binary representation.
Initialize variable prev with position of first right most 1, it basically stores the position of previously seen 1.
Now take another variable cur which stores the position of immediate 1 just after prev.
Now take difference of cur – prev – 1, it will be the number of 0’s between to immediate 1’s and compare it with previous max value of 0’s and update prev i.e; prev=cur for next iteration.
Use auxiliary variable setBit, which scans all bits of n and helps to detect if current bits is 0 or 1.
Initially check if N is 0 or power of 2.'''

# Python3 program to find maximum number of
# 0's in binary representation of a number
 
# Returns maximum 0's between two immediate
# 1's in binary representation of number
def maxZeros(n):
    # If there are no 1's or there is
    # only 1, then return -1
    if (n == 0 or (n & (n - 1)) == 0):
        return -1
 
    # loop to find position of right most 1
    # here sizeof is 4 that means total 32 bits
    setBit = 1
    prev = 0
    i = 1
    while(i < 33):
        prev += 1
 
        # we have found right most 1
        if ((n & setBit) == setBit):
            setBit = setBit << 1
            break
 
        # left shift setBit by 1 to check next bit
        setBit = setBit << 1
 
    # now loop through for remaining bits and find
    # position of immediate 1 after prev
    max0 = -10**9
    cur = prev
    for j in range(i + 1, 33):
        cur += 1
 
        # if current bit is set, then compare
        # difference of cur - prev -1 with
        # previous maximum number of zeros
        if ((n & setBit) == setBit):
            if (max0 < (cur - prev - 1)):
                max0 = cur - prev - 1
 
            # update prev
            prev = cur
        setBit = setBit << 1
 
    return max0
 
# Driver Code
n = 549
 
# Initially check that number must not
# be 0 and power of 2
print(maxZeros(n))
 
# 3

'''Lowest Common Ancestor of a Binary Tree
Medium

6934

225

Add to List

Share
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

 

Example 1:


Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.
Example 2:


Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
Example 3:

Input: root = [1,2], p = 1, q = 2
Output: 1
 

Constraints:

The number of nodes in the tree is in the range [2, 105].
-109 <= Node.val <= 109
All Node.val are unique.
p != q
p and q will exist in the tree.'''

def lowestCommonAncestor(self, root, p, q):
    stack = [root]
    parent = {root: None}
    while p not in parent or q not in parent:
        node = stack.pop()
        if node.left:
            parent[node.left] = node
            stack.append(node.left)
        if node.right:
            parent[node.right] = node
            stack.append(node.right)
    ancestors = set()
    while p:
        ancestors.add(p)
        p = parent[p]
    while q not in ancestors:
        q = parent[q]
    return q

'''Your input
[3,5,1,6,2,0,8,null,null,7,4]
5
1
Output
3
Expected
3'''

def lowestCommonAncestor(self, root, p, q):
        if not root: return None
        if p == root or q == root:
            return root
        left = self.lowestCommonAncestor(root.left, p , q)
        right = self.lowestCommonAncestor(root.right, p , q)
        
        if left and right:
            return root
        if not left:
            return right
        if not right:
            return left


'''Shortest Unique Prefix
Medium

35

0

Add to favorites
Asked In:
GOOGLE
Find shortest unique prefix to represent each word in the list.

Example:

Input: [zebra, dog, duck, dove]
Output: {z, dog, du, dov}
where we can see that
zebra = z
dog = dog
duck = du
dove = dov
NOTE : Assume that no word is prefix of another. In other words, the representation is always possible.'''

class Solution:
    # @param A : list of strings
    # @return a list of strings
    def prefix(self, words):
        prefixDict = {}
        incapablePrefixes = set() 
        for idx,word in enumerate(words):
            currPrefix = ''
            for char in word:
                currPrefix += char
                if currPrefix in prefixDict:
                    incapablePrefixes.add(currPrefix)
                    oldWordIndex = prefixDict[currPrefix]
                    oldWord = words[oldWordIndex]
                    extendedPrefix = oldWord[0:len(currPrefix)+1]
                    prefixDict[extendedPrefix] = oldWordIndex
                    del prefixDict[currPrefix]
                elif currPrefix in incapablePrefixes:
                    continue
                else:
                    prefixDict[currPrefix] = idx
                    break
        
        prefixList = [None]*len(words)
        for prefixItem in prefixDict:
            prefixList[prefixDict[prefixItem]] = prefixItem
        return prefixList

# 0r

class Solution:
    # @param A : list of strings
    # @return a list of strings
    def prefix(self, l):
        out = []
        for i in range(len(l)):
            temp = l[:i] + l[i+1:]
            point = 0
            flag = True
            while flag:
                for k in temp:
                    if(l[i][:point+1] == k[:point+1]):
                        point += 1
                        flag = True
                        break
                    else:
                        flag = False
            out.append(l[i][:point+1])
        return(out)


'''Maximum product of a triplet (subsequence of size 3) in array
Difficulty Level : Medium
Last Updated : 01 Jun, 2021
Given an integer array, find a maximum product of a triplet in array.

Examples: 

Input:  [10, 3, 5, 6, 20]
Output: 1200
Multiplication of 10, 6 and 20
 
Input:  [-10, -3, -5, -6, -20]
Output: -90

Input:  [1, -4, 3, -6, 7, 0]
Output: 168'''


import sys
 
# Function to find a maximum
# product of a triplet in array
# of integers of size n
def maxProduct(arr, n):
 
    # if size is less than 3,
    # no triplet exists
    if n < 3:
        return -1
 
    # will contain max product
    max_product = -(sys.maxsize - 1)
     
    for i in range(0, n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                max_product = max(
                    max_product, arr[i]
                    * arr[j] * arr[k])
 
    return max_product
 
# Driver Program
arr = [10, 3, 5, 6, 20]
n = len(arr)
 
max = maxProduct(arr, n)
 
if max == -1:
    print("No Tripplet Exits")
else:
    print("Maximum product is", max)


#OR

# A O(n) Python3 program to find maximum
# product pair in an array.
import sys
 
# Function to find a maximum product
# of a triplet in array of integers
# of size n
def maxProduct(arr, n):
 
    # If size is less than 3, no
    # triplet exists
    if (n < 3):
        return -1
 
    # Initialize Maximum, second
    # maximum and third maximum
    # element
    maxA = -sys.maxsize - 1
    maxB = -sys.maxsize - 1
    maxC = -sys.maxsize - 1
 
    # Initialize Minimum and
    # second mimimum element
    minA = sys.maxsize
    minB = sys.maxsize
 
    for i in range(n):
 
        # Update Maximum, second
        # maximum and third maximum
        # element
        if (arr[i] > maxA):
            maxC = maxB
            maxB = maxA
            maxA = arr[i]
             
        # Update second maximum and
        # third maximum element
        elif (arr[i] > maxB):
            maxC = maxB
            maxB = arr[i]
             
        # Update third maximum element
        elif (arr[i] > maxC):
            maxC = arr[i]
 
        # Update Minimum and second
        # mimimum element
        if (arr[i] < minA):
            minB = minA
            minA = arr[i]
 
        # Update second mimimum element
        elif (arr[i] < minB):
            minB = arr[i]
 
    return max(minA * minB * maxA,
               maxA * maxB * maxC)
 
# Driver Code
arr = [ 1, -4, 3, -6, 7, 0 ]
n = len(arr)
 
Max = maxProduct(arr, n)
 
if (Max == -1):
    print("No Triplet Exists")
else:
    print("Maximum product is", Max)


'''Minimum Window Substring
Hard

7795

487

Add to List

Share
Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.

A substring is a contiguous sequence of characters within the string.

 

Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
Example 2:

Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.
Example 3:

Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.
 

Constraints:

m == s.length
n == t.length
1 <= m, n <= 105
s and t consist of uppercase and lowercase English letters.'''

'''The idea is we use a variable-length sliding window which is gradually applied across the string. We use two pointers: start and end to mark the sliding window. We start by fixing the start pointer and moving the end pointer to the right. The way we determine the current window is a valid one is by checking if all the target letters have been found in the current window. If we are in a valid sliding window, we first make note of the sliding window of the most minimum length we have seen so far. Next we try to contract the sliding window by moving the start pointer. If the sliding window continues to be valid, we note the new minimum sliding window. If it becomes invalid (all letters of the target have been bypassed), we break out of the inner loop and go back to moving the end pointer to the right.
'''
def found_target(target_len):
    return target_len == 0

class Solution(object):
    def minWindow(self, search_string, target):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        target_letter_counts = collections.Counter(target)
        start = 0
        end = 0
        min_window = ""
        target_len = len(target)        
        
        for end in range(len(search_string)):
			# If we see a target letter, decrease the total target letter count
            if target_letter_counts[search_string[end]] > 0:
                target_len -= 1

            # Decrease the letter count for the current letter
			# If the letter is not a target letter, the count just becomes -ve
		    
            target_letter_counts[search_string[end]] -= 1
            
			# If all letters in the target are found:
            while found_target(target_len):
                window_len = end - start + 1
                if not min_window or window_len < len(min_window):
					# Note the new minimum window
                    min_window = search_string[start : end + 1]
                    
				# Increase the letter count of the current letter
                target_letter_counts[search_string[start]] += 1
                
				# If all target letters have been seen and now, a target letter is seen with count > 0
				# Increase the target length to be found. This will break out of the loop
                if target_letter_counts[search_string[start]] > 0:
                    target_len += 1
                    
                start+=1
                
        return min_window

#OR

def minWindow(self, s1: str, s2: str) -> str:
    temp = dict(Counter(s2))
    count = len(temp)

    i, j = 0, 0
    start = ""
    mini = 10 ** 6

    while j < len(s1):

        if s1[j] in temp:
            temp[s1[j]] -= 1
            if temp[s1[j]] == 0:
                count -= 1

        while count == 0 and i <= j:

            if (j - i + 1) < mini:
                mini = (j - i + 1)
                start = s1[i: j + 1]

            if s1[i] in temp:
                temp[s1[i]] += 1
                if temp[s1[i]] == 1:
                    count += 1

            i += 1
        j += 1
        
    return start

'''Find a triplet in an array whose sum is closest to a given number
Difficulty Level : Medium
Last Updated : 17 Aug, 2021
Given an array arr[] of N integers and an integer X, the task is to find three integers in arr[] such that the sum is closest to X.

Examples:

Input: arr[] = {-1, 2, 1, -4}, X = 1
Output: 2
Explanation:
Sums of triplets:
(-1) + 2 + 1 = 2
(-1) + 2 + (-4) = -3
2 + 1 + (-4) = -1
2 is closest to 1.

Input: arr[] = {1, 2, 3, 4, -5}, X = 10
Output: 9
Explanation:
Sums of triplets:
1 + 2 + 3 = 6
2 + 3 + 4 = 9
1 + 3 + 4 = 7
...
9 is closest to 10.'''

# Python3 implementation of the approach
 
import sys
 
# Function to return the sum of a
# triplet which is closest to x
def solution(arr, x) :
 
    # Sort the array
    arr.sort();
     
    # To store the closets sum
    closestSum = sys.maxsize;
 
    # Fix the smallest number among
    # the three integers
    for i in range(len(arr)-2) :
 
        # Two pointers initially pointing at
        # the last and the element
        # next to the fixed element
        ptr1 = i + 1; ptr2 = len(arr) - 1;
 
        # While there could be more pairs to check
        while (ptr1 < ptr2) :
 
            # Calculate the sum of the current triplet
            sum = arr[i] + arr[ptr1] + arr[ptr2];
 
            # If the sum is more closer than
            # the current closest sum
            if (abs(x - sum) < abs(x - closestSum)) :
                closestSum = sum;
 
            # If sum is greater then x then decrement
            # the second pointer to get a smaller sum
            if (sum > x) :
                ptr2 -= 1;
 
            # Else increment the first pointer
            # to get a larger sum
            else :
                ptr1 += 1;
 
    # Return the closest sum found
    return closestSum;
 
 
# Driver code
if __name__ == "__main__" :
 
    arr = [ -1, 2, 1, -4 ];
    x = 1;
    print(solution(arr, x))

'''Output: 
2
Complexity Analysis:

Time complexity: O(N2). 
There are only two nested loops traversing the array, so time complexity is O(n^2). Two pointer algorithm take O(n) time and the first element can be fixed using another nested traversal.
Space Complexity: O(1). 
As no extra space is required.'''

'''Reverse Words in a String
Medium

1968

3395

Add to List

Share
Given an input string s, reverse the order of the words.

A word is defined as a sequence of non-space characters. The words in s will be separated by at least one space.

Return a string of the words in reverse order concatenated by a single space.

Note that s may contain leading or trailing spaces or multiple spaces between two words. The returned string should only have a single space separating the words. Do not include any extra spaces.

 

Example 1:

Input: s = "the sky is blue"
Output: "blue is sky the"
Example 2:

Input: s = "  hello world  "
Output: "world hello"
Explanation: Your reversed string should not contain leading or trailing spaces.
Example 3:

Input: s = "a good   example"
Output: "example good a"
Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.
Example 4:

Input: s = "  Bob    Loves  Alice   "
Output: "Alice Loves Bob"
Example 5:

Input: s = "Alice does not even like bob"
Output: "bob like even not does Alice"'''

class Solution:
    def reverseWords(self, s: str) -> str:
        res = ""
        s = " "+s+ " "
        start = -1
        end = -1
        for i in range(len(s)-2, 0, -1):
            if s[i+1] == ' ' and s[i] != ' ':
                end = i

            if s[i-1] == ' ' and s[i] != ' ':
                start = i
                res = res + " " + s[start:end+1]

        return res[1:]

'''Print all pairs with given sum
Difficulty Level : Easy
Last Updated : 01 Jul, 2021
Given an array of integers, and a number ‘sum’, print all pairs in the array whose sum is equal to ‘sum’.

Examples :
Input  :  arr[] = {1, 5, 7, -1, 5}, 
          sum = 6
Output : (1, 5) (7, -1) (1, 5)

Input  :  arr[] = {2, 5, 17, -1}, 
          sum = 7
Output :  (2, 5)'''

def pairedElements(arr, sum):
   
    low = 0;
    high = len(arr) - 1;
 
    while (low < high):
        if (arr[low] +
            arr[high] == sum):
            print("The pair is : (", arr[low],
                  ", ", arr[high], ")");
        if (arr[low] + arr[high] > sum):
            high -= 1;
        else:
            low += 1;
 
#OR

def printPairs(arr, n, sum):
     
    # Store counts of all elements
    # in a dictionary
    mydict = dict()
 
    # Traverse through all the elements
    for i in range(n):
         
        # Search if a pair can be
        # formed with arr[i]
        temp = sum - arr[i]
         
        if temp in mydict:
            count = mydict[temp]
            for j in range(count):
                print("(", temp, ", ", arr[i],
                      ")", sep = "", end = '\n')
                       
        if arr[i] in mydict:
            mydict[arr[i]] += 1
        else:
            mydict[arr[i]] = 1
 
'''(1, 5)
(7, -1)
(1, 5)'''

'''Count pairs in array whose sum is divisible by K
Difficulty Level : Medium
Last Updated : 07 Jun, 2021
Given an array A[] and positive integer K, the task is to count the total number of pairs in the array whose sum is divisible by K. 
Note: This question is a generalized version of this 

Examples: 

Input : A[] = {2, 2, 1, 7, 5, 3}, K = 4
Output : 5
Explanation : 
There are five pairs possible whose sum
is divisible by '4' i.e., (2, 2), 
(1, 7), (7, 5), (1, 3) and (5, 3)

Input : A[] = {5, 9, 36, 74, 52, 31, 42}, K = 3
Output : 7'''


# Python3 code to count pairs whose
# sum is divisible by 'K'
 
# Function to count pairs whose
# sum is divisible by 'K'
def countKdivPairs(A, n, K):
     
    # Create a frequency array to count
    # occurrences of all remainders when
    # divided by K
    freq = [0] * K
     
    # Count occurrences of all remainders
    for i in range(n):
        freq[A[i] % K]+= 1
         
    # If both pairs are divisible by 'K'
    sum = freq[0] * (freq[0] - 1) / 2;
     
    # count for all i and (k-i)
    # freq pairs
    i = 1
    while(i <= K//2 and i != (K - i) ):
        sum += freq[i] * freq[K-i]
        i+= 1
 
    # If K is even
    if( K % 2 == 0 ):
        sum += (freq[K//2] * (freq[K//2]-1)/2);
     
    return int(sum)


'''Letter Combinations of a Phone Number
Medium

7090

562

Add to List

Share
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.



 

Example 1:

Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
Example 2:

Input: digits = ""
Output: []
Example 3:

Input: digits = "2"
Output: ["a","b","c"]
 

Constraints:

0 <= digits.length <= 4
digits[i] is a digit in the range ['2', '9'].'''

class Solution:
    # @param {string} digits
    # @return {string[]}
    def letterCombinations(self, digits):
        mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', 
                   '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        if len(digits) == 0:
            return []
        if len(digits) == 1:
            return list(mapping[digits[0]])
        prev = self.letterCombinations(digits[:-1])
        additional = mapping[digits[-1]]
        return [s + c for s in prev for c in additional]

#OR

class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        self.dfs(nums, [], res)
        return res
    
    def dfs(self, nums, path, res):
        res.append(path)
        for i in range(len(nums)):
            self.dfs(nums[i+1:], path + [nums[i]], res) 


#https://leetcode.com/problems/letter-combinations-of-a-phone-number/discuss/780232/Backtracking-Python-problems%2B-solutions-interview-prep

'''Find the minimum number of moves needed to move from one cell of matrix to another
Difficulty Level : Easy
Last Updated : 30 Mar, 2021
Given a N X N matrix (M) filled with 1 , 0 , 2 , 3 . Find the minimum numbers of moves needed to move from source to destination (sink) . while traversing through blank cells only. You can traverse up, down, right and left. 
A value of cell 1 means Source. 
A value of cell 2 means Destination. 
A value of cell 3 means Blank cell. 
A value of cell 0 means Blank Wall. 

Note : there is only single source and single destination.they may be more than one path from source to destination(sink).each move in matrix we consider as ‘1’ 

Examples: 

Input : M[3][3] = {{ 0 , 3 , 2 },
                   { 3 , 3 , 0 },
                   { 1 , 3 , 0 }};
Output : 4 

Input : M[4][4] = {{ 3 , 3 , 1 , 0 },
                   { 3 , 0 , 3 , 3 },
                   { 2 , 3 , 0 , 3 },
                   { 0 , 3 , 3 , 3 }};
Output : 4'''

class Graph:
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]
 
    # add edge to graph
    def addEdge (self, s , d ):
        self.adj[s].append(d)
        self.adj[d].append(s)
     
    # Level BFS function to find minimum
    # path from source to sink
    def BFS(self, s, d):
         
        # Base case
        if (s == d):
            return 0
     
        # make initial distance of all
        # vertex -1 from source
        level = [-1] * self.V
     
        # Create a queue for BFS
        queue = []
     
        # Mark the source node level[s] = '0'
        level[s] = 0
        queue.append(s)
     
        # it will be used to get all adjacent
        # vertices of a vertex
     
        while (len(queue) != 0):
             
            # Dequeue a vertex from queue
            s = queue.pop()
     
            # Get all adjacent vertices of the
            # dequeued vertex s. If a adjacent has
            # not been visited ( level[i] < '0') ,
            # then update level[i] == parent_level[s] + 1
            # and enqueue it
            i = 0
            while i < len(self.adj[s]):
                 
                # Else, continue to do BFS
                if (level[self.adj[s][i]] < 0 or
                    level[self.adj[s][i]] > level[s] + 1 ):
                    level[self.adj[s][i]] = level[s] + 1
                    queue.append(self.adj[s][i])
                i += 1
     
        # return minimum moves from source
        # to sink
        return level[d]
 
def isSafe(i, j, M):
    global N
    if ((i < 0 or i >= N) or
        (j < 0 or j >= N ) or M[i][j] == 0):
        return False
    return True
 
# Returns minimum numbers of moves from a
# source (a cell with value 1) to a destination
# (a cell with value 2)
def MinimumPath(M):
    global N
    s , d = None, None # source and destination
    V = N * N + 2
    g = Graph(V)
 
    # create graph with n*n node
    # each cell consider as node
    k = 1 # Number of current vertex
    for i in range(N):
        for j in range(N):
            if (M[i][j] != 0):
                 
                # connect all 4 adjacent cell to
                # current cell
                if (isSafe (i , j + 1 , M)):
                    g.addEdge (k , k + 1)
                if (isSafe (i , j - 1 , M)):
                    g.addEdge (k , k - 1)
                if (j < N - 1 and isSafe (i + 1 , j , M)):
                    g.addEdge (k , k + N)
                if (i > 0 and isSafe (i - 1 , j , M)):
                    g.addEdge (k , k - N)
 
            # source index
            if(M[i][j] == 1):
                s = k
 
            # destination index
            if (M[i][j] == 2):
                d = k
            k += 1
 
    # find minimum moves
    return g.BFS (s, d)


#OR

visited = {}
adj = [[] for i in range(16)]
 
# Performing the DFS for the minimum moves
def add_edges(u, v):
     
    global adj
    adj[u].append(v)
 
def DFS(s, d):
     
    global visited
 
    # Base condition for the recursion
    if (s == d):
        return 0
 
    # Initializing the result
    res = 10**9
    visited[s] = 1
     
    for item in adj[s]:
        if (item not in visited):
             
            # Comparing the res with
            # the result of DFS
            # to get the minimum moves
            res = min(res, 1 + DFS(item, d))
 
    return res
 
# Ruling out the cases where the element
# to be inserted is outside the matrix
def is_safe(arr, i, j):
     
    if ((i < 0 or i >= 4) or
        (j < 0 or j >= 4) or arr[i][j] == 0):
        return False
         
    return True
 
def min_moves(arr):
 
    s, d, V = -1,-1, 16
    # k be the variable which represents the
    # positions( 0 - 4*4 ) inside the graph.
     
    # k moves from top-left to bottom-right
    k = 0
    for i in range(4):
        for j in range(4):
             
            # Adding the edge
            if (arr[i][j] != 0):
                if (is_safe(arr, i, j + 1)):
                    add_edges(k, k + 1) # left
                if (is_safe(arr, i, j - 1)):
                    add_edges(k, k - 1) # right
                if (is_safe(arr, i + 1, j)):
                    add_edges(k, k + 4) # bottom
                if (is_safe(arr, i - 1, j)):
                    add_edges(k, k - 4) # top
 
            # Source from which DFS to be
            # performed
            if (arr[i][j] == 1):
                s = k
                 
            # Destination
            elif (arr[i][j] == 2):
                d = k
                 
            # Moving k from top-left
            # to bottom-right
            k += 1
 
    # DFS performed from
    # source to destination
    return DFS(s, d)
 
# Driver code
if __name__ == '__main__':
     
    arr = [ [ 3, 3, 1, 0 ],
            [ 3, 0, 3, 3 ],
            [ 2, 3, 0, 3 ],
            [ 0, 3, 3, 3 ] ]
 
    # If(min_moves(arr) == MAX) there
    # doesn't exist a path
    # from source to destination
    print(min_moves(arr))
 

'''4
'''

'''Kth Largest Element in a Stream
Easy

1479

939

Add to List

Share
Design a class to find the kth largest element in a stream. Note that it is the kth largest element in the sorted order, not the kth distinct element.

Implement KthLargest class:

KthLargest(int k, int[] nums) Initializes the object with the integer k and the stream of integers nums.
int add(int val) Appends the integer val to the stream and returns the element representing the kth largest element in the stream.
 

Example 1:

Input
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
Output
[null, 4, 5, 5, 8, 8]

Explanation
KthLargest kthLargest = new KthLargest(3, [4, 5, 8, 2]);
kthLargest.add(3);   // return 4
kthLargest.add(5);   // return 5
kthLargest.add(10);  // return 5
kthLargest.add(9);   // return 8
kthLargest.add(4);   // return 8
 '''

class KthLargest(object):

    
    def __init__(self, k, nums):
        self.pool = nums
        self.k = k
        heapq.heapify(self.pool)
        while len(self.pool) > k:
            heapq.heappop(self.pool)

            
    def add(self, val):
        if len(self.pool) < self.k:
            heapq.heappush(self.pool, val)
        elif val > self.pool[0]:
            heapq.heapreplace(self.pool, val)
        return self.pool[0]

#OR

import heapq
class KthLargest(object):

    def __init__(self, k, nums):
        """
        :type k: int
        :type nums: List[int]
        """
        self.heap = nums
        heapq.heapify(self.heap)
        self.k = k

    def add(self, val):
        """
        :type val: int
        :rtype: int
        """
        heapq.heappush(self.heap,val)
        # if heap grows bigger then k remove elements
        while len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]

'''Count all possible paths from top left to bottom right of a mXn matrix
Difficulty Level : Easy
Last Updated : 25 Jul, 2021
The problem is to count all the possible paths from top left to bottom right of a mXn matrix with the constraints that from each cell you can either move only to right or down
Examples : 
 

Input :  m = 2, n = 2;
Output : 2
There are two paths
(0, 0) -> (0, 1) -> (1, 1)
(0, 0) -> (1, 0) -> (1, 1)

Input :  m = 2, n = 3;
Output : 3
There are three paths
(0, 0) -> (0, 1) -> (0, 2) -> (1, 2)
(0, 0) -> (0, 1) -> (1, 1) -> (1, 2)
(0, 0) -> (1, 0) -> (1, 1) -> (1, 2)
 '''

def numberOfPaths(p, q):
     
    # Create a 1D array to store
    # results of subproblems
    dp = [1 for i in range(q)]
    for i in range(p - 1):
        for j in range(1, q):
            dp[j] += dp[j - 1]
    return dp[q - 1]
 
# Driver Code
print(numberOfPaths(3, 3))


def numberOfPaths(m, n) :
 
    # We have to calculate m + n-2 C n-1 here
    # which will be (m + n-2)! / (n-1)! (m-1)! path = 1;
    for i in range(n, (m + n - 1)):
        path *= i;
        path //= (i - n + 1);
     
    return path;
 
# Driver code
print(numberOfPaths(3, 3));

'''
Related Articles
Sum of all the child nodes with even grandparents in a Binary Tree
Print all nodes at distance k from a given node
Print all nodes that are at distance k from a leaf node
Print the longest leaf to leaf path in a Binary tree
Print path from root to a given node in a binary tree
Print root to leaf paths without using recursion
Print the nodes at odd levels of a tree
Print all full nodes in a Binary Tree
Print nodes between two given level numbers of a binary tree
Print nodes at k distance from root
Print Ancestors of a given node in Binary Tree
Check if a binary tree is subtree of another binary tree | Set 1
Check if a binary tree is subtree of another binary tree | Set 2
Check if a Binary Tree (not BST) has duplicate values
Check if a Binary Tree contains duplicate subtrees of size 2 or more
Serialize and Deserialize a Binary Tree
Construct BST from given preorder traversal | Set 2
Construct BST from given preorder traversal | Set 1
A program to check if a binary tree is BST or not
N Queen Problem | Backtracking-3
Printing all solutions in N-Queen Problem
Warnsdorff’s algorithm for Knight’s tour problem
The Knight’s tour problem | Backtracking-1
Rat in a Maze | Backtracking-2
Count number of ways to reach destination in a Maze
Top 50 Array Coding Problems for Interviews
Recursion
Difference between BFS and DFS
A* Search Algorithm
How to write a Pseudo Code?

Sum of all the child nodes with even grandparents in a Binary Tree
Difficulty Level : Medium
Last Updated : 22 Jun, 2021
Given a Binary Tree, calculate the sum of nodes with even valued Grandparents.
Examples: 

Input: 
      22
    /    \
   3      8
  / \    / \
 4   8  1   9
             \
              2
Output: 24
Explanation 
The nodes 4, 8, 2, 1, 9
has even value grandparents. 
Hence sum = 4 + 8 + 1 + 9 + 2 = 24.

Input:
        1
      /   \
     2     3
    / \   / \
   4   5 6   7
  /
 8
Output: 8
Explanation 
Only 8 has 2 as a grandparent.'''

# Python3 implementation to find sum
# of all the child nodes with
# even grandparents in a Binary Tree
 
# A binary tree node has data and
# pointers to the right and left children
class TreeNode():
     
    def __init__(self, data):
         
        self.data = data
        self.left = None
        self.right = None
 
sum = 0
 
# Function to calculate the sum
def getSum(curr, p, gp):
     
    global sum
     
    # Base condition
    if (curr == None):
        return
  
    # Check if node has a grandparent
    # if it does check
    # if they are even valued
    if (gp != None and gp.data % 2 == 0):
        sum += curr.data
  
    # Recurse for left child
    getSum(curr.left, curr, p)
  
    # Recurse for right child
    getSum(curr.right, curr, p)
     
# Driver code
if __name__=="__main__":
     
    root = TreeNode(22)
  
    root.left = TreeNode(3)
    root.right = TreeNode(8)
  
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(8)
  
    root.right.left = TreeNode(1)
    root.right.right = TreeNode(9)
    root.right.right.right = TreeNode(2)
  
    getSum(root, None, None)
     
    print(sum)

'''Output: 
24
 

Time Complexity: O(N)
Space Complexity: O(H)'''

'''Add all greater values to every node in a given BST
Difficulty Level : Medium
Last Updated : 21 Jul, 2021
Given a Binary Search Tree (BST), modify it so that all greater values in the given BST are added to every node. For example, consider the following BST.

              50
           /      \
         30        70
        /   \      /  \
      20    40    60   80 

The above tree should be modified to following 

              260
           /      \
         330        150
        /   \       /  \
      350   300    210   80'''

# Python3 program to add all greater values
# in every node of BST
 
# A utility function to create a
# new BST node
class newNode:
 
    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
 
# Recursive function to add all greater
# values in every node
def modifyBSTUtil(root, Sum):
     
    # Base Case
    if root == None:
        return
 
    # Recur for right subtree
    modifyBSTUtil(root.right, Sum)
 
    # Now Sum[0] has sum of nodes in right
    # subtree, add root.data to sum and
    # update root.data
    Sum[0] = Sum[0] + root.data
    root.data = Sum[0]
 
    # Recur for left subtree
    modifyBSTUtil(root.left, Sum)
 
# A wrapper over modifyBSTUtil()
def modifyBST(root):
    Sum = [0]
    modifyBSTUtil(root, Sum)
 
# A utility function to do inorder
# traversal of BST
def inorder(root):
    if root != None:
        inorder(root.left)
        print(root.data, end =" ")
        inorder(root.right)
 
# A utility function to insert a new node
# with given data in BST
def insert(node, data):
     
    # If the tree is empty, return a new node
    if node == None:
        return newNode(data)
 
    # Otherwise, recur down the tree
    if data <= node.data:
        node.left = insert(node.left, data)
    else:
        node.right = insert(node.right, data)
 
    # return the (unchanged) node pointer
    return node
 
# Driver Code
if __name__ == '__main__':
     
    # Let us create following BST
    # 50
    #     /     \
    # 30     70
    #     / \ / \
    # 20 40 60 80
    root = None
    root = insert(root, 50)
    insert(root, 30)
    insert(root, 20)
    insert(root, 40)
    insert(root, 70)
    insert(root, 60)
    insert(root, 80)
 
    modifyBST(root)
 
    # print inorder traversal of the
    # modified BST
    inorder(root)
     

#350 330 300 260 210 150 80


'''Sort linked list which is already sorted on absolute values
Difficulty Level : Medium
Last Updated : 13 May, 2021
Given a linked list that is sorted based on absolute values. Sort the list based on actual values.
Examples: 
 

Input :  1 -> -10 
output: -10 -> 1

Input : 1 -> -2 -> -3 -> 4 -> -5 
output: -5 -> -3 -> -2 -> 1 -> 4 

Input : -5 -> -10 
Output: -10 -> -5

Input : 5 -> 10 
output: 5 -> 10'''

# Python3 program to sort a linked list,
# already sorted by absolute values
     
# Linked list Node
class Node:
    def __init__(self, d):
        self.data = d
        self.next = None
 
class SortList:
    def __init__(self):
        self.head = None
         
    # To sort a linked list by actual values.
    # The list is assumed to be sorted by
    # absolute values.
    def sortedList(self, head):
         
        # Initialize previous and
        # current nodes
        prev = self.head
        curr = self.head.next
         
        # Traverse list
        while(curr != None):
             
            # If curr is smaller than prev,
            # then it must be moved to head
            if(curr.data < prev.data):
                 
                # Detach curr from linked list
                prev.next = curr.next
                 
                # Move current node to beginning
                curr.next = self.head
                self.head = curr
                 
                # Update current
                curr = prev
             
            # Nothing to do if current element
            # is at right place
            else:
                prev = curr
         
            # Move current
            curr = curr.next
        return self.head
     
    # Inserts a new Node at front of the list
    def push(self, new_data):
         
        # 1 & 2: Allocate the Node &
        #        Put in the data
        new_node = Node(new_data)
     
        # 3. Make next of new Node as head
        new_node.next = self.head
     
        # 4. Move the head to point to new Node
        self.head = new_node
     
    # Function to print linked list
    def printList(self, head):
        temp = head
        while (temp != None):
            print(temp.data, end = " ")
            temp = temp.next
        print()
     
# Driver Code
llist = SortList()
 
# Constructed Linked List is 
# 1->2->3->4->5->6->7->8->8->9->null
llist.push(-5)
llist.push(5)
llist.push(4)
llist.push(3)
llist.push(-2)
llist.push(1)
llist.push(0)
         
print("Original List :")
llist.printList(llist.head)
         
start = llist.sortedList(llist.head)
 
print("Sorted list :")
llist.printList(start)

'''Output: 

Original list :
0 -> 1 -> -2 -> 3 -> 4 -> 5 -> -5

Sorted list :
-5 -> -2 -> 0 -> 1 -> 3 -> 4 -> 5'''

