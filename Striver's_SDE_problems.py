


'''1 -- > (Sort Colors)
Medium


Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, 
with the colors in the order red, white, and blue.

We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.

 

Example 1:

Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
Example 2:

Input: nums = [2,0,1]
Output: [0,1,2]
Example 3:

Input: nums = [0]
Output: [0]
Example 4:

Input: nums = [1]
Output: [1]
 

Constraints:

n == nums.length
1 <= n <= 300
nums[i] is 0, 1, or 2.
 

Follow up: Could you come up with a one-pass algorithm using only constant extra space?
'''

'''This is a dutch partitioning problem. We are classifying the array into four groups: red, white, 
unclassified, and blue. Initially we group all elements into unclassified. 
We iterate from the beginning as long as the white pointer is less than the blue pointer.

If the white pointer is red (nums[white] == 0), we swap with the red pointer and move both white and red pointer forward. 
If the pointer is white (nums[white] == 1), the element is already in correct place, so we don't have to swap, 
just move the white pointer forward. If the white pointer is blue, we swap with the latest unclassified element.'''

def sortColors(self, nums):
    red, white, blue = 0, 0, len(nums)-1
    
    while white <= blue:
        if nums[white] == 0:
            nums[red], nums[white] = nums[white], nums[red]
            white += 1
            red += 1
        elif nums[white] == 1:
            white += 1
        else:
            nums[white], nums[blue] = nums[blue], nums[white]
            blue -= 1



'''Your input
    [2,0,2,1,1,0]
    Output
    [0,0,1,1,2,2]
    Expected
    [0,0,1,1,2,2]'''                                           
                                                
                                                        #*********************************#


'''2 --> Repeat and Missing Number 

method 1
Let x be the missing and y be the repeating element.
Let N is the size of array.
Get the sum of all numbers using formula S = N(N+1)/2
Get the sum of square of all numbers using formula Sum_Sq = N(N+1)(2N+1)/6
Iterate through a loop from i=1….N
S -= A[i]
Sum_Sq -= (A[i]*A[i])
It will give two equations 
x-y = S – (1) 
x^2 – y^2 = Sum_sq 
x+ y = (Sum_sq/S) – (2) 
 
Time Complexity: O(n) '''

def repeatedNumber(A):
     
    length = max(A)
    Sum_N = (length * (length + 1)) // 2
    Sum_NSq = ((length * (length + 1) *
                     (2 * length + 1)) // 6)
     
    missingNumber, repeating = 0, 0
     
    for i in range(length):
        Sum_N -= A[i]
        Sum_NSq -= A[i] * A[i]
         
    missingNumber = (Sum_N + Sum_NSq //
                             Sum_N) // 2
    repeating = missingNumber - Sum_N

    print('Repeating --> ', repeating)
    print('Missing --> ', missingNumber)
     
    ans = []
    ans.append(repeating)
    ans.append(missingNumber)
     
    return ans
 
# Driver code
'''v = [ 1,1,2,3,5]
res = repeatedNumber(v)
 
for i in res:
    print(i, end = " ")
'''
'''Repeating -->  1
Missing -->  4
1 4'''

'''method 2:
This method involves creating a Hashtable with the help of Map. In this, the elements are mapped to their natural index. 
In this process, if an element is mapped twice, then it is the repeating element. 
And if an element’s mapping is not there, then it is the missing element.'''
 
def repeatednum(A):
     maxi = max(A)
     H = {}
     for i in A:
        if i not in H:
             H[i] = True
        else:
            print('repeating --> ', i)
     for i in range(1, maxi+1):
         if i not in H:
             print('missing --> ', i)

#repeatednum([2,3,5,6,2])

'''repeating -->  2
missing -->  1
missing -->  4'''

                                                 #*********************************#


'''3 -- >  Merge two sorted array in O(1) space '''
# https://www.geeksforgeeks.org/efficiently-merging-two-sorted-arrays-with-o1-extra-space/

def nxtGap(gap):
    if gap <= 1:
        return 0
    return (gap//2) + (gap % 2)

def SortArr(a1, a2, m, n):
    gap = m+n
    gap = nxtGap(gap)

    while gap > 0:
        i = 0

        while i+gap < m:
            if a1[i+gap] < a1[i]:
                a1[i+gap], a1[i] = a1[i], a1[i+gap]
            i+=1

        j = gap - m if gap > m else 0
        while i < m and j < n:
            if a1[i] > a2[j]:
                a1[i], a2[j] = a2[j], a1[i]
            
            i+=1
            j+=1

        if j < n:
            j = 0

            while j + gap < n:
                if a2[j+gap] < a2[j]:
                    a2[j+gap] , a2[j] = a2[j], a2[j+gap]
                j+=1

        gap = nxtGap(gap)

    print(a1, ' ', a2)




#SortArr([8,7,6,1,2], [3,4,5], 5, 3)
            
#[1, 2, 3, 4, 5]   [6, 7, 8]


'''Merge Sorted Array

You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.

 

Example 1:

Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.
Example 2:

Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]
Explanation: The arrays we are merging are [1] and [].
The result of the merge is [1].
Example 3:

Input: nums1 = [0], m = 0, nums2 = [1], n = 1
Output: [1]
Explanation: The arrays we are merging are [] and [1].
The result of the merge is [1].
Note that because m = 0, there are no elements in nums1. The 0 is only there to ensure the merge result can fit in nums1.'''


class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        while n >0 and m > 0:
            if nums1[m-1] > nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        
            
        if n > 0:
            nums1[:n] = nums2[:n]
        
        return nums1
        
'''Your input
[1,2,3,0,0,0]
3
[2,5,6]
3
Output
[1,2,2,3,5,6]
Expected
[1,2,2,3,5,6]'''


''' Maximum Subarray

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

A subarray is a contiguous part of an array.

 

Example 1:

Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
Example 2:

Input: nums = [1]
Output: 1
Example 3:

Input: nums = [5,4,-1,7,8]
Output: 23
 

Constraints:

1 <= nums.length <= 3 * 104
-105 <= nums[i] <= 105
 

Follow up: If you have figured out the O(n) solution, 
try coding another solution using the divide and conquer approach, which is more subtle.'''

def maxArr(nums):
    s = 0
    maxi = float('-inf')
    for i in nums:
        s += i
        maxi = max(s, maxi)
        if s < 0:
            s = 0
    return maxi


# OR

def maxSubArray(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    for i in range(1, len(nums)):
        if nums[i-1] > 0:
            nums[i] += nums[i-1]
    return max(nums)
     

'''Your input
[-2,1,-3,4,-1,2,1,-5,4]
Output
6
Expected
6'''

def merge(intervals):
    """
    :type intervals: List[List[int]]
    :rtype: List[List[int]]
    """
    res = []
    arr = sorted(intervals)
    start = arr[0][0]
    end = arr[0][1]
    
    for it in arr:
        if it[0] <= end:
            end = max(end, it[1])
            
        else:
            res.append([start, end])
            start = it[0]
            end = it[1]
            
    res.append([start, end])
    return res

'''print(merge([[1,3],[2,6],[8,10],[15,18]]))'''

'''Your input
[[1,3],[2,6],[8,10],[15,18]]
Output
[[1,6],[8,10],[15,18]]
Expected
[[1,6],[8,10],[15,18]]'''

#OR

def mergeIntervals(arr):
         
        # Sorting based on the increasing order
        # of the start intervals
        arr.sort(key = lambda x: x[0])
         
        # array to hold the merged intervals
        m = []
        s = -10000
        max = -100000
        for i in range(len(arr)):
            a = arr[i]
            if a[0] > max:
                if i != 0:
                    m.append([s,max])
                max = a[1]
                s = a[0]
            else:
                if a[1] >= max:
                    max = a[1]
         
        #'max' value gives the last point of
        # that particular interval
        # 's' gives the starting point of that interval
        # 'm' array contains the list of all merged intervals
 
        if max != -100000 and [s, max] not in m:
            m.append([s, max])
        print("The Merged Intervals are :", end = " ")
        for i in range(len(m)):
            print(m[i], end = " ")
 
# Driver code
'''arr = [[6, 8], [1, 9], [2, 4], [4, 7]]
mergeIntervals(arr)'''

#OR

def merge(self, intervals):
    out = []
    for i in sorted(intervals, key=lambda x: x[0]):
        if out and i[0] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], i[1])
        else:
            out += i,
    return out



'''Find the Duplicate Number

Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

There is only one repeated number in nums, return this repeated number.

You must solve the problem without modifying the array nums and uses only constant extra space.

 

Example 1:

Input: nums = [1,3,4,2,2]
Output: 2
Example 2:

Input: nums = [3,1,3,4,2]
Output: 3
Example 3:

Input: nums = [1,1]
Output: 1
Example 4:

Input: nums = [1,1,2]
Output: 1
 

Constraints:

1 <= n <= 105
nums.length == n + 1
1 <= nums[i] <= n
All the integers in nums appear only once except for precisely one integer which appears two or more times.
 

Follow up:

How can we prove that at least one duplicate number must exist in nums?
Can you solve the problem in linear runtime complexity?'''

'''The main idea is the same with problem Linked List Cycle II,https://leetcode.com/problems/linked-list-cycle-ii/. 
Use two pointers the fast and the slow. The fast one goes forward two steps each time, while the slow one goes only step each time. 
They must meet the same item when slow==fast. In fact, they meet in a circle, the duplicate number must be the entry point of the circle when visiting the array from nums[0]. 
Next we just need to find the entry point. We use a point(we can use the fast one before) to visit form begining with one step each time, 
do the same job to slow. When fast==slow, they meet at the entry point of the circle. The easy understood code is as follows.'''

class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) > 1:
            slow = nums[0]
            fast = nums[nums[0]]

            while slow != fast:
                slow = nums[slow]
                fast = nums[nums[fast]]

            fast = 0
            while slow != fast:
                slow = nums[slow]
                fast = nums[fast]

            return slow
        
        return -1


'''Your input
[1,3,4,2,2]
Output
2
Expected
2'''

'''Set Matrix Zeroes

Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's, and return the matrix.

You must do it in place.

Example 1:


Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]
Example 2:


Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
 

Constraints:

m == matrix.length
n == matrix[0].length
1 <= m, n <= 200
-231 <= matrix[i][j] <= 231 - 1
 

Follow up:

A straightforward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?'''

class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        m, n, first = len(matrix), len(matrix[0]), not all(matrix[0])
        # Use first row/column as marker, scan the matrix
        for i in range(1, m):
            for j in range(n):
                if matrix[i][j] == 0:
                    matrix[0][j] = matrix[i][0] = 0
        # Set the zeros
        for i in range(1, m):
            for j in range(n - 1, -1, -1):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        # Set the zeros for the first row
        if first:
            matrix[0] = [0] * n
            
        
        return matrix
                
'''Your input
[[1,1,1],[1,0,1],[1,1,1]]
Output
[[1,0,1],[0,0,0],[1,0,1]]
Expected
[[1,0,1],[0,0,0],[1,0,1]]'''

def setZeroes(self, matrix):
    """
    Do not return anything, modify matrix in-place instead.
    """
    r = len(matrix)
    c = len(matrix[0])
    iscol = False
    for i in range(r):
        if matrix[i][0] == 0:
            iscol = True
        for j in range(1,c):
            if matrix[i][j] == 0:
                matrix[0][j] = 0
                matrix[i][0] = 0
    
    for i in range(1,r):
        for j in range(1,c):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    if matrix[0][0] == 0:
        for j in range(c):
            matrix[0][j] = 0
    if iscol: #for checking the first column needs to be turned or not
        for i in range(r):
            matrix[i][0] = 0


'''118. Pascal's Triangle

Given an integer numRows, return the first numRows of Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:
Example 1:

Input: numRows = 5
Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
Example 2:

Input: numRows = 1
Output: [[1]]
 

Constraints:

1 <= numRows <= 30'''

class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows == 1:
            return[[1]]
        elif numRows == 0:
            return []
        
        tri = [[1]]
        for i in range(1, numRows):
            row = [1]
            for j in range(1,i):
                row.append(tri[i-1][j-1] + tri[i-1][j])
            row.append(1)
            tri.append(row)
        return tri
        
#OR

"""
Given a non-negative integer numRows, generate the first numRows of Pascal's triangle.
In Pascal's triangle, each number is the sum of the two numbers directly above it.
Input: 5
Output:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
"""
#My method(by adding every two element of the last array)
def generate(self, n: int):
    
    if n == 0:
        return []
    ans = [[1]]
    for i in range(n-1):
        temp = ans[i]
        temp = temp
        new = [1]#because every new layer has 2 ones at the starting and end
        for i in range(len(temp)-1):
            temp1 = int(temp[i]) + int(temp[i+1])
            new.append(temp1)
        new.append(1) #because every new layer has 2 ones
        ans.append(new)
    return ans
#TC = O(n)
#SC = o(1) 
"""
If the question was given like this that given row and column number , give me the integer at that position in pascal triangle/
Then simply apply the NCR method
like if row = 4 and column = 3
you can compute like - (4*3*2) / (3*2*1)
the number of numbers on the numerator and denominator should remain same i.e if there are 3 numbers on the numerator there
will be 3 number in denominator.
TC= o(n) and SC = o(1)
"""

# Next permuatation

'''31. Next Permutation

Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such an arrangement is not possible, it must rearrange it as the lowest possible order (i.e., sorted in ascending order).

The replacement must be in place and use only constant extra memory.

 

Example 1:

Input: nums = [1,2,3]
Output: [1,3,2]
Example 2:

Input: nums = [3,2,1]
Output: [1,2,3]
Example 3:

Input: nums = [1,1,5]
Output: [1,5,1]
Example 4:

Input: nums = [1]
Output: [1]
 

Constraints:

1 <= nums.length <= 100
0 <= nums[i] <= 100'''

def nxtPermuatation(arr, n):
    if arr == None or n < 1:
        return
    if n == 1:
        print(arr)
        return
    i = j = n - 1
    while i > 0 and arr[i-1] >= arr[i]:
        i -= 1
    if i == 0:
        arr.reverse()
        return arr
    k = i-1
    while arr[j] <= arr[k]:
        j -= 1
    arr[j], arr[k] = arr[k], arr[j]
    l,r = k+1, n-1
    while l < r:
        arr[l], arr[r] = arr[r], arr[l]
        l += 1
        r -= 1
    return arr
    
    

'''p = nxtPermuatation([1,3,5,4,0], 5)
for i in p:
    print(i , end = ' ')'''

#1 4 0 3 5 

# Use two-pointers: two pointers start from back
# first pointer j stop at descending point
# second pointer i stop at value > nums[j]
# swap and sort rest


#count inversion


def countInversion(arr, n):
    temp = [0] * n
    return mergesort(arr, temp , 0, n-1)

def mergesort(arr, temp, left, right):
    inv_count = 0

    if left < right:
        
        mid = (left + right) // 2

        inv_count += mergesort(arr, temp, left, mid)

        inv_count += mergesort(arr, temp, mid+1, right)

        inv_count += merge(arr, temp, left, mid, right)

    return inv_count


def merge(arr, temp_arr, left, mid, right):
    i = left     # Starting index of left subarray
    j = mid + 1 # Starting index of right subarray
    k = left     # Starting index of to be sorted subarray
    inv_count = 0
 
    # Conditions are checked to make sure that
    # i and j don't exceed their
    # subarray limits.
 
    while i <= mid and j <= right:
 
        # There will be no inversion if arr[i] <= arr[j]
 
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            k += 1
            i += 1
        else:
            # Inversion will occur.
            temp_arr[k] = arr[j]
            inv_count += (mid-i + 1)
            k += 1
            j += 1
 
    # Copy the remaining elements of left
    # subarray into temporary array
    while i <= mid:
        temp_arr[k] = arr[i]
        k += 1
        i += 1
 
    # Copy the remaining elements of right
    # subarray into temporary array
    while j <= right:
        temp_arr[k] = arr[j]
        k += 1
        j += 1
 
    # Copy the sorted subarray into Original array
    for loop_var in range(left, right + 1):
        arr[loop_var] = temp_arr[loop_var]
         
    return inv_count



R = countInversion([5,4,3,2,1], 5)
print('count_inversion = ', R)



'''Best Time to Buy and Sell Stock

You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.
 

Constraints:

1 <= prices.length <= 105
0 <= prices[i] <= 104'''


'''The logic to solve this problem is same as "max subarray problem" using Kadane's Algorithm. 
Since no body has mentioned this so far, I thought it's a good thing for everybody to know.

All the straight forward solution should work, but if the interviewer twists the question slightly by giving the difference array of prices, 
Ex: for {1, 7, 4, 11}, if he gives {0, 6, -3, 7}, you might end up being confused.

Here, the logic is to calculate the difference (maxCur += prices[i] - prices[i-1]) of the original array, 
and find a contiguous subarray giving maximum profit. If the difference falls below 0, reset it to zero.'''

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        min_price = 0
        max_profit = 0
        for i in range(1,len(prices)):
            min_price += (prices[i] - prices[i-1])
            min_price = max(0, min_price )
            max_profit = max(max_profit, min_price)
        return max_profit
            

#or

def maxProfit(price):
    min_p = float('inf')
    max_p = 0
    for i in range(len(price)):
        min_p = min(min_p, price[i])
        max_p = max(max_p, price[i] - min_p)
    return max_p

'''Your input
[7,1,5,3,6,4]
Output
5
Expected
5'''

'''Rotate Image

You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

 

Example 1:


Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
Example 2:


Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
Example 3:

Input: matrix = [[1]]
Output: [[1]]
Example 4:

Input: matrix = [[1,2],[3,4]]
Output: [[3,1],[4,2]]
 

Constraints:

matrix.length == n
matrix[i].length == n
1 <= n <= 20
-1000 <= matrix[i][j] <= 1000'''
        
    
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n):
            for j in range(i, n):
                temp = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = temp
                
        for i in range(n):
            matrix[i] = matrix[i][::-1]
        
        return matrix

#OR

# reverse
def rotate(self, matrix):
    l = 0
    r = len(matrix) -1
    while l < r:
        matrix[l], matrix[r] = matrix[r], matrix[l]
        l += 1
        r -= 1
    # transpose 
    for i in range(len(matrix)):
        for j in range(i):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


'''Search a 2D Matrix

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

class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        n = len(matrix)
        m = len(matrix[0])
        
        i = 0 
        j = m-1
        
        while i < n and j >= 0:
            if matrix[i][j] == target:
                print('found at ', i, ' , ',j)
                return True
            
            if matrix[i][j] > target:
                j -= 1
                
            else:
                i += 1
        return False

#or

def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        n = len(matrix)
        m = len(matrix[0])
        
        i = 0 
        j = m*n-1
        
        while i != j:
            mid = (i + j-1) 
            if matrix[mid/m][mid%m] < target:
                i = mid + 1
            else:
                j = mid
        return matrix[j/m][j%m] == target
    

'''Pow(x, n)

Implement pow(x, n), which calculates x raised to the power n (i.e., xn).

Example 1:

Input: x = 2.00000, n = 10
Output: 1024.00000
Example 2:

Input: x = 2.10000, n = 3
Output: 9.26100
Example 3:

Input: x = 2.00000, n = -2
Output: 0.25000
Explanation: 2-2 = 1/22 = 1/4 = 0.25
 

Constraints:

-100.0 < x < 100.0
-231 <= n <= 231-1
-104 <= xn <= 104'''

def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        elif n < 0:
            return self.myPow(1/x, -n)
        elif n % 2 == 0:
            temp = self.myPow(x, n/2)
            return temp *temp
        else:
            return x * self.myPow(x, n-1)

#OR

def myPow(self, x, n):
        if n < 0:
            x = 1 / x
            n = -n
        pow = 1
        while n:
            if n & 1:
                pow *= x
            x *= x
            n >>= 1
        return pow


'''Majority Element

Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

Example 1:

Input: nums = [3,2,3]
Output: 3
Example 2:

Input: nums = [2,2,1,1,1,2,2]
Output: 2
 
Constraints:

n == nums.length
1 <= n <= 5 * 104
-231 <= nums[i] <= 231 - 1
 
Follow-up: Could you solve the problem in linear time and in O(1) space?'''

def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''We gonna use moore voting algorithm'''
        
        cnt = 0
        candidate = 0
        
        for it in nums:
            if cnt == 0:
                candidate = it
            if candidate == it:
                cnt += 1
            else:
                cnt -= 1
        return candidate

'''Your input
[3,2,3]
Output
3
Expected
3'''

#OR

class Solution(object):
    def majorityElement1(self, nums):
        nums.sort()
        return nums[len(nums)//2]
    
    def majorityElement2(self, nums):
        m = {}
        for n in nums:
            m[n] = m.get(n, 0) + 1
            if m[n] > len(nums)//2:
                return n
            
    def majorityElement(self, nums):
        candidate, count = nums[0], 0
        for num in nums:
            if num == candidate:
                count += 1
            elif count == 0:
                candidate, count = num, 1
            else:
                count -= 1
        return candidate

'''Majority Element II

Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.

Follow-up: Could you solve the problem in linear time and in O(1) space?

Example 1:

Input: nums = [3,2,3]
Output: [3]
Example 2:

Input: nums = [1]
Output: [1]
Example 3:

Input: nums = [1,2]
Output: [1,2]
 

Constraints:

1 <= nums.length <= 5 * 104
-109 <= nums[i] <= 109'''

def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        num1 = -1
        num2 = -1
        cnt1 = 0
        cnt2 = 0
        for it in nums:
            if it == num1:
                cnt1 += 1
            elif it == num2:
                cnt2 += 1
            elif cnt1 == 0:
                num1 = it
                cnt1 = 1
            elif cnt2 == 0:
                num2 = it
                cnt2 = 1
            else:
                cnt1 -=1
                cnt2 -= 1
        ans = []
        c1 = 0
        c2 = 0
        for it in nums:
            if it == num1:
                c1+=1
            elif it == num2:
                c2+=1
        
        n = len(nums)
        if c1 > n/3:
            ans.append(num1)
        if c2 > n/3:
            ans.append(num2)
            
        return ans

#OR

'''if element is found in our map of candidates, we +1 its value (count)

if there is room to add a new candidate (candidates number < k ), we add it

if our space is full and no room to add any new candidate we -1 all candidates

we want to remove the candidates that have reached zero and keep only the candidates who are >= 1, therefore we create a temp_dict and then we make the old dict (candidates) = temp_dict

At the end in linear time we go through the original array to count the number of times our suggest candidates appeared and check if they are > n//3
'''
class Solution:
    def majorityElement(self, nums):
        candidates = {}
        k = 3
        for num in nums:
            if num in candidates:
                candidates[num] += 1
            elif len(candidates) < k:
                candidates[num] = 1
            else:
                temp={}
                for c in candidates:
                    candidates[c]-=1
                    if candidates[c] >= 1:
                        temp[c] = candidates[c]
                candidates = temp
        out = [k for k in candidates if nums.count(k) > len(nums) // 3]          
        return out


'''Your input
[3,2,3]
Output
[3]
Expected
[3]'''

'''Unique Paths

A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

Example 1:
Input: m = 3, n = 7
Output: 28
Example 2:

Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down
Example 3:

Input: m = 7, n = 3
Output: 28
Example 4:

Input: m = 3, n = 3
Output: 6
 

Constraints:

1 <= m, n <= 100
It's guaranteed that the answer will be less than or equal to 2 * 109.'''

def uniquepath(n, m):
    N = m+n-2
    r = m-1
    ans = 1
    for i in range(1, m):
        ans = ans * (N-r + i)//i
        
    return ans

#print(uniquepath(7,3))

'''Your input
3
7
Output
28
Expected
28'''

#OR

# math C(m+n-2,n-1)
def uniquePaths1(self, m, n, i, j, HP):
    if i == m-1 and j == n-1:
        return 1
    if i >= n or j >= m:
        return 0
    if HP[i][j] != -1:
        return HP[i][j]
    
    else:
        return self.uniquePaths1(m, n, i+1, j, HP) + self.uniquePaths1(m, n, i, j+1, HP)

 
# O(m*n) space   
def uniquePaths2(self, m, n):
    if not m or not n:
        return 0
    dp = [[1 for _ in range(n)] for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[-1][-1]

# O(n) space 
def uniquePaths(self, m, n):
    if not m or not n:
        return 0
    cur = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            cur[j] += cur[j-1]
    return cur[-1]


#reverse pair

'''Reverse Pairs
Hard

Given an integer array nums, return the number of reverse pairs in the array.

A reverse pair is a pair (i, j) where 0 <= i < j < nums.length and nums[i] > 2 * nums[j].

 

Example 1:

Input: nums = [1,3,2,3,1]
Output: 2
Example 2:

Input: nums = [2,4,3,5,1]
Output: 3
 

Constraints:

1 <= nums.length <= 5 * 104
-231 <= nums[i] <= 231 - 1'''

class Solution(object):
    def merge(self, arr, l, mid, r):
        temp = [None] * (r-l+1)
        left = l
        right = mid+1
        inv = 0
        while left <= mid and right <= r:
            if arr[left] > (2 * arr[right]):
                inv += (mid - left + 1)
                right += 1
            else:
                left += 1
                
        left = l
        right = mid+1
        for i in range((r-l+1)):
            if mid < left:
                temp[i] = arr[right]
                right += 1
            elif right > r:
                temp[i] = arr[left]
                left += 1
                
            elif arr[left] <= arr[right]:
                temp[i] = arr[left]
                left += 1
            else:
                temp[i] = arr[right]
                right += 1
                
        for i in range((r-l+1)):
            arr[l+i] = temp[i]
            
        return inv
    
    def mergesort(self, arr, l, r):
        if l >= r:
            return 0
        cnt = 0
        mid = (l + r) // 2
        
        cnt += self.mergesort(arr,l, mid)
        
        cnt += self.mergesort(arr, mid+1, r)
        
        cnt += self.merge(arr, l, mid, r)
        
        return cnt
        
        
    
    def reversePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return self.mergesort(nums, 0, len(nums)-1)

'''Your input
[1,3,2,3,1]
Output
2
Expected
2'''

#OR

class Solution:
    def reversePairs(self, nums):
        cnt = 0

        def merge(left, right):
            nonlocal cnt              #work with variables inside nested function
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] <= 2*right[j]:               #this i can not make reverse pair
                    i += 1                              #so move the i pointer
                else:
                    cnt += len(left)-i                 #reverse pair exist so increase count
                    j += 1                             #(i,j) exist so we increase the j pointer

            return sorted(left+right)                  #returns sorted reverse pair


        def mergeSort(A):
            if len(A) == 1:                             #if there is only one element in the array
                return A
            mid=(len(A)) // 2
            return merge(mergeSort(A[:mid]), mergeSort(A[mid:]))      #check the left sub array and then the right sub array respectively

        mergeSort(nums)
        return cnt


'''Two Sum
Easy

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].
Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity?'''

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        result = []
        HP = {}
        for i in range(len(nums)):
            m = target - nums[i]
            if m in HP:
                result.append(HP[m])
                result.append(i)
                
                
            HP[nums[i]] = i
            
        return result
                
        
'''Your input
[2,7,11,15]
9
Output
[0,1]
Expected
[0,1]'''


#or

#https://leetcode.com/problems/two-sum/discuss/737092/Sum-MegaPost-Python3-Solution-with-a-detailed-explanation

def twoSum(self, nums, target):
        # Brute force
        
        for index, num in enumerate(nums):
            for other_index, other_num in enumerate(nums):
                if num + other_num == target and index != other_index:
                    return [index, other_index]
                
        # Two pass hash table
        
        hash_table = {n: i for i, n in enumerate(nums)}
        # Creates a hash table with each number and their indexes
        for i, n in enumerate(nums):
            complement = target - n # Gets the complement of the number
            if complement in hash_table.keys() and hash_table[complement] != i:
                # Check if the complement is in the hash table
                return [i, hash_table[complement]]
                # Returns the index and the index of the complement
                
                
        # One pass hash table
        
        hash_table = {}
        # Creates the hash table
        for i, n in enumerate(nums):
            complement = target - n
            # Gets the complement
            if complement in hash_table.keys():
                # Check if the complement exists, then returns
                return [i, hash_table[complement]]
            hash_table[n] = i
            # Otherwise it adds the number and index to the hash table

'''4Sum
Medium

Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:

0 <= a, b, c, d < n
a, b, c, and d are distinct.
nums[a] + nums[b] + nums[c] + nums[d] == target
You may return the answer in any order.

Example 1:

Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
Example 2:

Input: nums = [2,2,2,2,2], target = 8
Output: [[2,2,2,2]]

Constraints:

1 <= nums.length <= 200
-109 <= nums[i] <= 109
-109 <= target <= 109'''

"""
Given an array nums of n integers and an integer target, are there elements a, b, c, and d in nums such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.
Notice that the solution set must not contain duplicate quadruplets.
 
Example 1:
Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
"""

def fourSum(self, nums, target):
    ans = []
    nums.sort()
    for i in range(len(nums)-3):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        for j in range(i+1,len(nums)-2):
            if j > 0 and nums[j] == nums[j-1] and j-1 != i:
                continue
            l = j +1
            r = len(nums)-1
            while l < r:
                temp = nums[i]+nums[j]+nums[l]+nums[r]
                if temp < target:
                    l += 1
                elif temp > target:
                    r -= 1
                else:
                    ans.append([nums[i],nums[j],nums[l],nums[r]])
                    while l < r and nums[l] == nums[l+1]:
                        l += 1
                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                    l += 1
                    r-= 1
    return ans
"""
TC = o(n^3 + n log n)
sc = o(1)
"""

#OR


def fourSum(self, nums, target):
    def findNsum(l, r, target, N, result, results):
        if r-l+1 < N or N < 2 or target < nums[l]*N or target > nums[r]*N:  # early termination
            return
        if N == 2: # two pointers solve sorted 2-sum problem
            while l < r:
                s = nums[l] + nums[r]
                if s == target:
                    results.append(result + [nums[l], nums[r]])
                    l += 1
                    while l < r and nums[l] == nums[l-1]:
                        l += 1
                elif s < target:
                    l += 1
                else:
                    r -= 1
        else: # recursively reduce N
            for i in range(l, r+1):
                if i == l or (i > l and nums[i-1] != nums[i]):
                    findNsum(i+1, r, target-nums[i], N-1, result+[nums[i]], results)

    nums.sort()
    results = []
    findNsum(0, len(nums)-1, target, 4, [], results)
    return results

'''Longest Consecutive Sequence
Medium

Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.

Example 1:

Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
Example 2:

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
 
Constraints:

0 <= nums.length <= 105
-109 <= nums[i] <= 109'''

class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        H = {}
        best = 0
        for i in range(len(nums)):
            if nums[i] is not H:
                H[nums[i]] = 1
            else:
                H[nums[i]] += 1
            
        for it in nums:
            if it - 1 not in H:
                y = it + 1
                currS = 1
                while y in H:
                    y += 1
                    currS += 1
                best = max(best, currS)
        return best

'''Your input
[100,4,200,1,3,2]
Output
4
Expected
4'''
#OR

def longestConsecutive(self, num):
    num=set(num)
    maxLen=0
    while num:
        n=num.pop()
        i=n+1
        l1=0
        l2=0
        while i in num:
            num.remove(i)
            i+=1
            l1+=1
        i=n-1
        while i in num:
            num.remove(i)
            i-=1
            l2+=1
        maxLen=max(maxLen,l1+l2+1)
    return maxLen

'''Largest subarray with 0 sum 
Easy Accuracy: 46.94% Submissions: 63609 Points: 2
Given an array having both positive and negative integers. The task is to compute the length of the largest subarray with sum 0.

Example 1:

Input:
N = 8
A[] = {15,-2,2,-8,1,7,10,23}
Output: 5
Explanation: The largest subarray with
sum 0 will be -2 2 -8 1 7.
Your Task:
You just have to complete the function maxLen() which takes two arguments an array A and n, where n is the size of the array A and returns the length of the largest subarray with 0 sum.

Expected Time Complexity: O(N).
Expected Auxiliary Space: O(N).'''

def maxLen(n, arr):
    #Code here
    Hashmap = {}
    
    curr = 0
    maxi = 0
    
    for i in range(n):
        curr+=arr[i]
        
        if arr[i] == 0 and maxi == 0:
            maxi = 1
            
        if curr is 0:
            maxi = i + 1
            
        if  curr in Hashmap:
            maxi = max(maxi, i - Hashmap[curr])
            
        else:
            Hashmap[curr] = i
            
    return maxi

'''Count the number of subarrays having a given XOR
Difficulty Level : Hard
Last Updated : 13 May, 2021
Given an array of integers arr[] and a number m, count the number of subarrays having XOR of their elements as m.
Examples: 

Input : arr[] = {4, 2, 2, 6, 4}, m = 6
Output : 4
Explanation : The subarrays having XOR of 
              their elements as 6 are {4, 2}, 
              {4, 2, 2, 6, 4}, {2, 2, 6},
               and {6}

Input : arr[] = {5, 6, 7, 8, 9}, m = 5
Output : 2
Explanation : The subarrays having XOR of
              their elements as 5 are {5}
              and {5, 6, 7, 8, 9}'''

def cntSubarrays(arr, m):
    #Code here
    n = len(arr)
    res = 0
    for i in range(n):
        curr = 0
        for j in range(i, n):
            curr ^= arr[j]
            if curr == m:
                res += 1
    return res

print(cntSubarrays([4, 2, 2, 6, 4], 6))

#or
# 
def cntS(arr, m, n):
    ans = 0 # Initialize answer to be returned
 
    # Create a prefix xor-sum array such that
    # xorArr[i] has value equal to XOR
    # of all elements in arr[0 ..... i]
    xorArr =[0 for _ in range(n)]
 
    # Create map that stores number of prefix array
    # elements corresponding to a XOR value
    mp = dict()
 
    # Initialize first element
    # of prefix array
    xorArr[0] = arr[0]
 
    # Computing the prefix array.
    for i in range(1, n):
        xorArr[i] = xorArr[i - 1] ^ arr[i]
 
    # Calculate the answer
    for i in range(n):
         
        # Find XOR of current prefix with m.
        tmp = m ^ xorArr[i]
 
        # If above XOR exists in map, then there
        # is another previous prefix with same
        # XOR, i.e., there is a subarray ending
        # at i with XOR equal to m.
        if tmp in mp.keys():
            ans = ans + (mp[tmp])
 
        # If this subarray has XOR
        # equal to m itself.
        if (xorArr[i] == m):
            ans += 1
 
        # Add the XOR of this subarray to the map
        mp[xorArr[i]] = mp.get(xorArr[i], 0) + 1
 
    # Return total count of subarrays having
    # XOR of elements as given value m
    return ans
    
        
print(cntS([4, 2, 2, 6, 4], 6, 5))

'''Longest Substring Without Repeating Characters
Medium

Given a string s, find the length of the longest substring without repeating characters.

Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
Example 4:

Input: s = ""
Output: 0

Constraints:

0 <= s.length <= 5 * 104
s consists of English letters, digits, symbols and spaces.'''


'''the basic idea is, keep a hashmap which stores the characters in string as keys and their positions as values, and keep two pointers which define the max substring.
 move the right pointer to scan through the string , and meanwhile update the hashmap. If the character is already in the hashmap,
 then move the left pointer to the right of the same character last found. Note that the two pointers can only move forward.'''
def LongestSubstring(s, n):
    left = 0
    right = 0
    leng = 0
    H = {}
    while right < n:
        if s[right] in H:
            left = max(H[s[right]] + 1, left)

        H[s[right]] = right
        leng = max(leng, right - left + 1)
        right += 1

    return leng

print(LongestSubstring("abcabcbb", 6))

#OR


Grapes42's avatar
Grapes42
-10
February 18, 2020 9:13 AM

1.0K VIEWS

Python sliding window.

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        # create a window with left and right
        # keep a hash of elements already added
        left = 0
        right = 0
        seen = {} # keeps track of seen elements

        max_count = 0

        for i, character in enumerate(s):
            if character not in seen or seen[character] < left: # notice this logic
                seen[character] = i
                right = i
                if (right - left + 1) > max_count: # update the largest count
                    max_count = (right - left + 1)
            else:
                # we move start to one element after we found the character
                left = seen[character] + 1
                seen[character] = i # update index of last time we saw the character

        return max_count

