


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

#Python sliding window.

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


'''Reverse Linked List
Easy

Given the head of a singly linked list, reverse the list, and return the reversed list.

Example 1:

Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]
Example 2:

Input: head = [1,2]
Output: [2,1]
Example 3:

Input: head = []
Output: []


Constraints:

The number of nodes in the list is the range [0, 5000].
-5000 <= Node.val <= 5000
 

Follow up: A linked list can be reversed either iteratively or recursively. Could you implement both?'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        new = None
        while head != None:
            next = head.next
            head.next = new
            new = head
            head = next
            
        return new
        
#OR

class Solution:
# @param {ListNode} head
# @return {ListNode}
    def reverseList(self, head):
        prev = None
        while head:
            curr = head
            head = head.next
            curr.next = prev
            prev = curr
        return prev

#Recursion

class Solution:
# @param {ListNode} head
# @return {ListNode}
    def reverseList(self, head):
        return self._reverse(head)

    def _reverse(self, node, prev=None):
        if not node:
            return prev
        n = node.next
        node.next = prev
        return self._reverse(n, node)

'''Middle of the Linked List
Easy

Given the head of a singly linked list, return the middle node of the linked list.

If there are two middle nodes, return the second middle node.

Example 1:

Input: head = [1,2,3,4,5]
Output: [3,4,5]
Explanation: The middle node of the list is node 3.
Example 2:

Input: head = [1,2,3,4,5,6]
Output: [4,5,6]
Explanation: Since the list has two middle nodes with values 3 and 4, we return the second one.
 
Constraints:

The number of nodes in the list is in the range [1, 100].
1 <= Node.val <= 100'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        s = head 
        f = head
        while f != None and f.next != None:
            s = s.next
            f = f.next.next
        return s
        
#OR

def recusive(self, head):
        out = None
        def rec(head, curr):
            if head:
                total, node = rec(head.next, curr + 1)
                return total, node if total//2 != curr else head
            return curr, None
           
        return rec(head, 0)[1]


'''Merge Two Sorted Lists
Easy

Merge two sorted linked lists and return it as a sorted list. The list should be made by splicing together the nodes of the first two lists.

Example 1:

Input: l1 = [1,2,4], l2 = [1,3,4]
Output: [1,1,2,3,4,4]
Example 2:

Input: l1 = [], l2 = []
Output: []
Example 3:

Input: l1 = [], l2 = [0]
Output: [0]
 
Constraints:

The number of nodes in both lists is in the range [0, 50].
-100 <= Node.val <= 100
Both l1 and l2 are sorted in non-decreasing order.'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1 == None:
            return l2
        if l2 == None:
            return l1
        
        if l1.val > l2.val:
            temp = l1
            l1 = l2
            l2 = temp
            
        res = l1
        while l1 != None and l2 != None:
            tmp = None
            while l1 != None and l1.val <= l2.val:
                tmp = l1
                l1 = l1.next
                
            tmp.next = l2
            
            temp = l1
            l1 = l2
            l2 = temp
            
        return res
        
#OR

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1 == None:
            return l2
        if l2 == None:
            return l1
        
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
        
'''Remove Nth Node From End of List
Medium

Given the head of a linked list, remove the nth node from the end of the list and return its head.

Example 1:
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
Example 2:

Input: head = [1], n = 1
Output: []
Example 3:

Input: head = [1,2], n = 1
Output: [1]
 
Constraints:

The number of nodes in the list is sz.
1 <= sz <= 30
0 <= Node.val <= 100
1 <= n <= sz
 
Follow up: Could you do this in one pass?'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        curr = ListNode()
        curr.next = head
        fast = curr
        slow = curr
        i = 1
        while i <= n:
            fast = fast.next
            i+=1
            
        while fast.next != None:
            fast = fast.next
            slow = slow.next
            
        slow.next = slow.next.next
        
        return curr.next
        
#OR

'''My first solution is "cheating" a little. Instead of really removing the nth node,
 I remove the nth value. I recursively determine the indexes (counting from back), 
 then shift the values for all indexes larger than n, and then always drop the head.
'''
class Solution:
    def removeNthFromEnd(self, head, n):
        def index(node):
            if not node:
                return 0
            i = index(node.next) + 1
            if i > n:
                node.next.val = node.val
            return i
        index(head)
        return head.next
'''Index and Remove - AC in 56 ms

In this solution I recursively determine the indexes again, but this time my helper function removes the nth node. 
It returns two values. The index, as in my first solution, and the possibly changed head of the remaining list.
'''
class Solution:
    def removeNthFromEnd(self, head, n):
        def remove(head):
            if not head:
                return 0, head
            i, head.next = remove(head.next)
            return i+1, (head, head.next)[i+1 == n]
        return remove(head)[1]
'''n ahead - AC in 48 ms

The standard solution, but without a dummy extra node. Instead,
 I simply handle the special case of removing the head right after the fast cursor got its head start.
'''
class Solution:
    def removeNthFromEnd(self, head, n):
        fast = slow = head
        for _ in range(n):
            fast = fast.next
        if not fast:
            return head.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head

'''Delete Node in a Linked List
Easy

Write a function to delete a node in a singly-linked list. You will not be given access to the head of the list, instead you will be given access to the node to be deleted directly.

It is guaranteed that the node to be deleted is not a tail node in the list.

Example 1:

Input: head = [4,5,1,9], node = 5
Output: [4,1,9]
Explanation: You are given the second node with value 5, the linked list should become 4 -> 1 -> 9 after calling your function.
Example 2:

Input: head = [4,5,1,9], node = 1
Output: [4,5,9]
Explanation: You are given the third node with value 1, the linked list should become 4 -> 5 -> 9 after calling your function.
Example 3:

Input: head = [1,2,3,4], node = 3
Output: [1,2,4]
Example 4:

Input: head = [0,1], node = 0
Output: [1]
Example 5:

Input: head = [-3,5,-99], node = -3
Output: [5,-99]

Constraints:

The number of the nodes in the given list is in the range [2, 1000].
-1000 <= Node.val <= 1000
The value of each node in the list is unique.
The node to be deleted is in the list and is not a tail node'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next


'''Add Two Numbers
Medium

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
        
        dummy = ListNode(None)
        temp = dummy
        carry = 0
        while (l1 != None or l2 != None or carry == 1):
            sumi = 0
            
            if l1 != None:
                sumi += l1.val
                l1 = l1.next
            else:
                sumi += 0
                
                
            if l2 != None:
                sumi += l2.val
                l2 = l2.next
                
            else:
                sumi += 0
                
                
            sumi += carry
            carry = sumi / 10
        
            temp.next = ListNode(sumi%10)
            temp = temp.next
            
       
            
        return dummy.next
        
#or

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

#https://leetcode.com/problems/add-two-numbers/discuss/1102/Python-for-the-win


'''Intersection of Two Linked Lists
Easy

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
        
#or

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
#or

class Solution:
    # @param two ListNodes
    # @return the intersected ListNode
    def getIntersectionNode(self, headA, headB):
        curA,curB = headA,headB
        lenA,lenB = 0,0
        while curA is not None:
            lenA += 1
            curA = curA.next
        while curB is not None:
            lenB += 1
            curB = curB.next
        curA,curB = headA,headB
        if lenA > lenB:
            for i in range(lenA-lenB):
                curA = curA.next
        elif lenB > lenA:
            for i in range(lenB-lenA):
                curB = curB.next
        while curB != curA:
            curB = curB.next
            curA = curA.next
        return curA

'''Linked List Cycle
Easy

Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).
Example 2:


Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.
Example 3:


Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.
 
Constraints:

The number of the nodes in the list is in the range [0, 104].
-105 <= Node.val <= 105
pos is -1 or a valid index in the linked-list.
 
Follow up: Can you solve it using O(1) (i.e. constant) memory?'''
        
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head == None or head.next == None:
            return False
        slow = head
        fast = head
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
                
        return False

#OR

#The algorithm is of course Tortoise and hare.

def hasCycle(self, head):
    try:
        slow = head
        fast = head.next
        while slow is not fast:
            slow = slow.next
            fast = fast.next.next
        return True
    except:
        return False

#Dictionary/Hash table
class Solution:
    def hasCycle(self, head):
        dictionary = {}
        while head:
            if head in dictionary: 
                return True
            else: 
                dictionary[head]= True
            head = head.next
        return False

'''Reverse Nodes in k-Group
Hard

Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.

k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.

You may not alter the values in the list's nodes, only nodes themselves may be changed.

Example 1:
Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]
Example 2:


Input: head = [1,2,3,4,5], k = 3
Output: [3,2,1,4,5]
Example 3:

Input: head = [1,2,3,4,5], k = 1
Output: [1,2,3,4,5]
Example 4:

Input: head = [1], k = 1
Output: [1]
 
Constraints:

The number of nodes in the list is in the range sz.
1 <= sz <= 5000
0 <= Node.val <= 1000
1 <= k <= sz
 
Follow-up: Can you solve the problem in O(1) extra memory space?'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if k ==1 and head != None:
            return head
        
        dummy = ListNode(0, head)
        pre = dummy
        cur = dummy
        nex = dummy
        
        cnt =0
        while cur.next:
            cnt += 1
            cur = cur.next
            
        while cnt >= k:
            cur = pre.next
            nex = cur.next
            
            for i in range(1,k):
                cur.next = nex.next
                nex.next = pre.next
                pre.next = nex
                nex = cur.next
                
            pre = cur
            cnt -= k
            
        return dummy.next
        
#or

def reverseKGroup(self, head, k):
    dummy = jump = ListNode(0)
    dummy.next = l = r = head
    
    while True:
        count = 0
        while r and count < k:   # use r to locate the range
            r = r.next
            count += 1
        if count == k:  # if size k satisfied, reverse the inner linked list
            pre, cur = r, l
            for _ in range(k):
                cur.next, cur, pre = pre, cur.next, cur  # standard reversing
            jump.next, jump, l = pre, l, r  # connect two k-groups
        else:
            return dummy.next

def reverseKGroup(self, head, k):
    l, node = 0, head
    while node:
        l += 1
        node = node.next
    if k <= 1 or l < k:
        return head
    node, cur = None, head
    for _ in xrange(k):
        nxt = cur.next
        cur.next = node
        node = cur
        cur = nxt
    head.next = self.reverseKGroup(cur, k)
    return node

'''Palindrome Linked List
Easy

Given the head of a singly linked list, return true if it is a palindrome.

Example 1:

Input: head = [1,2,2,1]
Output: true
Example 2:

Input: head = [1,2]
Output: false
 
Constraints:

The number of nodes in the list is in the range [1, 105].
0 <= Node.val <= 9
 
Follow up: Could you do it in O(n) time and O(1) space?'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        s = slow
        prev = None
        while s:
            curr = s
            s = s.next
            curr.next = prev
            prev = curr
        dummy = head
        while prev:
            if dummy.val != prev.val:
                return False
            dummy = dummy.next
            prev = prev.next

        return True

#OR

'''Solution 1: Reversed first half == Second half?
Phase 1: Reverse the first half while finding the middle.
Phase 2: Compare the reversed first half with the second half.'''

def isPalindrome(self, head):
    rev = None
    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next
        rev, rev.next, slow = slow, rev, slow.next
    if fast:
        slow = slow.next
    while rev and rev.val == slow.val:
        slow = slow.next
        rev = rev.next
    return not rev

'''Solution 2: Play Nice
Same as the above, but while comparing the two halves, restore the list to its original state by reversing the first half back. 
Not that the OJ or anyone else cares.
'''
def isPalindrome(self, head):
    rev = None
    fast = head
    while fast and fast.next:
        fast = fast.next.next
        rev, rev.next, head = head, rev, head.next
    tail = head.next if fast else head
    isPali = True
    while rev:
        isPali = isPali and rev.val == tail.val
        head, head.next, rev = rev, head, rev.next
        tail = tail.next
    return isPali

'''Linked List Cycle II
Medium

Given a linked list, return the node where the cycle begins. If there is no cycle, return null.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Notice that you should not modify the linked list.

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: tail connects to node index 1
Explanation: There is a cycle in the linked list, where tail connects to the second node.
Example 2:

Input: head = [1,2], pos = 0
Output: tail connects to node index 0
Explanation: There is a cycle in the linked list, where tail connects to the first node.
Example 3:

Input: head = [1], pos = -1
Output: no cycle
Explanation: There is no cycle in the linked list.

Constraints:

The number of the nodes in the list is in the range [0, 104].
-105 <= Node.val <= 105
pos is -1 or a valid index in the linked-list.
 
Follow up: Can you solve it using O(1) (i.e. constant) memory?'''   

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None


''' Consider the following linked list, where E is the cylce entry and X, the crossing point of fast and slow.
        H: distance from head to cycle entry E
        D: distance from E to X
        L: cycle length
                          _____
                         /     \
        head_____H______E       \
                        \       /
                         X_____/   
        
    
        If fast and slow both start at head, when fast catches slow, slow has traveled H+D and fast 2(H+D). 
        Assume fast has traveled n loops in the cycle, we have:
        2H + 2D = H + D + L  -->  H + D = nL  --> H = nL - D
        Thus if two pointers start from head and X, respectively, one first reaches E, the other also reaches E. 
        In my solution, since fast starts at head.next, we need to move slow one step forward in the beginning of part 2'''

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None or head.next == None:
            return None
        
        slow = head
        fast =head
        entry = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                while slow != entry:
                    slow = slow.next
                    entry = entry.next
                return entry
        
        return None
            
'''Your input
[3,2,0,-4]
1
Output
tail connects to node index 1
Expected
tail connects to node index 1'''

#or

def detectCycle(self, head):
        s=set()
        while head:
            if head in s:
                return head
            s.add(head)
            head=head.next

'''             Flattening a Linked List 
Medium 

Given a Linked List of size N, where every node represents a sub-linked-list and contains two pointers:
(i) a next pointer to the next node,
(ii) a bottom pointer to a linked list where this node is head.
Each of the sub-linked-list is in sorted order.
Flatten the Link List such that all the nodes appear in a single level while maintaining the sorted order. 
Note: The flattened list will be printed using the bottom pointer instead of next pointer.

Example 1:

Input:
5 -> 10 -> 19 -> 28
|     |     |     | 
7     20    22   35
|           |     | 
8          50    40
|                 | 
30               45
Output:  5-> 7-> 8- > 10 -> 19-> 20->
22-> 28-> 30-> 35-> 40-> 45-> 50.
Explanation:
The resultant linked lists has every 
node in a single level.
(Note: | represents the bottom pointer.)
 
Example 2:

Input:
5 -> 10 -> 19 -> 28
|          |                
7          22   
|          |                 
8          50 
|                           
30              
Output: 5->7->8->10->19->20->22->30->50
Explanation:
The resultant linked lists has every
node in a single level.

(Note: | represents the bottom pointer.)
 
Your Task:
You do not need to read input or print anything. Complete the function flatten() that takes the head of the linked list as input parameter and returns the head of flattened link list.

Expected Time Complexity: O(N*M)
Expected Auxiliary Space: O(1)'''   

'''

class Node:
    def __init__(self, d):
        self.data=d
        self.next=None
        self.bottom=None
        
'''
def mergeTwoList(a, b):
    temp = Node(0)
    res = temp
    
    while a and b:
        if a.data <= b.data:
            temp.bottom = a
            temp = temp.bottom
            a = a.bottom
            
        else:
            temp.bottom = b
            temp = temp.bottom
            b = b.bottom
            
    if a != None:
        temp.bottom = a
    else:
        temp.bottom = b
        
    return res.bottom
    
        
def flatten(root):
    #Your code here
    if root == None or root.next == None:
        return root
        
    root.next = flatten(root.next)
    
    root = mergeTwoList(root, root.next)
    
    return root

'''For Input:
4 
4 2 3 4                  
5 7 8 30 10 20 19 22 50 28 35 40 45

Your Output is: 
5 7 8 10 19 20 22 28 30 35 40 45 50 '''

#OR

"""
Given a Linked List of size N, where every node represents a sub-linked-list and contains two pointers:
(i) a next pointer to the next node,
(ii) a bottom pointer to a linked list where this node is head.
Each of the sub-linked-list is in sorted order.
Flatten the Link List such that all the nodes appear in a single level while maintaining the sorted order. 
Note: The flattened list will be printed using the bottom pointer instead of next pointer.
In this we use merge sort on linked list approach.
"""
def merge(a,b):# to merge two linked list
    if a == None:
        return b
    if b == None:
        return a
    result = None
    if a.data < b.data:
        result = a
        result.bottom = merge(a.bottom,b)
    else:
        result = b
        result.bottom = merge(a,b.bottom)
    return result
def flatten(root):
    #Your code here
    if not root:
        return None
    return merge(root,flatten(root.next))#to merge all the linked lists

'''Rotate List
Medium

Given the head of a linked list, rotate the list to the right by k places.

Example 1:

Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]
Example 2:

Input: head = [0,1,2], k = 4
Output: [2,0,1]

Constraints:

The number of nodes in the list is in the range [0, 500].
-100 <= Node.val <= 100
0 <= k <= 2 * 109'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if head == None or head.next == None:
            return head
        
        cur = head
        l = 1
        
        while cur.next:
            cur = cur.next
            l += 1
            
        cur.next = head
        k = k % l
        k = l - k
        
        while k:
            cur = cur.next
            k-=1
            
        head = cur.next
        cur.next = None
        
        return head

'''Your input
[1,2,3,4,5]
2
Output
[4,5,1,2,3]
Expected
[4,5,1,2,3]'''

#OR

class Solution(object):
    def rotateRight(self, head, k):
        n, pre, current = 0, None, head
        while current:
            pre, current = current, current.next
            n += 1

        if not n or not k % n:
            return head

        tail = head
        for _ in range(n - k % n - 1):
            tail = tail.next

        next, tail.next, pre.next = tail.next, None, head
        return next

'''Copy List with Random Pointer
Medium

A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.

Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.

For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.

Return the head of the copied linked list.

The linked list is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:

val: an integer representing Node.val
random_index: the index of the node (range from 0 to n-1) that the random pointer points to, or null if it does not point to any node.
Your code will only be given the head of the original linked list.

Example 1:
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]

Example 2:
Input: head = [[1,1],[2,1]]
Output: [[1,1],[2,1]]

Example 3:
Input: head = [[3,null],[3,0],[3,null]]
Output: [[3,null],[3,0],[3,null]]
Example 4:

Input: head = []
Output: []
Explanation: The given linked list is empty (null pointer), so return null.
 
Constraints:
0 <= n <= 1000
-10000 <= Node.val <= 10000
Node.random is null or is pointing to some node in the linked list.'''

"""
# Definition for a Node.
class Node:
    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        # copy nodes
        itr = head
        front = head
        while itr:
            front = itr.next
            copy = Node(itr.val)
            itr.next = copy
            copy.next = front
            itr = front

        # copy random pointers
        itr = head
        while itr:
            if itr.random != None:
                itr.next.random = itr.random.next
                
            itr = itr.next.next

        # separate two parts   
        itr = head
        pseudo = Node(0)
        copy = pseudo
        while itr:
            front = itr.next.next
            copy.next = itr.next
            itr.next = front
            copy = copy.next
            itr = itr.next
            
        return pseudo.next
        
'''Your input
[[7,null],[13,0],[11,4],[10,2],[1,0]]
Output
[[7,null],[13,0],[11,4],[10,2],[1,0]]
Expected
[[7,null],[13,0],[11,4],[10,2],[1,0]]'''

#OR

 # using dictionary    
def copyRandomList(self, head):
    if not head:
        return 
    cur, dic = head, {}
    # copy nodes
    while cur:
        dic[cur] = RandomListNode(cur.label)
        cur = cur.next
    cur = head
    # copy random pointers
    while cur:
        if cur.random:
            dic[cur].random = dic[cur.random]
        if cur.next:
            dic[cur].next = dic[cur.next]
        cur = cur.next
    return dic[head]

#OR

class Solution(object):
    """
    """
    def copyRandomList(self, head):
        dic, prev, node = {}, None, head
        while node:
            if node not in dic:
                # Use a dictionary to map the original node to its copy
                dic[node] = Node(node.val, node.next, node.random)
            if prev:
                # Make the previous node point to the copy instead of the original.
                prev.next = dic[node]
            else:
                # If there is no prev, then we are at the head. Store it to return later.
                head = dic[node]
            if node.random:
                if node.random not in dic:
                    # If node.random points to a node that we have not yet encountered, store it in the dictionary.
                    dic[node.random] = Node(node.random.val, node.random.next, node.random.random)
                # Make the copy's random property point to the copy instead of the original.
                dic[node].random = dic[node.random]
            # Store prev and advance to the next node.
            prev, node = dic[node], node.next
        return head

'''Left View of Binary Tree 

Easy
Given a Binary Tree, print Left view of it. Left view of a Binary Tree is set of nodes visible when tree is visited from Left side. The task is to complete the function leftView(), which accepts root of the tree as argument.

Left view of following tree is 1 2 4 8.

          1
       /     \
     2        3
   /     \    /    \
  4     5   6    7
   \
     8   

Example 1:

Input:
   1
 /  \
3    2
Output: 1 3

Example 2:

Input:

Output: 10 20 40
Your Task:
You just have to complete the function leftView() that prints the left view. The newline is automatically appended by the driver code.
Expected Time Complexity: O(N).
Expected Auxiliary Space: O(Height of the Tree).

Constraints:
0 <= Number of nodes <= 100
1 <= Data of a node <= 1000'''

def LeftView(root):
    if root is None:
        return
    
    q = []
    q.append(root)
    res = []
    
    while len(q):
        
        n = len(q)
        
        for i in range(1, n+1):
            temp = q[0]
            q.pop(0)
            
            if i ==1:
                res.append(temp.data)
                
            if temp.left != None:
                q.append(temp.left)
    
            if temp.right != None:
                q.append(temp.right)
                
    return res

#OR

# Recursive function pritn left view of a binary tree
def leftViewUtil(root, level, max_level):
     
    # Base Case
    if root is None:
        return
 
    # If this is the first node of its level
    if (max_level[0] < level):
        print( "% d\t" %(root.data)),
        max_level[0] = level
 
    # Recur for left and right subtree
    leftViewUtil(root.left, level + 1, max_level)
    leftViewUtil(root.right, level + 1, max_level)
 
 
# A wrapper over leftViewUtil()
def leftView(root):
    max_level = [0]
    leftViewUtil(root, 1, max_level)

'''3Sum
Medium

Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Example 2:

Input: nums = []
Output: []
Example 3:

Input: nums = [0]
Output: []
 
Constraints:

0 <= nums.length <= 3000
-105 <= nums[i] <= 105'''

class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        nums = sorted(nums)
        n = len(nums)
        res = []
        
        for i in range(n-1):
            if i == 0 or ( i > 0 and nums[i] != nums[i-1]):
                low = i+1
                high = n-1
                sum = 0 - nums[i]
                
                while low <  high:
                    if nums[low] + nums[high] == sum:
                        res.append([nums[i], nums[low], nums[high]])
                        
                        while low < high and nums[low] == nums[low+1] :
                            low += 1
                        while low < high and nums[high] == nums[high-1]:
                            high -= 1
                            
                        low += 1
                        high -= 1
                        
                    elif nums[low] + nums[high] < sum:
                        low += 1
                        
                    else:
                        high -= 1
        
        return res

'''Your input
[-1,0,1,2,-1,-4]
Output
[[-1,-1,2],[-1,0,1]]
Expected
[[-1,-1,2],[-1,0,1]]'''

#or

def threeSum(self, nums):

        nums.sort()  # will make spoting of duplicates easy

        triplets = []
        length = len(nums)

        for i in range(length-2):  # ignore last two

            # check if element is a duplicate. the first cannot be a duplicate
            if i > 0 and nums[i] == nums[i-1]:
                # skip handling an element if it's similar to the one before it
                # because it is sorted, we effectively skip duplicates
                continue

            # TWO SUM for a sorted array
            # 1. find elements that will add up to 0
            # 2. check inner elements
            left = i + 1
            right = length - 1
            while left < right:

                # will be used to check if the sum is equal to 0
                total = nums[i] + nums[left] + nums[right]

                # if total is less than 0 we try to increase it's value
                if total < 0:
                    left += 1  # moving left to a lerger value

                # if total is more than 0 we try to decrease it's value
                elif total > 0:
                    right -= 1  # moving right to a smaller value

                # 1. add list of elements to triplets
                # 2. check inner elements
                else:
                    # add elements to triplets
                    triplets.append([nums[i], nums[left], nums[right]])

                    # check inner elements
                    # 1. skip similar elements
                    # 2. move to inner elements

                    # skip:
                    # no need to continue with an element with the same value as l/r
                    # Skip all similar to the current left and right so that,
                    # when we are moving to the next element, we dont move to an element with the same value
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1

                    # move to inner elements
                    left += 1
                    right -= 1

        return triplets

'''Trapping Rain Water
Hard

Given n non-negative integers representing an elevation map where the width of each bar is 1, 
compute how much water it can trap after raining.

Example 1:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case,
 6 units of rain water (blue section) are being trapped.
Example 2:

Input: height = [4,2,0,3,2,5]
Output: 9
 
Constraints:

n == height.length
1 <= n <= 2 * 104
0 <= height[i] <= 105'''

class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        pre = [0] * (n)
        pre[0] = height[0]
        suf = [0] * (n)
        suf[-1] = height[-1]
        
        for i in range(1,n):
            Q = max(pre[i-1] , height[i])
            pre[i] = Q
                
        for i in range(n-2, -1, -1):
            Q = max(height[i], suf[i+1])
            suf[i] = Q
                
        sum = 0
        for i in range(n):
            val = min(pre[i], suf[i]) - height[i]
            sum += val
            
        return sum
        
#or

class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        l = 0
        r = n-1
        sum = 0
        maxl = 0
        maxr = 0
        
        while l <= r:
            if height[l] <= height[r]:
                if height[l] > maxl:
                    maxl = height[l]
                else:
                    sum += (maxl - height[l])
                    
                l+=1
                
            else:
                if height[r] > maxr:
                    maxr = height[r]
                else:
                    sum += (maxr - height[r])
                    
                r-=1
        return sum

'''Remove Duplicates from Sorted Array
Easy

Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

Custom Judge:

The judge will test your solution with the following code:

int[] nums = [...]; // Input array
int[] expectedNums = [...]; // The expected answer with correct length

int k = removeDuplicates(nums); // Calls your implementation

assert k == expectedNums.length;
for (int i = 0; i < k; i++) {
    assert nums[i] == expectedNums[i];
}
If all assertions pass, then your solution will be accepted.

 

Example 1:

Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
Example 2:

Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
 

Constraints:

0 <= nums.length <= 3 * 104
-100 <= nums[i] <= 100
nums is sorted in non-decreasing order.'''

class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res =1
        n = len(nums)
        for i in range(1,n):
            if nums[i-1] != nums[i]:
                nums[res] = nums[i]
                res+=1
        return res
    
#OR

def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums[:] = sorted(set(nums))
        return len(nums)

'''Max Consecutive Ones
Easy

Given a binary array nums, return the maximum number of consecutive 1's in the array.

Example 1:

Input: nums = [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s. The maximum number of consecutive 1s is 3.
Example 2:

Input: nums = [1,0,1,1,0,1]
Output: 2
 
Constraints:

1 <= nums.length <= 105
nums[i] is either 0 or 1.'''

class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        maxi =0
        A = 0
        for i in nums:
            if i == 1:
                A += 1
                maxi = max(A, maxi)
                
            else:
                A = 0
                
        return maxi

#OR

class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        K = 0
        j = 0
        while i < len(nums):
            K -= 1 - nums[i]
            if K < 0:
                K += 1 - nums[j]
                j += 1
            i+=1
        return i - j

        
            
        
'''N meetings in one room 

Easy 
There is one meeting room in a firm. There are N meetings in the form of (start[i], end[i]) where start[i] is start time of meeting i and end[i] is finish time of meeting i.
What is the maximum number of meetings that can be accommodated in the meeting room when only one meeting can be held in the meeting room at a particular time?

Note: Start time of one chosen meeting can't be equal to the end time of the other chosen meeting.


Example 1:

Input:
N = 6
start[] = {1,3,0,5,8,5}
end[] =  {2,4,6,7,9,9}
Output: 
4
Explanation:
Maximum four meetings can be held with
given start and end timings.
The meetings are - (1, 2),(3, 4), (5,7) and (8,9)
Example 2:

Input:
N = 3
start[] = {10, 12, 20}
end[] = {20, 25, 30}
Output: 
1
Explanation:
Only one meetings can be held
with given start and end timings.

Your Task :
You don't need to read inputs or print anything. Complete the function maxMeetings() that takes two arrays start[] and end[] along with their size N as input parameters and returns the maximum number of meetings that can be held in the meeting room.


Expected Time Complexity : O(N*LogN)
Expected Auxilliary Space : O(N)


Constraints:
1 ≤ N ≤ 105
0 ≤ start[i] < end[i] ≤ 105'''

class Solution:
    
    #Function to find the maximum number of meetings that can
    #be performed in a meeting room.
    def maximumMeetings(self,n,start,end):
        # code here
        meet = [[None] *3] * n
        for i in range(n):
            meet[i] = [start[i], end[i], i+1]  
            
        meet.sort(key = lambda x : x[1])
        answer = []
        answer.append(meet[0][2])
        limit = meet[0][1]
        for i in range(0,n):
            if meet[i][0] > limit:
                limit = meet[i][1]
                answer.append(meet[i][2])
                
                
            
        return len(answer)


#OR

"""
There is one meeting room in a firm. There are N meetings in the form of (S[i], F[i]) 
where S[i] is start time of meeting i and F[i] is finish time of meeting i.
What is the maximum number of meetings that can be accommodated in the meeting room
when only one meeting can be held in the meeting room at a particular time? 
Also note start time of one chosen meeting can't be equal to the end time of the other chosen meeting.
"""
def maximumMeetings(n,start,end):
    # code here
    arr = list(zip(start,end))
    arr.sort(key = lambda x:x[1])
    cnt = 0
    prev = arr[0]
    for i in range(1,n):
        if prev[1] < arr[i][0]:
            prev = arr[i]
        else:
            cnt += 1
    return n - cnt
"""
TC = o(n*log(n))
SC = o(n)
"""

'''Minimum Platforms 
Medium 
Given arrival and departure times of all trains that reach a railway station. Find the minimum number of platforms required for the railway station so that no train is kept waiting.
Consider that all the trains arrive on the same day and leave on the same day. Arrival and departure time can never be the same for a train but we can have arrival time of one train equal to departure time of the other. At any given instance of time, same platform can not be used for both departure of a train and arrival of another train. In such cases, we need different platforms.


Example 1:

Input: n = 6 
arr[] = {0900, 0940, 0950, 1100, 1500, 1800}
dep[] = {0910, 1200, 1120, 1130, 1900, 2000}
Output: 3
Explanation: 
Minimum 3 platforms are required to 
safely arrive and depart all trains.
Example 2:

Input: n = 3
arr[] = {0900, 1100, 1235}
dep[] = {1000, 1200, 1240}
Output: 1
Explanation: Only 1 platform is required to 
safely manage the arrival and departure 
of all trains. 

Your Task:
You don't need to read input or print anything. Your task is to complete the function findPlatform() which takes the array arr[] (denoting the arrival times), array dep[] (denoting the departure times) and the size of the array as inputs and returns the minimum number of platforms required at the railway station such that no train waits.

Note: Time intervals are in the 24-hour format(HHMM) , where the first two characters represent hour (between 00 to 23 ) and the last two characters represent minutes (between 00 to 59).


Expected Time Complexity: O(nLogn)
Expected Auxiliary Space: O(n)


Constraints:
1 ≤ n ≤ 50000
0000 ≤ A[i] ≤ D[i] ≤ 2359'''

# similar problem : Minimum no of Railway Platforms problem

#https://www.geeksforgeeks.org/minimum-number-platforms-required-railwaybus-station/

# arrival time of the trains
arr = [900,940,950,1100,1500,1800]

#departure time of the trains
dep = [910,1200,1120,1130,1900,2000]

def minimumPlatform(self,n,arr,dep):
        # code here
        arr.sort()
        dep.sort()
        
        plat = result = 1
        
        i = 1
        j =0
        
        while i < n and j < n:
            if arr[i] <= dep[j]:
                plat += 1
                i+=1
                
            elif arr[i] > dep[j]:
                plat -= 1
                j += 1
                
            if plat > result:
                result = plat
                
        return result
'''For Input:
6
0900 0940 0950 1100 1500 1800
0910 1200 1120 1130 1900 2000

Your Output is: 
3'''

'''Job Sequencing Problem 
Medium Accuracy: 48.94% Submissions: 38045 Points: 4
Given a set of N jobs where each jobi has a deadline and profit associated with it. Each job takes 1 unit of time to complete and only one job can be scheduled at a time. We earn the profit if and only if the job is completed by its deadline. The task is to find the number of jobs done and the maximum profit.

Note: Jobs will be given in the form (Jobid, Deadline, Profit) associated with that Job.


Example 1:

Input:
N = 4
Jobs = {(1,4,20),(2,1,10),(3,1,40),(4,1,30)}
Output:
2 60
Explanation:
Job1 and Job3 can be done with
maximum profit of 60 (20+40).
Example 2:

Input:
N = 5
Jobs = {(1,2,100),(2,1,19),(3,2,27),
        (4,1,25),(5,1,15)}
Output:
2 127
Explanation:
2 jobs can be done with
maximum profit of 127 (100+27).

Your Task :
You don't need to read input or print anything. Your task is to complete the function JobScheduling() which takes an integer N and an array of Jobs(Job id, Deadline, Profit) as input and returns the count of jobs and maximum profit.


Expected Time Complexity: O(NlogN)
Expected Auxilliary Space: O(N)


Constraints:
1 <= N <= 105
1 <= Deadline <= 100
1 <= Profit <= 500'''

def JobScheduling(self,Jobs,n):
        
        # code here
        jobs = sorted(Jobs, key = lambda x : x.profit, reverse = True)
        
        maxi = 0
        for i in range(n):
            if jobs[i].deadline > maxi:
                maxi = jobs[i].deadline
                
        res = [-1] *(maxi+1)
        
        cnt = 0
        profit = 0
        
        for i in range(n):
            for j in range(jobs[i].deadline, 0, -1):
                if res[j] ==-1:
                    res[j] = i
                    cnt += 1
                    profit += jobs[i].profit
                    break
        
        ans = [cnt, profit]
        return ans

'''For Input:
4
1 4 20 2 1 10 3 1 40 4 1 30

Your Output is: 
2 60'''

# T = O(nlogn), O(n)
#OR

"""
Given a set of N jobs where each job i has a deadline and profit associated to it. Each job takes 1 unit of time to complete and only one job can be scheduled at a time. We earn the profit if and only if the job is completed by its deadline. The task is to find the maximum profit and the number of jobs done.
Input Format:
Jobs will be given in the form (Job id, Deadline, Profit) associated to that Job.
Example 1:
Input:
N = 4
Jobs = (1,4,20)(2,1,10)(3,1,40)(4,1,30)
Output: 2 60
Explanation: 2 jobs can be done with
maximum profit of 60 (20+40).
In this we sort the array and put all the the time slots in the hashmap and see if we can schedule a job there,
it is a greedy approach so we traverse from n to 0.
"""
def JobScheduling(Jobs,n):
    '''
    :param Jobs: list of "Job" class defined in driver code, with "profit" and "deadline".
    :param n: total number of jobs
    :return: A list of size 2 having list[0] = count of jobs and list[1] = max profit
    '''
    '''
    {
        class Job:.
        def __init__(self,profit=0,deadline=0):
            self.profit = profit
            self.deadline = deadline
            self.id = 0
    }
    '''
    # code here
    def solve(n,dic):
        for i in range(n,0,-1):
            if i not in dic:
                dic[i] = 1
                return True
        return False
        
    dead = {}
    ans = 0
    cnt = 0
    Jobs.sort(key = lambda x : x.profit,reverse = True)
    for i in Jobs:
        if i.deadline not in dead or solve(i.deadline,dead):
            ans += i.profit
            cnt += 1
            dead[i.deadline] = 1
        else:
            continue
    return [cnt,ans]
"""
TC = o(n^2)
SC = o(n)
"""

'''Fractional Knapsack 
Medium Accuracy: 45.14% Submissions: 37951 Points: 4
Given weights and values of N items, we need to put these items in a knapsack of capacity W to get the maximum total value in the knapsack.
Note: Unlike 0/1 knapsack, you are allowed to break the item. 

 

Example 1:

Input:
N = 3, W = 50
values[] = {60,100,120}
weight[] = {10,20,30}
Output:
240.00
Explanation:Total maximum value of item
we can have is 240.00 from the given
capacity of sack. 
Example 2:

Input:
N = 2, W = 50
values[] = {60,100}
weight[] = {10,20}
Output:
160.00
Explanation:
Total maximum value of item
we can have is 160.00 from the given
capacity of sack.
 

Your Task :
Complete the function fractionalKnapsack() that receives maximum capacity , array of structure/class and size n and returns a double value representing the maximum value in knapsack.
Note: The details of structure/class is defined in the comments above the given function.


Expected Time Complexity : O(NlogN)
Expected Auxilliary Space: O(1)


Constraints:
1 <= N <= 105
1 <= W <= 105'''

def fractionalknapsack(self, W,Items,n):
        
        # code here
        arr = sorted(Items, key = lambda x : x.value / x.weight, reverse = True)
        currW = 0
        finalval = 0.0
        
        for i in range(n):
            
            if currW + arr[i].weight <= W:
                currW += arr[i].weight
                finalval += arr[i].value
                
            else:
                remain = W - currW
                finalval += (arr[i].value/arr[i].weight) * remain
                break
                
        return finalval

#OR

"""
Given weights and values of N items,
 we need to put these items in a knapsack of capacity W to get the maximum total value in the knapsack.
Note: Unlike 0/1 knapsack, you are allowed to break the item. 
"""
def fractionalknapsack(w,arr,n):
    '''
    :param W: max weight which can be stored
    :param Items: List contianing Item class objects as defined in driver code, with value and weight
    :param n: size of Items
    :return: Float value of max possible value, two decimal place accuracy is expected by driver code
    
    {
        class Item:
        def __init__(self,val,w):
            self.value = val
            self.weight = w
    }
    '''
    # code zip(Items,W))
    arr.sort(key = lambda x:x.value/x.weight,reverse = True)
    ans = 0
    for i in arr:
        #print(i.value,i.weight,end = "")
        if i.weight < w:
            ans += i.value
            w -= i.weight
        else:
            ans += (w/i.weight) * i.value
            w = 0
            break
        #print(ans)
    #print()
    return ans
"""
TC = o(n*log n)
SC = o(1)
"""

'''Number of Coins 
Medium Accuracy: 47.78% Submissions: 23792 Points: 4
Given a value V and array coins[] of size M, the task is to make the change for V cents, given that you have an infinite supply of each of coins{coins1, coins2, ..., coinsm} valued coins. Find the minimum number of coins to make the change. If not possible to make change then return -1.


Example 1:

Input: V = 30, M = 3, coins[] = {25, 10, 5}
Output: 2
Explanation: Use one 25 cent coin
and one 5 cent coin
Example 2:
Input: V = 11, M = 4,coins[] = {9, 6, 5, 1} 
Output: 2 
Explanation: Use one 6 cent coin
and one 5 cent coin

Your Task:  
You don't need to read input or print anything. Complete the function minCoins() which takes V, M and array coins as input parameters and returns the answer.

Expected Time Complexity: O(V*M)
Expected Auxiliary Space: O(V)

Constraints:
1 ≤ V*M ≤ 106
All array elements are distinct'''

class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        if amount == 0:
            return 0
        
        dp = [amount + 1] *(amount +1)
        
        dp[0] = 0
        
        for i in range(amount+1):
            for j in range(len(coins)):
                if coins[j] <= i:
                    dp[i] = min(dp[i], 1+dp[i - coins[j]])
                    
        if dp[amount] > amount:
            return -1
        else:
            return dp[amount]
    
'''For Input:
30 3
25 10 5

Your Output is: 
2'''

#OR

def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        if amount == 0:
            return 0
        value1 = [0]
        value2 = []
        nc =  0
        visited = [False]*(amount+1)
        visited[0] = True
        while value1:
            nc += 1
            for v in value1:
                for coin in coins:
                    newval = v + coin
                    if newval == amount:
                        return nc
                    elif newval > amount:
                        continue
                    elif not visited[newval]:
                        visited[newval] = True
                        value2.append(newval)
            value1, value2 = value2, []
        return -1

#https://leetcode.com/problems/coin-change/discuss/?currentPage=1&orderBy=most_votes&query=&tag=python

'''Subset Sums 
Basic Accuracy: 63.02% Submissions: 8060 Points: 1
Given a list arr of N integers, print sums of all subsets in it. Output should be printed in increasing order of sums.

Example 1:

Input:
N = 2
arr[] = {2, 3}
Output:
0 2 3 5
Explanation:
When no elements is taken then Sum = 0.
When only 2 is taken then Sum = 2.
When only 3 is taken then Sum = 3.
When element 2 and 3 are taken then 
Sum = 2+3 = 5.
Example 2:

Input:
N = 3
arr = {5, 2, 1}
Output:
0 1 2 3 5 6 7 8
Your Task:  
You don't need to read input or print anything. Your task is to complete the function subsetSums() which takes a list/vector and an integer N as an input parameter and return the list/vector of all the subset sums in increasing order.

Expected Time Complexity: O(2N)
Expected Auxiliary Space: O(2N)

Constraints:
1 <= N <= 15
0 <= arr[i] <= 10000'''

def func(self,ind, sum, arr, n, subset):
        if ind == n:
            subset.append(sum)
            return
        
        self.func(ind+1, sum+arr[ind], arr, n, subset)
        
        self.func(ind+1, sum, arr, n, subset)
    
def subsetSums(self, arr, N):
    # code here
    subset = []
    self.func(0 , 0, arr, N, subset)
    subset.sort()
    return subset

'''For Input:
2
2 3

Your Output is: 
0 2 3 5 '''

'''Subsets II
Medium

Given an integer array nums that may contain duplicates, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

Example 1:

Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
Example 2:

Input: nums = [0]
Output: [[],[0]]
 
Constraints:

1 <= nums.length <= 10
-10 <= nums[i] <= 10'''

class Solution(object):
    def func(self,num, en, subset):
        subset.append(en)
        for i in range(0, len(num)):
            if i > 0 and num[i] == num[i-1]:
                continue
            self.func(num[i+1:], en+[num[i]], subset)
            
    
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        nums.sort()
        subset = []
        en = []
        self.func(nums, en, subset)
        return subset

'''Your input
[1,2,2]
Output
[[],[1],[1,2],[1,2,2],[2],[2,2]]
Expected
[[],[1],[1,2],[1,2,2],[2],[2,2]]'''

class Solution:
    # @param num, a list of integer
    # @return a list of lists of integer
    def subsetsWithDup(self, S):
        res = [[]]
        S.sort()
        for i in range(len(S)):
            if i == 0 or S[i] != S[i - 1]:
                l = len(res)
            for j in range(len(res) - l, len(res)):
                res.append(res[j] + [S[i]])
        return res


'''
Combination Sum
Medium

Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

Example 1:

Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
Example 2:

Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]
Example 3:

Input: candidates = [2], target = 1
Output: []
Example 4:

Input: candidates = [1], target = 1
Output: [[1]]
Example 5:

Input: candidates = [1], target = 2
Output: [[1,1]]

Constraints:
1 <= candidates.length <= 30
1 <= candidates[i] <= 200
All elements of candidates are distinct.
1 <= target <= 500'''

def func(self, ind, arr, tar, ds, ans):
    if ind == len(arr):
        if tar == 0:
            ans.append(ds)
        return
    
    if arr[ind] <= tar:
        self.func(ind, arr, tar - arr[ind], ds + [arr[ind]], ans)

    self.func(ind+1, arr, tar, ds, ans)

def CombinationSum(self, cands, tar):
    ans = []
    self.func(0, cands, tar, [], ans)
    return ans

'''Your input
[2,3,6,7]
7
Output
[[2,2,3],[7]]
Expected
[[2,2,3],[7]]'''

#OR

class Solution(object):
    def combinationSum(self, candidates, target):
        ret = []
        self.dfs(candidates, target, [], ret)
        return ret
    
    def dfs(self, nums, target, path, ret):
        if target < 0:
            return 
        if target == 0:
            ret.append(path)
            return 
        for i in range(len(nums)):
            self.dfs(nums[i:], target-nums[i], path+[nums[i]], ret)

#https://leetcode.com/problems/combination-sum/discuss/?currentPage=1&orderBy=most_votes&query=

'''Combination Sum II
Medium

Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

Each number in candidates may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.

Example 1:

Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
Example 2:

Input: candidates = [2,5,2,1,2], target = 5
Output: 
[
[1,2,2],
[5]
]

Constraints:
1 <= candidates.length <= 100
1 <= candidates[i] <= 50
1 <= target <= 30'''

def combinationSum2(self, candidates, target):
    # Sorting is really helpful, se we can avoid over counting easily
    candidates.sort()                      
    result = []
    self.combine_sum_2(candidates, 0, [], result, target)
    return result
    
def combine_sum_2(self, nums, start, path, result, target):
    # Base case: if the sum of the path satisfies the target, we will consider 
    # it as a solution, and stop there
    if not target:
        result.append(path)
        return
    
    for i in range(start, len(nums)):
        # Very important here! We don't use `i > 0` because we always want 
        # to count the first element in this recursive step even if it is the same 
        # as one before. To avoid overcounting, we just ignore the duplicates
        # after the first element.
        if i > start and nums[i] == nums[i - 1]:
            continue

        # If the current element is bigger than the assigned target, there is 
        # no need to keep searching, since all the numbers are positive
        if nums[i] > target:
            break

        # We change the start to `i + 1` because one element only could
        # be used once
        self.combine_sum_2(nums, i + 1, path + [nums[i]], 
                           result, target - nums[i])
        
#OR

def combinationSum2(self, candidates, target):
    candidates.sort()
    table = [None] + [set() for i in range(target)]
    for i in candidates:
        if i > target:
            break
        for j in range(target - i, 0, -1):
            table[i + j] |= {elt + (i,) for elt in table[j]}
        table[i].add((i,))
    return map(list, table[target])

#https://leetcode.com/problems/combination-sum-ii/discuss/16870/DP-solution-in-Python


'''Palindrome Partitioning
Medium

Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.
A palindrome string is a string that reads the same backward as forward.

Example 1:

Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]
Example 2:

Input: s = "a"
Output: [["a"]]
 
Constraints:

1 <= s.length <= 16
s contains only lowercase English letters.'''

class Solution(object):
    def partition(self, s):
        res = []
        self.dfs(s, [], res)
        return res

    def dfs(self, s, path, res):
        if not s:
            res.append(path)
            return
        for i in range(1, len(s)+1):
            if self.isPal(s[:i]):
                self.dfs(s[i:], path+[s[:i]], res)

    def isPal(self, s):
        return s == s[::-1]

'''Your input
"aab"
Output
[["a","a","b"],["aa","b"]]
Expected
[["a","a","b"],["aa","b"]]'''

#OR

class Solution(object):
    def partition(self, s):
        ret = []
        for i in range(1, len(s)+1):
            if s[:i] == s[i-1::-1]:
                for rest in self.partition(s[i:]):
                    ret.append([s[:i]]+rest)
        if not ret:
            return [[]]
        return ret

#https://leetcode.com/problems/palindrome-partitioning/discuss/?currentPage=1&orderBy=most_votes&query=&tag=python

'''Permutation Sequence
Hard

The set [1, 2, 3, ..., n] contains a total of n! unique permutations.

By listing and labeling all of the permutations in order, we get the following sequence for n = 3:

"123"
"132"
"213"
"231"
"312"
"321"
Given n and k, return the kth permutation sequence.

Example 1:

Input: n = 3, k = 3
Output: "213"
Example 2:

Input: n = 4, k = 9
Output: "2314"
Example 3:

Input: n = 3, k = 1
Output: "123"

Constraints:

1 <= n <= 9
1 <= k <= n!'''

class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        num = []
        fact = 1
        for i in range(1,n):
            fact *= i
            num.append(i)
            
        num.append(n)
        ans = ''
        k = k-1
        while True:
            ans += str(num[k/fact])
            num.pop(k/fact)
            if len(num) == 0:
                break
                
            k = k % fact
            fact = fact / len(num)
            
        return ans
        
'''Your input
3
3
Output
"213"
Expected
"213"'''

#OR

class Solution:
# @return a string
    def getPermutation(self, n, k):

        ll = [str(i) for i in range(1,n+1)] # build a list of ["1","2",..."n"]

        divisor = 1
        for i in range(1,n): # calculate 1*2*3*...*(n-1)
            divisor *= i

        answer = ""
        while k>0 and k<=divisor*n:  # there are only (divisor*n) solutions in total 
            group_num = k/divisor
            k %= divisor

            if k>0: # it's kth element of (group_num+1)th group
                choose = ll.pop(group_num)
                answer += choose
            else: # it's last element of (group_num)th group
                choose = ll.pop(group_num-1) 
                answer += choose
                ll.reverse() # reverse the list to get DESC order for the last element
                to_add = "".join(ll)
                answer += to_add
                break

            divisor/=len(ll)

        return answer

'''Permutations
Medium

7116

144

Add to List

Share
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

Example 1:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
Example 2:

Input: nums = [0,1]
Output: [[0,1],[1,0]]
Example 3:

Input: nums = [1]
Output: [[1]]
 

Constraints:

1 <= nums.length <= 6
-10 <= nums[i] <= 10
All the integers of nums are unique.'''

class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans = []
        self.func(0, nums, ans)
        return ans
    
    def func(self, ind, num, ans):
        n = len(num)
        if ind == n:
            ds = []
            for i in range(n):
                ds.append(num[i])
                
            ans.append(ds)
            return
        
        for i in range(ind, n):
            temp = num[i]
            num[i] = num[ind]
            num[ind] = temp
            
            self.func(ind+1, num, ans)
            
            temp = num[i]
            num[i] = num[ind]
            num[ind] = temp

'''Your input
[1,2,3]
Output
[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,2,1],[3,1,2]]
Expected
[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]'''

#OR

# DFS
def permute(self, nums):
    res = []
    self.dfs(nums, [], res)
    return res
    
def dfs(self, nums, path, res):
    if not nums:
        res.append(path)
        # return # backtracking
    for i in range(len(nums)):
        self.dfs(nums[:i]+nums[i+1:], path+[nums[i]], res)
        
#https://leetcode.com/problems/permutations/discuss/?currentPage=1&orderBy=most_votes&query=

'''N-Queens
Hard

3874

119

Add to List

Share
The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.

 

Example 1:


Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above
Example 2:

Input: n = 1
Output: [["Q"]]
 

Constraints:

1 <= n <= 9'''

class Solution(object):
    def func(self,col, board, ans, leftR, upperR, lowerR, n):
        if col == n:
            ans.append([''.join(row) for row in board])
            return

        for row in range(0,n):
            if leftR[row]== False and lowerR[row + col] == False and upperR[n-1 + col - row] == False:
                board[row] = [j for j in board[row]]
                board[row][col] = 'Q'
                leftR[row] = True
                upperR[n-1 + col - row] = True
                lowerR[row + col] = True
                self.func(col+1, board, ans, leftR, upperR, lowerR, n)

                leftR[row] = False
                upperR[n-1 + col - row] = False
                lowerR[row + col] = False
                board[row] = [j for j in board[row]]
                board[row][col] = '.'

    def solveNQueens(self,n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        ans = []
        board = [[0] *n]*n
        for i in range(n):
            for j in range(n):
                board[i][j] = '.'
        leftR = [False] * n
        upperR = [False] * (2*n)
        lowerR = [False] * (2*n)

        self.func(0, board, ans, leftR, upperR, lowerR, n)

        return ans

'''Your input
4
Output
[["..Q.","Q...","...Q",".Q.."],[".Q..","...Q","Q...","..Q."]]
Expected
[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]'''
                
#OR

def solveNQueens(self, n):
    res = []
    self.dfs([-1]*n, 0, [], res)
    return res
 
# nums is a one-dimension array, like [1, 3, 0, 2] means
# first queen is placed in column 1, second queen is placed
# in column 3, etc.
def dfs(self, nums, index, path, res):
    if index == len(nums):
        res.append(path)
        return  # backtracking
    for i in range(len(nums)):
        nums[index] = i
        if self.valid(nums, index):  # pruning
            tmp = "."*len(nums)
            self.dfs(nums, index+1, path+[tmp[:i]+"Q"+tmp[i+1:]], res)

# check whether nth queen can be placed in that column
def valid(self, nums, n):
    for i in range(n):
        if abs(nums[i]-nums[n]) == n -i or nums[i] == nums[n]:
            return False
    return True


#OR

class Solution:
    def __init__(self, ans = None):
        self.ans = []
        
    def Valid(self, queen: List[List[str]], n: int, row: int, col: int) -> bool:
        for i in range(n):
            if queen[i][col] == 'Q':
                return False
        for i in range(col,-1,-1):
            if queen[row][i] == 'Q':
                return False
        for (i,j) in zip(range(row,-1,-1),range(col,-1,-1)):
            if queen[i][j] == 'Q':
                return False
        for (i,j) in zip(range(row,n),range(col,-1,-1)):
            if queen[i][j] == 'Q':
                return False
        return True
    
    def SolveNQueens(self, queen: List[List[str]], n: int, col: int) -> None:
        if col >= n:
            self.ans.append(["".join(queen[i]) for i in range(n)])
            return None
        for row in range(n):
            if self.Valid(queen,n,row,col):
                queen[row][col] = 'Q'
                self.SolveNQueens(queen,n,col + 1)
                queen[row][col] = '.'
        return None
            
    def solveNQueens(self, n: int) -> List[List[str]]:
        queen = [['.' for i in range(n)] for i in range(n)]
        
        self.SolveNQueens(queen,n,0)
        return self.ans

#https://leetcode.com/problems/n-queens/discuss/?currentPage=1&orderBy=most_votes&query=&tag=python


'''Sudoku Solver
Hard

Write a program to solve a Sudoku puzzle by filling the empty cells.

A sudoku solution must satisfy all of the following rules:

Each of the digits 1-9 must occur exactly once in each row.
Each of the digits 1-9 must occur exactly once in each column.
Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
The '.' character indicates empty cells.

Example 1:
Input: board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
Output: [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
Explanation: The input board is shown above and the only valid solution is shown below:

Constraints:

board.length == 9
board[i].length == 9
board[i][j] is a digit or '.'.
It is guaranteed that the input board has only one solution.'''

class Solution:
    def solveSudoku(self, board):
        """
        Do not return anything, modify board in-place instead.
        """
        assert(self.backtrack(board))
        return
    
    
    def backtrack(self,board):
        n = len(board)
        m = len(board[0])
        for i in range(n):
            for j in range(m):
                if board[i][j] == '.':
                    for v in ['1','2','3','4','5','6','7','8','9']:
                        if self.isvalid(board,i, j, v):
                            board[i][j] = v
                            
                            if self.backtrack(board) == True:
                                return True
                            
                            else:
                                board[i][j] = '.'
                            
                    return False
                
        return True
    
    def isvalid(self,board, row, col, char):
        for i in range(9):
            if board[row][i] == char:
                return False
            
            if board[i][col] == char:
                return False
            if board[3 * (row//3)+ i//3][3 * (col//3)+ i%3] == char:
                return False
            
        return True
        
'''Your input
[["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],
["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],
[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
Output
[["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],
["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],
["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
Expected
[["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],
["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],
["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]'''

#OR

class Solution:
    # @param board, a 9x9 2D array
    # Solve the Sudoku by modifying the input board in-place.
    # Do not return any value.
    def solveSudoku(self, board):
        self.board = board
        self.solve()
    
    def findUnassigned(self):
        for row in range(9):
            for col in range(9):
                if self.board[row][col] == ".":
                    return row, col
        return -1, -1
    
    def solve(self):
        row, col = self.findUnassigned()
        #no unassigned position is found, puzzle solved
        if row == -1 and col == -1:
            return True
        for num in ["1","2","3","4","5","6","7","8","9"]:
            if self.isSafe(row, col, num):
                self.board[row][col] = num
                if self.solve():
                    return True
                self.board[row][col] = "."
        return False
            
    def isSafe(self, row, col, ch):
        boxrow = row - row%3
        boxcol = col - col%3
        if self.checkrow(row,ch) and self.checkcol(col,ch) and self.checksquare(boxrow, boxcol, ch):
            return True
        return False
    
    def checkrow(self, row, ch):
        for col in range(9):
            if self.board[row][col] == ch:
                return False
        return True
    
    def checkcol(self, col, ch):
        for row in range(9):
            if self.board[row][col] == ch:
                return False
        return True
       
    def checksquare(self, row, col, ch):
        for r in range(row, row+3):
            for c in range(col, col+3):
                if self.board[r][c] == ch:
                    return False
        return True
            
'''M-Coloring Problem 
Medium 
Given an undirected graph and an integer M. The task is to determine if the graph can be colored with at most M colors such that no two adjacent vertices of the graph are colored with the same color. Here coloring of a graph means the assignment of colors to all vertices. Print 1 if it is possible to colour vertices and 0 otherwise.

Example 1:

Input:
N = 4
M = 3
E = 5
Edges[] = {(0,1),(1,2),(2,3),(3,0),(0,2)}
Output: 1
Explanation: It is possible to colour the
given graph using 3 colours.
Example 2:

Input:
N = 3
M = 2
E = 3
Edges[] = {(0,1),(1,2),(0,2)}
Output: 0
Your Task:
Your task is to complete the function graphColoring() which takes the 2d-array graph[], the number of colours and the number of nodes as inputs and returns true if answer exists otherwise false. 1 is printed if the returned value is true, 0 otherwise. The printing is done by the driver's code.
Note: In the given 2d-array graph[], if there is an edge between vertex X and vertex Y graph[] will contain 1 at graph[X-1][Y-1], else 0. In 2d-array graph[ ], nodes are 0-based indexed, i.e. from 0 to N-1.

Expected Time Complexity: O(MN).
Expected Auxiliary Space: O(N).

Constraints:
1 ≤ N ≤ 20
1 ≤ E ≤ (N*(N-1))/2
1 ≤ M ≤ N'''    

def issafe(node, graph, color, k, v, i):
    for j in range(v):
        if graph[node][j] != 0 and color[j] == i:
            return False
    return True

def solve(node, graph, color, k, V):
    if node == V:
        return True
        
    for i in range(1,k+1):
        if issafe(node, graph, color, k,V, i):
            color[node] = i
            
            if solve(node+1, graph, color, k, V):
                return True
                
            color[node] = 0
            
    return False

def graphColoring(graph, k, V):
    
    #your code here
    n = len(graph)
    color = [0] * V
    return solve(0, graph,color,k,V)

'''For Input:
4
3
5
1 2 2 3 3 4 4 1 1 3

Your Output is: 
1'''
    
#OR

'''Rat in a Maze Problem - I 
Medium Accuracy: 37.73% Submissions: 76988 Points: 4
Consider a rat placed at (0, 0) in a square matrix of order N * N. It has to reach the destination at (N - 1, N - 1). Find all possible paths that the rat can take to reach from source to destination. The directions in which the rat can move are 'U'(up), 'D'(down), 'L' (left), 'R' (right). Value 0 at a cell in the matrix represents that it is blocked and rat cannot move to it while value 1 at a cell in the matrix represents that rat can be travel through it.
Note: In a path, no cell can be visited more than one time.

Example 1:

Input:
N = 4
m[][] = {{1, 0, 0, 0},
         {1, 1, 0, 1}, 
         {1, 1, 0, 0},
         {0, 1, 1, 1}}
Output:
DDRDRR DRDDRR
Explanation:
The rat can reach the destination at 
(3, 3) from (0, 0) by two paths - DRDDRR 
and DDRDRR, when printed in sorted order 
we get DDRDRR DRDDRR.
Example 2:
Input:
N = 2
m[][] = {{1, 0},
         {1, 0}}
Output:
-1
Explanation:
No path exists and destination cell is 
blocked.
Your Task:  
You don't need to read input or print anything. Complete the function printPath() which takes N and 2D array m[ ][ ] as input parameters and returns the list of paths in lexicographically increasing order. 
Note: In case of no path, return an empty list. The driver will output "-1" automatically.

Expected Time Complexity: O((N2)4).
Expected Auxiliary Space: O(L * X), L = length of the path, X = number of paths.'''   

class Solution:
    def findPath(self, m, n):
        # code here
        ans = []
        sol = [[0 for i in range(n)]for i in range(n)]
        self.solve(0,0,m,n,ans,'',sol)
        if ans == []:
            return []
        else:
            ans.sort()
            return ans
        
    def solve(self,row, col, arr, n, ans, st, vis):
        if row == n-1 and col == n-1 and arr[row][col] == 1:
            ans.append(st)
            return
        if row < 0 or col < 0 or row >= n or col >= n or arr[row][col] == 0 or vis[row][col] == 1:
            return 
        vis[row][col] = 1
        self.solve(row+1, col, arr, n, ans, st+"D", vis)
        self.solve(row, col+1, arr, n, ans, st+"R", vis)
        self.solve(row-1, col, arr, n, ans, st+"U", vis)
        self.solve(row, col-1, arr, n, ans, st+"L", vis)
        vis[row][col] = 0


#OR 

class Solution:
    def findPath(self, m, n):
        # code here
        ans = []
        sol = [[0 for i in range(n)]for i in range(n)]
        di = [(-1,0),(1,0),(0,-1),(0,1)]
        self.solve(0,0,m,n,ans,'',sol,di)
        if ans == []:
            return []
        else:
            ans.sort()
            return ans
    
    def solve(self, row, col, arr, n, ans, st, vis, di):
        if row == n-1 and col == n-1 and arr[row][col] == 1:
            ans.append(st)
            return
        
        dir = 'UDLR'
        for i in range(4):
            nr = row + di[i][0]
            nc = col + di[i][1]
            if nr >= 0 and nr < n and nc >= 0 and nc < n and arr[nr][nc] != 0 and vis[nr][nc] == 0:
                vis[row][col] = 1
                self.solve(nr, nc, arr, n, ans, st+dir[i], vis, di)
                vis[row][col] = 0

'''Word Break
Medium

7647

358

Add to List

Share
Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

 

Example 1:

Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
Example 2:

Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.
Example 3:

Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false
 

Constraints:

1 <= s.length <= 300
1 <= wordDict.length <= 1000
1 <= wordDict[i].length <= 20
s and wordDict[i] consist of only lowercase English letters.
All the strings of wordDict are unique.'''

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False] * (n+1)
        dp[n] = True
        
        for i in range(n-1, -1, -1):
            for w in wordDict:
                if (i + len(w)) <= n and s[i: i+len(w)] == w:
                    dp[i] = dp[i + len(w)]
                if dp[i]:
                    break
        return dp[0]

#OR

def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    from collections import deque
    q = deque([s])
    seen = set() 
    while q:
        s = q.popleft()    # popleft() = BFS ; pop() = DFS
        for word in wordDict:
            if s.startswith(word):
                new_s = s[len(word):]
                if new_s == "": 
                    return True
                if new_s not in seen:
                    q.append(new_s)
                    seen.add(new_s)
    return False

#https://leetcode.com/problems/word-break/discuss/?currentPage=1&orderBy=most_votes&query=

'''Find Nth root of M 
Easy Accuracy: 33.21% Submissions: 5782 Points: 2
You are given 2 numbers (n , m); the task is to find n√m (nth root of m).
 

Example 1:

Input: n = 2, m = 9
Output: 3
Explanation: 32 = 9
Example 2:

Input: n = 3, m = 9
Output: -1
Explanation: 3rd root of 9 is not
integer.
 

Your Task:
You don't need to read or print anyhting. Your task is to complete the function NthRoot() which takes n and m as input parameter and returns the nth root of m. If the root is not integer then returns -1.
 

Expected Time Complexity: O(n* log(m))
Expected Space Complexity: O(1)
 

Constraints:
1 <= n <= 30
1 <= m <= 109'''
            
class Solution:
    def multiply(self, num, n):
        ans = 1.0
        for i in range(1, n+1):
            ans = ans * num
        return ans
        
    def NthRoot(self, n, m):
		# Code here
        l = 1
        h = m
        k = 0.000001
        ans = 0
        while (l - h) > k:
            mid = (l+h) / 2.0
            ans = self.multiply(mid,n)
            if int(ans) == m:
                return int(mid)
                
            elif ans < m:
                l = mid
            else:
                h = mid
        return -1

'''Matrix Median
Medium

118

5

Add to favorites
Asked In:
AMAZON
Given a matrix of integers A of size N x M in which each row is sorted.

Find an return the overall median of the matrix A.

Note: No extra memory is allowed.

Note: Rows are numbered from top to bottom and columns are numbered from left to right.




Input Format

The first and only argument given is the integer matrix A.
Output Format

Return the overall median of the matrix A.
Constraints

1 <= N, M <= 10^5
1 <= N*M  <= 10^6
1 <= A[i] <= 10^9
N*M is odd
For Example

Input 1:
    A = [   [1, 3, 5],
            [2, 6, 9],
            [3, 6, 9]   ]
Output 1:
    5
Explanation 1:
    A = [1, 2, 3, 3, 5, 6, 6, 9, 9]
    Median is 5. So, we return 5.

Input 2:
    A = [   [5, 17, 100]    ]
Output 2:
    17 ``` Matrix='''

class Solution:
	# @param A : list of list of integers
	# @return an integer
	def findmid(self, row, mid):
		l = 0
		h = len(row) - 1
		while l <= h:
			md = (l+h)//2

			if row[md] <= mid:
				l = md + 1
			else:
				h = md-1
		return l

	def findMedian(self, A):
		low = 1
		high = 1e9
		n = len(A)
		m = len(A[0])
		while low <= high:
			mid = (low+high)//2
			cnt = 0
			for i in range(n):
				cnt += self.findmid(A[i],mid)

			if cnt <= ((n *m) /2):
				low = mid+1
			else:
				high = mid-1
		return int(low)

#OR

def findMedian(self, A):
        a = list(itertools.chain(*A))
        a.sort()
        if(len(a)%2==0):
            return (a[len(a)//2]+a[len(a)//2 -1])//2
        return a[len(a)//2]

#OR

class Solution:
    # @param A : list of list of integers
    # @return an integer
    def binary_search_count(self, array, target):
        """ Returns number of elements <= target in sorted array.
        Time complexity: O(lg(n)). Space complexity: O(1), n is len(array).
        """
        # special case
        if target < array[0]:  # target is less than min element
            return 0

        n = len(array)
        start, end = 0, n - 1
        while start < end:
            mid = (start + end + 1) // 2
            if target < array[mid]:
                end = mid - 1
            else:
                start = mid
        return start + 1

    def count_target(self, matrix, target):
        """ Returns number of elements <= target in matrix.
        Time complexity: O(n * lg(m)). Space complexity: O(1),
        n, m are dimensions of the matrix.
        """
        total = 0
        for arr in matrix:
            total += self.binary_search_count(arr, target)
        return total

    def findMedian(self, matrix):
        """ Returns matrix median.
        Time complexity: O(n * lg(m)). Space complexity: O(1),
        n, m are dimensions of the matrix.
        """
        n, m = len(matrix), len(matrix[0])
        # find min and max element in matrix
        min_num, max_num = float("inf"), float("-inf")
        for row in matrix:
            min_num = min(min_num, row[0])  # compare 1st element of each row
            max_num = max(max_num, row[-1])  # compare last element of each row

        goal = (n * m) // 2 + 1  # min count of <= elements for element to be median
        # find matrix median using binary search between min_num and max_num
        while min_num < max_num:
            mid = (min_num + max_num) // 2
            # mid = min_num + (max_num - min_num) // 2
            curr_count = self.count_target(matrix, mid)
            if curr_count < goal:
                min_num = mid + 1
            else:  # curr_count >= goal
                max_num = mid  # update the upper limit for median number
        return min_num


'''Single Element in a Sorted Array
Medium

2979

88

Add to List

Share
You are given a sorted array consisting of only integers where every element appears exactly twice, except for one element which appears exactly once. Find this single element that appears only once.

Follow up: Your solution should run in O(log n) time and O(1) space.

Example 1:

Input: nums = [1,1,2,3,3,4,4,8,8]
Output: 2
Example 2:

Input: nums = [3,3,7,7,10,11,11]
Output: 10
 

Constraints:

1 <= nums.length <= 10^5
0 <= nums[i] <= 10^5'''

class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        n = len(nums)
        l = 0
        h = n-2
        while(l <= h):
            mid = (l+h)//2
            
            if nums[mid] == nums[mid^1]:
                l = mid +1
                
            else:
                h = mid-1
            
        return nums[l]

'''odd xor 1 = odd-1
even xor 1 = even+1'''

#OR
class Solution(object):
    def singleNonDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left, right = 0, len(nums) - 1
        while left + 1 < right:
            mid = (left + right) // 2
            if mid % 2 == 1:
                if nums[mid] == nums[mid - 1]:
                    left = mid
                else:
                    right = mid
            else:
                if nums[mid] == nums[mid + 1]:
                    left = mid
                else:
                    right = mid
        #print(left, right)
        if left % 2 == 0:
            return nums[left]
        return nums[right]

#https://leetcode.com/problems/single-element-in-a-sorted-array/discuss/?currentPage=1&orderBy=most_votes&query=

'''Search in Rotated Sorted Array
Medium

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

'''Your input
[4,5,6,7,0,1,2]
0
Output
4
Expected
4'''

#Or

def search(self, nums, target):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) / 2
        if (nums[0] > target) ^ (nums[0] > nums[mid]) ^ (target > nums[mid]):
            lo = mid + 1
        else:
            hi = mid
    return lo if target in nums[lo:lo+1] else -1

'''Median of Two Sorted Arrays
Hard

11829

1650

Add to List

Share
Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).

 

Example 1:

Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.
Example 2:

Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.
Example 3:

Input: nums1 = [0,0], nums2 = [0,0]
Output: 0.00000
Example 4:

Input: nums1 = [], nums2 = [1]
Output: 1.00000
Example 5:

Input: nums1 = [2], nums2 = []
Output: 2.00000
 '''

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums2) < len(nums1):
            return self.findMedianSortedArrays(nums2, nums1)
        
        n1 = len(nums1)
        n2 = len(nums2)
        l = 0
        h = n1
        
        while l <= h:
            cut1 = (l+h) // 2
            cut2 = (n1+n2+1)//2 - cut1
            
            if cut1 == 0:
                left1 = float('-inf')
            else:
                left1 = nums1[cut1-1]
                
            if cut2 == 0:
                left2 = float('-inf')
            else:
                left2 = nums2[cut2-1]
                
            if cut1 == n1:
                right1 = float('inf')
            else:
                right1 = nums1[cut1]
                        
            if cut2 == n2:
                right2 = float('inf')
            else:
                right2 = nums2[cut2]
                
            if (left1 <= right2) and (left2 <= right1):
                if (n1+n2) % 2 == 0:
                    return (max(left1, left2) + min(right1, right2)) / 2.0
                else:
                    return (max(left1, left2))
                
            elif left1 > right2:
                h = cut1 -1
                
            else:
                l = cut1 + 1
                
        return 0.0
'''Accepted
Runtime: 44 ms
Your input
[1,3]
[2]
Output
2.00000
Expected
2.00000'''

#OR

def findMedianSortedArrays(self, nums1, nums2):
    a, b = sorted((nums1, nums2), key=len)
    m, n = len(a), len(b)
    after = (m + n - 1) / 2
    lo, hi = 0, m
    while lo < hi:
        i = (lo + hi) / 2
        if after-i-1 < 0 or a[i] >= b[after-i-1]:
            hi = i
        else:
            lo = i + 1
    i = lo
    nextfew = sorted(a[i:i+2] + b[after-i:after-i+2])
    return (nextfew[0] + nextfew[1 - (m+n)%2]) / 2.0

#https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/?currentPage=1&orderBy=most_votes&query=

'''K-th element of two sorted Arrays 
Medium Accuracy: 50.09% Submissions: 22500 Points: 4
Given two sorted arrays arr1 and arr2 of size N and M respectively and an element K. The task is to find the element that would be at the k’th position of the final sorted array.
 

Example 1:

Input:
arr1[] = {2, 3, 6, 7, 9}
arr2[] = {1, 4, 8, 10}
k = 5
Output:
6
Explanation:
The final sorted array would be -
1, 2, 3, 4, 6, 7, 8, 9, 10
The 5th element of this array is 6.
Example 2:
Input:
arr1[] = {100, 112, 256, 349, 770}
arr2[] = {72, 86, 113, 119, 265, 445, 892}
k = 7
Output:
256
Explanation:
Final sorted array is - 72, 86, 100, 112,
113, 119, 256, 265, 349, 445, 770, 892
7th element of this array is 256.

Your Task:  
You don't need to read input or print anything. Your task is to complete the function kthElement() which takes the arrays arr1[], arr2[], its size N and M respectively and an integer K as inputs and returns the element at the Kth position.


Expected Time Complexity: O(Log(N) + Log(M))
Expected Auxiliary Space: O(Log (N))


Constraints:
1 <= N, M <= 106
1 <= arr1i, arr2i <= 106
1 <= K <= N+M'''

#User function Template for python3

class Solution:
    def kthElement(self,  arr1, arr2, n, m, k):
        if m < n:
            return self.kthElement(arr2,arr1,m,n,k)
            
        low = max(0, k-m)
        high = min(n, k)
        
        while low <= high:
            cut1 = (low+high)//2
            cut2 = k -cut1
            
            if cut1 == 0:
                left1 = float('-inf')
            else:
                left1 = arr1[cut1-1]
                
            if cut2 == 0:
                left2 = float('-inf')
            else:
                left2 = arr2[cut2-1]
                
            if cut1 == n:
                right1 = float('inf')
            else:
                right1 = arr1[cut1]
                
            if cut2 == m:
                right2 = float('inf')
            else:
                right2 = arr2[cut2]
                
            if (left1 <= right2) and (left2 <= right1):
                return max(left1, left2)
                
            elif left1 > right2:
                high = cut1 -1
                
            else:
                low = cut1 + 1
                
        return 1
        
'''For Input:
5 4 5
2 3 6 7 9
1 4 8 10

Your Output is: 
6'''

#oR

def kth(arr1, arr2, m, n, k):
 
    sorted1 = [0] * (m + n)
    i = 0
    j = 0
    d = 0
    while (i < m and j < n):
 
        if (arr1[i] < arr2[j]):
            sorted1[d] = arr1[i]
            i += 1
        else:
            sorted1[d] = arr2[j]
            j += 1
        d += 1
 
    while (i < m):
        sorted1[d] = arr1[i]
        d += 1
        i += 1
    while (j < n):
        sorted1[d] = arr2[j]
        d += 1
        j += 1
    return sorted1[k - 1]

'''Allocate Books
Medium

Given an array of integers A of size N and an integer B.

College library has N bags,the ith book has A[i] number of pages.

You have to allocate books to B number of students so that maximum number of pages alloted to a student is minimum.

A book will be allocated to exactly one student.
Each student has to be allocated at least one book.
Allotment should be in contiguous order, for example: A student cannot be allocated book 1 and book 3, skipping book 2.
Calculate and return that minimum possible number.

NOTE: Return -1 if a valid assignment is not possible.

Input Format

The first argument given is the integer array A.
The second argument given is the integer B.
Output Format

Return that minimum possible number
Constraints

1 <= N <= 10^5
1 <= A[i] <= 10^5
For Example

Input 1:
    A = [12, 34, 67, 90]
    B = 2
Output 1:
    113
Explanation 1:
    There are 2 number of students. Books can be distributed in following fashion : 
        1) [12] and [34, 67, 90]
        Max number of pages is allocated to student 2 with 34 + 67 + 90 = 191 pages
        2) [12, 34] and [67, 90]
        Max number of pages is allocated to student 2 with 67 + 90 = 157 pages 
        3) [12, 34, 67] and [90]
        Max number of pages is allocated to student 1 with 12 + 34 + 67 = 113 pages

        Of the 3 cases, Option 3 has the minimum pages = 113.

Input 2:
    A = [5, 17, 100, 11]
    B = 4
Output 2:
    100
Note:You only need to implement the given function. Do not read input, instead use the arguments to the function. Do not print the output, 
instead return values as specified. Still have a doubt? Checkout Sample Codes for more details.'''

class Solution:
	# @param A : list of integers
	# @param B : integer
	# @return an integer
    def books(self, A, B):
        if B > len(A):
            return -1
        low = min(A)
        high = sum(A)
        res = -1
        while low <= high:
            mid = (low+high)// 2

            if self.ispossible(mid, A, B, len(A)):
                res = mid
                high = mid-1
            else:
                low = mid+1

        return res

    def ispossible(self, edge, arr, stu, n):
        alstu = 0
        pages = 0
        for i in range(n):
            if (pages + arr[i] > edge):
                alstu += 1
                pages = arr[i]
                if arr[i] > edge:
                    return False
            else:
                pages += arr[i]

        if alstu < stu:
            return True
        return False

#or

class Solution:
    # @param A : list of integers
    # @param B : integer
    # @return an integer
    def books(self, A, B):
        
        def func(A , mid ):
            sum1 = 0
            count = 1
            for i in range(0,len(A)):
                sum1 = sum1 + A[i]
        
                if(sum1 > mid):
                    sum1 = A[i]
                    count = count + 1
        
            return count
        
        if(B > len(A)):
            return -1
        else:
            l = max(A)
            r = sum(A)
            
            while(l<=r):
                mid = l + int((r-l)/2)
                # print(mid)
                k = func(A , mid)
                if(k <= B):
                    # print(k)
                    r = mid-1
                    ans = mid
                    # print("ans" , ans)
                elif(k > B):
                    l = mid+1

            return ans

        
#https://www.interviewbit.com/problems/allocate-books/

'''AGGRCOW - Aggressive cows
#binary-search
Farmer John has built a new long barn, with N (2 <= N <= 100,000) stalls. The stalls are located along a straight line at positions x1,...,xN (0 <= xi <= 1,000,000,000).

His C (2 <= C <= N) cows don't like this barn layout and become aggressive towards each other once put into a stall. To prevent the cows from hurting each other, FJ wants to assign the cows to the stalls, such that the minimum distance between any two of them is as large as possible. What is the largest minimum distance?
Input
t – the number of test cases, then t test cases follows.
* Line 1: Two space-separated integers: N and C
* Lines 2..N+1: Line i+1 contains an integer stall location, xi
Output
For each test case output one integer: the largest minimum distance.
Example
Input:

1
5 3
1
2
8
4
9
Output:

3
Output details:

FJ can put his 3 cows in the stalls at positions 1, 4 and 8,
resulting in a minimum distance of 3.
'''
# your code goes here
class sol:
    def aggresiveCows(self,arr, cows):
    	n = len(arr)
    	arr.sort()
    	low = 1
    	high = arr[n-1] - arr[0]
    	res = -1
    	while (low <= high):
    		mid = (low + high) //2
    		
    		if self.isPossible(arr, n, cows, mid):
    			res = mid
    			low = mid+1
    		else:
    			high = mid-1
    			
    		return high
    			
    def isPossible(self,arr, n, cows, dist):
    	cnt = 1
    	first = arr[0]
    	for i in range(1,n):
    		if (arr[i]- first >= dist):
    			cnt += 1
    			first = arr[i]
    			
    		if cnt == cows:
    			return True
    			
    		return False

l = sol()		
T = l.aggresiveCows([1,2,4,8,9], 3 )
print(T)

'3'

#similar question
'''Magnetic Force Between Two Balls
Medium

703

64

Add to List

Share
In universe Earth C-137, Rick discovered a special form of magnetic force between two balls if they are put in his new invented basket. Rick has n empty baskets, the ith basket is at position[i], Morty has m balls and needs to distribute the balls into the baskets such that the minimum magnetic force between any two balls is maximum.

Rick stated that magnetic force between two different balls at positions x and y is |x - y|.

Given the integer array position and the integer m. Return the required force.

 

Example 1:


Input: position = [1,2,3,4,7], m = 3
Output: 3
Explanation: Distributing the 3 balls into baskets 1, 4 and 7 will make the magnetic force between ball pairs [3, 3, 6]. The minimum magnetic force is 3. We cannot achieve a larger minimum magnetic force than 3.
Example 2:

Input: position = [5,4,3,2,1,1000000000], m = 2
Output: 999999999
Explanation: We can use baskets 1 and 1000000000.
 

Constraints:

n == position.length
2 <= n <= 10^5
1 <= position[i] <= 10^9
All integers in position are distinct.
2 <= m <= position.length'''

class Solution:
    def maxDistance(self, position: List[int], m: int) -> int:
        n = len(position)
        position.sort()
        
        def count(d):
            ans, curr = 1, position[0]
            for i in range(1, n):
                if position[i] - curr >= d:
                    ans += 1
                    curr = position[i]
            return ans
        
        l, r = 0, position[-1] - position[0]
        while l < r:
            mid = r - (r - l) // 2
            if count(mid) >= m:
                l = mid
            else:
                r = mid - 1
        return l


'''Implement Stack using Queues
Easy

1304

719

Add to List

Share
Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (push, top, pop, and empty).

Implement the MyStack class:

void push(int x) Pushes element x to the top of the stack.
int pop() Removes the element on the top of the stack and returns it.
int top() Returns the element on the top of the stack.
boolean empty() Returns true if the stack is empty, false otherwise.
Notes:

You must use only standard operations of a queue, which means that only push to back, peek/pop from front, size and is empty operations are valid.
Depending on your language, the queue may not be supported natively. You may simulate a queue using a list or deque (double-ended queue) as long as you use only a queue's standard operations.
 

Example 1:

Input
["MyStack", "push", "push", "top", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 2, 2, false]

Explanation
MyStack myStack = new MyStack();
myStack.push(1);
myStack.push(2);
myStack.top(); // return 2
myStack.pop(); // return 2
myStack.empty(); // return False
 

Constraints:

1 <= x <= 9
At most 100 calls will be made to push, pop, top, and empty.
All the calls to pop and top are valid.
 

Follow-up: Can you implement the stack using only one queue?'''

class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue1 = []
        

    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        
        queue1 = self.queue1
        queue1.append(x)
        for _ in range(len(queue1) - 1):
            queue1.append(queue1.pop(0))
            
        
    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.queue1.pop(0)

    def top(self) -> int:
        """
        Get the top element.
        """
        return self.queue1[0]

    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return len(self.queue1) == 0


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()

'''Your input
["MyStack","push","push","top","pop","empty"]
[[],[1],[2],[],[],[]]
Output
[null,null,null,2,2,false]
Expected
[null,null,null,2,2,false]'''

#https://leetcode.com/problems/implement-stack-using-queues/discuss/?currentPage=1&orderBy=most_votes&query=

'''Implement Queue using Stacks
Easy

2067

178

Add to List

Share
Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).

Implement the MyQueue class:

void push(int x) Pushes element x to the back of the queue.
int pop() Removes the element from the front of the queue and returns it.
int peek() Returns the element at the front of the queue.
boolean empty() Returns true if the queue is empty, false otherwise.
Notes:

You must use only standard operations of a stack, which means only push to top, peek/pop from top, size, and is empty operations are valid.
Depending on your language, the stack may not be supported natively. You may simulate a stack using a list or deque (double-ended queue) as long as you use only a stack's standard operations.
 

Example 1:

Input
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 1, 1, false]

Explanation
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false
 

Constraints:

1 <= x <= 9
At most 100 calls will be made to push, pop, peek, and empty.
All the calls to pop and peek are valid.
 

Follow-up: Can you implement the queue such that each operation is amortized O(1) time complexity? In other words, 
performing n operations will take overall O(n) time even if one of those operations may take longer.'''

class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.in_stack = []
        self.out_stack = []
        

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: None
        """
        self.in_stack.append(x)
        
    def refill_out_stack(self):
        """
        Check if the out_stack is empty. If it is, refill it.
        """
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        self.refill_out_stack()
        return self.out_stack.pop()
        

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        self.refill_out_stack()
        return self.out_stack[-1]

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return not (self.in_stack or self.out_stack)

'''Your input
["MyQueue","push","push","peek","pop","empty"]
[[],[1],[2],[],[],[]]
Output
[null,null,null,1,1,false]
Expected
[null,null,null,1,1,false]'''

'''Valid Parentheses
Easy

8600

344

Add to List

Share
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
 

Example 1:

Input: s = "()"
Output: true
Example 2:

Input: s = "()[]{}"
Output: true
Example 3:

Input: s = "(]"
Output: false
Example 4:

Input: s = "([)]"
Output: false
Example 5:

Input: s = "{[]}"
Output: true
 

Constraints:

1 <= s.length <= 104
s consists of parentheses only '()[]{}'.'''

class Solution:
    def isValid(self, s: str) -> bool:
        st = []
        dict = { ')':'(', ']':'[', '}':'{' }
        for it in s:
            if it in dict.values():
                st.append(it)
                
            elif it in dict.keys():
                if st == [] or dict[it] != st.pop():
                    return False
                
            else:
                return False
        
        return st == []

'''Your input
"()"
Output
true
Expected
true'''

#or

class Solution:
    def isValid(self, s: str) -> bool:
        while len(s) > 0:
            l = len(s)
            s = s.replace('()','').replace('{}','').replace('[]','')
            if l==len(s): return False
        return True

#https://leetcode.com/problems/valid-parentheses/discuss/?currentPage=1&orderBy=most_votes&query=&tag=python

'''Sort a stack 
Easy Accuracy: 50.51% Submissions: 49332 Points: 2
Given a stack, the task is to sort it such that the top of the stack has the greatest element.

Example 1:

Input:
Stack: 3 2 1
Output: 3 2 1
Example 2:

Input:
Stack: 11 2 32 3 41
Output: 41 32 11 3 2
Your Task: 
You don't have to read input or print anything. Your task is to complete the function sort() which sorts the elements present in the given stack.
 (The sorted stack is printed by the driver's code by popping the elements of the stack.)

Expected Time Complexity: O(N*N)
Expected Auxilliary Space: O(N) recursive.

Constraints:
1<=N<=100

Note:The Input/Ouput format and Example given are used for system's internal purpose, and should be used by a user for Expected Output only. 
As it is a function problem, hence a user should not read any input from stdin/console.
 The task is to complete the function specified, and not to write the full code.'''
                
def sortedI(s):
    s1 = []
    while s:
        tmp = s.pop()
        
        while s1 and s1[-1] > tmp:
            s.append(s1.pop())
        s1.append(tmp)
    return s1

print(sortedI([3,2,1,5,9,4]))

#[1, 2, 3, 4, 5, 9]

'''Next Smaller Element
Difficulty Level : Medium
Last Updated : 19 Jul, 2021
Given an array, print the Next Smaller Element (NSE) for every element. The Smaller smaller Element for an element x is the first smaller element on the right side of x in array. Elements for which no smaller element exist (on right side), consider next smaller element as -1. 
Examples: 
a) For any array, rightmost element always has next smaller element as -1. 
b) For an array which is sorted in increasing order, all elements have next smaller element as -1. 
c) For the input array [4, 8, 5, 2, 25}, the next smaller elements for each element are as follows.

Element       NSE
   4      -->   2
   8      -->   5
   5      -->   2
   2      -->   -1
   25     -->   -1
d) For the input array [13, 7, 6, 12}, the next smaller elements for each element are as follows.  

  Element        NSE
   13      -->    7
   7       -->    6
   6       -->    -1
   12     -->     -1'''

class Solution:
    def nextGreaterElements(self, nums):
        stack, res = [], [-1] * len(nums)
        n = len(nums)
        for i in range(0,(n)):              # 2
            while stack and nums[stack[-1]] > nums[i]:
                res[stack.pop()] = nums[i]
            stack.append(i)
            
        
        return res
        
        
l = Solution()

print(l.nextGreaterElements([10,4,7,3,2,8,6,11,1,5]))

#[4, 3, 3, 2, 1, 6, 1, 1, -1, -1]

'''Next Greater Element II
Medium

3076

105

Add to List

Share
Given a circular integer array nums (i.e., the next element of nums[nums.length - 1] is nums[0]), return the next greater number for every element in nums.

The next greater number of a number x is the first greater number to its traversing-order next in the array, which means you could search circularly to find its next greater number. If it doesn't exist, return -1 for this number.

 

Example 1:

Input: nums = [1,2,1]
Output: [2,-1,2]
Explanation: The first 1's next greater number is 2; 
The number 2 can't find next greater number. 
The second 1's next greater number needs to search circularly, which is also 2.
Example 2:

Input: nums = [1,2,3,4,3]
Output: [2,3,4,-1,4]
 

Constraints:

1 <= nums.length <= 104
-109 <= nums[i] <= 109'''

class Solution:
    def nextGreaterElements(self, nums):
        stack, res = [], [-1] * len(nums)
        n = len(nums)
        for i in range(0,(2*n-1)):              # 2
            while stack and nums[stack[-1]] < nums[i%n]:
                res[stack.pop()] = nums[i%n]
            stack.append(i%n)
            
        
        return res

#[11, 7, 8, 8, 8, 11, 11, -1, 5, 10]

#or

class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        stack, res = [], [-1] * len(nums)
        for i, num in enumerate(nums):              # 2
            while stack and nums[stack[-1]] < num:
                res[stack.pop()] = num
            stack.append(i)
        for i, num in enumerate(nums):              # 3
            while stack and nums[stack[-1]] < num:
                res[stack.pop()] = num
        return res
                
                
'''LRU Cache
Medium

9630

383

Add to List

Share
Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:

LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
int get(int key) Return the value of the key if the key exists, otherwise return -1.
void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.
The functions get and put must each run in O(1) average time complexity.

 

Example 1:

Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4
 

Constraints:

1 <= capacity <= 3000
0 <= key <= 104
0 <= value <= 105
At most 2 * 105 calls will be made to get and put.
Accepted
833,681
Submissions
2,213,943'''

class Node:
    def __init__(self, key= None, val =None):
        self.val = val
        self.key = key
        self.next = None
        self.prev = None
        
class LRUCache:

    def __init__(self, capacity: int):
        self.cap = capacity
        self.head = Node(0,0)
        self.tail = Node(0,0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.map = {}
        
    def addNode(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next = node
        node.next.prev = node
        
    def DelNode(self, node):
        delprev = node.prev
        delnext = node.next
        delprev.next = delnext
        delnext.prev = delprev
        
    def get(self, key: int) -> int:
        if key not in self.map:
            return -1
        
        node = self.map[key]
        res = node.val
        self.map.pop(key)
        self.DelNode(node)
        self.addNode(node)
        self.map[key] = self.head.next
        return res
        

    def put(self, key: int, value: int) -> None:
        if key in self.map:
            curr = self.map[key]
            self.map.pop(key)
            self.DelNode(curr)
            
        if(len(self.map) == self.cap):
            self.map.pop(self.tail.prev.key)
            self.DelNode(self.tail.prev)
            
        node = Node(key, value)    
        self.addNode(node)
        self.map[key] = self.head.next
        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

'''Your input
["LRUCache","put","put","get","put","get","put","get","get","get"]
[[2],[1,1],[2,2],[1],[3,3],[2],[4,4],[1],[3],[4]]
Output
[null,null,null,1,null,-1,null,-1,3,4]
Expected
[null,null,null,1,null,-1,null,-1,3,4]'''

class ListNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.dic = dict() # key to node
        self.capacity = capacity
        self.head = ListNode(0, 0)
        self.tail = ListNode(-1, -1)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key in self.dic:
            node = self.dic[key]
            self.removeFromList(node)
            self.insertIntoHead(node)
            return node.value
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.dic:             # similar to get()        
            node = self.dic[key]
            self.removeFromList(node)
            self.insertIntoHead(node)
            node.value = value         # replace the value len(dic)
        else: 
            if len(self.dic) >= self.capacity:
                self.removeFromTail()
            node = ListNode(key,value)
            self.dic[key] = node
            self.insertIntoHead(node)
			
    def removeFromList(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def insertIntoHead(self, node):
        headNext = self.head.next 
        self.head.next = node 
        node.prev = self.head 
        node.next = headNext 
        headNext.prev = node
    
    def removeFromTail(self):
        if len(self.dic) == 0: return
        tail_node = self.tail.prev
        del self.dic[tail_node.key]
        self.removeFromList(tail_node)

'''LFU Cache
Hard

2354

164

Add to List

Share
Design and implement a data structure for a Least Frequently Used (LFU) cache.

Implement the LFUCache class:

LFUCache(int capacity) Initializes the object with the capacity of the data structure.
int get(int key) Gets the value of the key if the key exists in the cache. Otherwise, returns -1.
void put(int key, int value) Update the value of the key if present, or inserts the key if not already present. When the cache reaches its capacity, it should invalidate and remove the least frequently used key before inserting a new item. For this problem, when there is a tie (i.e., two or more keys with the same frequency), the least recently used key would be invalidated.
To determine the least frequently used key, a use counter is maintained for each key in the cache. The key with the smallest use counter is the least frequently used key.

When a key is first inserted into the cache, its use counter is set to 1 (due to the put operation). The use counter for a key in the cache is incremented either a get or put operation is called on it.

The functions get and put must each run in O(1) average time complexity.

 

Example 1:

Input
["LFUCache", "put", "put", "get", "put", "get", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [3], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, 3, null, -1, 3, 4]

Explanation
// cnt(x) = the use counter for key x
// cache=[] will show the last used order for tiebreakers (leftmost element is  most recent)
LFUCache lfu = new LFUCache(2);
lfu.put(1, 1);   // cache=[1,_], cnt(1)=1
lfu.put(2, 2);   // cache=[2,1], cnt(2)=1, cnt(1)=1
lfu.get(1);      // return 1
                 // cache=[1,2], cnt(2)=1, cnt(1)=2
lfu.put(3, 3);   // 2 is the LFU key because cnt(2)=1 is the smallest, invalidate 2.
                 // cache=[3,1], cnt(3)=1, cnt(1)=2
lfu.get(2);      // return -1 (not found)
lfu.get(3);      // return 3
                 // cache=[3,1], cnt(3)=2, cnt(1)=2
lfu.put(4, 4);   // Both 1 and 3 have the same cnt, but 1 is LRU, invalidate 1.
                 // cache=[4,3], cnt(4)=1, cnt(3)=2
lfu.get(1);      // return -1 (not found)
lfu.get(3);      // return 3
                 // cache=[3,4], cnt(4)=1, cnt(3)=3
lfu.get(4);      // return 4
                 // cache=[3,4], cnt(4)=2, cnt(3)=3
 

Constraints:

0 <= capacity <= 104
0 <= key <= 105
0 <= value <= 109
At most 2 * 105 calls will be made to get and put.'''

import collections
class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = None
        self.prev = None
        self.cnt = 1
        
class DLL:
    def __init__(self):
        self.head = Node(0,0)
        self.tail = Node(0,0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
    
    def addfront(self, node):
        temp = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = temp
        temp.prev = node
        self.size += 1
        
    def delNode(self, node):
        delprev = node.prev
        delnext = node.next
        delprev.next = delnext
        delnext.prev = delprev
        self.size -= 1
    
    def removeTail(self):
        tail = self.tail.prev
        self.delNode(tail)
        return tail

class LFUCache:

    def __init__(self, capacity: int):
        self.cap = capacity
        self.minfreq = 0
        self.currfreq = 0
        self.keynode = {}
        self.freqmap = collections.defaultdict(DLL)
        
    def updateFreq(self, node, key, value):
        node = self.keynode[key]
        node.val = value
        prevcnt = node.cnt
        node.cnt += 1
        self.freqmap[prevcnt].delNode(node)
        self.freqmap[node.cnt].addfront(node)
        if prevcnt == self.minfreq and self.freqmap[prevcnt].size == 0:
            self.minfreq += 1
            
        return node.val
            
    def get(self, key: int) -> int:
        if key not in self.keynode:
            return -1
        return self.updateFreq(self.keynode[key], key, self.keynode[key].val)
        

    def put(self, key: int, value: int) -> None:
        if not self.cap:
            return
        if key in self.keynode:
            self.updateFreq(self.keynode[key], key, value)
        
        else:
            if len(self.keynode) == self.cap:
                prevnode = self.freqmap[self.minfreq].removeTail()
                del self.keynode[prevnode.key]
                
            node = Node(key,value)
            self.freqmap[1].addfront(node)
            self.keynode[key] = node
            self.minfreq = 1
                
            
        


# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

'''Your input
["LFUCache","put","put","get","put","get","get","put","get","get","get"]
[[2],[1,1],[2,2],[1],[3,3],[2],[3],[4,4],[1],[3],[4]]
Output
[null,null,null,1,null,-1,3,null,-1,3,4]
Expected
[null,null,null,1,null,-1,3,null,-1,3,4]'''

#OR

'''Approach:

Data Structures:

Frequency Table: A dictionary to store the mapping of different frequency values with values as DLLs storing (key, value) pairs as nodes
Cache Dicitionary: Nodes in the DLL are stored as values for each key pushed into the cache
Algorithm:

get(key):

If key is not present in cache, return -1
Get the node from the cache
Update the node frequency
Remove the node from the DLL of node's previous frequency
Add the node to the DLL with the node's updated frequency
Update min frequency value
put(key, value):

If key is present in cache
Similar logic to that of get function
Only difference being that we need to update the value here
If key not present in cache
If the cache has already reached it's capacity, delete the tail node from the DLL with least frequency
Create the new node with the (key, value) pair passed as arguments
Add the node to the frequency table with frequency key = 1
Add the node to the cache
Update min frequency to be 1'''

class ListNode:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.freq = 1
        self.prev = None
        self.next = None
        
class DLL:
    def __init__(self):
        self.head = ListNode(0, 0)
        self.tail = ListNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
        
    def insertHead(self, node):
        headNext = self.head.next
        headNext.prev = node
        self.head.next = node
        node.prev = self.head
        node.next = headNext
        self.size += 1
        
    def removeNode(self, node):
        node.next.prev = node.prev
        node.prev.next = node.next
        self.size -= 1
        
    def removeTail(self):
        tail = self.tail.prev
        self.removeNode(tail)
        return tail
    

class LFUCache:

    def __init__(self, capacity: int):
        self.cache = {}
        self.freqTable = collections.defaultdict(DLL)
        self.capacity = capacity
        self.minFreq = 0
        
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        return self.updateCache(self.cache[key], key, self.cache[key].val)
        

    def put(self, key: int, value: int) -> None:
        if not self.capacity:
            return
        if key in self.cache:
            self.updateCache(self.cache[key], key, value)
        else:
            if len(self.cache) == self.capacity:
                prevTail = self.freqTable[self.minFreq].removeTail()
                del self.cache[prevTail.key]
            node = ListNode(key, value)
            self.freqTable[1].insertHead(node)
            self.cache[key] = node
            self.minFreq = 1
        
    
    def updateCache(self, node, key, value):
        node = self.cache[key]
        node.val = value
        prevFreq = node.freq
        node.freq += 1
        self.freqTable[prevFreq].removeNode(node)
        self.freqTable[node.freq].insertHead(node)
        if prevFreq == self.minFreq and self.freqTable[prevFreq].size == 0:
            self.minFreq += 1
        return node.val

#https://leetcode.com/problems/lfu-cache/discuss/?currentPage=1&orderBy=most_votes&query=

'''Largest Rectangle in Histogram
Hard

6858

116

Add to List

Share
Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

 

Example 1:


Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The above is a histogram where width of each bar is 1.
The largest rectangle is shown in the red area, which has an area = 10 units.
Example 2:


Input: heights = [2,4]
Output: 4
 

Constraints:

1 <= heights.length <= 105
0 <= heights[i] <= 104'''

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        n = len(heights)
        lefti = [0] * n
        righti = [0] * n
        st =[]
        
        for i in range(n):
            while st and heights[st[-1]] >= heights[i]:
                st.pop()
            
            if st:
                lefti[i] = st[-1] + 1
            else:
                lefti[i] = 0
            
            st.append(i)
            
        print(lefti)
        
        while st != []:
            st.pop()
            
        for i in range(n-1, -1, -1):
            while st and heights[st[-1]] >= heights[i]:
                st.pop()
                
            if st:
                righti[i] = st[-1] - 1
            else:
                righti[i] = n-1
            
            st.append(i)
            
        print(righti)   
        maxA = 0
        
        for i in range(n):
            maxA = max(maxA, (heights[i] * (righti[i] - lefti[i] + 1)))
            
        return maxA

'''Your input
[2,1,5,6,2,3]
stdout
[0, 0, 2, 3, 2, 5]
[0, 5, 3, 3, 5, 5]

Output
10
Expected
10'''

#or

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        n = len(heights)
        st =[]
        maxA = 0
        for i in range(n+1):
            while st and (i == n or heights[st[-1]] >= heights[i]):
                H = heights[st.pop()]
                
                if st:
                    W = i - st[-1] -1
                else:
                    W = i
                    
                maxA = max(maxA, (W * H))
                
            st.append(i)
            
        return maxA

'''The stack maintain the indexes of buildings with ascending height. Before adding a new building pop the building who is taller than the new one.
 The building popped out represent the height of a rectangle with the new building as the right boundary and the current stack top as the left boundary.
 Calculate its area and update ans of maximum area. Boundary is handled using dummy buildings.'''

'''Sliding Window Maximum
Hard

6939

256

Add to List

Share
You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.

 

Example 1:

Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
Example 2:

Input: nums = [1], k = 1
Output: [1]
Example 3:

Input: nums = [1,-1], k = 1
Output: [1,-1]
Example 4:

Input: nums = [9,11], k = 2
Output: [11]
Example 5:

Input: nums = [4,-2], k = 2
Output: [4]
 

Constraints:

1 <= nums.length <= 105
-104 <= nums[i] <= 104
1 <= k <= nums.length
'''

def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    dq = []
    n = len(nums)
    ans = []
    for i in range(n):
        if dq and dq[0] == i-k:
            dq.pop(0)
            
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
            
        dq.append(i)
        
        if i >= k-1:
            ans.append(nums[dq[0]])
    return ans

'''Your input
[1,3,-1,-3,5,3,6,7]
3
Output
[3,3,5,5,6,7]
Expected
[3,3,5,5,6,7]'''

class Solution:
    def maxSlidingWindow(self, nums, k):
        d = collections.deque()
        out = []
        for i, n in enumerate(nums):
            while d and nums[d[-1]] < n:
                d.pop()
            d += i,
            if d[0] == i - k:
                d.popleft()
            if i >= k - 1:
                out += nums[d[0]],
        return out

'''Min Stack
Easy

5726

510

Add to List

Share
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:

MinStack() initializes the stack object.
void push(val) pushes the element val onto the stack.
void pop() removes the element on the top of the stack.
int top() gets the top element of the stack.
int getMin() retrieves the minimum element in the stack.
 

Example 1:

Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]

Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
 

Constraints:

-231 <= val <= 231 - 1
Methods pop, top and getMin operations will always be called on non-empty stacks.
At most 3 * 104 calls will be made to push, pop, top, and getMin.'''

class MinStack:

    def __init__(self):
        self.stack= []

    def push(self, x):
        """
        :type x: int
        :rtype: nothing
        """
        if not self.stack:self.stack.append((x,x)) 
        else:
            self.stack.append((x,min(x,self.stack[-1][1])))

    def pop(self):
        """
        :rtype: nothing
        """
        if self.stack: self.stack.pop()

    def top(self):
        """
        :rtype: int
        """
        if self.stack: return self.stack[-1][0]
        else: return None

    def getMin(self):
        """
        :rtype: int
        """
        if self.stack: return self.stack[-1][1]
        else: return None


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

'''Your input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]
Output
[null,null,null,null,-3,null,0,-2]
Expected
[null,null,null,null,-3,null,0,-2]'''


'''Rotting Oranges
Medium

4085

227

Add to List

Share
You are given an m x n grid where each cell can have one of three values:

0 representing an empty cell,
1 representing a fresh orange, or
2 representing a rotten orange.
Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return -1.

Example 1:

Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
Example 2:

Input: grid = [[2,1,1],[0,1,1],[1,0,1]]
Output: -1
Explanation: The orange in the bottom left corner (row 2, column 0) is never rotten, because rotting only happens 4-directionally.
Example 3:

Input: grid = [[0,2]]
Output: 0
Explanation: Since there are already no fresh oranges at minute 0, the answer is just 0.
 

Constraints:

m == grid.length
n == grid[i].length
1 <= m, n <= 10
grid[i][j] is 0, 1, or 2.'''

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        if grid == None:
            return 0
        
        n = len(grid)
        m = len(grid[0])
        
        q = []
        tlt = 0
        days = 0
        cnt = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] != 0:
                    tlt += 1
                if grid[i][j] == 2:
                    q.append([i,j])
                    
        di = [-1,1,0,0]
        dj = [0,0,-1,1]
        
        while q != []:
            k = len(q)
            print(q)
            cnt += k
            while k:
                k-=1
                first = q.pop(0)
                x = first[0]
                y = first[1]
                for i in range(4):
                    nx = x + di[i]
                    ny = y + dj[i]
                    if nx < 0 or ny < 0 or nx >= n or ny >= m or grid[nx][ny] != 1:
                        continue
                    grid[nx][ny] = 2
                    q.append([nx,ny])
                
            if q != []:
                days += 1
                
        if tlt == cnt:
            return days
        else: 
            return -1

'''Accepted
Runtime: 53 ms
Your input
[[2,1,1],[1,1,0],[0,1,1]]
stdout
[[0, 0]]
[[1, 0], [0, 1]]
[[1, 1], [0, 2]]
[[2, 1]]
[[2, 2]]

Output
4
Expected
4'''

#or

from collections import deque

# Time complexity: O(rows * cols) -> each cell is visited at least once
# Space complexity: O(rows * cols) -> in the worst case if all the oranges are rotten they will be added to the queue

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        
        # number of rows
        rows = len(grid)
        if rows == 0:  # check if grid is empty
            return -1
        
        # number of columns
        cols = len(grid[0])
        
        # keep track of fresh oranges
        fresh_cnt = 0
        
        # queue with rotten oranges (for BFS)
        rotten = deque()
        
        # visit each cell in the grid
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    # add the rotten orange coordinates to the queue
                    rotten.append((r, c))
                elif grid[r][c] == 1:
                    # update fresh oranges count
                    fresh_cnt += 1
        
        # keep track of minutes passed.
        minutes_passed = 0
        
        # If there are rotten oranges in the queue and there are still fresh oranges in the grid keep looping
        while rotten and fresh_cnt > 0:

            # update the number of minutes passed
            # it is safe to update the minutes by 1, since we visit oranges level by level in BFS traversal.
            minutes_passed += 1
            
            # process rotten oranges on the current level
            for _ in range(len(rotten)):
                x, y = rotten.popleft()
                
                # visit all the adjacent cells
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    # calculate the coordinates of the adjacent cell
                    xx, yy = x + dx, y + dy
                    # ignore the cell if it is out of the grid boundary
                    if xx < 0 or xx == rows or yy < 0 or yy == cols:
                        continue
                    # ignore the cell if it is empty '0' or visited before '2'
                    if grid[xx][yy] == 0 or grid[xx][yy] == 2:
                        continue
                        
                    # update the fresh oranges count
                    fresh_cnt -= 1
                    
                    # mark the current fresh orange as rotten
                    grid[xx][yy] = 2
                    
                    # add the current rotten to the queue
                    rotten.append((xx, yy))

        
        # return the number of minutes taken to make all the fresh oranges to be rotten
        # return -1 if there are fresh oranges left in the grid (there were no adjacent rotten oranges to make them rotten)
        return minutes_passed if fresh_cnt == 0 else -1

#https://leetcode.com/problems/rotting-oranges/discuss/?currentPage=1&orderBy=most_votes&query=

'''Online Stock Span
Medium

1764

183

Add to List

Share
Design an algorithm that collects daily price quotes for some stock and returns the span of that stock's price for the current day.

The span of the stock's price today is defined as the maximum number of consecutive days (starting from today and going backward) for which the stock price was less than or equal to today's price.

For example, if the price of a stock over the next 7 days were [100,80,60,70,60,75,85], then the stock spans would be [1,1,1,2,1,4,6].
Implement the StockSpanner class:

StockSpanner() Initializes the object of the class.
int next(int price) Returns the span of the stock's price given that today's price is price.
 

Example 1:

Input
["StockSpanner", "next", "next", "next", "next", "next", "next", "next"]
[[], [100], [80], [60], [70], [60], [75], [85]]
Output
[null, 1, 1, 1, 2, 1, 4, 6]

Explanation
StockSpanner stockSpanner = new StockSpanner();
stockSpanner.next(100); // return 1
stockSpanner.next(80);  // return 1
stockSpanner.next(60);  // return 1
stockSpanner.next(70);  // return 2
stockSpanner.next(60);  // return 1
stockSpanner.next(75);  // return 4, because the last 4 prices (including today's price of 75) were less than or equal to today's price.
stockSpanner.next(85);  // return 6
 

Constraints:

1 <= price <= 105
At most 104 calls will be made to next.'''

class StockSpanner:

    def __init__(self):
        self.st = []
        

    def next(self, price: int) -> int:
        res = 1
        while self.st and self.st[-1][0] <= price:
            res += self.st.pop()[1]
        self.st.append([price,res])
        return res


# Your StockSpanner object will be instantiated and called as such:
# obj = StockSpanner()
# param_1 = obj.next(price)

'''Your input
["StockSpanner","next","next","next","next","next","next","next"]
[[],[100],[80],[60],[70],[60],[75],[85]]
Output
[null,1,1,1,2,1,4,6]
Expected
[null,1,1,1,2,1,4,6]'''

#OR

class StockSpanner:

    def __init__(self):
        self.prices = []
        self.spans = []

    def next(self, price: int) -> int:
        span = 1
        index = len(self.spans) - 1
        while index >= 0 and price >= self.prices[index]:
            span += self.spans[index]
            index -= self.spans[index]
        self.spans.append(span)
        self.prices.append(price)
        return span

'''Maximum of minimum for every window size 
Hard Accuracy: 55.24% Submissions: 8237 Points: 8
Given an integer array. The task is to find the maximum of the minimum of every window size in the array.
Note: Window size varies from 1 to the size of the Array.

Example 1:

Input:
N = 7
arr[] = {10,20,30,50,10,70,30}
Output: 70 30 20 10 10 10 10 
Explanation: First element in output
indicates maximum of minimums of all
windows of size 1. Minimums of windows
of size 1 are {10}, {20}, {30}, {50},
{10}, {70} and {30}. Maximum of these
minimums is 70. 
Second element in output indicates
maximum of minimums of all windows of
size 2. Minimums of windows of size 2
are {10}, {20}, {30}, {10}, {10}, and
{30}. Maximum of these minimums is 30 
Third element in output indicates
maximum of minimums of all windows of
size 3. Minimums of windows of size 3
are {10}, {20}, {10}, {10} and {10}.
Maximum of these minimums is 20. 
Similarly other elements of output are
computed.
Example 2:

Input:
N = 3
arr[] = {10,20,30}
Output: 30 20 10
Explanation: First element in output
indicates maximum of minimums of all
windows of size 1.Minimums of windows
of size 1 are {10} , {20} , {30}.
Maximum of these minimums are 30 and
similarly other outputs can be computed
Your Task:
The task is to complete the function maxOfMin() which takes the array arr[] and its size N as inputs and finds the maximum of minimum of every window size and returns an array containing the result. 

Expected Time Complxity : O(N)
Expected Auxilliary Space : O(N)

Constraints:
1 <= N <= 105
1 <= arr[i] <= 106'''


# An efficient Python3 program to find
# maximum of all minimums of windows of 
# different sizes

def printMaxOfMin(arr, n):
    
    s = [] # Used to find previous 
           # and next smaller 

    # Arrays to store previous and next 
    # smaller. Initialize elements of 
    # left[] and right[]
    left = [-1] * (n + 1) 
    right = [n] * (n + 1) 

    # Fill elements of left[] using logic discussed on 
    # https:#www.geeksforgeeks.org/next-greater-element
    for i in range(n):
        while (len(s) != 0 and 
               arr[s[-1]] >= arr[i]): 
            s.pop() 

        if (len(s) != 0):
            left[i] = s[-1]

        s.append(i)

    # Empty the stack as stack is going 
    # to be used for right[] 
    while (len(s) != 0):
        s.pop()

    # Fill elements of right[] using same logic
    for i in range(n - 1, -1, -1):
        while (len(s) != 0 and arr[s[-1]] >= arr[i]): 
            s.pop() 

        if(len(s) != 0): 
            right[i] = s[-1] 

        s.append(i)

    # Create and initialize answer array 
    ans = [0] * (n + 1)
    for i in range(n + 1):
        ans[i] = 0

    # Fill answer array by comparing minimums 
    # of all. Lengths computed using left[] 
    # and right[]
    for i in range(n):
        
        # Length of the interval 
        Len = right[i] - left[i] - 1

        # arr[i] is a possible answer for this
        #  Length 'Len' interval, check if arr[i] 
        # is more than max for 'Len' 
        ans[Len] = max(ans[Len], arr[i])

    # Some entries in ans[] may not be filled 
    # yet. Fill them by taking values from
    # right side of ans[]
    for i in range(n - 1, 0, -1):
        ans[i] = max(ans[i], ans[i + 1]) 

    # Print the result
    for i in range(1, n + 1):
        print(ans[i], end = " ")

# Driver Code
if __name__ == '__main__':

    arr = [10, 20, 30, 50, 10, 70, 30] 
    n = len(arr) 
    printMaxOfMin(arr, n)

# This code is contributed by PranchalK


'''For Input:
7
10 20 30 50 10 70 30

Your Output is: 
70 30 20 10 10 10 10 '''

''''''

#O(N)

'''The Celebrity Problem 
Medium Accuracy: 39.46% Submissions: 82153 Points: 4
A celebrity is a person who is known to all but does not know anyone at a party. If you go to a party of N people, find if there is a celebrity in the party or not.
A square NxN matrix M[][] is used to represent people at the party such that if an element of row i and column j  is set to 1 it means ith person knows jth person. Here M[i][i] will always be 0.
Note: Follow 0 based indexing.
 

Example 1:

Input:
N = 3
M[][] = {{0 1 0},
         {0 0 0}, 
         {0 1 0}}
Output: 1
Explanation: 0th and 2nd person both
know 1. Therefore, 1 is the celebrity. 

Example 2:

Input:
N = 2
M[][] = {{0 1},
         {1 0}}
Output: -1
Explanation: The two people at the party both
know each other. None of them is a celebrity.

Your Task:
You don't need to read input or print anything. Complete the function celebrity() which takes the matrix M and its size N as input parameters and returns the index of the celebrity. If no such celebrity is present, return -1.


Expected Time Complexity: O(N)
Expected Auxiliary Space: O(1)


Constraints:
2 <= N <= 3000
0 <= M[][] <= 1'''


class Solution:
    def know(self, a, b, M):
        return M[a][b]
    #Function to find if there is a celebrity in the party or not.
    def celebrity(self, M, n):
        # code here 
        st =[]
        for i in range(n):
            st.append(i)
            
        count = 0
        while count < n-1:
            first = st.pop()
            second = st.pop()
            
            if self.know(first, second, M):
                st.append(second)
            else:
                st.append(first)
                
            count += 1
        if st == []:
            return -1
        celeb = st.pop()  
        
        for i in range(n):
            if i != celeb and (self.know(celeb, i, M) or self.know(i, celeb, M) != True):
                return -1
    
        return celeb

'''For Input:
3
0 1 0 0 0 0 0 1 0

Your Output is: 
1'''

#https://www.geeksforgeeks.org/the-celebrity-problem/

'''Reverse Words in a String
Medium

1965

3393

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
Output: "bob like even not does Alice"
 

Constraints:

1 <= s.length <= 104
s contains English letters (upper-case and lower-case), digits, and spaces ' '.
There is at least one word in s.
 

Follow-up: If the string data type is mutable in your language, can you solve it in-place with O(1) extra space?'''

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

'''Your input
"the sky is blue"
Output
"blue is sky the"
Expected
"blue is sky the"'''

'''Longest Palindromic Substring
Medium

12769

772

Add to List

Share
Given a string s, return the longest palindromic substring in s.

 

Example 1:

Input: s = "babad"
Output: "bab"
Note: "aba" is also a valid answer.
Example 2:

Input: s = "cbbd"
Output: "bb"
Example 3:

Input: s = "a"
Output: "a"
Example 4:

Input: s = "ac"
Output: "a"
 

Constraints:

1 <= s.length <= 1000
s consist of only digits and English letters.'''

class Solution:
    def longestPalindrome(self, s: str) -> str:
        res = ""
        for i in range(len(s)):
            tmp = self.helpher(s, i, i)
            if len(tmp) > len(res):
                res = tmp
                
            tmp = self.helpher(s, i, i+1)
            if len(tmp) > len(res):
                res = tmp
                
        return res
    def helpher(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l-=1
            r+=1
        
        return s[l+1:r]

'''Your input
"babad"
Output
"bab"
Expected
"bab"'''

#or

class Solution:
    def longestPalindrome(self, s: str) -> str:
        longestPalSub = ''
        for i in range(len(s)):
            center = self.expandAroundCenter(s, i, i)
            inBetween = self.expandAroundCenter(s, i, i+1)
            longestPalSub = max(longestPalSub, center, inBetween, key = len)
        return longestPalSub
    
    def expandAroundCenter(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1 : right]

'''13. Roman to Integer
Easy

1506

119

Add to List

Share
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two one's added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.

 

Example 1:

Input: s = "III"
Output: 3
Example 2:

Input: s = "IV"
Output: 4
Example 3:

Input: s = "IX"
Output: 9
Example 4:

Input: s = "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.
Example 5:

Input: s = "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
 

Constraints:

1 <= s.length <= 15
s contains only the characters ('I', 'V', 'X', 'L', 'C', 'D', 'M').
It is guaranteed that s is a valid roman numeral in the range [1, 3999].'''

class Solution:
    def romanToInt(self, s: str) -> int:
        roman = {'I': 1, 'V':5,'X':10, 'L':50, 'C':100,'D':500,'M':1000}
        
        res = 0
        prev = ''
        for i in s:
            res += roman[i]
            if (prev == 'I') and (i == 'V' or i == 'X'):
                res-=2
            elif (prev == 'X') and (i == 'L' or i == 'C'):
                res-=20
            elif (prev == 'C') and (i == 'D' or i == 'M'):
                res-=200
            prev = i
        return res

'''Your input
"III"
Output
3
Expected
3'''

#OR

class Solution:
    def romanToInt(self, s: str) -> int:
        dict = {'I':1, 
                'V': 5,
                'X' :10,
                'L' :50,
                'C': 100,
                'D': 500,
                'M': 1000}
        
        number = 0
        s = s.replace("IV", "IIII").replace("IX", "VIIII")
        s = s.replace("XL", "XXXX").replace("XC", "LXXXX")
        s = s.replace("CD", "CCCC").replace("CM", "DCCCC")
        for i in s:
            number += dict[i]
            
        return number

'''Implement Atoi 
Medium Accuracy: 32.9% Submissions: 77054 Points: 4
Your task  is to implement the function atoi. The function takes a string(str) as argument and converts it to an integer and returns it.

Note: You are not allowed to use inbuilt function.

Example 1:

Input:
str = 123
Output: 123
Example 2:

Input:
str = 21a
Output: -1
Explanation: Output is -1 as all
characters are not digit only.
Your Task:
Complete the function atoi() which takes a string as input parameter and returns integer value of it. if the input string is not a numerical string then returns -1.

Expected Time Complexity: O(|S|), |S| = length of string str.
Expected Auxiliary Space: O(1)
'''

def isnumeric(self, x):
        if (x >= '0' and  x <= '9'):
            return True
        return False
        
def atoi(self,string):
    # Code here
    res = 0
    sign = 1
    i = 0
    
    if string[0] == '-':
        sign = -1
        i+=1
        
    for j in range(i, len(string)):
        
        if self.isnumeric(string[j]) == False:
            return -1
        
        res = res * 10 + (ord(string[j]) - ord('0'))
        
    return sign * res

'''For Input:
123

Your Output is: 
123'''

#https://www.geeksforgeeks.org/write-your-own-atoi/

'''Longest Common Prefix
Easy

5126

2425

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
Explanation: There is no common prefix among the input strings.
 

Constraints:

1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] consists of only lower-case English letters.'''

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

'''Your input
["flower","flow","flight"]
Output
"fl"
Expected
"fl"'''

#https://leetcode.com/problems/longest-common-prefix/discuss/?currentPage=1&orderBy=most_votes&query=

'''Repeated String Match
Medium

1139

861

Add to List

Share
Given two strings a and b, return the minimum number of times you should repeat string a so that string b is a substring of it. If it is impossible for b​​​​​​ to be a substring of a after repeating it, return -1.

Notice: string "abc" repeated 0 times is "",  repeated 1 time is "abc" and repeated 2 times is "abcabc".

 

Example 1:

Input: a = "abcd", b = "cdabcdab"
Output: 3
Explanation: We return 3 because by repeating a three times "abcdabcdabcd", b is a substring of it.
Example 2:

Input: a = "a", b = "aa"
Output: 2
Example 3:

Input: a = "a", b = "a"
Output: 1
Example 4:

Input: a = "abc", b = "wxyz"
Output: -1
 

Constraints:

1 <= a.length <= 104
1 <= b.length <= 104
a and b consist of lower-case English letters.'''

class Solution:
    def repeatedStringMatch(self, a: str, b: str) -> int:
        res = ''
        cnt = 0
        while len(res) < len(b):
            res += a
            cnt += 1
            if b in res:
                return cnt
        res += a
        if b in res:
            return cnt+1
        return -1

'''Your input
"abcd"
"cdabcdab"
Output
3
Expected
3'''

#https://www.geeksforgeeks.org/rabin-karp-algorithm-for-pattern-searching/

'''KMP Algorithm for Pattern Searching'''

def KMPSearch(pat, txt):
    M = len(pat)
    N = len(txt)
  
    # create lps[] that will hold the longest prefix suffix 
    # values for pattern
    lps = [0]*M
    j = 0 # index for pat[]
  
    # Preprocess the pattern (calculate lps[] array)
    computeLPSArray(pat, M, lps)
  
    i = 0 # index for txt[]
    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1
  
        if j == M:
            print ("Found pattern at index " + str(i-j))
            j = lps[j-1]
  
        # mismatch after j matches
        elif i < N and pat[j] != txt[i]:
            # Do not match lps[0..lps[j-1]] characters,
            # they will match anyway
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
  
def computeLPSArray(pat, M, lps):
    len = 0 # length of the previous longest prefix suffix
  
    lps[0] # lps[0] is always 0
    i = 1
  
    # the loop calculates lps[i] for i = 1 to M-1
    while i < M:
        if pat[i]== pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            # This is tricky. Consider the example.
            # AAACAAAA and i = 7. The idea is similar 
            # to search step.
            if len != 0:
                len = lps[len-1]
  
                # Also, note that we do not increment i here
            else:
                lps[i] = 0
                i += 1
  
txt = "ABABDABACDABABCABAB"
pat = "ABABCABAB"
KMPSearch(pat, txt)

'''Minimum characters to be added at front to make string palindrome
Difficulty Level : Hard
Last Updated : 08 Mar, 2021
Given a string str we need to tell minimum characters to be added at front of string to make string palindrome.'''


#Found pattern at index 10

# Python3 program for getting minimum
# character to be added at the front
# to make string palindrome
 
# Returns vector lps for given string str
def computeLPSArray(string):
 
    M = len(string)
    lps = [None] * M
 
    length = 0
    lps[0] = 0 # lps[0] is always 0
 
    # the loop calculates lps[i]
    # for i = 1 to M-1
    i = 1
    while i < M:
     
        if string[i] == string[length]:
         
            length += 1
            lps[i] = length
            i += 1
         
        else: # (str[i] != str[len])
         
            # This is tricky. Consider the example.
            # AAACAAAA and i = 7. The idea is
            # similar to search step.
            if length != 0:
             
                length = lps[length - 1]
 
                # Also, note that we do not
                # increment i here
             
            else: # if (len == 0)
             
                lps[i] = 0
                i += 1
 
    return lps
 
# Method returns minimum character
# to be added at front to make
# string palindrome
def getMinCharToAddedToMakeStringPalin(string):
 
    revStr = string[::-1]
 
    # Get concatenation of string,
    # special character and reverse string
    concat = string + "$" + revStr
 
    # Get LPS array of this
    # concatenated string
    lps = computeLPSArray(concat)
 
    # By subtracting last entry of lps
    # vector from string length, we
    # will get our result
    return len(string) - lps[-1]
 
# Driver Code
if __name__ == "__main__":
 
    string = "AACECAAAA"
    print(getMinCharToAddedToMakeStringPalin(string))
#2

'''Anagram 
Easy Accuracy: 50.99% Submissions: 45235 Points: 2
Given two strings a and b consisting of lowercase characters. The task is to check whether two given strings are an anagram of each other or not. An anagram of a string is another string that contains the same characters, only the order of characters can be different. For example, “act” and “tac” are an anagram of each other.

Example 1:

Input:
a = geeksforgeeks, b = forgeeksgeeks
Output: YES
Explanation: Both the string have same
characters with same frequency. So, 
both are anagrams.
Example 2:

Input:
a = allergy, b = allergic
Output: NO
Explanation:Characters in both the strings
are not same, so they are not anagrams.
Your Task:
You don't need to read input or print anything.Your task is to complete the function isAnagram() which takes the string a and string b as input parameter and check if the two strings are an anagram of each other. The function returns true if the strings are anagram else it returns false.

Expected Time Complexity: O(|a|+|b|).
Expected Auxiliary Space: O(Number of distinct characters).

Note: |s| represents the length of string s.

Constraints:
1 ≤ |a|,|b| ≤ 105'''

class Solution:
    
    #Function is to check whether two strings are anagram of each other or not.
    def isAnagram(self,a,b):
        #code here
        
        n1 = len(a)
        n2 = len(b)
        
        temp = ''
        
        if n1 != n2:
            return False
            
        temp = a+a
        
        if temp.count(b) > 0:
            return True
            
        else:
            return False

'''For Input:
geeksforgeeks forgeeksgeeks

Your Output is: 
YES'''

#OR

# Python program to check whether two strings are
# anagrams of each other
 
# function to check whether two strings are anagram
# of each other
 
 
def areAnagram(str1, str2):
    # Get lengths of both strings
    n1 = len(str1)
    n2 = len(str2)
 
    # If lenght of both strings is not same, then
    # they cannot be anagram
    if n1 != n2:
        return 0
 
    # Sort both strings
    str1 = sorted(str1)
    str2 = sorted(str2)
 
    # Compare sorted strings
    for i in range(0, n1):
        if str1[i] != str2[i]:
            return 0
 
    return 1
 
 
# Driver code
str1 = "test"
str2 = "ttew"
 
# Function Call
if areAnagram(str1, str2):
    print("The two strings are anagram of each other")
else:
    print("The two strings are not anagram of each other")


'''Count and Say
Medium

742

2199

Add to List

Share
The count-and-say sequence is a sequence of digit strings defined by the recursive formula:

countAndSay(1) = "1"
countAndSay(n) is the way you would "say" the digit string from countAndSay(n-1), which is then converted into a different digit string.
To determine how you "say" a digit string, split it into the minimal number of groups so that each group is a contiguous section all of the same character. Then for each group, say the number of characters, then say the character. To convert the saying into a digit string, replace the counts with a number and concatenate every saying.

For example, the saying and conversion for digit string "3322251":


Given a positive integer n, return the nth term of the count-and-say sequence.

 

Example 1:

Input: n = 1
Output: "1"
Explanation: This is the base case.
Example 2:

Input: n = 4
Output: "1211"
Explanation:
countAndSay(1) = "1"
countAndSay(2) = say "1" = one 1 = "11"
countAndSay(3) = say "11" = two 1's = "21"
countAndSay(4) = say "21" = one 2 + one 1 = "12" + "11" = "1211"'''

class Solution:
    def countAndSay(self, n):
        s = '1'
        for _ in range(n-1):
            let, temp, count = s[0], '', 0
            for l in s:
                if let == l:
                    count += 1
                else:
                    temp += str(count)+let
                    let = l
                    count = 1
            temp += str(count)+let
            s = temp
        return s
        
'''Your input
1
Output
"1"
Expected
"1"'''

'''Compare Version Numbers
Medium

897

1764

Add to List

Share
Given two version numbers, version1 and version2, compare them.

Version numbers consist of one or more revisions joined by a dot '.'. Each revision consists of digits and may contain leading zeros. Every revision contains at least one character. Revisions are 0-indexed from left to right, with the leftmost revision being revision 0, the next revision being revision 1, and so on. For example 2.5.33 and 0.1 are valid version numbers.

To compare version numbers, compare their revisions in left-to-right order. Revisions are compared using their integer value ignoring any leading zeros. This means that revisions 1 and 001 are considered equal. If a version number does not specify a revision at an index, then treat the revision as 0. For example, version 1.0 is less than version 1.1 because their revision 0s are the same, but their revision 1s are 0 and 1 respectively, and 0 < 1.

Return the following:

If version1 < version2, return -1.
If version1 > version2, return 1.
Otherwise, return 0.
 

Example 1:

Input: version1 = "1.01", version2 = "1.001"
Output: 0
Explanation: Ignoring leading zeroes, both "01" and "001" represent the same integer "1".
Example 2:

Input: version1 = "1.0", version2 = "1.0.0"
Output: 0
Explanation: version1 does not specify revision 2, which means it is treated as "0".
Example 3:

Input: version1 = "0.1", version2 = "1.1"
Output: -1
Explanation: version1's revision 0 is "0", while version2's revision 0 is "1". 0 < 1, so version1 < version2.
Example 4:

Input: version1 = "1.0.1", version2 = "1"
Output: 1
Example 5:

Input: version1 = "7.5.2.4", version2 = "7.5.3"
Output: -1
 

Constraints:

1 <= version1.length, version2.length <= 500
version1 and version2 only contain digits and '.'.
version1 and version2 are valid version numbers.
All the given revisions in version1 and version2 can be stored in a 32-bit integer.'''

class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        v1 = version1.split('.')
        v2 = version2.split('.')
        
        while v1 or v2:
            v1val = 0
            v2val = 0
            
            if v1:
                v1val = int(v1.pop(0))
                
            if v2:
                v2val = int(v2.pop(0))
                
            if v1val > v2val:
                return 1
            if v2val > v1val:
                return -1
            
        return 0

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
'''

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
        mid = (left + right) / 2
 
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
 
    return res
 
 
# Driver code
if __name__ == '__main__':
    arr = [1, 2, 8, 4, 9]
    n = len(arr)
    k = 3
    print(largestMinDist(arr, n, k))
 

'''Word Boggle 
Medium Accuracy: 48.98% Submissions: 15861 Points: 4
Given a dictionary of distinct words and an M x N board where every cell has one character. Find all possible words from the dictionary that can be formed by a sequence of adjacent characters on the board. We can move to any of 8 adjacent characters, but a word should not have multiple instances of the same cell.


Example 1:

Input: 
N = 1
dictionary = {"CAT"}
R = 3, C = 3
board = {{C,A,P},{A,N,D},{T,I,E}}
Output:
CAT
Explanation: 
C A P
A N D
T I E
Words we got is denoted using same color.
Example 2:

Input:
N = 4
dictionary = {"GEEKS","FOR","QUIZ","GO"}
R = 3, C = 3 
board = {{G,I,Z},{U,E,K},{Q,S,E}}
Output:
GEEKS QUIZ
Explanation: 
G I Z
U E K
Q S E 
Words we got is denoted using same color.

Your task:
You don’t need to read input or print anything. Your task is to complete the function wordBoggle() which takes the dictionary contaning N space-separated strings and R*C board as input parameters and returns a list of words that exist on the board in lexicographical order.


Expected Time Complexity: O(N*W + R*C^2)
Expected Auxiliary Space: O(N*W + R*C)


Constraints:
1 ≤ N ≤ 15
1 ≤ R, C ≤ 50
1 ≤ length of Word ≤ 60
Each word can consist of both lowercase and uppercase letters.'''

class Solution:
    def searchBoggle(self,board, words, processed, i, j,M,N, path=""):
 
        # mark the current node as processed
        processed[i][j] = True
     
        # update the path with the current character and insert it into the set
        path = path + board[i][j]
        words.add(path)
     
        # check for all eight possible movements from the current cell
        for k in range(8):
            row = [-1, -1, -1, 0, 1, 0, 1, 1]
            col = [-1, 1, 0, -1, -1, 1, 0, 1]
            # skip if a cell is invalid, or it is already processed
            ni = i + row[k]
            nj = j + col[k]
            if ni >= 0 and nj >= 0 and ni < M and nj < N and processed[ni][nj] == False:
                self.searchBoggle(board, words, processed, ni, nj,M,N, path)
     
        # backtrack: mark the current node as unprocessed
        processed[i][j] = False
     
 
# Function to search for a given set of words in a boggle
    def wordBoggle(self,board, input):
        M = len(board)
        N = len(board[0])
        # construct a matrix to store whether a cell is processed or not
        processed = [[False for x in range(N)] for y in range(M)]
     
        # construct a set to store all possible words constructed from the matrix
        words = set()
     
        # generate all possible words in a boggle
        for i in range(M):
            for j in range(N):
                # consider each character as a starting point and run DFS
                self.searchBoggle(board, words, processed, i, j,M,N)
     
        # for each word in the input list, check whether it is present in the set
        return ([word for word in input if word in words])

'''['QUIZ', 'GEEKS']'''

'''Maximum Profit in Job Scheduling
Hard

1763

19

Add to List

Share
We have n jobs, where every job is scheduled to be done from startTime[i] to endTime[i], obtaining a profit of profit[i].

You're given the startTime, endTime and profit arrays, return the maximum profit you can take such that there are no two jobs in the subset with overlapping time range.

If you choose a job that ends at time X you will be able to start another job that starts at time X.

Example 1:

Input: startTime = [1,2,3,3], endTime = [3,4,5,6], profit = [50,10,40,70]
Output: 120
Explanation: The subset chosen is the first and fourth job. 
Time range [1-3]+[3-6] , we get profit of 120 = 50 + 70.
Example 2:

Input: startTime = [1,2,3,4,6], endTime = [3,5,10,6,9], profit = [20,20,100,70,60]
Output: 150
Explanation: The subset chosen is the first, fourth and fifth job. 
Profit obtained 150 = 20 + 70 + 60.
Example 3:

Input: startTime = [1,1,1], endTime = [2,3,4], profit = [5,6,4]
Output: 6
 

Constraints:

1 <= startTime.length == endTime.length == profit.length <= 5 * 104
1 <= startTime[i] < endTime[i] <= 109
1 <= profit[i] <= 104'''

class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        n = len(startTime)
        
        jobs = []
        for i in range(n):
            jobs.append([startTime[i], endTime[i], profit[i]])
            
        jobs.sort(key = lambda x : x[1])
        
        dp = [0] * n
        dp[0] = jobs[0][2]
        
        for i in range(1,n):
            inc = jobs[i][2]
            l = self.binary(jobs,i)
            if l != -1:
                inc += dp[l]
            dp[i] = max(inc, dp[i-1])
        
        return dp[n-1]
    
    def binary(self, jobs, i):
        low = 0
        high = i-1
        while low <= high:
            mid = (low + high)//2
            
            if jobs[mid][1] <= jobs[i][0]:
                if jobs[mid+1][1] <= jobs[i][0]:
                    low = mid + 1
                else:
                    return mid
            else:
                high = mid - 1
                
        return -1

'''Your input
[1,2,3,3]
[3,4,5,6]
[50,10,40,70]
Output
120
Expected
120'''

#OR

class Solution(object):
    def jobScheduling(self, startTime, endTime, profit):
        """
        :type startTime: List[int]
        :type endTime: List[int]
        :type profit: List[int]
        :rtype: int
        """
        # O(nlog(n))
        lst = sorted(zip(startTime, endTime, profit), key = lambda x: x[1])
        dpEndTime = [0]
        dpProfit = [0]
        
        for start, end, pro in lst:
            # find rightMost idx to insert this start time
            # idx is where this new start needs to be inserted
            # idx - 1 is the one that doesn't overlap
            idx = self.bSearch(dpEndTime, start)
            lastProfit = dpProfit[-1]
            currProfit = dpProfit[idx-1] + pro # they don't overlap
            
            # whener we find currProfit greater than last, we update
            if currProfit > lastProfit:
                dpEndTime.append(end)
                dpProfit.append(currProfit)
        
        return dpProfit[-1]
            
    
    def bSearch(self, dp, target):
        left, right = 0, len(dp)
        
        while left < right:
            mid = (left + right)/2
            
            if dp[mid] <= target:
                left = mid + 1
            else:
                right = mid
        
        return left

'''Palindrome Partitioning II
Hard

2505

68

Add to List

Share
Given a string s, partition s such that every substring of the partition is a palindrome.

Return the minimum cuts needed for a palindrome partitioning of s.

 

Example 1:

Input: s = "aab"
Output: 1
Explanation: The palindrome partitioning ["aa","b"] could be produced using 1 cut.
Example 2:

Input: s = "a"
Output: 0
Example 3:

Input: s = "ab"
Output: 1
 

Constraints:

1 <= s.length <= 2000
s consists of lower-case English letters only.'''

class Solution:
    def minCut(self, s: str) -> int:
        # acceleration
        if s == s[::-1]: return 0
        for i in range(1, len(s)):
            if s[:i] == s[:i][::-1] and s[i:] == s[i:][::-1]:
                return 1
        # algorithm
        cut = [x for x in range(-1,len(s))]  # cut numbers in worst case (no palindrome)
        for i in range(len(s)):
            r1, r2 = 0, 0
            # use i as origin, and gradually enlarge radius if a palindrome exists
            # odd palindrome
            while i-r1 >= 0 and i+r1 < len(s) and s[i-r1] == s[i+r1]:
                cut[i+r1+1] = min(cut[i+r1+1], cut[i-r1]+1)
                r1 += 1
            # even palindrome
            while i-r2 >= 0 and i+r2+1 < len(s) and s[i-r2] == s[i+r2+1]:
                cut[i+r2+2] = min(cut[i+r2+2], cut[i-r2]+1)
                r2 += 1
        return cut[-1]

'''Your input
"aab"
Output
1
Expected
1'''

'''Egg Dropping Puzzle 
Medium Accuracy: 54.38% Submissions: 24069 Points: 4
You are given N identical eggs and you have access to a K-floored building from 1 to K.

There exists a floor f where 0 <= f <= K such that any egg dropped at a floor higher than f will break, and any egg dropped at or below floor f will not break. There are few rules given below. 

An egg that survives a fall can be used again.
A broken egg must be discarded.
The effect of a fall is the same for all eggs.
If the egg doesn't break at a certain floor, it will not break at any floor below.
If the eggs breaks at a certain floor, it will break at any floor above.
Return the minimum number of moves that you need to determine with certainty what the value of f is.

For more description on this problem see wiki page

Example 1:

Input:
N = 1, K = 2
Output: 2
Explanation: 
1. Drop the egg from floor 1. If it 
   breaks, we know that f = 0.
2. Otherwise, drop the egg from floor 2.
   If it breaks, we know that f = 1.
3. If it does not break, then we know f = 2.
4. Hence, we need at minimum 2 moves to
   determine with certainty what the value of f is.
Example 2:

Input:
N = 2, K = 10
Output: 4
Your Task:
Complete the function eggDrop() which takes two positive integer N and K as input parameters and returns the minimum number of attempts you need in order to find the critical floor.

Expected Time Complexity : O(N*K)
Expected Auxiliary Space: O(N*K)

Constraints:
1<=N<=200
1<=K<=200'''

def eggDrop(self,n, k):
    # code here
    if k == 0 or k ==1:
        return k
    
    if n == 1:
        return k
        
    min = float('inf')
    for i in range(1, k+1):
        res = max(self.eggDrop(n-1, i-1), self.eggDrop(n, k-i))
        
        if res < min:
            min = res
            
    return min+1

'''Cutting a Rod | DP-13
Difficulty Level : Medium
Last Updated : 10 Aug, 2021
 
Given a rod of length n inches and an array of prices that includes prices of all pieces of size smaller than n. Determine the maximum value obtainable by cutting up the rod and selling the pieces. For example, if the length of the rod is 8 and the values of different pieces are given as the following, then the maximum obtainable value is 22 (by cutting in two pieces of lengths 2 and 6) 

length   | 1   2   3   4   5   6   7   8  
--------------------------------------------
price    | 1   5   8   9  10  17  17  20'''


# A Dynamic Programming solution for Rod cutting problem
INT_MIN = -32767
 
# Returns the best obtainable price for a rod of length n and
# price[] as prices of different pieces
def cutRod(price, n):
    val = [0 for x in range(n+1)]
    val[0] = 0
 
    # Build the table val[] in bottom up manner and return
    # the last entry from the table
    for i in range(1, n+1):
        max_val = INT_MIN
        for j in range(i):
             max_val = max(max_val, price[j] + val[i-j-1])
        val[i] = max_val
 
    return val[n]


"""
Given an array arr[] of size N, check if it can be partitioned into two parts such that the sum of elements in both parts is the same.
Example 1:
Input: N = 4
arr = {1, 5, 11, 5}
Output: YES
Explaination: 
The two parts are {1, 5, 5} and {11}.
"""
def equalPartition(self, n, arr):
    # code here
    s = sum(arr)
    if s%2 != 0:
        return 0
    else:
        target = s//2
        dp = [[False for i in range(target + 1)]for i in range(n+1)]
        
        for i in range(n+1):
            for j in range(target+1):
                if i == 0:
                    dp[i][j] = False
                if j == 0:
                    dp[i][j] = True
                else:
                    if j >= arr[i-1]:
                        dp[i][j] = dp[i-1][j] or dp[i-1][j-arr[i-1]]
                    else:
                        dp[i][j] = dp[i-1][j]
        return int(dp[n][target])
"""
TC = o(n*s)
SC = o(n*s)
"""

"""
You are given coins of different denominations and a total amount of money amount.
Write a function to compute the fewest number of coins that you need to make up that amount.
If that amount of money cannot be made up by any combination of the coins, return -1.
You may assume that you have an infinite number of each kind of coin.
 
Example 1:
Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
"""
def coinChange(self, arr, s: int) -> int:
    n = len(arr)
    dp = [[0 for i in range(s+1)]for i in range(n+1)]
    for i in range(s+1):
        dp[0][i] = float("inf") # Because with 0 coins you can not make any number
    for i in range(1,s+1):
        if i%arr[0]==0:
            dp[1][i] = i//arr[0] # With 1 coin you either can divide with it or not
        else:
            dp[1][i] = float('inf')
    for i in range(2,n+1):
        for j in range(1,s+1):
            if j >= arr[i-1]:
                dp[i][j] = min(dp[i-1][j],1+dp[i][j-arr[i-1]]) # To calculate min. number
            else:
                dp[i][j] = dp[i-1][j]
    if dp[n][s] == float('inf'):
        return -1
    return dp[n][s]
"""
TC = o(n*s)
SC = o(n*s)
"""

"""
Given an n*m matrix, the task is to find the maximum sum of elements of cells starting from the cell (0, 0) to cell (n-1, m-1). 
However, the allowed moves are right, downwards or diagonally right, i.e, from location (i, j) next move can be (i+1, j), or, (i, j+1), or (i+1, j+1). Find the maximum sum of elements satisfying the allowed moves.
Examples: 
Input:
mat[][] = {{100, -350, -200},
           {-100, -300, 700}}
Output: 500
Explanation: 
Path followed is 100 -> -300 -> 700
"""
def maximum_path_sum(mat,vis,dp,i,j):
    n = len(mat)
    m = len(mat[0])
    if i == n-1 and j == m - 1:
        dp[i][j] = mat[i][j]
        return dp[i][j]
    if vis[i][j]:
        return dp[i][j]
    vis[i][j] = 1
    total_sum = dp[i][j]

    if i < n-1 and j < m-1:
        currentsum = max(maximum_path_sum(mat,vis,dp,i,j+1),maximum_path_sum(mat,vis,dp,i+1,j+1),maximum_path_sum(mat,vis,dp,i+1,j))
        total_sum = currentsum + mat[i][j]
        dp = total_sum
    elif i == n-1:
        total_sum = mat[i][j] + maximum_path_sum(mat,vis,dp,i,j+1)
        dp[i][j] = total_sum
    else:
        total_sum = mat[i][j] + maximum_path_sum(mat,vis,dp,i+1,j)
        dp[i][j] = total_sum
    return total_sum
"""
TC = o(n*m)
SC = o(n*m)
"""


'''Populating Next Right Pointers in Each Node
Medium

4038

181

Add to List

Share
You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

 

Example 1:


Input: root = [1,2,3,4,5,6,7]
Output: [1,#,2,3,#,4,5,6,7,#]
Explanation: Given the above perfect binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.
Example 2:

Input: root = []
Output: []
 

Constraints:

The number of nodes in the tree is in the range [0, 212 - 1].
-1000 <= Node.val <= 1000
 

Follow-up:

You may only use constant extra space.
The recursive approach is fine. You may assume implicit stack space does not count as extra space for this problem.'''

class Solution(object):
    def connect(self, root):
        """
        :type root: TreeLinkNode
        :rtype: nothing
        """
        
        if not root:
            return None
        cur  = root
        next = root.left

        while cur.left :
            cur.left.next = cur.right
            if cur.next:
                cur.right.next = cur.next.left
                cur = cur.next
            else:
                cur = next
                next = cur.left


#OR

def connect1(self, root):
    if root and root.left and root.right:
        root.left.next = root.right
        if root.next:
            root.right.next = root.next.left
        self.connect(root.left)
        self.connect(root.right)
 
# BFS       
def connect2(self, root):
    if not root:
        return 
    queue = [root]
    while queue:
        curr = queue.pop(0)
        if curr.left and curr.right:
            curr.left.next = curr.right
            if curr.next:
                curr.right.next = curr.next.left
            queue.append(curr.left)
            queue.append(curr.right)
    
# DFS 
def connect(self, root):
    if not root:
        return 
    stack = [root]
    while stack:
        curr = stack.pop()
        if curr.left and curr.right:
            curr.left.next = curr.right
            if curr.next:
                curr.right.next = curr.next.left
            stack.append(curr.right)
            stack.append(curr.left)


'''A program to check if a binary tree is BST or not
Difficulty Level : Medium
Last Updated : 15 Jul, 2021
A binary search tree (BST) is a node based binary tree data structure which has the following properties. 
• The left subtree of a node contains only nodes with keys less than the node’s key. 
• The right subtree of a node contains only nodes with keys greater than the node’s key. 
• Both the left and right subtrees must also be binary search trees.
From the above properties it naturally follows that: 
• Each node (item in the tree) has a distinct key.

'''


INT_MAX = 4294967296
INT_MIN = -4294967296
 
# A binary tree node
class Node:
 
    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
 
 
# Returns true if the given tree is a binary search tree
# (efficient version)
def isBST(node):
    return (isBSTUtil(node, INT_MIN, INT_MAX))
 
# Retusn true if the given tree is a BST and its values
# >= min and <= max
def isBSTUtil(node, mini, maxi):
     
    # An empty tree is BST
    if node is None:
        return True
 
    # False if this node violates min/max constraint
    if node.data < mini or node.data > maxi:
        return False
 
    # Otherwise check the subtrees recursively
    # tightening the min or max constraint
    return (isBSTUtil(node.left, mini, node.data -1) and
          isBSTUtil(node.right, node.data+1, maxi))
 
# Driver program to test above function
root = Node(4)
root.left = Node(2)
root.right = Node(5)
root.left.left = Node(1)
root.left.right = Node(3)
 
if (isBST(root)):
    print "Is BST"
else:
    print "Not a BST"

''''''
