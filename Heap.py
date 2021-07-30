class Heap:
    def __init__(self):
        self.heap = []

    def Max_heap_insert(self, A, n):
        i = n
        temp = A[i]
        while i > 1 and temp >A[i//2]:
            A[i] = A[i//2]
            i = i//2

        A[i] = temp

        return A


    def Min_heap_insert(self, A, n):
        i = n
        temp = A[i]
        while i > 1 and temp < A[(i//2)]:
            A[i] = A[(i//2)]
            i = (i//2)

        A[i] = temp
        return A

    def list_insert(self, arr):
        n = (len(arr))
        for i in range(2,n):
            self.Max_heap_insert(arr, i)
        return arr

    def delete_max(self, arr, n):
        val = arr[1]
        arr[1] = arr[n]
        arr[n] = val
        i = 1
        j = i *2
        while j < n-1:
            if arr[j+1] > arr[j]:
                j = j+1

            if arr[i] < arr[j]:
                temp = arr[i]
                arr[i] = arr[j]
                arr[j] = temp
                i = j
                j = 2*j

            else:
                break
        print(arr)
        return val

    def heap_sort(self,arr, n):
        for k in range(n-1, 1, -1):
            val = arr[1]
            arr[1] = arr[k]
            arr[k] = val
            i = 1
            j = i *2
            while j < k-1:
                if arr[j+1] > arr[j]:
                    j += 1
                
                if arr[j] > arr[i]:
                    temp = arr[i]
                    arr[i] = arr[j]
                    arr[j] = temp

                else:
                    break
        return arr


    def heapify(self, arr, n):
        for i in range(n//2, 0, -1):
            temp = arr[i]
            while i > 1 and temp > arr[i//2]:
                arr[i] = arr[i//2]
                i = i//2

            arr[i] = temp
        return arr

    
                
    

H  = Heap()
A = H.list_insert([0,10,20,30,25,5,40,35])

print(A)

B = H.delete_max(A, 7)
print(B)
            
C = H.heap_sort(A, 7)
print(C)

D = H.heapify([0,5,30,20,35,40,15], len([0,5,30,20,35,40,15]))
print(D)