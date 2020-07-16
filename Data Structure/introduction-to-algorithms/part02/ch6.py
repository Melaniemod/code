# -*-coding:utf-8-*-

import numpy as np

# todo 最大堆排序
class HeapSort():
    def __init__(self,data):
        self.A = data

    def parent(self,i):
        return int((i-1)/2)

    def left(self,i):
        return 2*i+1

    def right(self,i):
        return 2*i+2

    def max_heapify(self,i,j):
        l = self.left(i)
        r = self.right(i)
        if (l<j) and (self.A[l]>self.A[i]):
            largest = l
        else:
            largest = i
        if (r<j) and (self.A[r]>self.A[largest]):
            largest = r
        if largest != i:
            self.A[i],self.A[largest] = self.A[largest],self.A[i]
            self.max_heapify(largest,j)

    def build_max_head(self):
        for i in range(int(len(self.A)/2),-1,-1):
            self.max_heapify(i,len(self.A))

    def heap_sort(self):
        self.build_max_head()
        for i in range(len(self.A)-1,1,-1):
            print("=====",self.A,i,self.A[i])
            self.A[0],self.A[i] = self.A[i],self.A[0]
            self.max_heapify(0,i-1)
        return self.A

    def heap_max(self):
        return self.A[0]

    def heap_extract_max(self,j):
        if j<0:
            raise IndexError("指针溢出")
        my_max = self.A[1]
        self.A[0] = self.A[j]
        self.max_heapify(0)
        return my_max

    def heap_increase_key(self,i,key):
        if self.A[i] < key:
            raise AssertionError("new key is smaller than current key")
        self.A[i] = key
        while (i > 0) and (self.A[i] > self.A[self.parent(i)]):
            self.A[i],self.A[self.parent(i)] = self.A[self.parent(i)],self.A[i]
            i = self.parent(i)

    def max_heap_insert(self,j,key):
        j += 1
        self.A[j] = np.float("-inf")
        self.heap_increase_key(j,key)




# todo 快速排序
class QuickSort():
    def __init__(self,data):
        self.A = data

    def quick_sort(self,p,r):
        if p<r:
            q = self.partition(p,r)
            self.quick_sort(p,q-1)
            self.quick_sort(q+1,r)
        return self.A

    def partition(self,p,r):
        x = self.A[r]
        i = p-1
        for j in range(p,r):
            # print("p,r",p,r)
            if self.A[j] <= x:
                i += 1
                self.A[i],self.A[j] = self.A[j],self.A[i]
        self.A[i+1],self.A[r] = self.A[r],self.A[i+1]
        return i+1

    def random_partition(self,p,r):
        i = np.random(p,r)
        self.A[r],self.A[i] = self.A[i],self.A[r]
        return self.partition(p,r)

    def random_quick_sort(self,p,r):
        if p<r:
            q = self.random_partition(p,r)
            self.random_quick_sort(p,q)
            self.random_quick_sort(q,r)
        return self.A


# todo 计数排序
class CountingSort():
    def __init__(self,data):
        self.A = data
        self.k = max(self.A)
        self.B = [-1]*len(self.A)
        self.C = [0]*(self.k+1)

    def counting_sort(self):
        for i in range(len(self.A)):
            self.C[self.A[i]] += 1
        for j in range(1,len(self.C),1):
            self.C[j] += self.C[j-1]
        for j in range(len(self.A)-1,-1,-1):
            self.B[self.C[self.A[j]]-1] = self.A[j]
            self.C[self.A[j]] -= 1
        return self.B____


# todo 桶排序
class BucketSort():
    pass


# todo 第i个顺序统计量
class OrderStatistic():
    def __init__(self,data):
        self.A = data

    def partition(self,p,r):
        x = self.A[r]
        i = p-1
        for j in range(p,r):
            if self.A[j] <= x:
                i += 1
                self.A[i],self.A[j] = self.A[j],self.A[i]
        self.A[i+1],self.A[r] = self.A[r],self.A[i+1]
        return i+1

    def random_partition(self,p,r):
        print("p,r",p,r)
        i = np.random.randint(p,r)
        print("i",i)
        self.A[r],self.A[i] = self.A[i],self.A[r]
        return self.partition(p,r)

    def minimum(self):
        my_min = self.A[0]
        for i in range(1,len(self.A)-1,1):
            if my_min < self.A[i]:
                my_min = self.A[i]
        return my_min

    def random_select(self,p,r,i):
        if p == r:
            return self.A[p]
        q = self.random_partition(p,r)
        k = q-p+1
        if k == i:
            return self.A[q]
        elif k<i:
            return self.random_select(q+1,r,i)
        else:
            return self.random_select(p,q-1,i)











if __name__ == '__main__':
    data = [4,3, 5, 3,9,8,7, 2, 9, 6, 0, 8]
    # heap = HeapSort(data)
    # heap_sort = heap.heap_sort()
    # print(heap_sort)

    # quick = QuickSort(data)
    # quick_sort = quick.quick_sort(0, len(data) - 1)
    # print(quick_sort)

    # my_counting = CountingSort(data)
    # counting_sort = my_counting.counting_sort()
    # print(counting_sort)

    select = OrderStatistic(data)
    my_select = select.random_select(0,len(data)-1,4)
    print(my_select)