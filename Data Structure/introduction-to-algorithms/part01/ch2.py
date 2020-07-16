# -*-coding:utf-8-*-
import numpy as np

# 插入排序
def inser_sort(A):
    for i in range(1,len(A)):
        key = A[i]
        j = i-1
        while (j>=0) and (A[j] > key):
            A[j+1] = A[j]
            j -= 1
        A[j+1] = key
    return A

# a = inser_sort([1,5,2,3,8,0])
# print(a)


# todo 寻找和最大子序列
class MaxSubarray():
    def find_max_crossing_subarray(self,A,l,m,r):
        left_sum = float('-inf')
        sum = 0
        max_l = m
        for i in range(m,l-1,-1):
            sum = sum+A[i]
            if sum > left_sum:
                left_sum = sum
                max_l = i
                # print("-----------",left_sum,max_l)

        right_sum = float("-inf")
        sum = 0
        max_r = m+1
        for j in range(m+1,r+1):
            sum += A[j]
            if right_sum < sum:
                right_sum = sum
                max_r = j
        return max_l,max_r,left_sum+right_sum


    def find_max_subarray(self,A,low,high):
        if low == high:
            return low,high,A[low]
        else:
            mid = int((low+high)/2)
            l_low,l_high,l_sum = self.find_max_subarray(A,low,mid)
            r_low,r_high,r_sum =self.find_max_subarray(A,mid+1,high)
            corss_low,cross_high,cross_sum = self.find_max_crossing_subarray(A,low,mid,high)
            # print("&&&&&&&&&&&",corss_low,cross_high,cross_sum )

            if (l_sum >= r_sum) & (l_sum >= cross_sum):
                # print("===1===",l_low,l_high,l_sum,'\n 1=',r_low,r_high,r_sum,'\n 1===',corss_low,cross_high,cross_sum)
                return l_low,l_high,l_sum
            elif (r_sum >= l_sum) & (r_sum >= cross_sum):
                # print("===2===",l_low,l_high,l_sum,'\n 2===',r_low,r_high,r_sum,'\n 2===',corss_low,cross_high,cross_sum)
                return r_low,r_high,r_sum
            else:
                # print("===3===",l_low,l_high,l_sum,'\n 3===',r_low,r_high,r_sum,'\n 3===',corss_low,cross_high,cross_sum)
                return corss_low,cross_high,cross_sum

ls = [3,5,-6,4,-9,-10,4,1,7,8]
max_subarray = MaxSubarray()
a,b,c = max_subarray.find_max_subarray(ls,0,9)
print(a,b,c)





