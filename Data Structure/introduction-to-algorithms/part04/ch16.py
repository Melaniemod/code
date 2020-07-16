# -*-coding:utf-8-*-

# todo 迭代贪心算法
def greedActSel(s,f):
    n = len(s)
    A = [1]
    k = 1
    for m in range(2,n):
        if s[m] > f[k]:
            A.append(m)
            k = m
    return A


# todo 递归贪心算法
def recuActSel(s,f,k,n,A):
    m = k+1
    while (m < n) and (s[m] < f[k]):
        m += 1
    if m < n:
        A.append(m)
        recuActSel(s,f,m,n,A)
    return A


s = [0, 1, 3, 0, 5, 3, 5, 6, 8, 8, 2, 12, ]
f = [0, 4, 5, 6, 7, 9, 9, 10, 11, 12, 14, 16, ]
print("======== 迭代方法 =======")
print(greedActSel(s, f))

A = recuActSel(s, f, 0, len(s),[])
print("======== 递归方法 =======")
print(A)
