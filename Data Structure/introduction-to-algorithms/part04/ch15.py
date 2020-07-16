# -*-coding:utf-8-*-


# todo 钢条切割问题，带备忘的自顶向下
def memoized_cut_rod(p, n):
    r = [-1 for i in range(n+1)]
    return memoized_cut_rod_aux(p, n, r),r


def memoized_cut_rod_aux(p, n, r):
    if r[n] >= 0:
        return r[n]
    q = -1
    if n == 0:
        q = 0
    else:
        for i in range(1, n + 1):
            q = max(q, memoized_cut_rod_aux(p, n - i, r)+p[i])
    r[n] = q
    return q


# todo 钢条切割问题，自底向上
def bottom_up_cut_rod(p,n):
    r = [0 for i in range(n+1)]
    for i in range(1,n+1):
        if n == 0:
            return 0
        q = 0
        for j in range(1,i+1):
            q = max(q,r[i-j]+p[j])
        r[i] = q
    return r[n],r


# todo 矩阵乘法，自底向上
def matrix_chain_order(p):
    n = len(p) - 1
    m = [[0 for i in range(n)] for j in range(n)]
    s = [[0 for i in range(n)] for j in range(n)]
    for j in range(1,n):
        for i in range(j-1,-1,-1):
            m[i][j] = float("inf")
            for k in range(i,j):
                q = m[i][k]+ m[k+1][j]+p[i]*p[k+1]*p[j+1]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k
    return m,s


def print_option_parens(s,i,j):
    if i==j:
        print("A"+str(i),end='')
    else:
        print("(",end="")
        print_option_parens(s,i,s[i][j])
        print_option_parens(s,s[i][j]+1,j)
        print(")",end="")


# todo 矩阵乘法，自顶向下
def memoized_matrix(p):
    n = len(p)-1
    m = [[float("inf") for i in range(n)] for j in range(n)]
    s = [[0 for i in range(n)] for j in range(n)]
    lookup_chain(m,p,s,0,n-1)
    return m,s


def lookup_chain(m,p,s,i,j):
    if m[i][j] < float('inf'):
        return m[i][j]
    if i == j:
        m[i][j] = 0
    else:
        for k in range(i,j):
            q = lookup_chain(m,p,s,i,k)+lookup_chain(m,p,s,k+1,j)+p[i]*p[k+1]*p[j+1]
            if q<m[i][j]:
                m[i][j] = q
                s[i][j] = k
    return m[i][j]


# todo 最长公共子序列
def lcs_length(x,y):
    m,n = len(x),len(y)
    c = [[0 for i in range(n+1)] for j in range(m+1)]
    b = [[" " for i in range(n+1)] for j in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if x[i-1] == y[j-1]:
                c[i][j] = c[i-1][j-1]+1
                b[i][j] = 'left up'
            elif c[i-1][j] >= c[i][j-1]:
                c[i][j] = c[i-1][j]
                b[i][j] = 'up'
            else:
                c[i][j] = c[i][j-1]
                b[i][j] = 'left'
    return c,b


def print_lcs(b,x,i,j):
    if i == 0 or j == 0:
        return
    if b[i][j] == 'left up':
        print_lcs(b,x,i-1,j-1)
        print(x[i-1],end="")
    elif b[i][j] == "up":
        print_lcs(b,x,i-1,j)
    else:
        print_lcs(b,x,i,j-1)


# todo 最优二叉搜索树
def optimal_bst(p,q,n):
    e = [[0 for j in range(n+1)] for i in range(n+2)]
    w = [[0 for j in range(n+1)] for i in range(n+2)]
    root = [[0 for j in range(n+1)] for i in range(n+1)]
    for i in range(n+2):
        e[i][i-1] = q[i-1]
        w[i][i-1] = q[i-1]
    for l in range(1,n+1):
        for i in range(1,n-l+2):
            j = i+l-1
            e[i][j] = float("inf")
            w[i][j] = w[i][j-1] + p[j] +q[j]
            for r in range(i,j+1):
                t = e[i][r-1] + e[r+1][j]+w[i][j]
                if t < e[i][j]:
                    e[i][j] = t
                    root[i][j] = r
    return e,root





if __name__ == "__main__":
    p = [0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30]
    r = memoized_cut_rod(p,10)
    # print(r)
    r = bottom_up_cut_rod(p,10)
    # print(r)

    p = [30, 35, 15, 5, 10, 20, 25]
    # m,s = matrix_chain_order(p)
    # print_option_parens(s,0,5)


    p = [30, 35, 15, 5, 10, 20, 25]
    m,s = memoized_matrix(p)
    # print_option_parens(s,0,5)

    x = ["A", "B", "C", "B", "D", "A", "B"]
    y = ["B", "D", "C", "A", "B", "A"]
    c,b=lcs_length(x,y)
    print("bbbbbb\n",b,"\n\n")

    print_lcs(b, x, len(x), len(y))

