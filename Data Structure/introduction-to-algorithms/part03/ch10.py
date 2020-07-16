# -*-coding:utf-8-*-

import numpy as np

# todo 栈
class Stack():
    def __init__(self):
        self.S = []

    def stack_empty(self):
        return bool(self.S)

    def push(self,value):
        self.S.append(value)

    def pop(self):
        if self.S:
            self.S.pop()
        else:
            raise LookupError("stack is empty")

    def peek(self):
        if not self.isEmpty():
            return self.S[len(self.S)-1]

    def top(self):
        return self.S[-1]


# todo 队列 https://blog.csdn.net/hjhmpl123/article/details/52965908
class Head():
    def __init__(self):
        self.left = None
        self.right = None

class Node():
    def __init__(self,value):
        self.value = value
        self.next = None

class Queue():
    def __init__(self):
        # 初始化节点
        self.head = Head()

    def enqueue(self,value):
        # 插入元素
        new_node = Node(value)
        p = self.head
        if p.right:
            tmp = p.right
            p.right = new_node
            tmp.next = new_node
        else:
            p.left = new_node
            p.right = new_node

    def dequeue(self):
        # 取出元素
        p = self.head
        if p.left and (p.left == p.right):
            tmp = p.left
            p.left = None
            p.right = None
            return tmp.value
        elif p.left and (p.left != p.right):
            tmp = p.left
            p.left = tmp.next
            return tmp.value
        else:
            raise LookupError("queue is empty")

    def is_empty(self):
        if self.head.left:
            return False
        else:
            return True

    def top(self):
        if self.head.left:
            return self.head.left.value
        else:
            raise LookupError("queue is empty")


# todo 队列 https://blog.csdn.net/hfutdog/article/details/95001240
class Queue(object):
    def __init__(self):
        """创建一个空的队列"""
        self.__list = []

    def enqueue(self, elem):
        """往队列中添加一个item元素"""
        self.__list.append(elem)

    def dequeue(self):
        """从队列头部删除一个元素"""
        if self.is_empty():
            return None
        else:
            return self.__list.pop(0)

    def is_empty(self):
        """判断一个队列是否为空"""
        return [] == self.__list

    def size(self):
        """返回队列的大小"""
        return len(self.__list)


# todo 双链表 https://zhuanlan.zhihu.com/p/60057180  https://www.cnblogs.com/reader/p/9547621.html     https://www.cnblogs.com/reader/p/9547621.html
class Node():
    def __init__(self,item):
        self.item = item
        self.next = None
        self.prev = None

class Chain():
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head is None

    def length(self):
        cur = self.head
        count = 0
        while cur is not None:
            count += 1
            cur = cur.next
        return count

    def items(self):
        cur = self.head
        while cur is not None:
            yield cur.item
            cur = cur.next

    # todo 向链表头部添加元素
    def add(self,item):
        node = Node(item)
        if self.is_empty():
            self.head = node
        else:
            node.next = self.head
            # 原头部 prev 指向 新结点
            self.head.prev = node
            # head 指向新结点
            self.head = node

    def append(self,item):
        node = Node(item)
        if self.head is None:
            self.head = node
        else:
            cur = self.head
            while cur.next is not None:
                cur = cur.next
            node.prev = cur
            cur.next = node

    def insert(self,index,item):
        if index <= 0:
            self.add(item)
        elif index > self.items() - 1:
            self.append(item)
        else:
            node = Node(item)
            cur = self.head
            for i in range(index):
                cur = cur.next
            node.next =cur
            node.prev = cur.prev
            cur.prev =  node
            cur.prev.next = node

    def remove(self,item):
        if self.is_empty():
            return
        cur = self.head
        if cur.item == item:
            if cur.next == None:
                self.head = None
                return True
            else:
                self.head = cur.next
                cur.next.prev = None
            return True
        while cur.next is not None:
            if cur.item == item:
                cur.prev.next = cur.next
                cur.next.prev = cur.prev
                return True
            cur = cur.next
        if cur.item == item:
            cur.prev.next = None
            return True

    def find(self,item):
        return item in self.items()


# todo 二叉树，插入，广度遍历（），深度遍历（前，中，后）
class TreeNode():
    def __init__(self,item):
        self.key = item
        self.left = None
        self.right = None
        self.p = None
        self.color = "black"

class Tree():
    def __init__(self):
        self.root = None
        self.nil = TreeNode(0)

    # todo 插入
    def insert(self,item):
        z = TreeNode(item)
        y = None
        x = self.root
        while x != None:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.p = y
        if y == None:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z

#     todo 中序便利
    def middle_walk(self,x):
        if (x is not None) and (x != self.nil):
            print("middle_walk: ",x.key)
            self.middle_walk(x.left)
            self.middle_walk(x.right)

#     todo 广度优先
    def level_walk(self,root):
        if (root == None) or (root == self.nil):
            return
        q = []
        q.append(root)
        while len(q) >0 :
            length = len(q)
            for i in range(length):
                # 同层次节点依次出队
                r = q.pop(0)
                if (r.left is not None) and (r.left != self.nil):
                    q.append(r.left)
                if (r.right is not None) and (r.right != self.nil):
                    q.append(r.right)
                print("level_walk:",r.key)

    # todo 递归查找
    def search1(self,x,k):
        if (x is None) or (x.key == k) or (x == self.nil):
            return x
        if k < x.key:
            return self.search1(x.left,k)
        else:
            return self.search1(x.right,k)

    # TODO 循环查找
    def search2(self,x,k):
        while (x != None) and (x.key != k):
            if k < x.key:
                x = x.left
            else:
                x = x.right
        return x

    # todo 查找最小值
    def TreeMinimum(self,x):
        while (x.left != None) and (x.left != self.nil):
            x = x.left
        return x

    # TODO 查找后继
    def successor(self,x):
        if x.right is not None:
            return self.TreeMinimum(x.right)
        y = x.p
        if (y is not None) and (y.right == x):
            x = y
            y = y.p
        return y

    # todo 交换两棵子树
    def transplant(self,u,v):
        if u.p is None:
            self.root = u
        elif u == u.p.left:
            u.p.left = v
        else:
            u.p.right = v
        if v is not None:
            v.p = u.p

    # TODO 删除
    def delete(self,x):
        if x.left is None:
            self.transplant(x,x.right)
        elif x.right is None:
            self.transplant(x,x.left)
        else:
            y = self.TreeMinimum(x.right)
            if y.p!= x:
                self.transplant(y,y.right)
                y.right = x.right
            self.transplant(x,y)
            y.left = x.left
            x.left.p = y


# TODO 红黑树 https://blog.csdn.net/z649431508/article/details/78034751
class TreeNode():
    def __init__(self,item):
        self.key = item
        self.left = None
        self.right = None
        self.p = None
        self.color = "black"

class RBTree(Tree):
    def __init__(self):
        self.nil = TreeNode(0)
        self.root = self.nil

    # todo 左旋转
    def LeftRotate(self,x):
        y = x.right
        x.right = y.left
        if y.left != self.nil:
            y.left.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.left = x
        x.p = y

    # todo 右旋转
    def RightRotate(self,x):
        y = x.left
        x.left = y.right
        if y.right != self.nil:
            y.right.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.right = x
        x.p = y

    # todo 红黑树的插入
    def insert(self,item):
        y = self.nil
        x = self.root
        z = TreeNode(item)
        while x != self.nil:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.p = y
        if y == self.nil:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.left = self.nil
        z.right = self.nil
        z.color = 'red'
        self.RBInsertFixup(z)
        return z.key,z.color

    # todo 插入颜色调整
    def RBInsertFixup(self,z):
        while z.p.color == "red":
            if z.p == z.p.p.left:
                y = z.p.p.right
                if y.color == 'red':
                    z.p.color = "black"
                    z.p.p.color = "red"
                    y.color = "black"
                    z = z.p.p
                else:
                    if z == z.p.right:
                        z = z.p
                        self.LeftRotate(z)
                    z.p.color = "black"
                    z.p.p.color = "red"
                    self.RightRotate(z.p.p)
            else:
                y = z.p.p.left
                if y.color == 'red':
                    z.p.color = 'black'
                    z.p.p.color = "red"
                    y.color = "black"
                    z = z.p.p
                else:
                    if z == z.p.left:
                        z = z.p
                        self.RightRotate(z)
                    z.p.color = "black"
                    z.p.p.color = "red"
                    self.LeftRotate(z.p.p)
        self.root.color = "black"

    def RBTransplant(self,u,v):
        if u.p == self.nil:
            self.root = v
        elif u == u.p.left:
            u.p.left = v
        else:
            u.p.right = v
        v.p = u.p

    # todo 红黑树的删除
    def RBDelete(self,z):
        y = z
        y_original_color = y.color
        if z.left == self.nil:
            x = z.right
            self.RBTransplant(z,z.right)
        elif z.right == self.nil:
            x = z.left
            self.RBTransplant(z,z.left)
        else:
            y = self.TreeMinimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.p == z:
                x.p = z
            else:
                self.RBTransplant(y,y.right)
                y.right = z.right
                y.right.p = z.right.p
            self.RBTransplant(z,y)
            y.left = z.left
            y.left.p = y
            y.color = z.color
        if y_original_color == 'black':
            self.RBDeleteFixup(x)

    def RBDeleteFixup(self,x):
        while x != self.root and x.color == "black":
            if x == x.p.left:
                w = x.p.right
                if w.color == 'red':
                    w.color = "black"
                    x.p.color = 'red'
                    self.LeftRotate(x.p)
                    w = x.p.right
                if w.left.color == 'black' and w.right.color == 'balck':
                    w.color = "red"
                    x = x.p
                else:
                    if w.right.color == 'black':
                        w.left.color = "black"
                        w.color = 'red'
                        self.RightRotate(w)
                        w = x.p.right
                    w.color = x.p.color
                    x.p.color = "black"
                    w.right.color = "black"
                    self.LeftRotate(x.p)
                    x = self.root
            else:
                w = x.p.left
                if w.color == 'red':
                    w.color = "black"
                    x.p = 'red'
                    self.RightRotate(x.p)
                    w = x.p.left
                if w.right.color == 'black' and w.left.color == 'black':
                    w.color = "red"
                    x = x.p
                else:
                    if w.left.color == 'black':
                        w.right.color = 'black'
                        w.color = 'red'
                        self.LeftRotate(w)
                        w = x.p.left
                    w.color = x.color
                    x.p.color = 'black'
                    w.right.color = 'black'
                    self.RightRotate(x.p)
                    x = self.root
                x.color = 'balck'







if __name__ == '__main__':

    # todo 红黑树
    datas = [4,2,6,1,3,5,7,6.5,7.5]
    # tree = Tree()
    # for data in datas:
    #     tree.insert(data)
    #
    # print("root",tree.root.key)
    # tree.middle_walk(tree.root)
    # print("level")
    # tree.level_walk(tree.root)
    # node1 = tree.search1(tree.root,6)
    # print("res",node1,node1.key)
    #
    # node1 = tree.search2(tree.root,6)
    # print("res",node1,node1.key)
    #
    # tree.delete(node1)
    # tree.level_walk(tree.root)


    rbtree = RBTree()
    for data in datas:
        rbtree.insert(data)
    rbtree.insert(0)
    print("root",rbtree.root.key)
    rbtree.middle_walk(rbtree.root)
    node1 = rbtree.search1(rbtree.root,6)
    rbtree.level_walk(rbtree.root)
    rbtree.RBDelete(node1)


