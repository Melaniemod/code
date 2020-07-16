import tensorflow as tf
import numpy as np

sess = tf.Session()




c = tf.range(0, 5, dtype=tf.int32)
c1 = tf.random_shuffle(c)
a = np.random.random((5, 4))
c2 = sess.run(c1)
a1 = sess.run(tf.gather(a,c1))
print(c2,'\n',a,'\n\n',a1)



b1 = tf.constant([1,2,3],dtype=tf.float16)
b2 = sess.run(tf.expm1(b1))
print(b2)



from tqdm import trange
from time import time
for i in trange(100):
    #do something
    pass








sess.close()
