import tensorflow as tf
from itertools import count

dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])

sess = tf.Session()

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

try:
    for x in count():
        data_numpy=sess.run(el)
        print(data_numpy)
except tf.errors.OutOfRangeError:
        print("Process finished")
        pass