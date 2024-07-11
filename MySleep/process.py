import numpy as np
import random
data = np.array([[ 0,  1,  2,  3],
                 [ 4,  5,  6,  7],
                 [ 8,  9, 10, 11],
                 [12, 13, 14, 15]]) # shape:(4,4)
label = np.array([1,2,3,4]) # shape:(4,)

sample_num = int(0.5 * len(data)) # 假设取50%的数据
sample_list = [i for i in range(len(data))] # [0, 1, 2, 3]
sample_list = random.sample(sample_list, sample_num) # [1, 2]

data = data[sample_list,:] # array([[ 4,  5,  6,  7], [ 8,  9, 10, 11]])
label = label[sample_list] # array([2, 3])
print(data)
