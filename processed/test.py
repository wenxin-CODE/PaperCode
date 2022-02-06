import scipy.sparse as sp
import numpy as np
import pandas as pd
# d_A = np.array([[1, 0, 3],
#                 [0, 54, 6],
#                 [7, 0, 0]])
# s_A = sp.coo_matrix(d_A)
# print(s_A.shape)
# print(np.ones(3))

# print(pd.read_csv("C:\\Users\\dly\\Desktop\\RNA链路预测\\0CircR2Disease--雷\\0CircR2Disease--雷\\2.feat"))
path="G:\\python\\processed\\ARGA-master\\data\\facebook\\"
dataset="facebook"
idx_features_labels = np.genfromtxt("{}2.feat".format(path, dataset),
                                    dtype=np.dtype(str))  # 读取数据
features1 = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
# print(idx_features_labels.dtype)
a = []
for i in range(len(idx_features_labels)):
    for j in range(len(idx_features_labels[0])):
        if(idx_features_labels[i][j]!='0.0'):
            # print(type(idx_features_labels[i][j]))
            # print(i,j,idx_features_labels[i][j])
            a.append([i,j,idx_features_labels[i][j]])
df = pd.DataFrame(a)
df.to_csv("3.edges")