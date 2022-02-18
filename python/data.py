# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# data1 = pd.read_excel("E:\\RNA链路预测\\0CircR2Disease--雷\\0CircR2Disease--雷\\outputfu4-1.xlsx").values
# #.values将DataFrame转化为矩阵
# print(type(data1))
# print(data1.shape)
# data2 = np.delete(data1,[0,1,92],1)
# print(data2[4,0])
# plt.scatter(range(0,1000),data2,marker='+',color='g')
# plt.show()

# ============从csv文件中随机选择一些行另存为一个csv文件=================

import random
from random import randint

# oldf = open('features.csv', 'r', encoding='UTF-8')
# newf = open('features_Random.csv', 'w', encoding='UTF-8')
# n = 0
# # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素
# resultList = random.sample(range(0, 6400), 1000)
#
# lines = oldf.readlines()
# for i in resultList:
#     newf.write(lines[i])
#
# oldf.close()
# newf.close()

# 两个csv文件对比删除重复的
import pandas as pd
# a = pd.read_csv("features.csv")
# b = pd.read_csv("features_Random.csv")
# print(a not in b)
# # print(a)
# num = 0
# # dataframe逐行读取数据并输出
# # for indexs in a.index:
# #     print(a.loc[indexs].values[0:-1])
#
# for indexs in a.index:
#     if(all(a.loc[indexs].values[0:-1]==b.loc[num].values[0:-1])):
#         # 删除某一行https://blog.csdn.net/qq_18351157/article/details/105785367
#         print(a.drop(index=indexs))
#         num=num+1
# print(num)
#

#==============生成数据================================

#先用random生成随机数据，然后利用一个函数得到对应random的评分，把这些数据分成训练集&测试集，看算法性能
# 生成两个数组，第一个数组中的每一个元素对第二个数组的所有元素做运算

# 还有个空行问题
from numpy.matlib import rand
import numpy as np
import pandas as pd
import csv

# a = rand(5000)
# b = rand(5000)
# c = a*a.T+2.5*a+21
a = [1,2,3]
b = [4,5,6]
c = [7,8,9]

# dataframe = pd.DataFrame({"item_id":a,"score":b})
# dataframe.to_csv("features1.csv",index=False)
with open("features1.csv","w") as csvfile:
    write = csv.writer(csvfile)
    write.writerow(["user_id","item_id","score"])
    write.writerows(np.array([a,b,c]).T)