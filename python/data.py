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
a = pd.read_csv("features.csv")
b = pd.read_csv("features_Random.csv")
print(a not in b)
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
