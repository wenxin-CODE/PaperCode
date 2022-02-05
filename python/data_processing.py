import numpy as np
import pandas as pd

data = pd.read_csv("features.csv")
print(len(data))
for i in range(len(data)):
    if (data.loc[i,"score"]<100.0):
        data.loc[i,"score"] = 1
    elif (100.0<data.loc[i,"score"]<1000.0):
        data.loc[i,"score"] = 2
    elif (1000.0<data.loc[i, "score"] < 10000.0):
        data.loc[i, "score"] = 3
    elif (10000.0<data.loc[i, "score"] < 100000.0):
        data.loc[i, "score"] = 4

print(data)
# df.loc[index_val].values.tolist()输出某一行