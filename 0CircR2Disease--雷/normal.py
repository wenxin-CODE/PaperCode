import pandas as pd
import numpy as np
import xlrd as xd


def select1_1():
    path = "C:\\Users\\dly\\Desktop\\RNA链路预测\\0CircR2Disease--雷\\0CircR2Disease--雷\\CircR2Disease_circRNA-disease associations - 副本.xlsx"
    data = xd.open_workbook(path)  # 打开excel表所在路径
    sheet = data.sheet_by_name('Sheet1')  # 读取数据，以excel表名来打开
    d = []
    for r in range(sheet.nrows):  # 将表中数据按行逐步添加到列表中，最后转换为list结构
        data1 = []
        for c in range(sheet.ncols):
            data1.append(sheet.cell_value(r, c))
        d.append(list(data1))

    data2 = []
    for i in range(len(d)):
        # print(d[i][3])
        # 检测CircBase Link和Gene Symbol是否为空
        if (d[i][4] == 'N/A' or d[i][3] == 'N/A'):
            continue
        else:
            data2.append(d[i])

    print(len(data2))
    pd.DataFrame(data2).to_excel('output1-1.xlsx', header=False, index=False)
    # print(data2)
    # for i in range(len(data2)):
    #     print(data2[i][3])

    # df = pd.read_excel("C:\\Users\\dly\\Desktop\\RNA链路预测\\0CircR2Disease--雷\\0CircR2Disease--雷\CircR2Disease_circRNA-disease associations - 副本.xlsx")
    # ndata = np.array(df)
    # reportsList = ndata.tolist()
    # print(reportsList[:,4])

def select1_2():
    path = "C:\\Users\\dly\\Desktop\\RNA链路预测\\0CircR2Disease--雷\\0CircR2Disease--雷\\CircAtlas.xlsx"
    data = xd.open_workbook(path)  # 打开excel表所在路径
    sheet = data.sheet_by_name('circRNADisease')  # 读取数据，以excel表名来打开
    d = []
    for r in range(sheet.nrows):  # 将表中数据按行逐步添加到列表中，最后转换为list结构
        data1 = []
        for c in range(sheet.ncols):
            data1.append(sheet.cell_value(r, c))
        d.append(list(data1))

    data2 = []
    for i in range(len(d)):
        # print(d[i][3])
        # host gene是Gene Symbol
        if (d[i][5] == '-' or d[i][13] == '-'):
            continue
        else:
            data2.append(d[i])
    print(len(data2))
    #
    pd.DataFrame(data2).to_excel('output1-2.xlsx', header=False, index=False)

def select2_1():
    # 网上查询疾病ID是否存在（search_result最后一列），然后比对
    path = 'C:\\Users\\dly\\Desktop\\RNA链路预测\\0CircR2Disease--雷\\0CircR2Disease--雷\\output1-1.xlsx'
    df = pd.read_excel(path)
    path2 = 'C:/Users/dly/Desktop/RNA链路预测/test/diease_ID_result.xlsx'
    df2 = pd.read_excel(path2)

    ndata = np.array(df)
    ndata1 = ndata[:,5]
    reportsList = ndata.tolist()
    reportsList1 = ndata1.tolist()
    # capitalize()将字符串的第一个字母变成大写,其他字母变小写
    reportsList1 = [i.capitalize() for i in reportsList1]
    print(len(reportsList1))

    ndata2 = np.array(df2)
    ndata2 = ndata2[:,0]
    reportsList2 = ndata2.tolist()
    print(len(reportsList2))

    result = []
    # 比对是否存在ID
    for i in range(len(reportsList1)):
        if reportsList1[i] in reportsList2:
            result.append(reportsList[i])
    print(len(result))
    df = pd.DataFrame(result,columns=['circRNA Name','Region','Strand','Gene Symbol','CircBase Link','Disease Name','Expression Pattern','Experimental Techniques','Species','Brief description','Year','PMID'])
    df.to_excel("output2-1.xlsx",index=False)

def select2_2():
    path = 'C:\\Users\\dly\\Desktop\\RNA链路预测\\0CircR2Disease--雷\\0CircR2Disease--雷\\output1-2.xlsx'
    df = pd.read_excel(path)
    path2 = 'C:/Users/dly/Desktop/RNA链路预测/test/diease_ID_result.xlsx'
    df2 = pd.read_excel(path2)

    ndata = np.array(df)
    ndata1 = ndata[:,8]
    reportsList = ndata.tolist()
    reportsList1 = ndata1.tolist()
    reportsList1 = [i.capitalize() for i in reportsList1]
    print(len(reportsList1))

    ndata2 = np.array(df2)
    ndata2 = ndata2[:,0]
    reportsList2 = ndata2.tolist()
    print(len(reportsList2))

    result = []
    for i in range(len(reportsList1)):
        if reportsList1[i] in reportsList2:
            result.append(reportsList[i])
    print(len(result))
    df = pd.DataFrame(result,columns=['id','title','journal','pub time','pmid','circRNA id','circRNA name','circRNA synonyms',' disease','method of circRNA detection','species','expression','association','host gene','tissue/cell line','functional describution'])
    df.to_excel("output2-2.xlsx",index=False)

def select3_1():
    # 找到了两种ID的转换表，然后确定前面的数据是否在样本上有表达数据（即前面的ID是否存在于exoRBase.xlsx）
    path = 'C:/Users/dly/Desktop/RNA链路预测/0CircR2Disease--雷/0CircR2Disease--雷/exoRBase.xlsx'
    df = pd.read_excel(path)
    ndata = np.array(df)
    reportsList = ndata.tolist()

    path2 = 'C:/Users/dly/Desktop/RNA链路预测/0CircR2Disease--雷/0CircR2Disease--雷/outputfu2-1.xlsx'
    df2 = pd.read_excel(path2)
    ndata2 = np.array(df2)
    reportsList2 = ndata2.tolist()
    print(len(reportsList2))
    result = []
    result1 = []
    for i in range(len(reportsList2)):
        str = reportsList2[i][0]
        for j in range(len(reportsList)):
            str2 = reportsList[j][92]
            if str == str2:
                result.append(reportsList[j])
                result1.append(reportsList2[i])
    print(len(result))
    df = pd.DataFrame(result, columns=['exo_circ_ID', 'CircRNA', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18', 'N19', 'N20', 'N21', 'N22', 'N23', 'N24', 'N25', 'N26', 'N27', 'N28', 'N29', 'N30', 'N31', 'N32', 'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CRC1', 'CRC2', 'CRC3', 'CRC4', 'CRC5', 'CRC6', 'CRC7', 'CRC8', 'CRC9', 'CRC10', 'CRC11', 'CRC12', 'HCC1', 'HCC2', 'HCC3', 'HCC4', 'HCC5', 'HCC6', 'HCC7', 'HCC8', 'HCC9', 'HCC10', 'HCC11', 'HCC12', 'HCC13', 'HCC14', 'HCC15', 'HCC16', 'HCC17', 'HCC18', 'HCC19', 'HCC20', 'HCC21', 'PAAD1', 'PAAD2', 'PAAD3', 'PAAD4', 'PAAD5', 'PAAD6', 'PAAD7', 'PAAD8', 'PAAD9', 'PAAD10', 'PAAD11', 'PAAD12', 'PAAD13', 'PAAD14', 'WhB1', 'WhB2', 'WhB3', 'WhB4', 'WhB5','circBase_ID'])
    df.to_excel("output3-1.xlsx",index=False)
    df = pd.DataFrame(result1,
                      columns=['circRNA Name', 'Region', 'Strand', 'Gene Symbol', 'CircBase Link', 'Disease Name',
                               'Expression Pattern', 'Experimental Techniques', 'Species', 'Brief description', 'Year',
                               'PMID'])
    df.to_excel("output3-1-1.xlsx", index=False)

def select3_2():
    path = 'C:/Users/dly/Desktop/RNA链路预测/0CircR2Disease--雷/0CircR2Disease--雷/exoRBase.xlsx'
    df = pd.read_excel(path)
    ndata = np.array(df)
    reportsList = ndata.tolist()

    path2 = 'C:/Users/dly/Desktop/RNA链路预测/0CircR2Disease--雷/0CircR2Disease--雷/output2-2.xlsx'
    df2 = pd.read_excel(path2)
    ndata2 = np.array(df2)
    reportsList2 = ndata2.tolist()
    print(len(reportsList2))
    result = []
    result1 = []
    for i in range(len(reportsList2)):
        str = reportsList2[i][5]
        for j in range(len(reportsList)):
            str2 = reportsList[j][92]
            if str == str2:
                result.append(reportsList[j])
                result1.append(reportsList2[i])
    print(len(result))
    df = pd.DataFrame(result, columns=['exo_circ_ID', 'CircRNA', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18', 'N19', 'N20', 'N21', 'N22', 'N23', 'N24', 'N25', 'N26', 'N27', 'N28', 'N29', 'N30', 'N31', 'N32', 'CHD1', 'CHD2', 'CHD3', 'CHD4', 'CHD5', 'CHD6', 'CRC1', 'CRC2', 'CRC3', 'CRC4', 'CRC5', 'CRC6', 'CRC7', 'CRC8', 'CRC9', 'CRC10', 'CRC11', 'CRC12', 'HCC1', 'HCC2', 'HCC3', 'HCC4', 'HCC5', 'HCC6', 'HCC7', 'HCC8', 'HCC9', 'HCC10', 'HCC11', 'HCC12', 'HCC13', 'HCC14', 'HCC15', 'HCC16', 'HCC17', 'HCC18', 'HCC19', 'HCC20', 'HCC21', 'PAAD1', 'PAAD2', 'PAAD3', 'PAAD4', 'PAAD5', 'PAAD6', 'PAAD7', 'PAAD8', 'PAAD9', 'PAAD10', 'PAAD11', 'PAAD12', 'PAAD13', 'PAAD14', 'WhB1', 'WhB2', 'WhB3', 'WhB4', 'WhB5','circBase_ID'])
    df.to_excel("output3-2.xlsx",index=False)
    df = pd.DataFrame(result1, columns=['id', 'title', 'journal', 'pub time', 'pmid', 'circRNA id', 'circRNA name',
                                       'circRNA synonyms', ' disease', 'method of circRNA detection', 'species',
                                       'expression', 'association', 'host gene', 'tissue/cell line',
                                       'functional describution'])
    df.to_excel("output3-2-2.xlsx", index=False)

def select4_1():
    # 合并3-1-2&3-2-2,为4-2.有一些重复
    # 4-2包含RNA212，疾病67
    path = 'C:\\Users\\dly\\Desktop\\RNA链路预测\\0CircR2Disease--雷\\0CircR2Disease--雷\\output4-2.xlsx'
    df = pd.read_excel(path)
    # print(df)
    ndata = np.array(df)
    reportsList = ndata.tolist()
    RNA = set()
    Diease = set()
    # print(reportsList[0][0])
    for i in range(len(reportsList)):
        # print(reportsList[i][0])
        RNA.add(reportsList[i][0])
        Diease.add(reportsList[i][1])
    RNA = list(RNA)
    Diease = list(Diease)
    # print(len(RNA),len(Diease))
    RNA1 = {}
    Diease1 = {}
    for i in range(212):
        RNA1[i]=RNA[i]
    for i in range(67):
        Diease1[i]=Diease[i]
    # print(Diease1)
    CD = np.zeros((212,67))
    print(CD[199][54])
    for i in range(len(reportsList)):
        a = [k for k,v in RNA1.items() if v==reportsList[i][0]]
        b = [k for k,v in Diease1.items() if v==reportsList[i][1]]
        CD[a[0]][b[0]]=1
    print(CD)
    print(sum(CD.T))
    print(sum(sum(CD.T).T))

def select4_2():
    path = 'C:\\Users\\dly\\Desktop\\RNA链路预测\\0CircR2Disease--雷\\0CircR2Disease--雷\\outputfu4-1.xlsx'
    df = pd.read_excel(path)
    ndata = np.array(df)
    reportsList = ndata.tolist()
    print(reportsList[0][2])
    CT = np.zeros((212, 90),dtype=float)

    for i in range(212):
        for j in range(90):
            CT[i][j] = reportsList[i][j+2]
    print(CT)

def calcu4():

    pass

if __name__ == '__main__':
    select4_1()
    # path = 'C:\\Users\\dly\\Desktop\\RNA链路预测\\0CircR2Disease--雷\\0CircR2Disease--雷\\outputfu4-1.xlsx'
    # df = pd.read_excel(path)
    # print(type(df))
    # ndata = np.array(df)
    # reportsList = ndata.tolist()
    # print("hello python!!!")