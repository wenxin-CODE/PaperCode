import os
import pandas as pd

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data_three(dname='pubmed', dtype='citation'):
    print('Loading {} dataset.........'.format(dname))
    if dtype == 'citation':
        candidate = ['x','y','tx','ty','allx','ally','graph']
        obj = []
        dpath="../data/pubmed/"
        for name in candidate:
            with open("{}/ind.{}.{}".format(dpath,dname,name),'rb') as f:
                 obj.append(np.load(f, allow_pickle=True,encoding='latin1'))
        x, y, tx, ty, allx, ally, graph = tuple(obj)
        test_index = []
        for line in open("{}/ind.{}.test.index".format(dpath,dname)):
            test_index.append(int(line.strip()))
            # sort the test
        test_index_sorted = np.sort(test_index)
        if dname == 'cora':
            full_test_index = range(test_index_sorted[0], test_index_sorted[-1]+1)
            tx_ = sp.lil_matrix((len(full_test_index), x.shape[1]))
            tx_[test_index_sorted-min(test_index_sorted),:] = tx
            tx = tx_
            ty_ = np.zeros((len(full_test_index), y.shape[1]))
            ty_[test_index_sorted-min(test_index_sorted),:] = ty
            ty = ty_
        features1 = sp.vstack((allx, tx)).tolil()
        features1[test_index, :] = features1[test_index_sorted, :]
        m = features1.shape[0]
        edges=np.array(nx.edges(nx.from_dict_of_lists(graph)))
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(m, m),
                            dtype=np.float32)  # 邻接矩阵

    return adj,features1
def load_data_two():
    object_to_idx = {}
    idx_counter = 0
    edges = []
    dataset_str='twitch'
    data_path="../data/twitch"
    with open(os.path.join(data_path, "edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    data = pd.read_csv(r'E:\Ealine\论文资料\论文\对比实验-成功\ARGA-master\data\twitch\features.csv',
                       encoding="utf8",
                       sep=",",
                       dtype={"switch": np.int32})
    # data = np.array(data.loc[:, :])
    row = np.array(data["node_id"])
    col = np.array(data["feature_id"])
    values = np.array(data["value"])
    node_count = max(row) + 1
    feature_count = max(col) + 1
    shape = (node_count, feature_count)
    features1 = sp.csr_matrix((values, (row, col)), shape=shape)
    m=features1.shape[0]
    edges=np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(m, m),
                        dtype=np.float32)  # 邻接矩阵

    return adj,features1
# def load_data_four(path="../data/facebook/", dataset="facebook"):
#     """Load citation network dataset (cora only for now)"""
#     print('Loading {} dataset...'.format(dataset))
#     idx_features_labels = np.genfromtxt("{}107.feat".format(path, dataset),
#                                         dtype=np.dtype(str))#读取数据
#     print(type(idx_features_labels))#<class 'numpy.ndarray'>
#     features1 = sp.csr_matrix(idx_features_labels[:,1:-1], dtype=np.float32)#第一步处理--去掉矩阵的第一列，并将矩阵转化为邻接表
#     # build graph
#     m = features1.shape[0]#查看去掉的第一列有多少个元素
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)#节点id--就是前面去掉的第一列所有元素
#     idx_map = {j: i for i, j in enumerate(idx)}#节点对应字典--把上面的idx变成字典896: 0,idx是键
#     # print(idx_map)
#     edges_unordered = np.genfromtxt("{}107.edges".format(path, dataset),
#                                     dtype=np.int32)#直接把107.edges变成 n行2列 的二维矩阵
#     print(type(edges_unordered))#<class 'numpy.ndarray'>
#     # print(list(map(idx_map.get, edges_unordered.flatten())))
#     # 下面的idx就是RNA个数，edges_unordered就是CD矩阵的邻接表
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)#节点边--list结果是一维列表,edges是二维矩阵 n行2列
#     # print(edges)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                               shape=(m, m),
#                               dtype=np.float32)  # 邻接矩阵
#     # coo_matrix((data, (row, col)), shape=(4, 4))结果是row[i],col[i]位置赋值为data[i]
#     print(adj.shape,"********",features1.shape)
#     return adj,features1

def load_data_four(path="../data/facebook/", dataset="facebook"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}2.feat".format(path, dataset),
                                        dtype=np.dtype(str))#读取数据
    print(idx_features_labels.shape)
    features1 = sp.csr_matrix(idx_features_labels[:,1:-1], dtype=np.float32)#第一步处理--去掉矩阵的第一列，并将矩阵转化为邻接表
    # build graph
    m = features1.shape[0]#查看去掉的第一列有多少个元素
    idx = np.array(idx_features_labels[:, 0], dtype=np.float32)#节点id--就是前面去掉的第一列所有元素
    idx_map = {j: i for i, j in enumerate(idx)}#节点对应字典--把上面的idx变成字典896: 0,idx是键
    # print(idx_map)
    edges_unordered = np.genfromtxt("{}1.edges".format(path, dataset),
                                    dtype=np.int32)#直接把107.edges变成 n行2列 的二维矩阵
    # print(edges_unordered)
    # print(list(map(idx_map.get, edges_unordered.flatten())))
    # 下面的idx就是RNA个数，edges_unordered就是CD矩阵的邻接表
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)#节点边--list结果是一维列表,edges是二维矩阵 n行2列
    # print(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                              shape=(m, m),
                              dtype=np.int32)  # 邻接矩阵
    # coo_matrix((data, (row, col)), shape=(4, 4))结果是row[i],col[i]位置赋值为data[i]
    print(adj.shape,"********",features1.shape)
    return adj,features1


def load_alldata(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset_str, names[i]))))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, np.argmax(labels, 1)
