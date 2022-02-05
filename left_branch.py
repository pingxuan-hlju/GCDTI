
import scipy.sparse as sp
import heapq
import argparse
import networkx as nx
from AGithub.model import nodeattention as MI, node2vec2
from numpy import *
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.cuda.manual_seed(1)
random.seed(1)
torch.backends.cudnn.enabled = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='../big_data/adj_Graph',
                        help='Input graph path')
    parser.add_argument('--input_b', nargs='?', default='../big_data/adj_Graph2',
                        help='Input graph path')
    parser.add_argument('--input_c', nargs='?', default='../big_data/adj_Graph3',
                        help='Input graph path')
    parser.add_argument('--input_d', nargs='?', default='../big_data/adj_Graph4',
                        help='Input graph path')
    parser.add_argument('--input1', nargs='?', default='../big_data/adj.cites',
                        help='Input graph path')
    parser.add_argument('--input2', nargs='?', default='../big_data/adj2.cites',
                        help='Input graph path')
    parser.add_argument('--input3', nargs='?', default='../big_data/s_drug.cites',
                        help='Input graph path')
    parser.add_argument('--input4', nargs='?', default='../big_data/s_protein.cites',
                        help='Input graph path')
    parser.add_argument('--input5', nargs='?', default='../big_data/m_drug.cites',
                        help='Input graph path')
    parser.add_argument('--input6', nargs='?', default='../big_data/s_protein2.cites',
                        help='Input graph path')
    parser.add_argument('--input7', nargs='?', default='../big_data/s_drug2.cites',
                        help='Input graph path')
    parser.add_argument('--input8', nargs='?', default='../big_data/m_protein.cites',
                        help='Input graph path')
    parser.add_argument('--output', nargs='?', default='karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=8,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=0.8,
                        help='Return hyperparameter. Default is 1.txt.')

    parser.add_argument('--q', type=float, default=1.3,
                        help='Inout hyperparameter. Default is 1.txt.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=True)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()

def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    if not args.directed:
        G = G.to_undirected()
    return G
def read_graph1():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input_b, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input_b, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    if not args.directed:
        G = G.to_undirected()
    return G
def read_graph2():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input_c, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input_c, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    if not args.directed:
        G = G.to_undirected()
    return G
def read_graph3():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input_d, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input_d, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    if not args.directed:
        G = G.to_undirected()
    return G

def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph()
    G = node2vec2.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks, appearnum, nodes = G.simulate_walks(args.num_walks, args.walk_length)

    nx_G2 = read_graph1()
    G = node2vec2.Graph(nx_G2, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks2, appearnum2, nodes2 = G.simulate_walks(args.num_walks, args.walk_length)

    nx_G3 = read_graph2()
    G = node2vec2.Graph(nx_G3, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks3, appearnum3, nodes3 = G.simulate_walks(args.num_walks, args.walk_length)

    nx_G4 = read_graph3()
    G = node2vec2.Graph(nx_G4, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks4, appearnum4, nodes4 = G.simulate_walks(args.num_walks, args.walk_length)
    print('1-----------------------------------------------------')
    for i in range(2220):
        for j in range(i + 2220, i + 19981, 2220):
            appearnum[i] = appearnum[i] + appearnum[j]
    max_index2 = []
    for i in range(2220):
        a = appearnum[i].tolist()
        max_number = heapq.nlargest(20, a)
        max_index = []
        for t in max_number:
            index = a.index(t)
            max_index.append(index)
            a[index] = 0
        max_index2.append(max_index)
    print('2----------------------------------------------------------')
    for i in range(2220):
        for j in range(i + 2220, i + 19981, 2220):
            appearnum2[i] = appearnum2[i] + appearnum2[j]
    max_index3 = []
    for i in range(2220):
        a = appearnum2[i].tolist()
        max_number = heapq.nlargest(20, a)
        max_index = []
        for t in max_number:
            index = a.index(t)
            max_index.append(index)
            a[index] = 0
        max_index3.append(max_index)
    print('3-------------------------------------------------------------')
    for i in range(2220):
        for j in range(i + 2220, i + 19981, 2220):
            appearnum3[i] = appearnum3[i] + appearnum3[j]
    max_index4 = []
    for i in range(2220):
        a = appearnum2[i].tolist()
        max_number = heapq.nlargest(20, a)
        max_index = []
        for t in max_number:
            index = a.index(t)
            max_index.append(index)
            a[index] = 0
        max_index4.append(max_index)
    print('4-------------------------------------------------------------')
    for i in range(2220):
        for j in range(i + 2220, i + 19981, 2220):
            appearnum4[i] = appearnum4[i] + appearnum4[j]
    max_index5 = []
    for i in range(2220):
        a = appearnum2[i].tolist()
        max_number = heapq.nlargest(20, a)
        max_index = []
        for t in max_number:
            index = a.index(t)
            max_index.append(index)
            a[index] = 0
        max_index5.append(max_index)
    return max_index2,max_index3,max_index4,max_index5

def get_neighbor(max_index2):
    drug_neighbor = []
    protein_neighbor = []
    for i in range(len(max_index2)):
        protein_neighbor2 = []
        drug_neighbor2 = []
        drug_neighbor2.append(i)
        protein_neighbor2.append(i)
        for j in range(0, len(max_index2[i])):
            if (max_index2[i][j] <= 707):
                drug_neighbor2.append(max_index2[i][j])
            else:
                protein_neighbor2.append(max_index2[i][j])
        drug_neighbor.append(drug_neighbor2)
        protein_neighbor.append(protein_neighbor2)
    return drug_neighbor, protein_neighbor

def attentionadj():
    dp_D=[]
    dp_P=[]
    for i in range(2220):
        s = 0
        dp_D.append((i,i))
        for j in range(708):
            if j in drug_neighbor[i]:
               # if(s<=8):
                  dp_D.append((i, j))
                  s+=1
    #药物和蛋白的蛋白邻居
    for i in range(2220):
        s = 0
        dp_P.append((i,i))
        for j in range(708, 2220):
            if j in protein_neighbor[i]:
               # if(s<=8):
                  dp_P.append((i, j))
                  s+=1
    np.savetxt('../big_data/adj.cites', dp_D, fmt="%.0f", delimiter=" ")
    np.savetxt('../big_data/adj2.cites', dp_P, fmt="%.0f", delimiter=" ")
    print('-------------------------------------------------------------------------')
    a = []
    b = []
    # 药物和蛋白的药物邻居
    for i in range(2220):
        s = 0
        a.append((i, i))
        for j in range(708):
            if j in drug_neighbor2[i]:
                # if(s<=8):
                a.append((i, j))
                s += 1
    # 药物和蛋白的蛋白邻居
    for i in range(2220):
        s = 0
        b.append((i, i))
        for j in range(708, 2220):
            if j in protein_neighbor2[i]:
                # if(s<=8):
                b.append((i, j))
                s += 1
    np.savetxt('../big_data/s_drug.cites', a, fmt="%.0f", delimiter=" ")  # adj是药物和蛋白的药物邻居
    np.savetxt('../big_data/s_protein.cites', b, fmt="%.0f", delimiter=" ")  # adj2是药物和蛋白的蛋白邻居
    print('-----------------------------------------------------------------------------')
    a = []
    b = []
    # 药物和蛋白的药物邻居
    for i in range(2220):
        s = 0
        a.append((i, i))
        for j in range(708):
            if j in drug_neighbor3[i]:
                # if(s<=8):
                a.append((i, j))
                s += 1
    # 药物和蛋白的蛋白邻居
    for i in range(2220):
        s = 0
        b.append((i, i))
        for j in range(708, 2220):
            if j in protein_neighbor3[i]:
                # if(s<=8):
                b.append((i, j))
                s += 1
    np.savetxt('../big_data/m_drug.cites', a, fmt="%.0f", delimiter=" ")
    np.savetxt('../big_data/s_protein2.cites', b, fmt="%.0f", delimiter=" ")
    print('-----------------------------------------------------------------------------')
    a = []
    b = []
    # 药物和蛋白的药物邻居
    for i in range(2220):
        s = 0
        a.append((i, i))
        for j in range(708):
            if j in drug_neighbor4[i]:
                # if(s<=8):
                a.append((i, j))
                s += 1
    # 药物和蛋白的蛋白邻居
    for i in range(2220):
        s = 0
        b.append((i, i))
        for j in range(708, 2220):
            if j in protein_neighbor4[i]:
                # if(s<=8):
                b.append((i, j))
                s += 1
    np.savetxt('../big_data/s_drug2.cites', a, fmt="%.0f", delimiter=" ")
    np.savetxt('../big_data/m_protein.cites', b, fmt="%.0f", delimiter=" ")

# Load data
def load_data3(path="../big_data/", dataset="Drug_Protein_FeatureMatrix",dataset2="Drug_Protein_FeatureMatrix2",dataset3="Drug_Protein_FeatureMatrix3",dataset4="Drug_Protein_FeatureMatrix4"):
    """Load citation network dataset (cora only for now)"""
    features1 = np.genfromtxt("{}{}.txt".format(path, dataset), dtype=np.dtype(str))
    features1 = sp.csr_matrix(features1, dtype=np.float32)  # 转换为稀疏矩阵
    features2 = np.genfromtxt("{}{}.txt".format(path, dataset2), dtype=np.dtype(str))
    features2 = sp.csr_matrix(features2, dtype=np.float32)  # 转换为稀疏矩阵
    features3 = np.genfromtxt("{}{}.txt".format(path, dataset3), dtype=np.dtype(str))
    features3 = sp.csr_matrix(features3, dtype=np.float32)  # 转换为稀疏矩阵
    features4 = np.genfromtxt("{}{}.txt".format(path, dataset4), dtype=np.dtype(str))
    features4 = sp.csr_matrix(features4, dtype=np.float32)  # 转换为稀疏矩阵

    edges = np.genfromtxt("{}{}.cites".format(path, 'adj'), dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(features1.shape[0], features1.shape[0]), dtype=np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))

    edges = np.genfromtxt("{}{}.cites".format(path, 's_drug'), dtype=np.int32)
    s_drug = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(features1.shape[0], features1.shape[0]), dtype=np.float32)
    s_drug = normalize_adj(s_drug + sp.eye(s_drug.shape[0]))
    s_drug = torch.FloatTensor(np.array(s_drug.todense()))

    edges = np.genfromtxt("{}{}.cites".format(path, 's_protein'), dtype=np.int32)
    s_protein = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                           shape=(features1.shape[0], features1.shape[0]), dtype=np.float32)
    s_protein = normalize_adj(s_protein + sp.eye(s_protein.shape[0]))
    s_protein = torch.FloatTensor(np.array(s_protein.todense()))

    edges = np.genfromtxt("{}{}.cites".format(path, 'm_drug'), dtype=np.int32)
    m_drug = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                              shape=(features1.shape[0], features1.shape[0]), dtype=np.float32)
    m_drug = normalize_adj(m_drug + sp.eye(m_drug.shape[0]))
    m_drug = torch.FloatTensor(np.array(m_drug.todense()))


    edges = np.genfromtxt("{}{}.cites".format(path, 's_protein2'), dtype=np.int32)
    s_protein2 = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                              shape=(features1.shape[0], features1.shape[0]), dtype=np.float32)
    s_protein2 = normalize_adj(s_protein2 + sp.eye(s_protein2.shape[0]))
    s_protein2 = torch.FloatTensor(np.array(s_protein2.todense()))

    edges = np.genfromtxt("{}{}.cites".format(path, 's_drug2'), dtype=np.int32)
    s_drug2 = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                               shape=(features1.shape[0], features1.shape[0]), dtype=np.float32)
    s_drug2 = normalize_adj(s_drug2 + sp.eye(s_drug2.shape[0]))
    s_drug2 = torch.FloatTensor(np.array(s_drug2.todense()))

    edges = np.genfromtxt("{}{}.cites".format(path, 'm_protein'), dtype=np.int32)
    m_protein = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                               shape=(features1.shape[0], features1.shape[0]), dtype=np.float32)
    m_protein = normalize_adj(m_protein + sp.eye(m_protein.shape[0]))
    m_protein = torch.FloatTensor(np.array(m_protein.todense()))  # 将其返回此矩阵的密集矩阵表示形式 并转换成floattensor的形式

    features1 = torch.FloatTensor(np.array(features1.todense()))  # 将其返回此矩阵的密集矩阵表示形式 并转换成floattensor的形式
    features2 = torch.FloatTensor(np.array(features2.todense()))  # 将其返回此矩阵的密集矩阵表示形式 并转换成floattensor的形式
    features3 = torch.FloatTensor(np.array(features3.todense()))  # 将其返回此矩阵的密集矩阵表示形式 并转换成floattensor的形式
    features4 = torch.FloatTensor(np.array(features4.todense()))  # 将其返回此矩阵的密集矩阵表示形式 并转换成floattensor的形式

    return adj, s_drug,s_protein,m_drug,s_protein2,s_drug2,m_protein,features1,features2,features3,features4
# Load data
def load_data2(path="../big_data/", dataset="Drug_Protein_FeatureMatrix"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    features1 = np.genfromtxt("{}{}.txt".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(features1, dtype=np.float32)
    edges = np.genfromtxt("{}{}.cites".format(path, 'adj2'), dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))

    return adj
# 标准化图
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
# 标准化特征
def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))           # 求和
    r_inv = np.power(rowsum, -1).flatten() # 取倒数
    r_inv[np.isinf(r_inv)] = 0.            # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)            # 把它变成对角矩阵
    mx = r_mat_inv.dot(mx)                 # 点乘起来
    return mx
# --- lncRNA similarity
def get_index(lncrna):
    index = []
    for i in range(len(lncrna)):
        if lncrna[i] == 1:
            index.append(i)
    return index

def read_data_flies():
    A = np.loadtxt("../big_data2/mat_drug_protein.txt")
    Sd_pp = np.loadtxt("../big_data2/Similarity_Matrix_Proteins.txt")
    Sm_dd = np.loadtxt("../big_data2/Similarity_Matrix_Drugs.txt")
    mat_dd = np.loadtxt("../big_data2/mat_drug_drug.txt")
    mat_pp = np.loadtxt("../big_data2/mat_protein_protein.txt")
    Sm_dd_protein = np.loadtxt("../big_data2/Drug_Protein_SIM.txt")
    Sm_pp_protein = np.loadtxt("../big_data2/Protein_Drug_SIM.txt")
    return A, Sd_pp, Sm_dd, mat_dd, mat_pp, Sm_dd_protein, Sm_pp_protein


# define a class
class value_index():
    def __init__(self, num, i, j):
        self.value = num  # save value
        self.value_x = i  # save index of row
        self.value_y = j  # save index of column


# define input data
class data_input():
    def __init__(self, value, x, y):
        self.value = value  # value 0 or 1
        self.index_x = x  # the row in A_association   A(x,y)
        self.index_y = y  # the column in A_association A(x,y)
    def add_probability_predict(self, probability, predict_value):
        self.probability = probability
        self.predict_value = predict_value

def testgetxandytofixtestresulttothematrix(data):
    list2 = []  # 纵坐标
    # print(len(data),'data')       3762个1       1
    for j in range(len(data)):
        temp_save = []  # cat features
        x_A = data[j].index_x  # 横坐标
        y_A = data[j].index_y
        list = [x_A, y_A]
        list2.append(list)
    # print(list2, 'list')
    return list2
def load_data(data, A, Rna_matrix, disease_matrix, BATCH_SIZE):
    # load_data2(all_k_Sd_associated[i], save_all_count_A[i], Sm_dd, Sd_pp, batchsize, Lm_ddi, Md_pdi)
    x = []  # all_k_development[i],save_all_count_A[i],Sm,Sd,batchsize,Lm,Md
    y = []
    z=[]
    print(len(data),'data')
    for j in range(len(data)):
        x.append(data[j].index_x)
        y.append(data[j].index_y)
        z.append(data[j].value)

    x = torch.LongTensor(np.array(x))
    y = torch.LongTensor(np.array(y))
    value = torch.LongTensor(np.array(z))
    torch_dataset = Data.TensorDataset(x, y,value)
    # 数据集
    data2_loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=64,  # mini batch size
        shuffle=False,  # random shuffle for training
        num_workers=1,  # subprocesses for loading data 工作进程
        drop_last=False  # ,drop_last告诉如何处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留
    )
    return data2_loader
def load_datatrue(data,labelmatrix,lncanddiseasezxw):

    x = []  # all_k_development[i],save_all_count_A[i],Sm,Sd,batchsize,Lm,Md
    y = []
    z = []
    for j in range(len(data)):
                x.append(data[j].index_x)
                y.append(data[j].index_y)
                z.append(data[j].value)
    x = torch.LongTensor(np.array(x))
    y = torch.LongTensor(np.array(y))
    value = torch.LongTensor(np.array(z))
    torch_dataset = Data.TensorDataset(x, y,value)

    # 数据集
    data2_loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=1000,  # mini batch size
        shuffle=True,  # random shuffle for training
        num_workers=1,  # subprocesses for loading data 工作进程
        drop_last=True  # ,drop_last告诉如何处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留
    )
    return data2_loader,labelmatrix,lncanddiseasezxw


class Attention2(nn.Module):
    def __init__(self, input_size, output_size):  # 240 100
        super(Attention2, self).__init__()

        self.node = nn.Linear(input_size, output_size, bias=True)
        nn.init.xavier_normal_(self.node.weight)
        self.h_n_parameters = nn.Parameter(torch.randn(output_size, input_size))
        nn.init.xavier_normal_(self.h_n_parameters)

    def forward(self, h_n_states):  # print(h_n_states.shape,'hnstatr')   # 50 1 240

        temp_nodes = self.node(h_n_states)  # torch.Size([50, 1, 100])
        temp_nodes = torch.tanh(temp_nodes)  # torch.Size([50, 1, 100])
        nodes_score = torch.matmul(temp_nodes,
                                   self.h_n_parameters)  # self.h_n_parameters:torch.Size([100, 240])   nodes_score.size():torch.Size([50, 1, 240])
        alpha = F.softmax(nodes_score, dim=2)  # torch.Size([50, 1, 240])
        y_i = alpha * h_n_states
        return y_i


class Beta_score2(nn.Module):
    def __init__(self, input_size_lnc, input_size_A,  input_size_AT, input_size_dis,
                 output_size, batch_size):
        super(Beta_score2, self).__init__()
        self.node1 = nn.Linear(input_size_lnc + input_size_A, output_size, bias=True)  # 1140 100
        nn.init.xavier_normal_(self.node1.weight)
        self.h_n_parameters = nn.Parameter(torch.randn(output_size, 1))  # 生成100*1的矩阵
        nn.init.xavier_normal_(self.h_n_parameters)
    def forward(self, result_ls, result_A, result_AT, result_ds):  # result: 50*1**240/405/495
        length = result_ls.size()[2] + result_A.size()[2] # length  240+405+495=1140
        batch_size_ = result_ls.size()[0]  # 50个 batchsize
        ls_pad = Variable(torch.zeros(batch_size_, result_ls.size()[1],
                                      length - result_ls.size()[2]))  # 进行拼接 拼接成 【50，1，1140】的长度样式
        ls_pad = torch.cat((result_ls, ls_pad), dim=2)  # [50, 1, 1140]
        result_A_pad = Variable(torch.zeros(batch_size_, result_A.size()[1], length - result_A.size()[2]))
        result_A_pad = torch.cat((result_A, result_A_pad), dim=2)
        result_AT_pad = Variable(torch.zeros(batch_size_, result_AT.size()[1], length - result_AT.size()[2]))
        result_AT_pad = torch.cat((result_AT, result_AT_pad), dim=2)
        result_ds_pad = Variable(torch.zeros(batch_size_, result_ds.size()[1], length - result_ds.size()[2]))
        result_ds_pad = torch.cat((result_ds, result_ds_pad), dim=2)
        reslut = torch.cat((ls_pad, result_A_pad), dim=1)
        reslut = torch.cat((reslut, result_AT_pad), dim=1)
        reslut = torch.cat((reslut, result_ds_pad), dim=1)
        temp_nodes = self.node1(reslut)
        temp_nodes = torch.tanh(temp_nodes)
        nodes_score = torch.matmul(temp_nodes, self.h_n_parameters)
        print(nodes_score.size())
        nodes_score = nodes_score.view(-1, 1, 4)
        beta = F.softmax(nodes_score, dim=2)
        z_i = torch.matmul(beta, reslut)
        print(z_i.size(),'zi')
        return z_i

def setlabelmatrixtofuone(lncanddiseasezxw, alloneexcepttestzxw):
    # print(len(alloneexcepttestzxw))
    listzxw = testgetxandytofixtestresulttothematrix(alloneexcepttestzxw)
    B = lncanddiseasezxw
    sum = 0
    for i in range(708):
        for j in range(1512):
            for kk in range(len(alloneexcepttestzxw)):
                if (i == (listzxw[kk][0]) and j == (listzxw[kk][1])):
                    B[i][j] = -50
                    # print(i,j)
                    sum = sum + 1
    print(sum, 'sum')
    return B


def testzxw(model, test_loader, LR, lncanddiseasezxw, all_k_test_data,adj_drug,
            features1, adj_protein, features2,features3,features4,
            adj_drug2, adj_protein2, adj_drug3,adj_protein3,adj_drug4,adj_protein4):

    listzxw = testgetxandytofixtestresulttothematrix(all_k_test_data)
    loss_func = nn.CrossEntropyLoss()
    model.eval()
    same_number1 = 0.0
    same_number_length1 = 0.0
    kk = 0
    for step, (x_A, y_A, y) in enumerate(test_loader):
        X_A = Variable(x_A)
        Y_A = Variable(y_A)
        b_y = Variable(y)
        output = model(features1, features2, features3, features4, adj_drug, adj_protein, X_A, Y_A,adj_drug2, adj_protein2, adj_drug3, adj_protein3, adj_drug4,
         adj_protein4)
        output=F.softmax(output,dim=1)
        loss = loss_func(output, b_y)
        pred_train_y = torch.max(output, 1)[1].data.squeeze().int()  # torch.Size([50])
        for j in range(1000):
            lncanddiseasezxw[X_A[j]][Y_A[j]] = output[j][1].item()
            kk = kk + 1
        same_number1 += sum(np.array((pred_train_y).cpu()) == np.array((b_y).cpu()))
        same_number_length1 += b_y.size(0)
        same_number1 = float(same_number1)
        same_number_length1 = float(same_number_length1)
        accuracy1 = same_number1 / same_number_length1
        print('Epoch: ', step, '| test loss: %.4f' % loss.item(), '| test accuracy: %.4f' % accuracy1)
    return lncanddiseasezxw


def train_top(modell, epoch,lr, adj_drug, features, adj_protein, features2,features3,features4,train_loader,adj_drug2, adj_protein2, adj_drug3,adj_protein3,adj_drug4,adj_protein4):
    optimizer = torch.optim.SGD(modell.parameters(), lr=0.001, weight_decay=1e-8)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(90):
        same_number1 = 0.0
        same_number_length1 = 0.0
        modell.train()
        for step, (x_A, y_A, y) in enumerate(train_loader):
            modell.train()
            X_A = Variable(x_A)
            Y_A = Variable(y_A)
            b_y = Variable(y)
            output = modell(features, features2,features3,features4,
                            adj_drug, adj_protein, X_A, Y_A,adj_drug2, adj_protein2, adj_drug3,adj_protein3,adj_drug4,
                            adj_protein4)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_train_y = torch.max(output, 1)[1].data.squeeze().int()  # torch.Size([50])
            hduias = torch.max(output, 1)[0].data.squeeze()
            same_number1 += sum(np.array((pred_train_y).cpu()) == np.array((b_y).cpu()))
            same_number_length1 += b_y.size(0)
        same_number1 = float(same_number1)
        same_number_length1 = float(same_number_length1)
        accuracy1 = same_number1 / same_number_length1
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| train accuracy: %.4f' % accuracy1)
        print(output)
# files operator
def save_to_file2(file_name, contents):
    with open(file_name, 'a') as f:
        f.write(contents + '\n')

def createdrug_matrix(Sm_dd,save_all_count_A):
    dd=torch.Tensor(Sm_dd)
    A=torch.Tensor(save_all_count_A)
    feature_matrix=torch.cat((dd,A),1)
    return feature_matrix

def createprotein_matrix(Sm_pp, save_all_count_A):
    pp = torch.Tensor(Sm_pp)
    A = torch.Tensor(save_all_count_A)
    A = A.t()
    feature_matrix = torch.cat((A, pp), 1)
    return feature_matrix

def argsjj():
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser1.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser1.add_argument('--sparse', action='store_true', default=True, help='GAT with sparse version or not.')
    parser1.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser1.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')  # 应该设为几？训练轮数 是否要训练
    parser1.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser1.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser1.add_argument('--hidden', type=int, default=1000, help='Number of hidden units.')  # 应该设为几？中间隐藏层的结点个数
    parser1.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')  # 应该设为几？多头注意力
    parser1.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1.txt - keep probability).')
    parser1.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser1.add_argument('--patience', type=int, default=100, help='Patience')

    args1 = parser1.parse_args()
    return args1

def tes8_tes(k):
    A, Sd_pp, Sm_dd, mat_dd, mat_pp, Sm_dd_protein, Sm_pp_protein = read_data_flies()
    alloneexcepttestzxw, all_k_Sd_associated, all_k_count_positive, all_k_test_data, num, A_length, sava_association_A, \
    all_k_development, save_all_count_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, k_Sd_associated = data_partitioning1(
        A, k)
    _LR = 0.0005
    for i in range(k):
        feature_matrix_drug  = createdrug_matrix(Sm_dd, save_all_count_A[i])
        feature_matrix_protein = createprotein_matrix(Sd_pp,save_all_count_A[i])
        hhh=torch.cat((feature_matrix_drug,feature_matrix_protein),0)
        np.savetxt('../big_data/Drug_Protein_FeatureMatrix.txt', hhh, fmt="%.6f",
                   delimiter=" ")
        feature_matrix_drug2 = createdrug_matrix(mat_dd, save_all_count_A[i])
        feature_matrix_protein2 = createprotein_matrix(Sd_pp, save_all_count_A[i])
        hhh = torch.cat((feature_matrix_drug2, feature_matrix_protein2), 0)
        np.savetxt('../big_data/Drug_Protein_FeatureMatrix2.txt', hhh, fmt="%.6f",
                   delimiter=" ")
        feature_matrix_drug3 = createdrug_matrix(mat_dd,save_all_count_A[i])
        feature_matrix_protein3 = createprotein_matrix(mat_pp,save_all_count_A[i])
        hhh = torch.cat((feature_matrix_drug3, feature_matrix_protein3), 0)
        np.savetxt('../big_data/Drug_Protein_FeatureMatrix3.txt', hhh, fmt="%.6f",
                   delimiter=" ")
        feature_matrix_drug4 = createdrug_matrix(Sm_dd,save_all_count_A[i])
        feature_matrix_protein4 = createprotein_matrix(mat_pp,save_all_count_A[i])
        hhh = torch.cat((feature_matrix_drug4, feature_matrix_protein4), 0)
        np.savetxt('../big_data/Drug_Protein_FeatureMatrix4.txt', hhh, fmt="%.6f",
                   delimiter=" ")
        train_loader = load_data(all_k_Sd_associated[i], save_all_count_A[i], Sm_dd, Sd_pp, 50)
        adj_drug ,adj_drug2, adj_protein2, adj_drug3,adj_protein3,adj_drug4,adj_protein4, features1,features2,features3,features4 =  load_data3()
        adj_protein = load_data2()
        args11 = argsjj()
        model = MI.the_modell(nfeat=features1.shape[1],
                               nhid=args11.hidden,
                               nclass=500,
                               dropout=args11.dropout,
                               nheads=args11.nb_heads,
                               alpha=args11.alpha
                               )
        train_top(model, 80,0.001, adj_drug, features1, adj_protein, features2,features3,features4,train_loader,adj_drug2, adj_protein2, adj_drug3,adj_protein3,adj_drug4,adj_protein4)
        lncanddiseasezxw = np.loadtxt("../big_data2/mat_drug_protein.txt")  #
        lncanddiseasezxw2 = np.loadtxt("../big_data2/mat_drug_protein.txt")  # 240 * 405  lncRNA * diseases
        labelmatrix = setlabelmatrixtofuone(lncanddiseasezxw2, alloneexcepttestzxw[i])  # 把测试的数据都设为-1
        lncanddiseasezxw = setlabelmatrixtofuone(lncanddiseasezxw, alloneexcepttestzxw[i])
        test_loader,labelmatrix,lncanddiseasezxw = load_datatrue(all_k_test_data[i], labelmatrix, lncanddiseasezxw)
        if (i == 0):
            scoresmatrix = testzxw(model, test_loader, 0.0005, lncanddiseasezxw, all_k_test_data[i],adj_drug, features1, adj_protein, features2,features3,features4,
                                   adj_drug2, adj_protein2, adj_drug3,adj_protein3,adj_drug4,adj_protein4)
    return labelmatrix, scoresmatrix


if __name__ == '__main__':

    k = 5
    args = parse_args()
    max_index,max_index2,max_index3,max_index4 = main(args)
    drug_neighbor, protein_neighbor   = get_neighbor(max_index)
    drug_neighbor2, protein_neighbor2 = get_neighbor(max_index2)
    drug_neighbor3, protein_neighbor3 = get_neighbor(max_index3)
    drug_neighbor4, protein_neighbor4 = get_neighbor(max_index4)
    attentionadj()
    labelmatrix, scoresmatrix = tes8_tes(k)

"""Final"""




















