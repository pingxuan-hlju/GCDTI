
import matplotlib.pyplot as plt
from numpy import *
from sklearn.metrics import auc
import random
import os
import numpy as np
import copy
from operator import attrgetter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def read_data_flies():

    A = np.loadtxt("../big_data2/mat_drug_protein.txt")
    Sd_pp = np.loadtxt("../big_data2/Similarity_Matrix_Proteins.txt")
    Sm_dd = np.loadtxt("../big_data2/Similarity_Matrix_Drugs.txt")
    mat_dd = np.loadtxt("../big_data2/mat_drug_drug.txt")
    mat_pp = np.loadtxt("../big_data2/mat_protein_protein.txt")
    Sm_dd_protein= np.loadtxt("../big_data2/Drug_Protein_SIM.txt")
    Sm_pp_protein = np.loadtxt("../big_data2/Protein_Drug_SIM.txt")
    return A, Sd_pp, Sm_dd, mat_dd, mat_pp, Sm_dd_protein,Sm_pp_protein

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
        self.index_x = x
        self.index_y = y
    def add_probability_predict(self, probability, predict_value):
        self.probability = probability
        self.predict_value = predict_value

def get_test_data(data, A, Rna_matrix, disease_matrix, lnc_mi, mi_dis):
    x = []
    y = []
    for j in range(len(data)):
        temp_save = []
        x_A = data[j].index_x
        y_A = data[j].index_y
        rna_disease_mi = np.concatenate((Rna_matrix[x_A], A[x_A], lnc_mi[x_A]), axis=0)
        disease_rna_mi = np.concatenate((disease_matrix[y_A], A[:, y_A], mi_dis[:, y_A]), axis=0)
        temp_save.append(rna_disease_mi)
        temp_save.append(disease_rna_mi)
        x.append([temp_save])
        y.append(data[j].value)
    test_x = Variable(torch.FloatTensor(np.array(x))).cuda()
    test_y = torch.IntTensor(np.array(y)).cuda()
    return test_x, test_y

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


def load_data2(data, A, Rna_matrix, disease_matrix, BATCH_SIZE, drop=False):
    x = []
    y = []
    z = []
    for j in range(len(data)):
        x.append(data[j].index_x)
        y.append(data[j].index_y)
        z.append(data[j].value)
    x = torch.LongTensor(np.array(x))
    y = torch.LongTensor(np.array(y))
    value = torch.LongTensor(np.array(z))


    torch_dataset = Data.TensorDataset(x, y, value)
    # 数据集
    data2_loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=False,  # random shuffle for training
        num_workers=1,  # subprocesses for loading data 工作进程
        drop_last=drop
    )
    return data2_loader


def load_data3(labelmatrix, lncanddiseasezxw, data, A, Rna_matrix, disease_matrix, BATCH_SIZE,
               drop=False):
    x = []
    y = []
    z = []

    for j in range(len(data)):
        x.append(data[j].index_x)
        y.append(data[j].index_y)
        z.append(data[j].value)
    x = torch.LongTensor(np.array(x))
    y = torch.LongTensor(np.array(y))
    value = torch.LongTensor(np.array(z))
    torch_dataset = Data.TensorDataset(x, y, value)
    # 数据集
    data2_loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=1000,  # mini batch size
        shuffle=False,  # random shuffle for training
        num_workers=1,  # subprocesses for loading data 工作进程
        drop_last=True  # ,drop_last告诉如何处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留
    )
    return data2_loader, labelmatrix, lncanddiseasezxw


class the_model(nn.Module):
    def __init__(self, input_size_lnc, input_size_A, input_size_dis, output_size,
                 batchsize):
        super(the_model, self).__init__()

        self.kua1=nn.Sequential(
            nn.Conv3d(in_channels=1,
                 out_channels=8,
                 kernel_size=(3,2,2),
                 stride=1,
                 padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(1, 1, 2))
           )
        
        self.kua2=nn.Sequential(
            nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3,2, 2),
                stride=1,
                padding=1,
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3,2,2),stride=(2,2,2)),
        )
        self.fc= nn.Sequential(
            nn.Linear(26640,2),
        )
    def forward(self, x3):
        x3=x3.unsqueeze(1)
        kua = self.kua1(x3)
        kua2 = self.kua2(kua)
        X = kua2.view(kua2.size(0),-1)
        output = self.fc(X)
        return output

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


def changenorm(data1, data2):
    data3 = torch.stack((data1, data2), dim=1)
    return data3
def changenorm2(data1, data2,data3,data4,data5, data6,data7,data8,data9):
    data3 = torch.stack((data1, data2,data3,data4,data5, data6,data7,data8,data9), dim=1)
    return data3
def changenorm3(data1,data2,data3,data4,data5,data6,data7,data8):
    data3 = torch.stack((data1,data2,data3,data4,data5,data6,data7,data8), dim=1)
    return data3


def testzxw(model, test_loader, LR, lncanddiseasezxw, all_k_test_data, feature_matrix_drug, feature_matrix_protein,
                  feature_matrix_drug2,feature_matrix_protein2,
                  feature_matrix_drug4,feature_matrix_protein4,
                  feature_matrix_drug5, feature_matrix_protein5,
                  feature_matrix_drug6, feature_matrix_protein6,
                  feature_matrix_drug8, feature_matrix_protein8,
                  feature_matrix_drug10, feature_matrix_protein10,
                  feature_matrix_drug11, feature_matrix_protein11,
                  feature_matrix_drug12, feature_matrix_protein12,

           ):

    listzxw = testgetxandytofixtestresulttothematrix(all_k_test_data)
    loss_func = nn.CrossEntropyLoss()
    model.eval()
    same_number1 = 0.0
    same_number_length1 = 0.0
    kk = 0
    jishu=0

    for step, (xx1, yy1, z) in enumerate(test_loader):
        # print('---------------------------')
        x = Variable(xx1)
        y = Variable(yy1)
        train_y = Variable(z)

        f1 = feature_matrix_drug[x]  # 序列相似性
        f111 = feature_matrix_protein[y]
        f2 = feature_matrix_drug2[x]  # 互作
        f22 = feature_matrix_protein2[y]
        f4 = feature_matrix_drug4[x]  # 根据互作算出的相似性
        f44 = feature_matrix_protein4[y]
        f5 = feature_matrix_drug5[x]  # 序列相似性
        f55 = feature_matrix_protein5[y]
        f6 = feature_matrix_drug6[x]  # 互作
        f66 = feature_matrix_protein6[y]
        f8 = feature_matrix_drug8[x]  # 根据互作算出的相似性
        f88 = feature_matrix_protein8[y]
        f10 = feature_matrix_drug10[x]  # 互作
        f1010 = feature_matrix_protein10[y]
        f11 = feature_matrix_drug11[x]  # 根据疾病算出的相似性
        f1111 = feature_matrix_protein11[y]
        f12 = feature_matrix_drug12[x]  # 根据互作算出的相似性
        f1212 = feature_matrix_protein12[y]
        f = changenorm(f1, f111)  # 50  2 2220
        fa = changenorm(f2, f22)  # 50  2 2220
        fc = changenorm(f4, f44)  # 50  2 2220
        fd = changenorm(f5, f55)
        fe = changenorm(f6, f66)
        fg = changenorm(f8, f88)
        gl = changenorm(f10, f1010)
        li = changenorm(f11, f1111)
        aw = changenorm(f12, f1212)
        data3 = changenorm2(f, fa, fc, fd, fe, fg, gl, li, aw)  # fd,aw,sa f
        train_output = model(data3)
        train_output=F.softmax(train_output)

        loss = loss_func(train_output, train_y)
        # print(loss)
        pred_train_y = torch.max(train_output, 1)[1].data.squeeze().int()  # torch.Size([50])
        for j in range(1000):
            lncanddiseasezxw[listzxw[kk][0]][listzxw[kk][1]] = train_output[j][1].item()
            kk = kk + 1
        same_number1 += sum(np.array((pred_train_y).cpu()) == np.array((train_y).cpu()))
        same_number_length1 += train_y.size(0)
        same_number1 = float(same_number1)
        same_number_length1 = float(same_number_length1)
        accuracy1 = same_number1 / same_number_length1
        jishu=jishu+1
        print(jishu,':| test accuracy: %.4f' % accuracy1, '| test loss: %.4f' % loss)
    return lncanddiseasezxw


def train_top(model, EPOCH, train_loader, LR, feature_matrix_drug, feature_matrix_protein,
              feature_matrix_drug2, feature_matrix_protein2,
              feature_matrix_drug4, feature_matrix_protein4,
              feature_matrix_drug5, feature_matrix_protein5,
              feature_matrix_drug6, feature_matrix_protein6,
              feature_matrix_drug8, feature_matrix_protein8,
              feature_matrix_drug10, feature_matrix_protein10,
              feature_matrix_drug11, feature_matrix_protein11,
              feature_matrix_drug12, feature_matrix_protein12,

              ):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-8)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(200):
        for step, (x, y, z) in enumerate(train_loader):
            model.train()
            f1 = feature_matrix_drug[x]
            f111 = feature_matrix_protein[y]
            f2=feature_matrix_drug2[x]
            f22 =feature_matrix_protein2[y]
            f4=feature_matrix_drug4[x]
            f44=feature_matrix_protein4[y]
            f5 = feature_matrix_drug5[x]
            f55= feature_matrix_protein5[y]
            f6 = feature_matrix_drug6[x]
            f66 = feature_matrix_protein6[y]
            f8 = feature_matrix_drug8[x]  #
            f88 = feature_matrix_protein8[y]
            f10 = feature_matrix_drug10[x]
            f1010 = feature_matrix_protein10[y]
            f11 = feature_matrix_drug11[x]
            f1111 = feature_matrix_protein11[y]
            f12 = feature_matrix_drug12[x]
            f1212 = feature_matrix_protein12[y]
            f  = changenorm(f1, f111)
            fa = changenorm(f2,f22)
            fc = changenorm(f4,f44)
            fd=changenorm(f5,f55)
            fe=changenorm(f6,f66)
            fg=changenorm(f8,f88)
            gl=changenorm(f10,f1010)
            li=changenorm(f11,f1111)
            aw=changenorm(f12,f1212)
            data3=changenorm2(f,fa,fc,fd,fe,fg,gl,li,aw)#fd,aw,sa f
            data3=Variable(data3)
            output = model(data3)
            loss = loss_func(output, z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                sum2 = 0
                print(epoch, 'epoch')
                print(step, 'step')
                same_number1 = 0.0
                same_number_length1 = 0.0
                for _, (x, y, z) in enumerate(train_loader):
                    x = Variable(x)
                    y=Variable(y)
                    train_y = z.int()
                    f1 = feature_matrix_drug[x]  # 序列相似性
                    f111 = feature_matrix_protein[y]
                    f2 = feature_matrix_drug2[x]  # 互作
                    f22 = feature_matrix_protein2[y]
                    f4 = feature_matrix_drug4[x]  # 根据互作算出的相似性
                    f44 = feature_matrix_protein4[y]
                    f5 = feature_matrix_drug5[x]  # 序列相似性
                    f55 = feature_matrix_protein5[y]
                    f6 = feature_matrix_drug6[x]  # 互作
                    f66 = feature_matrix_protein6[y]
                    f8 = feature_matrix_drug8[x]  # 根据互作算出的相似性
                    f88 = feature_matrix_protein8[y]
                    f10 = feature_matrix_drug10[x]  # 互作
                    f1010 = feature_matrix_protein10[y]
                    f11 = feature_matrix_drug11[x]  # 根据疾病算出的相似性
                    f1111 = feature_matrix_protein11[y]
                    f12 = feature_matrix_drug12[x]  # 根据互作算出的相似性
                    f1212 = feature_matrix_protein12[y]
                    f = changenorm(f1, f111)  # 50  2 2220
                    fa = changenorm(f2, f22)  # 50  2 2220
                    fc = changenorm(f4, f44)  # 50  2 2220
                    fd = changenorm(f5, f55)
                    fe = changenorm(f6, f66)
                    fg = changenorm(f8, f88)
                    gl = changenorm(f10, f1010)
                    li = changenorm(f11, f1111)
                    aw = changenorm(f12, f1212)
                    data3 = changenorm2(f, fa, fc, fd, fe, fg, gl, li, aw)  # fd,aw,sa f
                    train_output = model(data3)
                    pred_train_y = torch.max(train_output, 1)[1].data.squeeze().int()  # torch.Size([50])
                    hduias = torch.max(train_output, 1)[0].data.squeeze()
                    same_number1 += sum(np.array((pred_train_y).cpu()) == np.array((train_y).cpu()))
                    same_number_length1 += train_y.size(0)
                same_number1 = float(same_number1)
                same_number_length1 = float(same_number_length1)
                accuracy1 = same_number1 / same_number_length1
                print('Epoch: ', epoch, '|| train loss: %.8f' % loss.item(), '| train accuracy: %.6f' % accuracy1)


def save_to_file2(file_name, contents):
    with open(file_name, 'a') as f:
        f.write(contents + '\n')



def createdrug_matrix(Sm_dd, save_all_count_A):
    dd = torch.Tensor(Sm_dd)
    A = torch.Tensor(save_all_count_A)
    feature_matrix = torch.cat((dd, A), 1)


    return feature_matrix


def createprotein_matrix(save_all_count_A, Sd_pp):
    a = torch.Tensor(save_all_count_A)
    a = a.t()
    pp = torch.Tensor(Sd_pp)
    feature_matrix = torch.cat((a, pp), 1)
    return feature_matrix


def tes8_tes(k):
    print('!')
    A, Sd_pp, Sm_dd, mat_dd, mat_pp, Sm_dd_protein,Sm_pp_protein= read_data_flies()
    alloneexcepttestzxw, all_k_Sd_associated, all_k_count_positive, all_k_test_data, num, A_length, sava_association_A, \
    all_k_development, save_all_count_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, k_Sd_associated = data_partitioning1(
        A, k)
    epoch1 = 80
    epoch2 = 10
    _LR = 0.0005
    batchsize = 50
    input_size_lnc = len(A)
    input_size_A = len(A[0])
    input_size_dis = input_size_A

    output_size = 100


    for i in range(k):
        np.savetxt('/home/zhangxiaowen/RGA/result2/save_all_count_A2.txt',save_all_count_A[i])
        feature_matrix_drug = createdrug_matrix(Sm_dd, save_all_count_A[i])
        feature_matrix_protein = createprotein_matrix(save_all_count_A[i], Sd_pp)
        feature_matrix_drug2 = createdrug_matrix(Sm_dd, save_all_count_A[i])
        feature_matrix_protein2 = createprotein_matrix(save_all_count_A[i], mat_pp)
        feature_matrix_drug3 = createdrug_matrix(Sm_dd, save_all_count_A[i])
        feature_matrix_protein3 = createprotein_matrix(save_all_count_A[i], Sm_pp_protein)


        feature_matrix_drug4 = createdrug_matrix(mat_dd,save_all_count_A[i])
        feature_matrix_protein4=createprotein_matrix(save_all_count_A[i],mat_pp)
        feature_matrix_drug5 = createdrug_matrix(mat_dd, save_all_count_A[i])
        feature_matrix_protein5 = createprotein_matrix(save_all_count_A[i], Sd_pp)
        feature_matrix_drug6 = createdrug_matrix(mat_dd, save_all_count_A[i])
        feature_matrix_protein6 = createprotein_matrix(save_all_count_A[i], Sm_pp_protein)


        feature_matrix_drug7=createdrug_matrix(Sm_dd_protein,save_all_count_A[i])
        feature_matrix_protein7=createprotein_matrix(save_all_count_A[i],mat_pp)
        feature_matrix_drug8 = createdrug_matrix(Sm_dd_protein, save_all_count_A[i])
        feature_matrix_protein8 = createprotein_matrix(save_all_count_A[i], Sd_pp)
        feature_matrix_drug9 = createdrug_matrix(Sm_dd_protein, save_all_count_A[i])
        feature_matrix_protein9 = createprotein_matrix(save_all_count_A[i], Sm_pp_protein)

        lncanddiseasezxw = np.loadtxt("../big_data2/mat_drug_protein.txt")
        lncanddiseasezxw2 = np.loadtxt("../big_data2/mat_drug_protein.txt")
        lncanddiseasezxw = setlabelmatrixtofuone(lncanddiseasezxw, alloneexcepttestzxw[i])  #
        labelmatrix = setlabelmatrixtofuone(lncanddiseasezxw2, alloneexcepttestzxw[i])  #  把除了测试集那一部分 置为-50
        train_loader = load_data2(all_k_Sd_associated[i], save_all_count_A[i], Sm_dd, Sd_pp, batchsize)
        np.savetxt('/home/zhangxiaowen/RGA/result2/labelmatrix2.txt', labelmatrix, fmt="%d", delimiter=" ")
        test_loader, labelmatrix, lncanddiseasezxw = load_data3(labelmatrix, lncanddiseasezxw, all_k_test_data[i],
                                                                save_all_count_A[i], Sm_dd, Sd_pp, batchsize)
        model = the_model(input_size_lnc, input_size_A, input_size_dis, output_size,batchsize)
        train_top(model, 20, train_loader, 0.0001, feature_matrix_drug, feature_matrix_protein,
                  feature_matrix_drug2,feature_matrix_protein2,
                  feature_matrix_drug3,feature_matrix_protein3,
                  feature_matrix_drug4,feature_matrix_protein4,
                  feature_matrix_drug5, feature_matrix_protein5,
                  feature_matrix_drug6, feature_matrix_protein6,
                  feature_matrix_drug7, feature_matrix_protein7,
                  feature_matrix_drug8, feature_matrix_protein8,
                  feature_matrix_drug9, feature_matrix_protein9,
                 )
        if (i == 0):
            scoresmatrix = testzxw(model, test_loader, 0.0005, lncanddiseasezxw, all_k_test_data[i],
                                   feature_matrix_drug, feature_matrix_protein,
                                   feature_matrix_drug2, feature_matrix_protein2,
                                   feature_matrix_drug3, feature_matrix_protein3,
                                   feature_matrix_drug4, feature_matrix_protein4,
                                   feature_matrix_drug5, feature_matrix_protein5,
                                   feature_matrix_drug6, feature_matrix_protein6,
                                   feature_matrix_drug7, feature_matrix_protein7,
                                   feature_matrix_drug8, feature_matrix_protein8,
                                   feature_matrix_drug9, feature_matrix_protein9,)
    return labelmatrix, scoresmatrix


if __name__ == '__main__':
    k = 5
    labelmatrix, scoresmatrix = tes8_tes(k)
"""Final"""




















