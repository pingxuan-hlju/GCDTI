

from numpy import *

import torch
import torch.nn as nn
import torch.nn.functional as F
def argsjj():
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser1.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser1.add_argument('--sparse', action='store_true', default=True, help='GAT with sparse version or not.')
    parser1.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser1.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    parser1.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser1.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser1.add_argument('--hidden', type=int, default=1000, help='Number of hidden units.')
    parser1.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')
    parser1.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1.txt - keep probability).')
    parser1.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser1.add_argument('--patience', type=int, default=100, help='Patience')
    args1 = parser1.parse_args()
    return args1


def changenorm(data1, data2):
    data3 = torch.stack((data1, data2), dim=1)
    return data3


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        # self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def forward(self, input, adj):   # input是输入矩阵的特征即2220*2220  adj是药物和蛋白的药物邻居图/药物和蛋白的蛋白邻居图
        dv = 'cuda' if input.is_cuda else 'cpu'
        N = input.size()[0]          # 2220
        edge = adj.nonzero().t()     # 2*23656
        h = torch.mm(input, self.W)  # 2220*2220 2220*1000 -> 2220*1000
        # h: N x out
        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()  # 2000*23656
        # edge: 2*D x E
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))     # 1*2000 2000*23656-> 1*23656
        # print('edge_e.shape',edge_e.shape)
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))  #2220*1
        # edge_e = self.dropout(edge_e)
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)#2220*1000
        # print(h_prime.shape,h_prime)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)  # 2220*1000
        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):# edge edgee nn
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class Beta_score3(nn.Module):
    def __init__(self):
        super(Beta_score3, self).__init__()
        self.h_n_parameters = nn.Parameter(torch.randn(500, 1))
        nn.init.xavier_normal_(self.h_n_parameters)
    def forward(self, result_A, result_B):
        result_A = result_A.unsqueeze(1)
        result_B = result_B.unsqueeze(1)
        result = torch.cat((result_A, result_B),1)
        nodes_score = torch.matmul(result, self.h_n_parameters)
        nodes_score = nodes_score.view(-1,1, 2)
        beta = F.softmax(nodes_score, dim=2)
        z_i = torch.matmul(beta, result)
        return z_i



class Beta_score4(nn.Module):
    def __init__(self):
        super(Beta_score4, self).__init__()
        self.h_n_parameters = nn.Parameter(torch.randn(500, 1))
        nn.init.xavier_normal_(self.h_n_parameters)
    def forward(self, result_A, result_B):
        result_A = result_A.unsqueeze(1)
        result_B = result_B.unsqueeze(1)
        result = torch.cat((result_A, result_B), 1)
        nodes_score = torch.matmul(result, self.h_n_parameters)
        nodes_score = nodes_score.view(-1, 1, 2)
        beta = F.softmax(nodes_score, dim=2)
        z_i = torch.matmul(beta, result)
        print(z_i.shape)
        return z_i


class Beta_score2(nn.Module):
    def __init__(self, result_A, result_B, output_size, batch_size):
        super(Beta_score2, self).__init__()
        self.node1 = nn.Linear(result_A , output_size, bias=True)
        nn.init.xavier_normal_(self.node1.weight)
        self.h_n_parameters = nn.Parameter(torch.randn(1, 2))
        nn.init.xavier_normal_(self.h_n_parameters)

    def forward(self, result_A, result_B):
        length = result_A.size() + result_B.size()
        result_A=result_A.reshape(1,-1)
        result_B=result_B.reshape(1,-1)
        result = torch.cat((result_A, result_B),0)
        nodes_score = torch.matmul(self.h_n_parameters,result )
        temp_nodes = torch.tanh(nodes_score)
        return temp_nodes



def getdrug(f):
    h = f[0]
    for i in range(1, 708):
      h = torch.cat((h, f[i]), 0)  # 竖着进行拼接，拼接第0-707个坐标数据 药物和药物的邻居节点特征  ([708, 6])
    features_drug_drug = h.reshape(708, -1)
    return features_drug_drug

def getpro(f):
    hi = f[708]
    for i in range(709, 2220):
        hi = torch.cat((hi, f[i]), 0)  # 竖着进行拼接，拼接第0-707个坐标数据 药物和药物的邻居节点特征  ([708, 6])
    features_protein_drug = hi.reshape(1512, -1)
    return features_protein_drug

class the_modell(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(the_modell, self).__init__()
        self.attentions = [SpGraphAttentionLayer(nfeat,        # 2220
                                                 nhid,         # 1000
                                                 dropout=dropout,
                                                 alpha=alpha,  # 0.2
                                                 concat=True) for _ in range(nheads)]  # 多头注意力进行循环 conncat=true

        for i, attention in enumerate(self.attentions):       # 加入pytorch的Module模块
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid * nheads,   # 1000*1
                                             nclass,          # 500
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)  # 多头注意力进行循环 conncat=true
        self.attention_soures = Beta_score3()
        self.attention_soures2 = Beta_score4()
        self.fully_connected2 = nn.Sequential(nn.Linear(16000,2), nn.BatchNorm1d(2))
        self.fully_connected3 = nn.Sequential(nn.Linear(16000, 8000))
        self.fully_connected4 = nn.Sequential(nn.Linear(8000, 4000))
        self.fully_connected5 = nn.Sequential(nn.Linear(4000, 2))
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )



    def forward(self, features, features2, features3, features4, adj_drug, adj_protein, x_A , y_A,adj_drug2, adj_protein2, adj_drug3,adj_protein3,adj_drug4,adj_protein4):
            a=features
            b=features2
            c=features3
            d=features4
            '''
            I.这是经过自注意力的部分
            '''

            featuresa = torch.cat([att(features, adj_drug2) for att in self.attentions], dim=1)  # 2220*1000
            featuresa = F.elu(self.out_att(featuresa, adj_drug2))                                # 2220*500
            featuresb = torch.cat([att(a, adj_protein2) for att in self.attentions],dim=1)       # 2220*1000
            featuresb = F.elu(self.out_att(featuresb, adj_protein2))                             # 2220*500

            featuresc = torch.cat([att(features2, adj_drug3) for att in self.attentions], dim=1)
            featuresc = F.elu(self.out_att(featuresc, adj_drug3))  # 改了
            featuresd = torch.cat([att(b, adj_protein3) for att in self.attentions], dim=1)
            featuresd = F.elu(self.out_att(featuresd, adj_protein3))

            featurese = torch.cat([att(features3, adj_drug) for att in self.attentions], dim=1)
            featurese = F.elu(self.out_att(featurese, adj_drug))  # 改了
            featuresf = torch.cat([att(c, adj_protein) for att in self.attentions], dim=1)
            featuresf = F.elu(self.out_att(featuresf, adj_protein))

            featuresg = torch.cat([att(features4, adj_drug4) for att in self.attentions], dim=1)
            featuresg = F.elu(self.out_att(featuresg, adj_drug4))  # 改了
            featuresh = torch.cat([att(d, adj_protein4) for att in self.attentions], dim=1)
            featuresh = F.elu(self.out_att(featuresh, adj_protein4))

            features_drug_drug = getdrug(featuresa)
            features_protein_drug = getpro(featuresa)
            features_drug_protein = getdrug(featuresb)
            features_protein_protein = getpro(featuresb)


            features_drug_drug2 = getdrug(featuresc)
            features_protein_drug2 = getpro(featuresc)
            features_drug_protein2 = getdrug(featuresd)
            features_protein_protein2 = getpro(featuresd)


            features_drug_drug3 = getdrug(featurese)
            features_protein_drug3 = getpro(featurese)
            features_drug_protein3 = getdrug(featuresf)
            features_protein_protein3 = getpro(featuresf)


            features_drug_drug4 =getdrug(featuresg)
            features_protein_drug4 = getpro(featuresg)
            features_drug_protein4=getdrug(featuresh)
            features_protein_protein4=getpro(featuresh)


            features_drug_drug = features_drug_drug[x_A]
            features_drug_protein = features_drug_protein[x_A]
            features_drug_drug2 = features_drug_drug2[x_A]
            features_drug_protein2 = features_drug_protein2[x_A]
            features_drug_drug3 = features_drug_drug3[x_A]
            features_drug_protein3 = features_drug_protein3[x_A]
            features_drug_drug4 = features_drug_drug4[x_A]
            features_drug_protein4 = features_drug_protein4[x_A]

            features_protein_drug = features_protein_drug[y_A]
            features_protein_protein = features_protein_protein[y_A]
            features_protein_drug2 = features_protein_drug2[y_A]
            features_protein_protein2 = features_protein_protein2[y_A]
            features_protein_drug3 = features_protein_drug3[y_A]
            features_protein_protein3 = features_protein_protein3[y_A]
            features_protein_drug4 = features_protein_drug4[y_A]
            features_protein_protein4 = features_protein_protein4[y_A]


            ddd1 = self.attention_soures(features_drug_drug,features_drug_protein)
            ddd11 = self.attention_soures(features_drug_drug2,features_drug_protein2)
            ddd111 = self.attention_soures(features_drug_drug3, features_drug_protein3)
            ddd1111 = self.attention_soures(features_drug_drug4, features_drug_protein4)

            ddd2 = self.attention_soures2(features_protein_drug, features_protein_protein)
            ddd22 = self.attention_soures2(features_protein_drug2, features_protein_protein2)
            ddd222 = self.attention_soures2(features_protein_drug3, features_protein_protein3)
            ddd2222 = self.attention_soures2(features_protein_drug4, features_protein_protein4)


            drug_protein_pair = torch.cat((ddd1,ddd2),1)
            drug_protein_pair=drug_protein_pair.unsqueeze(1)

            drug_protein_pair2 = torch.cat((ddd11,ddd22),1)
            drug_protein_pair2 = drug_protein_pair2.unsqueeze(1)

            drug_protein_pair3 = torch.cat((ddd111,ddd222),1)
            drug_protein_pair3 = drug_protein_pair3.unsqueeze(1)

            drug_protein_pair4 =  torch.cat((ddd1111,ddd2222),1)
            drug_protein_pair4 = drug_protein_pair4.unsqueeze(1)
            final=self.encoder(drug_protein_pair)
            final2 = self.encoder(drug_protein_pair2)
            final3 = self.encoder(drug_protein_pair3)
            final4 = self.encoder(drug_protein_pair4)
            f_c = final.view(final.size()[0], -1)
            f_c2 = final2.view(final2.size()[0], -1)
            f_c3 = final3.view(final3.size()[0], -1)
            f_c4 = final4.view(final4.size()[0], -1)
            final = torch.cat((f_c, f_c2,f_c3,f_c4), 1)
            sum = f_c.mul(f_c.t()) +f_c.mul(f_c2.t()) + f_c.mul(f_c3.t()) + f_c.mul(f_c4.t())
            sum2 = f_c2.mul(f_c.t()) + f_c2.mul(f_c2.t()) + f_c2.mul(f_c3.t()) + f_c2.mul(f_c4.t())
            sum3 = f_c3.mul(f_c.t()) + f_c3.mul(f_c2.t()) + f_c3.mul(f_c3.t()) + f_c3.mul(f_c4.t())
            sum4 = f_c4.mul(f_c.t()) + f_c4.mul(f_c2.t()) + f_c4.mul(f_c3.t()) + f_c4.mul(f_c4.t())
            f_c1 = f_c + F.softmax(f_c.mul(f_c.t())/sum)*f_c+F.softmax(f_c.mul(f_c2.t())/sum)*f_c2 + F.softmax(f_c.mul(f_c3.t())/sum)*f_c3 + F.softmax(f_c.mul(f_c4.t())/sum)*f_c4
            f_c22 = f_c2+ F.softmax(f_c2.mul(f_c.t())/sum2)*f_c+F.softmax(f_c2.mul(f_c2.t())/sum2)*f_c2 + F.softmax(f_c2.mul(f_c3.t())/sum2)*f_c3 + F.softmax( f_c2.mul(f_c4.t()) / sum2) * f_c4
            f_c33 = f_c3 + F.softmax(f_c3.mul(f_c.t()) / sum3) * f_c + F.softmax(f_c3.mul(f_c2.t()) / sum3) * f_c2 + F.softmax(f_c3.mul(f_c3.t()) / sum3) * f_c3 + F.softmax(f_c3.mul(f_c4.t()) / sum3) * f_c4
            f_c44 = f_c4 + F.softmax(f_c4.mul(f_c.t()) / sum4) * f_c + F.softmax(f_c4.mul(f_c2.t()) / sum4) * f_c2 + F.softmax(f_c4.mul(f_c3.t()) / sum4) * f_c3 + F.softmax(f_c4.mul(f_c4.t()) / sum4) * f_c4
  
            final2 = torch.cat((f_c1, f_c22, f_c33, f_c44), 1)
            out = self.fully_connected2(final)
            return out



