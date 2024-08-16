import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from dgl import function as fn
import tqdm
import math
from dgl.utils import check_eq_shape, expand_as_pair
from dgl.utils import check_eq_shape, expand_as_pair
from dgl.base import DGLError
from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm
from torch import Tensor
from .base import MLP
from conf import *
import pandas as pd
import os

def print_modlist(modlist):
    # 遍历 modlist 中的每一层，并输出参数值
    for idx, layer in enumerate(modlist):
        print(f"Layer {idx}: {layer}")
        # for name, param in layer.named_parameters(recurse=False):
        #     print(f"  Parameter: {name} | Size: {param.size()} | Values: {param.data}")




class TSTConv(nn.Module):
    def __init__(self, in_size, 
	      			   hid_size, 
					   out_size):
        super(TSTConv,self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        #cal label and new feats
        # nn.init.xavier_uniform_(self.cal_label.weight)
        self.explain = MLP(self.in_size,self.out_size,self.in_size,num_layers=2,dropout=0.05,tail_activation=True,xariver=True, gn=True)
        self.cal_label = nn.Conv1d(self.in_size, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        # self.cuda1 = True
        # self.pos_vector = None #label==1
        # self.neg_vector = None #label==0
        self.pos_vector = nn.Parameter(torch.full((1,self.in_size),float('nan')), requires_grad=False) #label==1
        self.neg_vector = nn.Parameter(torch.full((1,self.in_size),float('nan')), requires_grad=False) #label==0

        self.train_pos = None
        self.train_neg = None
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.bncos = GraphNorm(self.in_size)
        #混合
        self.fc_balance1      = MLP(self.in_size, hidden_channels=self.hid_size, output_channels=1,num_layers=2,xariver=True)
        

        self.balance_w1       = nn.Sigmoid()
        

    def cal_simi_scores(self, feats):
        epsilon = 1e-10
        pos_vector1 = torch.log(self.pos_vector.data + epsilon).repeat(len(feats),1)
        neg_vector1 = torch.log(self.neg_vector.data + epsilon).repeat(len(feats),1)
        self_feats = feats + epsilon


        # 对目标分布 Q 进行对数变换
        cosine_pos = 1-self.balance(self.KLDiv(pos_vector1, self_feats,reduction='none').sum(dim=1))
        cosine_neg = 1-self.balance(self.KLDiv(neg_vector1, self_feats,reduction='none').sum(dim=1))
        simi_scores = torch.cat((cosine_neg.view(-1, 1), cosine_pos.view(-1, 1)), 1)
        return simi_scores

    def cal_simi_scores_cos(self, feats):
        # feats = self.bncos(feats) #让小于0的出现
        
        pos_vector1 = (self.pos_vector.data).repeat(len(feats), 1)
        neg_vector1 = (self.neg_vector.data).repeat(len(feats), 1)
        self_feats = feats

        # 对目标分布 Q 进行对数变换
        cosine_pos = self.cos(pos_vector1, self_feats)
        cosine_neg = self.cos(neg_vector1, self_feats)
        simi_scores = torch.cat((cosine_neg.view(-1, 1), cosine_pos.view(-1, 1)), 1)
        # print((simi_scores<0).any())
        return simi_scores

    def softmax_with_temperature(self, input, t=1, axis=-1):
        ex = torch.exp(input / t)
        sum = torch.sum(ex, axis=axis)
        return ex / sum
    
    @torch.no_grad()
    def update_label_vector(self, x):
        # pdb.set_trace()
        # if isinstance(x, torch.Tensor):
        x_pos = x[self.train_pos]
        x_neg = x[self.train_neg]
        # elif isinstance(x, torch.nn.Embedding):
        # 	x_pos = x(self.pos_index)
        # 	x_neg = x(self.neg_index)
        if torch.all(self.pos_vector != self.pos_vector): 
            self.pos_vector.data = torch.mean(x_pos, dim=0, keepdim=True)#need to change to parameter
            self.neg_vector.data = torch.mean(x_neg, dim=0, keepdim=True)
        else:
            cosine_pos = self.cos(self.pos_vector.data, x_pos)
            cosine_neg = self.cos(self.neg_vector.data, x_neg)
            weights_pos = self.softmax_with_temperature(cosine_pos, t=5).reshape(1, -1)
            weights_neg = self.softmax_with_temperature(cosine_neg, t=5).reshape(1, -1)
            self.pos_vector.data = torch.mm(weights_pos, x_pos)
            self.neg_vector.data = torch.mm(weights_neg, x_neg)
            
    @torch.no_grad()
    def update_label_vector_mean(self, x):
        # pdb.set_trace()
        # if isinstance(x, torch.Tensor):
        x_pos = x[self.train_pos]
        x_neg = x[self.train_neg]
        # elif isinstance(x, torch.nn.Embedding):
        # 	x_pos = x(self.pos_index)
        # 	x_neg = x(self.neg_index)
        # if self.pos_vector is None:
        self.pos_vector.data = torch.mean(x_pos, dim=0, keepdim=True)
        self.neg_vector.data = torch.mean(x_neg, dim=0, keepdim=True)
        # else:
        #     cosine_pos = self.cos(self.pos_vector.data, x_pos)
        #     cosine_neg = self.cos(self.neg_vector.data, x_neg)
        #     weights_pos = self.softmax_with_temperature(cosine_pos, t=5).reshape(1, -1)
        #     weights_neg = self.softmax_with_temperature(cosine_neg, t=5).reshape(1, -1)
        #     self.pos_vector.data = torch.mm(weights_pos, x_pos)
        #     self.neg_vector.data = torch.mm(weights_neg, x_neg)

    def forward(self, block, feats):
        h = feats 

        #update trainpos and neg
        self.train_pos = torch.where(block.srcdata['label_unk'] == 1)
        self.train_neg = torch.where(block.srcdata['label_unk'] == 0)
        org_feat = h

        #feature optimization
        feat_sim = self.explain(h)
        bn_feat_sim = self.bncos(feat_sim)
        #update prototype
        if self.training:
            self.update_label_vector(bn_feat_sim)
            pass
        simi = self.cal_simi_scores_cos(bn_feat_sim)


        h = feat_sim.unsqueeze(0)
        h = h.permute((0, 2, 1))
        self.pred_label =  self.cal_label(h)
        self.pred_label = self.pred_label.permute((0, 2, 1)).squeeze()

        self.pred_label_batch = self.pred_label[:block.num_dst_nodes()]# use for loss cal

        pred_label_fraud = self.sigmoid(self.pred_label)[:,1]
        self.pred_label_fraud = pred_label_fraud[:block.num_dst_nodes()].unsqueeze(-1)
        self.train_pos_feat = feat_sim[block.srcdata['label_unk']==1]
        self.train_neg_feat = feat_sim[block.srcdata['label_unk']==0]
        #cal loss?
        simi_weight = simi[:,0]*torch.tensor(block.srcdata['label_unk'] == 0,dtype=torch.float32)\
                        + simi[:,1]*torch.tensor(block.srcdata['label_unk'] == 1,dtype=torch.float32)\
                        + torch.tensor(block.srcdata['label_unk'] == 2,dtype=torch.float32) * (torch.max(simi, dim=1)[0])#change to be max#(pred_label_fraud*simi[:,1]+(1-pred_label_fraud)*simi[:,0])

        # #generate edge weight
        balance1 = self.balance_w1(self.fc_balance1(org_feat))
        feat_sim = balance1*org_feat + (1-balance1)*feat_sim
        

        return feat_sim,simi_weight
    
class TSTConvOld1(nn.Module):
    def __init__(self, in_size, 
	      			   hid_size, 
					   out_size):
        super(TSTConv,self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        #cal label and new feats
        from model.base import MLP
        # nn.init.xavier_uniform_(self.cal_label.weight)
        self.explain = MLP(self.in_size,self.out_size,self.in_size,num_layers=2,dropout=0.05,tail_activation=True,xariver=True,gn=True)
        self.cal_label = nn.Conv1d(self.in_size, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.cuda1 = True
        self.pos_vector = None #label==1
        self.neg_vector = None #label==0
        self.train_pos = None
        self.train_neg = None
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.bncos = nn.BatchNorm1d(self.in_size)
        #混合
        self.fc_balance1      = MLP(self.in_size, hidden_channels=self.hid_size, output_channels=1,num_layers=2,xariver=True)
        

        self.balance_w1       = nn.Sigmoid()

    def cal_simi_scores(self, feats):
        epsilon = 1e-10
        pos_vector1 = torch.log(self.pos_vector + epsilon).repeat(len(feats),1)
        neg_vector1 = torch.log(self.neg_vector + epsilon).repeat(len(feats),1)
        self_feats = feats + epsilon


        # 对目标分布 Q 进行对数变换
        cosine_pos = 1-self.balance(self.KLDiv(pos_vector1, self_feats,reduction='none').sum(dim=1))
        cosine_neg = 1-self.balance(self.KLDiv(neg_vector1, self_feats,reduction='none').sum(dim=1))
        simi_scores = torch.cat((cosine_neg.view(-1, 1), cosine_pos.view(-1, 1)), 1)
        return simi_scores

    def cal_simi_scores_cos(self, feats):
        # feats = self.bncos(feats) #让小于0的出现
        feats = self.bncos(feats)

        
        pos_vector1 = (self.pos_vector).repeat(len(feats), 1)
        neg_vector1 = (self.neg_vector).repeat(len(feats), 1)
        self_feats = feats

        # 对目标分布 Q 进行对数变换
        cosine_pos = self.cos(pos_vector1, self_feats)
        cosine_neg = self.cos(neg_vector1, self_feats)
        simi_scores = torch.cat((cosine_neg.view(-1, 1), cosine_pos.view(-1, 1)), 1)
        # print((simi_scores<0).any())
        return simi_scores

    def softmax_with_temperature(self, input, t=1, axis=-1):
        ex = torch.exp(input / t)
        sum = torch.sum(ex, axis=axis)
        return ex / sum
    
    @torch.no_grad()
    def update_label_vector(self, x):
        # pdb.set_trace()
        # if isinstance(x, torch.Tensor):
        x_pos = x[self.train_pos]
        x_neg = x[self.train_neg]
        # elif isinstance(x, torch.nn.Embedding):
        # 	x_pos = x(self.pos_index)
        # 	x_neg = x(self.neg_index)
        if self.pos_vector is None:
            self.pos_vector = torch.mean(x_pos, dim=0, keepdim=True)
            self.neg_vector = torch.mean(x_neg, dim=0, keepdim=True)
        else:
            cosine_pos = self.cos(self.pos_vector, x_pos)
            cosine_neg = self.cos(self.neg_vector, x_neg)
            weights_pos = self.softmax_with_temperature(cosine_pos, t=5).reshape(1, -1)
            weights_neg = self.softmax_with_temperature(cosine_neg, t=5).reshape(1, -1)
            self.pos_vector = torch.mm(weights_pos, x_pos)
            self.neg_vector = torch.mm(weights_neg, x_neg)

    def forward(self, block, feats):
        #返回对应每个节点的置信度
        h = feats 
        # def feature_test(feat):
        #     for i in range(feat.shape[0]):
        #         feat = nn.BatchNorm1d(feat.shape[1]).cuda()(feat)
        #         test_feat = feat[i].repeat(len(feat),1)
        #         cos_test = self.cos(test_feat,feat)
        #         if (cos_test<0).any():
        #             print(cos_test[cos_test<0])
        # feature_test(feats)
        # feats = self.bncos(feats) #让小于0的出现

        #update trainpos and neg
        self.train_pos = torch.where(block.srcdata['label_unk'] == 1)
        self.train_neg = torch.where(block.srcdata['label_unk'] == 0)
        org_feat = h
        # feat_sim = self.explain(feat)
        feat_sim = self.explain(h)
        # feat_sim = feat
        # if self.training:
        self.update_label_vector(feat_sim)

        simi = self.cal_simi_scores_cos(feat_sim)
        h = feat_sim.unsqueeze(0)
        h = h.permute((0, 2, 1))
        self.pred_label =  self.cal_label(h)
        self.pred_label = self.pred_label.permute((0, 2, 1)).squeeze()

        self.pred_label_batch = self.pred_label[:block.num_dst_nodes()]# use for loss cal

        pred_label_fraud = self.sigmoid(self.pred_label)[:,1]
        self.pred_label_fraud = pred_label_fraud[:block.num_dst_nodes()].unsqueeze(-1)
        self.train_pos_feat = feat_sim[block.srcdata['label_unk']==1]
        self.train_neg_feat = feat_sim[block.srcdata['label_unk']==0]
        #cal loss?
        simi_weight = simi[:,0]*torch.tensor(block.srcdata['label_unk'] == 0,dtype=torch.float32)\
                        + simi[:,1]*torch.tensor(block.srcdata['label_unk'] == 1,dtype=torch.float32)\
                        + torch.tensor(block.srcdata['label_unk'] == 2,dtype=torch.float32) * (pred_label_fraud*simi[:,1]+(1-pred_label_fraud)*simi[:,0])

        # #generate edge weight
        # blocks[0].srcdata['simi_weight'] = simi_weight
        # #change_feat?
        # h = feat_sim
        # block.srcdata['feature'] = feat_sim
        balance1 = self.balance_w1(self.fc_balance1(org_feat))
        feat_sim = balance1*org_feat + (1-balance1)*feat_sim
        

        return feat_sim,simi_weight
    
class TSTConvHighPerform(nn.Module):
    def __init__(self, in_size, 
	      			   hid_size, 
					   out_size):
        super(TSTConvHighPerform,self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        #cal label and new feats
        from model.base import MLP
        # nn.init.xavier_uniform_(self.cal_label.weight)
        self.explain = MLP(self.in_size,self.out_size,self.in_size,num_layers=2,dropout=0.05,tail_activation=True,xariver=True,gn=True)
        self.cal_label = nn.Conv1d(self.in_size, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.cuda1 = True
        self.pos_vector = None #label==1
        self.neg_vector = None #label==0
        self.train_pos = None
        self.train_neg = None
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.bncos = nn.BatchNorm1d(self.in_size)
        #混合
        self.fc_balance1      = MLP(self.in_size, hidden_channels=self.hid_size, output_channels=1,num_layers=2,xariver=True)
        

        self.balance_w1       = nn.Sigmoid()

    def cal_simi_scores(self, feats):
        epsilon = 1e-10
        pos_vector1 = torch.log(self.pos_vector + epsilon).repeat(len(feats),1)
        neg_vector1 = torch.log(self.neg_vector + epsilon).repeat(len(feats),1)
        self_feats = feats + epsilon


        # 对目标分布 Q 进行对数变换
        cosine_pos = 1-self.balance(self.KLDiv(pos_vector1, self_feats,reduction='none').sum(dim=1))
        cosine_neg = 1-self.balance(self.KLDiv(neg_vector1, self_feats,reduction='none').sum(dim=1))
        simi_scores = torch.cat((cosine_neg.view(-1, 1), cosine_pos.view(-1, 1)), 1)
        return simi_scores

    def cal_simi_scores_cos(self, feats):
        # feats = self.bncos(feats) #让小于0的出现
        feats = self.bncos(feats)

        
        pos_vector1 = (self.pos_vector).repeat(len(feats), 1)
        neg_vector1 = (self.neg_vector).repeat(len(feats), 1)
        self_feats = feats

        # 对目标分布 Q 进行对数变换
        cosine_pos = self.cos(pos_vector1, self_feats)
        cosine_neg = self.cos(neg_vector1, self_feats)
        simi_scores = torch.cat((cosine_neg.view(-1, 1), cosine_pos.view(-1, 1)), 1)
        # print((simi_scores<0).any())
        return simi_scores

    def softmax_with_temperature(self, input, t=1, axis=-1):
        ex = torch.exp(input / t)
        sum = torch.sum(ex, axis=axis)
        return ex / sum
    
    @torch.no_grad()
    def update_label_vector(self, x):
        # pdb.set_trace()
        # if isinstance(x, torch.Tensor):
        x_pos = x[self.train_pos]
        x_neg = x[self.train_neg]
        # elif isinstance(x, torch.nn.Embedding):
        # 	x_pos = x(self.pos_index)
        # 	x_neg = x(self.neg_index)
        if self.pos_vector is None:
            self.pos_vector = torch.mean(x_pos, dim=0, keepdim=True)
            self.neg_vector = torch.mean(x_neg, dim=0, keepdim=True)
        else:
            cosine_pos = self.cos(self.pos_vector, x_pos)
            cosine_neg = self.cos(self.neg_vector, x_neg)
            weights_pos = self.softmax_with_temperature(cosine_pos, t=5).reshape(1, -1)
            weights_neg = self.softmax_with_temperature(cosine_neg, t=5).reshape(1, -1)
            self.pos_vector = torch.mm(weights_pos, x_pos)
            self.neg_vector = torch.mm(weights_neg, x_neg)

    def forward(self, block, feats):
        #返回对应每个节点的置信度
        h = feats 
        # def feature_test(feat):
        #     for i in range(feat.shape[0]):
        #         feat = nn.BatchNorm1d(feat.shape[1]).cuda()(feat)
        #         test_feat = feat[i].repeat(len(feat),1)
        #         cos_test = self.cos(test_feat,feat)
        #         if (cos_test<0).any():
        #             print(cos_test[cos_test<0])
        # feature_test(feats)
        # feats = self.bncos(feats) #让小于0的出现

        #update trainpos and neg
        self.train_pos = torch.where(block.srcdata['label_unk'] == 1)
        self.train_neg = torch.where(block.srcdata['label_unk'] == 0)
        org_feat = h
        # feat_sim = self.explain(feat)
        feat_sim = self.explain(h)
        # feat_sim = feat

        self.update_label_vector(feat_sim)

        simi = self.cal_simi_scores_cos(feat_sim)
        h = feat_sim.unsqueeze(0)
        h = h.permute((0, 2, 1))
        self.pred_label =  self.cal_label(h)
        self.pred_label = self.pred_label.permute((0, 2, 1)).squeeze()

        self.pred_label_batch = self.pred_label[:block.num_dst_nodes()]# use for loss cal

        pred_label_fraud = self.sigmoid(self.pred_label)[:,1]
        self.pred_label_fraud = pred_label_fraud[:block.num_dst_nodes()].unsqueeze(-1)
        self.train_pos_feat = feat_sim[block.srcdata['label_unk']==1]
        self.train_neg_feat = feat_sim[block.srcdata['label_unk']==0]
        #cal loss?
        simi_weight = simi[:,0]*torch.tensor(block.srcdata['label_unk'] == 0,dtype=torch.float32)\
                        + simi[:,1]*torch.tensor(block.srcdata['label_unk'] == 1,dtype=torch.float32)\
                        + torch.tensor(block.srcdata['label_unk'] == 2,dtype=torch.float32) * (pred_label_fraud*simi[:,1]+(1-pred_label_fraud)*simi[:,0])

        # #generate edge weight
        # blocks[0].srcdata['simi_weight'] = simi_weight
        # #change_feat?
        # h = feat_sim
        # block.srcdata['feature'] = feat_sim
        balance1 = self.balance_w1(self.fc_balance1(org_feat))
        feat_sim = balance1*org_feat + (1-balance1)*feat_sim
        

        return feat_sim,simi_weight


class TSTConvOld(nn.Module):
    def __init__(self, in_size, 
	      			   hid_size, 
					   out_size):
        super(TSTConvOld,self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        #cal label and new feats
        from model.base import MLP
        # nn.init.xavier_uniform_(self.cal_label.weight)
        self.explain = MLP(self.in_size,self.out_size,self.in_size,num_layers=2,dropout=0.05,tail_activation=True,xariver=True,gn=True)
        self.cal_label = nn.Conv1d(self.in_size, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.cuda1 = True
        self.pos_vector = None #label==1
        self.neg_vector = None #label==0
        self.train_pos = None
        self.train_neg = None
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.bncos = nn.BatchNorm1d(self.in_size)

    def cal_simi_scores(self, feats):
        epsilon = 1e-10
        pos_vector1 = torch.log(self.pos_vector + epsilon).repeat(len(feats),1)
        neg_vector1 = torch.log(self.neg_vector + epsilon).repeat(len(feats),1)
        self_feats = feats + epsilon


        # 对目标分布 Q 进行对数变换
        cosine_pos = 1-self.balance(self.KLDiv(pos_vector1, self_feats,reduction='none').sum(dim=1))
        cosine_neg = 1-self.balance(self.KLDiv(neg_vector1, self_feats,reduction='none').sum(dim=1))
        simi_scores = torch.cat((cosine_neg.view(-1, 1), cosine_pos.view(-1, 1)), 1)
        return simi_scores

    def cal_simi_scores_cos(self, feats):
        feats = self.bncos(feats) #让小于0的出现
        
        pos_vector1 = (self.pos_vector).repeat(len(feats), 1)
        neg_vector1 = (self.neg_vector).repeat(len(feats), 1)
        self_feats = feats

        # 对目标分布 Q 进行对数变换
        cosine_pos = self.cos(pos_vector1, self_feats)
        cosine_neg = self.cos(neg_vector1, self_feats)
        simi_scores = torch.cat((cosine_neg.view(-1, 1), cosine_pos.view(-1, 1)), 1)
        # print((simi_scores<0).any())
        return simi_scores

    def softmax_with_temperature(self, input, t=1, axis=-1):
        ex = torch.exp(input / t)
        sum = torch.sum(ex, axis=axis)
        return ex / sum

    def update_label_vector(self, x):
        # pdb.set_trace()
        # if isinstance(x, torch.Tensor):
        x_pos = x[self.train_pos]
        x_neg = x[self.train_neg]
        # elif isinstance(x, torch.nn.Embedding):
        # 	x_pos = x(self.pos_index)
        # 	x_neg = x(self.neg_index)
        if self.pos_vector is None:
            self.pos_vector = torch.mean(x_pos, dim=0, keepdim=True)
            self.neg_vector = torch.mean(x_neg, dim=0, keepdim=True)
        else:
            cosine_pos = self.cos(self.pos_vector, x_pos)
            cosine_neg = self.cos(self.neg_vector, x_neg)
            weights_pos = self.softmax_with_temperature(cosine_pos, t=5).reshape(1, -1)
            weights_neg = self.softmax_with_temperature(cosine_neg, t=5).reshape(1, -1)
            self.pos_vector = torch.mm(weights_pos, x_pos)
            self.neg_vector = torch.mm(weights_neg, x_neg)

    def forward(self, block, feats):
        #返回对应每个节点的置信度
        h = feats 
        # def feature_test(feat):
        #     for i in range(feat.shape[0]):
        #         feat = nn.BatchNorm1d(feat.shape[1]).cuda()(feat)
        #         test_feat = feat[i].repeat(len(feat),1)
        #         cos_test = self.cos(test_feat,feat)
        #         if (cos_test<0).any():
        #             print(cos_test[cos_test<0])
        # feature_test(feats)

        #update trainpos and neg
        self.train_pos = torch.where(block.srcdata['label_unk'] == 1)
        self.train_neg = torch.where(block.srcdata['label_unk'] == 0)

        # feat_sim = self.explain(feat)
        feat_sim = self.explain(h)
        # feat_sim = feat

        self.update_label_vector(feat_sim)

        simi = self.cal_simi_scores_cos(feat_sim)
        h = feat_sim.unsqueeze(0)
        h = h.permute((0, 2, 1))
        self.pred_label =  self.cal_label(h)
        self.pred_label = self.pred_label.permute((0, 2, 1)).squeeze()

        self.pred_label_batch = self.pred_label[:block.num_dst_nodes()]# use for loss cal

        pred_label_fraud = self.sigmoid(self.pred_label)[:,1]
        self.pred_label_fraud = pred_label_fraud[:block.num_dst_nodes()].unsqueeze(-1)
        self.train_pos_feat = feat_sim[block.srcdata['label_unk']==1]
        self.train_neg_feat = feat_sim[block.srcdata['label_unk']==0]
        #cal loss?
        simi_weight = simi[:,0]*torch.tensor(block.srcdata['label_unk'] == 0,dtype=torch.float32)\
                        + simi[:,1]*torch.tensor(block.srcdata['label_unk'] == 1,dtype=torch.float32)\
                        + torch.tensor(block.srcdata['label_unk'] == 2,dtype=torch.float32) * (pred_label_fraud*simi[:,1]+(1-pred_label_fraud)*simi[:,0])

        # #generate edge weight
        # blocks[0].srcdata['simi_weight'] = simi_weight
        # #change_feat?
        # h = feat_sim
        # block.srcdata['feature'] = feat_sim
        return feat_sim,simi_weight