import torch
import numpy as np
from sklearn import metrics
from collections import namedtuple
import pandas as pd
from dgl.dataloading import (
	MultiLayerFullNeighborSampler,
	NeighborSampler
)

def prob2pred(logits_fraud, thres=0.5):
    """
    Convert probability to predicted results according to given threshold
    :param y_prob: numpy array of probability in [0, 1]
    :param thres: binary classification threshold, default 0.5
    :returns: the predicted result with the same shape as y_prob
    """
    y_pred = np.zeros_like(logits_fraud, dtype=np.int32)  
    y_pred[logits_fraud >= thres] = 1
    y_pred[logits_fraud < thres] = 0
    return y_pred

def convert_probs(labels, logits, threshold_moving=True, thres=0.5):
    logits = torch.nn.Sigmoid()(logits)  
    logits = logits.detach().cpu().numpy() # logits
    
    logits_fraud = logits[:, 1]         
    if threshold_moving:
        preds = prob2pred(logits_fraud, thres=thres)
    else:
        preds = logits.argmax(axis=1) 

    return labels, logits_fraud, preds

def calc_acc(y_true, y_pred):
    """
    Compute the accuracy of prediction given the labels.
    """
    # return (y_pred == y_true).sum() * 1.0 / len(y_pred)
    return metrics.accuracy_score(y_true, y_pred)


def calc_f1(y_true, y_pred):
    f1_binary_1_gnn = metrics.f1_score(y_true, y_pred, pos_label=1, average='binary')
    f1_binary_0_gnn = metrics.f1_score(y_true, y_pred, pos_label=0, average='binary')
    f1_micro_gnn = metrics.f1_score(y_true, y_pred, average='micro')
    f1_macro_gnn = metrics.f1_score(y_true, y_pred, average='macro')

    return f1_binary_1_gnn, f1_binary_0_gnn, f1_micro_gnn, f1_macro_gnn


def calc_roc_and_thres(y_true, y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
    auc_gnn = metrics.auc(fpr, tpr)

    J = tpr - fpr
    ks_val = max(abs(J))
    
    idx = J.argmax(axis=0)
    best_thres = thresholds[idx]
    return auc_gnn, best_thres


def calc_ap_and_thres(y_true, y_prob):
    ap_gnn = metrics.average_precision_score(y_true, y_prob)

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_prob)
    F1 = 2 * precision * recall / (precision + recall)
    idx = F1.argmax(axis=0)
    best_thres = thresholds[idx]

    return ap_gnn, best_thres


def calc_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5


def eval_model(self, y_true, y_prob, y_pred):
    """
    :param y_true: torch.Tensor
    :param y_prob: torch.Tensor
    :param y_pred: torch.Tensor
    :return: namedtuple
    """
    acc = calc_acc(y_true, y_pred)
    f1_binary_1, f1_binary_0, f1_micro, f1_macro = calc_f1(y_true, y_pred)

    auc_gnn, best_roc_thres = calc_roc_and_thres(y_true, y_prob)
    # auc_gnn = metrics.roc_auc_score(labels, probs_1)

    ap_gnn, best_pr_thres = calc_ap_and_thres(y_true, y_prob)

    precision_1 = metrics.precision_score(y_true, y_pred, pos_label=1, average="binary")
    recall_1 = metrics.recall_score(y_true, y_pred, pos_label=1, average='binary')
    recall_macro = metrics.recall_score(y_true, y_pred, average='macro')

    conf_gnn = metrics.confusion_matrix(y_true, y_pred)
    gmean_gnn = calc_gmean(conf_gnn)
    tn, fp, fn, tp = conf_gnn.ravel()

    DataType = namedtuple('Metrics', ['f1_binary_1', 'f1_binary_0', 'f1_macro', 'auc_gnn',
                                      'gmean_gnn', 'recall_1', 'precision_1', 'ap_gnn',
                                      'best_roc_thres', 'best_pr_thres', 'recall_macro'])
    # 1:fraud->positive, 0:benign->negtive
    results = DataType(f1_binary_1=f1_binary_1, f1_binary_0=f1_binary_0, f1_macro=f1_macro,
                       auc_gnn=auc_gnn, gmean_gnn=gmean_gnn, ap_gnn=ap_gnn,
                       recall_1=recall_1, precision_1=precision_1, recall_macro=recall_macro,
                       best_pr_thres=best_pr_thres, best_roc_thres=best_roc_thres)

    return results

def get_tst_res(model,data):
    # 目标节点（假设我们要对节点6进行采样）
    # seed_nodes = torch.tensor([6])

    # 创建DataLoader
    sampler = MultiLayerFullNeighborSampler(num_layers=2)
    from dgl.dataloading import DataLoader as DGLDataLoader 
    # data = data.to('cuda')
    dataloader = DGLDataLoader(data,
                                torch.arange(data.num_nodes()),
                                sampler,
                                data.num_nodes(), 
                                shuffle = True,
                                drop_last = False,
                                num_workers=1)
    model.eval()
    
    logits_list = []
    label_list  = []  
    #use loader
    # for (input_nodes, output_nodes, blocks) in dataloader:
    #     res = model.tstconv(blocks.cuda(), dataloader.data.ndata['feature'].cuda())       # relations = datasetHelper.relations
    #     res = pd.DataFrame(res.cpu().detach().numpy()).transpose()
    #     res.to_csv('./tensor.csv')

@torch.no_grad()
def evaluate(self, datasetHelper, 
                   val_test_loader, 
                   model, 
                   threshold_moving=True,  
                   thres = 0.5, 
                   dataset = None,
                   idseries = None):

    model.eval()

    logits_list = []
    label_list  = []  
    if self.config['model_name'] in ['GCDGNN', 'GCDGNN-L']:
        relations = datasetHelper.relations
        for step, (input_nodes, output_nodes, blocks) in enumerate(val_test_loader):
            blocks = [b.to(torch.cuda.current_device()) for b in blocks]
            val_test_feats = blocks[0].srcdata['feature']
            val_test_label = blocks[-1].dstdata['label']
            batch_logits = model(blocks, relations, val_test_feats)
            
            logits_list.append(batch_logits.cpu())
            label_list .append(val_test_label.cpu())
            if idseries is not None:
                idseries.extend(blocks[-1].dstdata['_ID'].cpu().numpy())
        # get_tst_res(model, datasetHelper.data)
        
    # shape=(len(eval_loader), 2)
    logits = torch.cat(logits_list, dim=0)  # predicted logits torch.Size([4595, 2])
    # for label alighment when using train_loader
    eval_labels   = torch.cat(label_list, dim=0)  # ground truth labels
    eval_labels   = eval_labels.detach().cpu().numpy()
    eval_labels, fraud_probs, preds = convert_probs(eval_labels, logits, threshold_moving=threshold_moving, thres=thres)
    return eval_labels, fraud_probs, preds


@torch.no_grad()
def evaluate_tune(self, datasetHelper, 
                   val_test_loader, 
                   model, 
                   loss_func, 
                   threshold_moving=True,  
                   thres = 0.5, 
                   dataset = None,
                   idseries = None):

    model.eval()

    logits_list = []
    label_list  = []  
    if self.config['model_name'] == 'GAGA':
        for (batch_seq, batch_labels) in val_test_loader:
            batch_seq    = batch_seq.cuda()
            batch_logits = model(batch_seq)
            logits_list.append(batch_logits.cpu())
            label_list .append(batch_labels.cpu())
    elif self.config['model_name'] in ['GraphSAGE', 'LA-SAGE', 'LA-SAGE2', 'LA-SAGE-LI', 'LA-SAGE-S','SAGE', 'SIMPMP', 'SAGE_ORG']:
        relations = datasetHelper.relations
        eval_loss = 0
        for step, (input_nodes, output_nodes, blocks) in enumerate(val_test_loader):
            blocks = [b.to(torch.cuda.current_device()) for b in blocks]
            val_test_feats = blocks[0].srcdata['feature']
            val_test_label = blocks[-1].dstdata['label']
            batch_logits = model(blocks, relations, val_test_feats)
            
            logits_list.append(batch_logits.cpu())
            label_list .append(val_test_label.cpu())
            if idseries is not None:
                idseries.extend(blocks[-1].dstdata['_ID'].cpu().numpy())
        # get_tst_res(model, datasetHelper.data)
    # shape=(len(eval_loader), 2)
    #cal loss
            eval_loss += loss_func(batch_logits, val_test_label, model)



    logits = torch.cat(logits_list, dim=0)  # predicted logits torch.Size([4595, 2])
    # for label alighment when using train_loader
    eval_labels   = torch.cat(label_list, dim=0)  # ground truth labels
    eval_labels   = eval_labels.detach().cpu().numpy()
    eval_labels, fraud_probs, preds = convert_probs(eval_labels, logits, threshold_moving=threshold_moving, thres=thres)
    return eval_labels, fraud_probs, preds, eval_loss