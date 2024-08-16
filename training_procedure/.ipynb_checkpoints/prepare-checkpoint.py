import torch
import torch.nn as nn
from importlib import import_module
import os.path as osp
from DataHelper.datasetHelper import DatasetHelper
from model.ProtoSAGE import GraphSAGE_DGL

from model.LASAGE_S import LASAGE_S
def prepare_train(self, model, datasetHelper: DatasetHelper):
    config = self.config
    scheduler = None
    optimizer = getattr(torch.optim, config['optimizer'])(  params          = model.parameters(), 
                                                            lr              = config['lr'] ,
                                                            weight_decay    = config.get('weight_decay', 0) )
    if config.get('lr_scheduler', False):
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['step_size'],gamma=config['gamma'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = config['resi'], min_lr=1e-3)
    weight = (1-datasetHelper.labels[datasetHelper.train_mask]).sum() / datasetHelper.labels[datasetHelper.train_mask].sum()
    loss_func = nn.CrossEntropyLoss(weight = torch.tensor([1., weight]).cuda() if config['weighted_loss'] else None, reduction=config['reduction'])
    return optimizer, loss_func, scheduler

def prepare_model(self, datasetHelper: DatasetHelper):
    config = self.config
    model_name = config['model_name']



    if model_name == 'GCDGNN':
        #添加模型直接在斜面进行修改，多加一个参数方便调试
        # from model.SAGE import GraphSAGE_DGL
        mlp_act = config.get('mlp_activation', 'relu')
        if mlp_act == 'relu':
            mlp_activation = nn.ReLU(inplace=True)
        elif mlp_act == 'elu':
            mlp_activation = nn.ELU(inplace=True)
        model = GraphSAGE_DGL(
            in_size=datasetHelper.feat_dim,
            hid_size=config['hid_dim'],
            out_size=datasetHelper.num_classes if not config['proj'] else config['hid_dim'],
            num_layers=config['n_layer'],
            dropout=config['dropout'],
            proj=config['proj'],
            num_relations=datasetHelper.num_relations,
            out_proj_size=datasetHelper.num_classes,
            agg=config['agg'],
            relation_agg=config['relation_agg']
        ).cuda()


    return model


def init(self, datasetHelper: DatasetHelper):
    config = self.config
    model = self.prepare_model(datasetHelper)
    optimizer, loss_func, scheduler = self.prepare_train(model, datasetHelper)
    
    return model, optimizer, loss_func, scheduler