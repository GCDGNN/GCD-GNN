import torch
import torch.nn as nn
from importlib import import_module
import os.path as osp
from DataHelper.datasetHelper import DatasetHelper
from model.ProtoSAGE import GraphSAGE_DGL

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
            attn_drop= config['attn_drop'],
            num_relations=datasetHelper.num_relations,
            out_proj_size=datasetHelper.num_classes,
            agg=config['agg'],
            relation_agg=config['relation_agg']
        ).cuda()
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                if m.weight is not None:
                    torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Conv2d):
                if m.weight is not None:
                    torch.nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        model.apply(init_weights)
    elif model_name=='GCDGNN-L':
        from model.SAGE_ORG import GraphSAGE_DGL_ORG

        mlp_act = config.get('mlp_activation', 'relu')
        if mlp_act == 'relu':
            mlp_activation = nn.ReLU(inplace=True)
        elif mlp_act == 'elu':
            mlp_activation = nn.ELU(inplace=True)
        model = GraphSAGE_DGL_ORG(in_size=datasetHelper.feat_dim,
                         hid_size=config['hid_dim'],
                         out_size=datasetHelper.num_classes if not config['proj'] else config['hid_dim'],
                         num_layers=config['n_layer'],
                         dropout=config['dropout'],
                         proj=config['proj'],
                         out_proj_size=datasetHelper.num_classes,
                         agg=config['agg'],
                         num_relations=datasetHelper.num_relations,
                         relation_agg=config['relation_agg'],
                       ).cuda()

    elif model_name=='BWGNN':
        from model.BWGNN import BWGNN, BWGNN_Hetero
        if config['homo']:
            model = BWGNN(in_feats=datasetHelper.feat_dim,
                          h_feats=config['hid_dim'],
                          num_classes=datasetHelper.num_classes,
                          graph= datasetHelper.data,
                          d = 3).cuda()
        else:
            model = BWGNN_Hetero(in_feats=datasetHelper.feat_dim,
                                h_feats=config['hid_dim'],
                                num_classes=datasetHelper.num_classes,
                                graph= datasetHelper.data,
                                d = 3).cuda()

    return model




def init(self, datasetHelper: DatasetHelper):
    config = self.config
    model = self.prepare_model(datasetHelper)
    optimizer, loss_func, scheduler = self.prepare_train(model, datasetHelper)
    
    return model, optimizer, loss_func, scheduler