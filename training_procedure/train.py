
import torch.nn as nn
from DataHelper.datasetHelper import DatasetHelper
import torch
import pandas as pd
from datetime import datetime


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'data_{timestamp}.csv'


def train(self, epoch, model, loss_func, optimizer, train_loader = None, datasetHelper: DatasetHelper = None):
    model.train()
    config = self.config 
    total_loss = 0.0

    if config['model_name'] in ['GCDGNN', 'GCDGNN-L']:
        relations = datasetHelper.relations
        # if 'homo' in relations:
        #     relations.remove('homo')
        for step, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            # train on blocks
            blocks = [b.to(torch.cuda.current_device()) for b in blocks]
            train_feats = blocks[0].srcdata['feature']
            train_label = blocks[-1].dstdata['label']
            # if self.ice == 0 and epoch>50:
            #     model.tstconv.eval()
            #     for name, child in model.named_children():
            #         if name in ['tstconv']:
            #             for param in child.parameters():
            #                 param.requires_grad = False
            #     print(epoch, 'arg iced!')
            #     ice = 1
            optimizer.zero_grad()
            batch_logits = model(blocks, relations, train_feats)
            if config['model_name'] in ['SAGE'] and config['load_model_path']!='LASAGE_S_del_self_inf':
                loss = loss_func(batch_logits, train_label, model)
            else :
                loss = loss_func(batch_logits, train_label)

            if config['train_auc']:
                from training_procedure.evaluate import calc_roc_and_thres
                auc,thres = calc_roc_and_thres(train_label.cpu().detach().numpy(),batch_logits[:,1].cpu().detach().numpy())
                print(auc)
            # print(loss)
            total_loss += loss 
            loss.backward(retain_graph=False)
            optimizer.step()
    
    return model, total_loss