import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import random
from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn.pytorch.hetero import HeteroGraphConv
from dgl.data import AsNodePredDataset
from dgl import function as fn
import tqdm
import dgl
from dgl.utils import check_eq_shape, expand_as_pair
from dgl.dataloading import (
	DataLoader,
	MultiLayerFullNeighborSampler,
	NeighborSampler,
	BlockSampler
)
from dgl.utils import check_eq_shape, expand_as_pair
import dgl.nn as dglnn
from model.Layer import TSTConv
from dgl.ops import edge_softmax
from conf import *
import pandas as pd
# from main import writer/
import pickle
import os



import torchmetrics.functional as MF

from ogb.nodeproppred import DglNodePropPredDataset
"""
	GraphSAGE implementations
	Paper: Inductive Representation Learning on Large Graphs
	Source: https://github.com/williamleif/graphsage-simple/
"""


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class GraphSage(nn.Module):
	"""
	Vanilla GraphSAGE Model
	Code partially from https://github.com/williamleif/graphsage-simple/
	"""
	def __init__(self, num_classes, enc):
		super(GraphSage, self).__init__()
		self.enc = enc
		self.xent = nn.CrossEntropyLoss()
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes):
		embeds = self.enc(nodes)
		scores = self.weight.mm(embeds)
		return scores.t()

	def to_prob(self, nodes):
		pos_scores = torch.sigmoid(self.forward(nodes))
		return pos_scores

	def loss(self, nodes, labels):
		scores = self.forward(nodes)
		return self.xent(scores, labels.squeeze())


class MeanAggregator(nn.Module):
	"""
	Aggregates a node's embeddings using mean of neighbors' embeddings
	"""

	def __init__(self, features, cuda=False, gcn=False):
		"""
		Initializes the aggregator for a specific graph.

		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		cuda -- whether to use GPU
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""

		super(MeanAggregator, self).__init__()

		self.features = features
		self.cuda = cuda
		self.gcn = gcn

	def forward(self, nodes, to_neighs, num_sample=10):
		"""
		nodes --- list of nodes in a batch
		to_neighs --- list of sets, each set is the set of neighbors for node in batch
		num_sample --- number of neighbors to sample. No sampling if None.
		"""
		# Local pointers to functions (speed hack)
		_set = set
		if not num_sample is None:
			_sample = random.sample
			samp_neighs = [_set(_sample(to_neigh,
										num_sample,
										)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
		else:
			samp_neighs = to_neighs

		if self.gcn:
			samp_neighs = [samp_neigh.union(set([int(nodes[i])])) for i, samp_neigh in enumerate(samp_neighs)]
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh)
		if self.cuda1:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		to_feats = mask.mm(embed_matrix)
		return to_feats


class Encoder(nn.Module):
	"""
	Vanilla GraphSAGE Encoder Module
	Encodes a node's using 'convolutional' GraphSage approach
	"""

	def __init__(self, features, feature_dim,
				 embed_dim, adj_lists, aggregator,
				 num_sample=10,
				 base_model=None, gcn=False, cuda=False,
				 feature_transform=False):
		super(Encoder, self).__init__()

		self.features = features
		self.feat_dim = feature_dim
		self.adj_lists = adj_lists
		self.aggregator = aggregator
		self.num_sample = num_sample
		if base_model != None:
			self.base_model = base_model

		self.gcn = gcn
		self.embed_dim = embed_dim
		self.cuda = cuda
		self.aggregator.cuda = cuda
		self.weight = nn.Parameter(
			torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes):
		"""
		Generates embeddings for a batch of nodes.

		nodes     -- list of nodes
		"""
		neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
											  self.num_sample)

		if isinstance(nodes, list):
			index = torch.LongTensor(nodes)
		else:
			index = nodes

		if not self.gcn:
			if self.cuda:
				self_feats = self.features(index).cuda()
			else:
				self_feats = self.features(index)
			combined = torch.cat((self_feats, neigh_feats), dim=1)
		else:
			combined = neigh_feats
		combined = F.relu(self.weight.mm(combined.t()))
		return combined



class GCN(nn.Module):
	"""
	Vanilla GCN Model
	Code partially from https://github.com/williamleif/graphsage-simple/
	"""
	def __init__(self, num_classes, enc):
		super(GCN, self).__init__()
		self.enc = enc
		self.xent = nn.CrossEntropyLoss()
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
		init.xavier_uniform_(self.weight)


	def forward(self, nodes):
		embeds = self.enc(nodes)
		scores = self.weight.mm(embeds)
		return scores.t()

	def to_prob(self, nodes):
		pos_scores = torch.sigmoid(self.forward(nodes))
		return pos_scores

	def loss(self, nodes, labels):
		scores = self.forward(nodes)
		return self.xent(scores, labels.squeeze())


class GCNAggregator(nn.Module):
	"""
	Aggregates a node's embeddings using normalized mean of neighbors' embeddings
	"""

	def __init__(self, features, cuda=False, gcn=False):
		"""
		Initializes the aggregator for a specific graph.

		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		cuda -- whether to use GPU
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""

		super(GCNAggregator, self).__init__()

		self.features = features
		self.cuda = cuda
		self.gcn = gcn

	def forward(self, nodes, to_neighs):
		"""
		nodes --- list of nodes in a batch
		to_neighs --- list of sets, each set is the set of neighbors for node in batch
		"""
		# Local pointers to functions (speed hack)
		
		samp_neighs = to_neighs

		#  Add self to neighs
		samp_neighs = [samp_neigh.union(set([int(nodes[i])])) for i, samp_neigh in enumerate(samp_neighs)]
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1.0  # Adjacency matrix for the sub-graph
		if self.cuda:
			mask = mask.cuda()
		row_normalized = mask.sum(1, keepdim=True).sqrt()
		col_normalized = mask.sum(0, keepdim=True).sqrt()
		mask = mask.div(row_normalized).div(col_normalized)
		if self.cuda:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		to_feats = mask.mm(embed_matrix)
		return to_feats

class GCNEncoder(nn.Module):
	"""
	GCN Encoder Module
	"""
	def __init__(self, features, feature_dim,
				 embed_dim, adj_lists, aggregator,
				 num_sample=10,
				 base_model=None, gcn=False, cuda=False,
				 feature_transform=False):
		super(GCNEncoder, self).__init__()

		self.features = features
		self.feat_dim = feature_dim
		self.adj_lists = adj_lists
		self.aggregator = aggregator
		self.num_sample = num_sample
		if base_model != None:
			self.base_model = base_model

		self.gcn = gcn
		self.embed_dim = embed_dim
		self.cuda = cuda
		self.aggregator.cuda = cuda
		self.weight = nn.Parameter(
			torch.FloatTensor(embed_dim, self.feat_dim ))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes):
		"""
		Generates embeddings for a batch of nodes.
		Input:
			nodes -- list of nodes
		Output:
			embed_dim*len(nodes)
		"""
		neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes])

		if isinstance(nodes, list):
			index = torch.LongTensor(nodes)
		else:
			index = nodes

		combined = F.relu(self.weight.mm(neigh_feats.t()))
		return combined




def agg_func(nodes):
    # return {'neighbor_fraud': , }

    # src_feats = nodes.mailbox['m']  # [9, 1, 32]
    # src_nodes = nodes.mailbox['src'] # [9, 1]
    # src_fake_labels = nodes.mailbox['src_fake_label'] # [9, 1]
    #[1,3,32] * [1,3,1]

    return {'neigh_homo': (nodes.mailbox['m'] * (nodes.mailbox['simi_weight_mask']).unsqueeze(-1)).sum(1),
            'neigh_hete': (nodes.mailbox['m'] * (~nodes.mailbox['simi_weight_mask']).unsqueeze(-1)).sum(1)}

def mp_func(edges):
	src_simi_weight = edges.src['simi_weight']
	# src =  edges.edges()[0]
	src = edges.src['_ID']  
	simi_weight_mask = (src_simi_weight>0)

	message =  edges.src['feature']*edges.data['_edge_weight_homo'].unsqueeze(-1)*(simi_weight_mask.unsqueeze(-1)) \
				+ edges.src['feature']*edges.data['_edge_weight_hete'].unsqueeze(-1)*(~simi_weight_mask.unsqueeze(-1)) 
	return {'m': message, 'src': src, 'simi_weight_mask': simi_weight_mask}


class MySAGEConvTSTPartv2(SAGEConv):
	def __init__(
			self,
			in_feats,
			out_feats,
			aggregator_type,
			feat_drop=0,
			attn_drop=0.05,
			bias=True,
			norm=None,
			activation=None,
	):
		super(MySAGEConvTSTPartv2,self).__init__(in_feats,
			out_feats,
			aggregator_type,
			feat_drop,
			bias,
			norm,
			activation)
		
		# Functions
		self.softmax = nn.Softmax(dim=-1)
		# self.KLDiv = torch.nn.functional.kl_div
		self.balance = nn.Sigmoid()
		# self.balance1 = nn.Sigmoid()
		self.attn = nn.Parameter(torch.FloatTensor(size=(1, 3, out_feats)))
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2) 
		self.attn_drop=nn.Dropout(attn_drop)
		from model.LASAGE_S import LIMLP
		self.fc_neigh_homo =  LIMLP(self._in_src_feats, hidden_channels=out_feats, output_channels=out_feats,
                                     num_layers=1, batch_size = 512, origin_infeat = self._in_src_feats, activation=nn.ReLU(inplace=True))

		self.fc_neigh_hete =  LIMLP(self._in_src_feats, hidden_channels=out_feats, output_channels=out_feats,
									num_layers=1, batch_size = 512, origin_infeat = self._in_src_feats, activation=nn.ReLU(inplace=True))

		self.relu = nn.ReLU()



	def fetch_feat(self, nodes):
		if self.cuda and isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = torch.LongTensor(nodes)
		return self.features(index)


	def generate_edge_attn(self, graph, simi, edge_attr="_edge_weight"):
		graph.edata["e"] = simi
		e = self.leaky_relu(
			graph.edata.pop("e")
		)  # (num_src_edge, num_heads, out_dim)
		# e = (
		# 	(e * self.attn).sum(dim=-1).unsqueeze(dim=2)
		# )  # (num_edge, num_heads, 1)
		# print(edge_weight)

		graph.edata["a"] = self.attn_drop(
			edge_softmax(graph, e)
		)  # (num_edge, num_heads)
		graph.edata[edge_attr] = graph.edata.pop("a")
		
	

	def forward(self, graph, feat, edge_weight=None):
		r"""

		Description
		-----------
		Compute GraphSAGE layer.

		Parameters
		----------
		graph : DGLGraph
			The graph. # block
		feat : torch.Tensor or pair of torch.Tensor
			If a torch.Tensor is given, it represents the input feature of shape
			:math:`(N, D_{in})`
			where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
			If a pair of torch.Tensor is given, the pair must contain two tensors of shape
			:math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
		edge_weight : torch.Tensor, optional
			Optional tensor on the edge. If given, the convolution will weight
			with regard to the message.

		Returns
		-------
		torch.Tensor
			The output feature of shape :math:`(N_{dst}, D_{out})`
			where :math:`N_{dst}` is the number of destination nodes in the input graph,
			:math:`D_{out}` is the size of the output feature.
		"""
		with (graph.local_scope()):
			# graph.srcdata["h"] = feat
			if isinstance(feat, tuple):
				feat_src = self.feat_drop(feat[0])
				feat_dst = self.feat_drop(feat[1])
			else:
				feat_src = feat_dst = self.feat_drop(feat)
				if graph.is_block:
					feat_dst = feat_src[: graph.number_of_dst_nodes()]
			msg_fn = fn.copy_u("h", "m")

			
			edge_weight = graph.srcdata['simi_weight'][graph.edges()[0]]


			#GCD attention
			if edge_weight is not None:
				assert edge_weight.shape[0] == graph.num_edges()
				# self.generate_edge_attn(graph,edge_weight)
				self.generate_edge_attn(graph,edge_weight,"_edge_weight_homo")
				self.generate_edge_attn(graph,-edge_weight,"_edge_weight_hete")
				# msg_fn = fn.u_mul_e("h", "_edge_weight", "m")
			h_self = feat_dst

			# Handle the case of graphs without edges
			if graph.num_edges() == 0:
				graph.dstdata["neigh"] = torch.zeros(
					feat_dst.shape[0], self._in_src_feats
				).to(feat_dst)

			# Determine whether to apply linear transformation before message passing A(XW)
			lin_before_mp = self._in_src_feats > self._out_feats

			# Message Passing
			if self._aggre_type == "mean":
				graph.srcdata["h"] = (
					self.fc_neigh(feat_src) if lin_before_mp else feat_src
				)
				graph.update_all(mp_func, agg_func)
				h_neigh_homo = graph.dstdata["neigh_homo"]
				h_neigh_hete = graph.dstdata["neigh_hete"]
				if not lin_before_mp:
					# h_neigh = self.fc_neigh(h_neigh)
					h_neigh_homo = self.fc_neigh_homo(h_neigh_homo,h_self)
					h_neigh_hete = self.fc_neigh_hete(h_neigh_hete,h_self)
				
				h_neigh = h_neigh_homo + h_neigh_hete
				
			elif self._aggre_type == "gcn":
				check_eq_shape(feat)
				graph.srcdata["h"] = (
					self.fc_neigh(feat_src) if lin_before_mp else feat_src
				)
				if isinstance(feat, tuple):  # heterogeneous
					graph.dstdata["h"] = (
						self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
					)
				else:
					if graph.is_block:
						graph.dstdata["h"] = graph.srcdata["h"][
							: graph.num_dst_nodes()
						]
					else:
						graph.dstdata["h"] = graph.srcdata["h"]
				#message generation
				graph.update_all(msg_fn, fn.sum("m", "neigh"))
				# divide in_degrees
				degs = graph.in_degrees().to(feat_dst)
				h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
					degs.unsqueeze(-1) + 1
				)
				if not lin_before_mp:
					h_neigh = self.fc_neigh(h_neigh)
			elif self._aggre_type == "pool":
				graph.srcdata["h"] = F.relu(self.fc_pool(feat_src))
				graph.update_all(msg_fn, fn.max("m", "neigh"))
				h_neigh = self.fc_neigh(graph.dstdata["neigh"])
			elif self._aggre_type == "lstm":
				graph.srcdata["h"] = feat_src
				graph.update_all(msg_fn, self._lstm_reducer)
				h_neigh = self.fc_neigh(graph.dstdata["neigh"])
			else:
				raise KeyError(
					"Aggregator type {} not recognized.".format(
						self._aggre_type
					)
				)

			# GraphSAGE GCN does not require fc_self.
			if self._aggre_type == "gcn":
				rst = h_neigh
				# add bias manually for GCN
				if self.bias is not None:
					rst = rst + self.bias
			else:
				rst = self.fc_self(h_self) + h_neigh

			# activation
			if self.activation is not None:
				rst = self.activation(rst)
			# normalization
			if self.norm is not None:
				rst = self.norm(rst)
			return rst



#改变MYSAGECONV来查看不同的实验架构
class GraphSAGE_DGL(nn.Module):
	def __init__(self, in_size, 
	      			   hid_size, 
					   out_size, 
					   num_layers, 
					   dropout, 
					   proj, 
					   num_relations,
					   out_proj_size = None, 
					   agg = "mean", 
					   relation_agg = None,
					   attn_drop = 0,
				 	   TST = True):
		super(GraphSAGE_DGL, self).__init__()
		self.layers = nn.ModuleList()
		self.in_size = in_size
		self.hid_size = hid_size
		self.out_size = out_size
		self.proj = proj
		self.relation_agg = relation_agg
		self.num_relations = num_relations
		for i in range(num_layers):
			if i == num_layers-1:
				hid_size = out_size
			if TST:
				self.layers.append(MySAGEConvTSTPartv2(in_size, hid_size, agg, feat_drop=0, attn_drop= attn_drop))
				print(self.layers)
			else:
				self.layers.append(MySAGEConv(in_size, hid_size, agg))
			in_size = hid_size

		self.dropout = dropout
		
		# if self.relation_agg == 'cat':
		self.relation_mlp = nn.ModuleList()
		for j in range(num_layers):
			if i == num_layers-1:
				hid_size = out_size
			self.relation_mlp.append(nn.Linear(hid_size*num_relations if self.relation_agg == 'cat' else hid_size,hid_size))
		
		if proj:
			self.Conv = nn.Conv1d(out_size, out_proj_size, kernel_size=1)

		self.tstconv = TSTConv(self.in_size,self.hid_size,self.out_size) #old1 amazon比较高？



	def forward(self, blocks, relations, feats):   # blocks 
		h = feats 
		#calculate edge_weight


		for l, (layer, block) in enumerate(zip(self.layers, blocks)):  # blocks of one relation in one layer
			# feature optimization, GCD generation module
			feat_sim, simi_weight = self.tstconv(block,h)
			
			#generate GCD
			blocks[0].srcdata['simi_weight'] = simi_weight
			h = feat_sim #mixed feat?!!!
			


			block.srcdata['feature'] = feat_sim
			
			layer_emb = []  
			#aggregation
			for r, etype in enumerate(relations):
				b_graph = block[etype]  
				layer_emb.append(layer(b_graph, h))
			if self.relation_agg == 'cat':
				relation_agg_emb = torch.cat(layer_emb, dim=1)
			elif self.relation_agg == 'mean':
				relation_agg_emb = torch.mean(layer_emb, dim=0)
			elif self.relation_agg == 'add':
				relation_agg_emb = torch.sum(torch.stack(layer_emb), dim=0)
			#inter-relation MLP
			h = self.relation_mlp[l](relation_agg_emb)
				 
			if l != len(self.layers) - 1:
				h = F.relu(h)
				# h = self.dropout(h)  #
		if self.proj:
			if self.dropout > 0:
				h = F.dropout(h, self.dropout, training=self.training)
			h = h.unsqueeze(0)
			h = h.permute((0,2,1))
			h = self.Conv(h)
			h = h.permute((0,2,1)).squeeze()
		return h

	def inference(self, g, device, batch_size):
		"""Conduct layer-wise inference to get all the node embeddings."""
		feat = g.ndata["feat"]
		sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
		dataloader = DataLoader(
			g,
			torch.arange(g.num_nodes()).to(g.device),
			sampler,
			device=device, 
			batch_size=batch_size, 
			shuffle=False, 
			drop_last=False, 
			num_workers=0
		)
		buffer_device = torch.device("cpu")
		pin_memory = buffer_device != device

		for l, layer in enumerate(self.layers):
			y = torch.empty(
				g.num_nodes(),
				self.hid_size if l != len(self.layers) - 1 else self.out_size,
				device=buffer_device,
				pin_memory=pin_memory,
			)
			feat = feat.to(device)
			for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
				x = feat[input_nodes]
				h = layer(blocks[0], x)  # len(blocks) = 1
				if l != len(self.layers) - 1:
					h = F.relu(h)
					h = self.dropout(h)
				# by design, our output nodes are contiguous
				y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
			feat = y
		return y


class GraphSAGE_Former(GraphSAGE_DGL):
	def __init__(self, in_size, hid_size, out_size, num_layers, dropout, proj, num_relations, out_proj_size=None, agg="mean", relation_agg=None):
		super().__init__(in_size, hid_size, out_size, num_layers, dropout, proj, num_relations, out_proj_size, agg, relation_agg)



def train(args, device, g, dataset, model, num_classes):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):

            x = blocks[0].srcdata["feat"]    
            y = blocks[-1].dstdata["label"]  
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader, num_classes)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc.item()
            )
        )


def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):  

        with torch.no_grad():
            x = blocks[0].srcdata["feat"] 
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )