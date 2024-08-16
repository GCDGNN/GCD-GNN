import dgl
import numpy as np
import torch
from tqdm import tqdm


def add_meta_path(g, rel1, rel2):
    # 添加新关系
    # 找到所有的 'click' 和 'buy' 边
    rel1_src, rel1_dst = g.edges(etype=rel1)
    rel2_src, rel2_dst = g.edges(etype=rel2)

    # 创建一个新关系的字典来存储新边
    new_edges = {'src': [], 'dst': []}

    print('start adding meta path')
    # 查找符合 user -> click -> user -> buy -> user 的路径
    for i in tqdm(range(len(rel1_src))):
        intermediate_user = rel1_dst[i]
        # 查找所有 intermediate_user 作为起始，'buy' 类型的边
        indices = (rel2_src == intermediate_user).nonzero(as_tuple=True)[0]
        for idx in indices:
            final_user = rel2_dst[idx]
            # 将原始起始用户和最终用户作为一条新边添加
            new_edges['src'].append(rel1_src[i].item())
            new_edges['dst'].append(final_user.item())

    #处理图
    return new_edges

def new_heterograph(g, relations):
    new_rel = {}
    node_type = g.ntypes[0]
    for rel in g.etypes:
        src,dst = g.all_edges(etype=rel)
        new_rel[(node_type,rel,node_type)] = (src, dst)

    #新的
    for i,rel in enumerate(relations):
        src,dst = rel['src'], rel['dst']
        new_rel[(node_type,f'rel{i}',node_type)] = (src, dst)

    #创建新图
    new_g = dgl.heterograph(new_rel)

    # 复制节点数据
    for ntype in g.ntypes:
        new_g.nodes[ntype].data.update(g.nodes[ntype].data)

    print('starting saving')
    dgl.save_graphs('./datasets/new_yelp.bin', [new_g])
    print('save done!')
    return new_g


def process_yelp_metapath(g):
    relations = []
    for rel1 in g.etypes:
        for rel2 in g.etypes:
            if rel1 != rel2:
                relations.append(add_meta_path(g,rel1,rel2))

    return new_heterograph(g,relations)

def process_yelp_metapath_mini(g):
    relations = []
    # for rel1 in g.etypes:
    #     for rel2 in g.etypes:
    #         if rel1 != rel2:
    rel1 = g.etypes[0]
    rel2 = g.etypes[1]
    print(rel1, rel2)
    relations.append(add_meta_path(g,rel1,rel2))

    return new_heterograph(g,relations)
