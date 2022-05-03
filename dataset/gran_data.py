import torch
import time
import os
import pickle
import glob
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from utils.data_helper import *


class GRANData(object):

  def __init__(self, config, graphs, tag='train'):
    self.config = config
    self.data_path = config.dataset.data_path
    self.model_name = config.model.name
    self.max_num_nodes = config.model.max_num_nodes
    self.block_size = config.model.block_size
    self.stride = config.model.sample_stride

    self.graphs = graphs
    self.num_graphs = len(graphs)
    self.npr = np.random.RandomState(config.seed)
    self.node_order = config.dataset.node_order
    self.num_canonical_order = config.model.num_canonical_order
    self.tag = tag
    self.num_fwd_pass = config.dataset.num_fwd_pass
    self.is_sample_subgraph = config.dataset.is_sample_subgraph
    self.num_subgraph_batch = config.dataset.num_subgraph_batch
    self.is_overwrite_precompute = config.dataset.is_overwrite_precompute

    if self.is_sample_subgraph:
      assert self.num_subgraph_batch > 0

    self.save_path = os.path.join(
        self.data_path, '{}_{}_{}_{}_{}_{}_{}_precompute'.format(
            config.model.name, config.dataset.name, tag, self.block_size,
            self.stride, self.num_canonical_order, self.node_order))

    if not os.path.isdir(self.save_path) or self.is_overwrite_precompute:
      self.file_names = []
      if not os.path.isdir(self.save_path):
        os.makedirs(self.save_path)

      self.config.dataset.save_path = self.save_path
      for index in tqdm(range(self.num_graphs)):
        # One graph.
        G = self.graphs[index]
        # A list of the adj matrices of the graph in different node orders.
        data = self._get_graph_data(G)
        tmp_path = os.path.join(self.save_path, '{}_{}.p'.format(tag, index))
        pickle.dump(data, open(tmp_path, 'wb'))
        # One file per graph.
        self.file_names += [tmp_path]
    else:
      self.file_names = glob.glob(os.path.join(self.save_path, '*.p'))

  def _get_graph_data(self, G):
    node_degree_list = [(n, d) for n, d in G.degree()]

    adj_0 = np.array(nx.to_numpy_matrix(G))

    ### Degree descent ranking
    # N.B.: largest-degree node may not be unique
    degree_sequence = sorted(
        node_degree_list, key=lambda tt: tt[1], reverse=True)
    adj_1 = np.array(
        nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))

    ### Degree ascent ranking
    degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1])
    adj_2 = np.array(
        nx.to_numpy_matrix(G, nodelist=[dd[0] for dd in degree_sequence]))

    ### BFS & DFS from largest-degree node
    CGs = [G.subgraph(c) for c in nx.connected_components(G)]

    # rank connected componets from large to small size
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

    node_list_bfs = []
    node_list_dfs = []
    for ii in range(len(CGs)):
      node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
      degree_sequence = sorted(
          node_degree_list, key=lambda tt: tt[1], reverse=True)

      bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
      dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])

      node_list_bfs += list(bfs_tree.nodes())
      node_list_dfs += list(dfs_tree.nodes())

    adj_3 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_bfs))
    adj_4 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_dfs))

    ### k-core
    num_core = nx.core_number(G)
    core_order_list = sorted(list(set(num_core.values())), reverse=True)
    degree_dict = dict(G.degree())
    core_to_node = defaultdict(list)
    for nn, kk in num_core.items():
      core_to_node[kk] += [nn]

    node_list = []
    for kk in core_order_list:
      sort_node_tuple = sorted(
          [(nn, degree_dict[nn]) for nn in core_to_node[kk]],
          key=lambda tt: tt[1],
          reverse=True)
      node_list += [nn for nn, dd in sort_node_tuple]

    adj_5 = np.array(nx.to_numpy_matrix(G, nodelist=node_list))

    if self.num_canonical_order == 5:
      adj_list = [adj_0, adj_1, adj_3, adj_4, adj_5]
    else:
      if self.node_order == 'degree_decent':
        adj_list = [adj_1]
      elif self.node_order == 'degree_accent':
        adj_list = [adj_2]
      elif self.node_order == 'BFS':
        adj_list = [adj_3]
      elif self.node_order == 'DFS':
        adj_list = [adj_4]
      elif self.node_order == 'k_core':
        adj_list = [adj_5]
      elif self.node_order == 'DFS+BFS':
        adj_list = [adj_4, adj_3]
      elif self.node_order == 'DFS+BFS+k_core':
        adj_list = [adj_4, adj_3, adj_5]
      elif self.node_order == 'DFS+BFS+k_core+degree_decent':
        adj_list = [adj_4, adj_3, adj_5, adj_1]
      elif self.node_order == 'all':
        adj_list = [adj_4, adj_3, adj_5, adj_1, adj_0]
      else:
        adj_list = [adj_0]

    # print('number of nodes = {}'.format(adj_0.shape[0]))

    return adj_list

  def __getitem__(self, index):
    # @congrui's comment: This data loader is implemented in a rather strange way. According to the paper,
    # the goal is to train a network of p(edges to newly added nodes | existing graph), 
    # maximizing a log-likelihood lower bound computed by summing the likelihood
    # under a set of pre-defined node orderings. This seems achievable by simply
    # sampling as follows:
    # Select a graph -> select an ordering -> select a subgraph -> generate a training pair
    # Which could be implemented much easier than the brainfuck here.
    # 
    # The data loader here though is implemented in a weird manner.
    # Weird 1): __getitem__ returns a batch, size=50 something, then collate_fn does some post-processing.
    #           and the batch_size for pytorch is actually 1 X(
    # Weird 2): Each batch then seems to come from the same graph. Shouldn't we mix them together?
    # Weird 3): Items in the batch keeps the order of the specified canonical orders. Is this necessary?
    # Weird 4): The subgraphs in a batch are randomly selected, but they keep an increasing-node-num order
    #           in the batch. Is this necessary?

    K = self.block_size
    N = self.max_num_nodes
    S = self.stride

    # load graph
    # List of adj matrices of one graph in different node orders.
    adj_list = pickle.load(open(self.file_names[index], 'rb'))
    num_nodes = adj_list[0].shape[0]
    num_subgraphs = int(np.floor((num_nodes - K) / S) + 1)

    if self.is_sample_subgraph:
      if self.num_subgraph_batch < num_subgraphs:
        num_subgraphs_pass = int(
            np.floor(self.num_subgraph_batch / self.num_fwd_pass))
      else:
        num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))

      end_idx = min(num_subgraphs, self.num_subgraph_batch)
    else:
      num_subgraphs_pass = int(np.floor(num_subgraphs / self.num_fwd_pass))
      end_idx = num_subgraphs

    ### random permute subgraph
    rand_perm_idx = self.npr.permutation(num_subgraphs).tolist()

    start_time = time.time()
    data_batch = []
    for ff in range(self.num_fwd_pass):
      ff_idx_start = num_subgraphs_pass * ff
      if ff == self.num_fwd_pass - 1:
        ff_idx_end = end_idx
      else:
        ff_idx_end = (ff + 1) * num_subgraphs_pass

      # Randomly samples (without replacement) up to num_subgraphs_pass subgraphs.
      rand_idx = rand_perm_idx[ff_idx_start:ff_idx_end]

      edges = []
      node_idx_gnn = []
      node_idx_feat = []
      label = []
      subgraph_size = []
      subgraph_idx = []
      att_idx = []
      subgraph_count = 0

      for ii in range(len(adj_list)):
        # Loops over each node ordering.
        adj_full = adj_list[ii]
        # adj_tril = np.tril(adj_full, k=-1)

        idx = -1
        # jj is the number of 'context' nodes.
        for jj in range(0, num_nodes, S):
          # loop over different subgraphs
          idx += 1

          ### for each size-(jj+K) subgraph, we generate edges for the new block of K nodes
          if jj + K > num_nodes:
            # Each step predicts the next K nodes. Stops if no sufficient nodes left.
            break

          if idx not in rand_idx:
            # Only produces subgraphs sampled for this group.
            continue

          ### get graph for GNN propagation
          adj_block = np.pad(
              adj_full[:jj, :jj], ((0, K), (0, K)),
              'constant',
              constant_values=1.0)  # assuming fully connected for the new block
          adj_block = np.tril(adj_block, k=-1)
          adj_block = adj_block + adj_block.transpose()
          adj_block = torch.from_numpy(adj_block).to_sparse()
          # [Tensor(2, N_entries)] Each column for one edge.
          edges += [adj_block.coalesce().indices().long()]

          ### Gets attention index. att_idx is an 1-d array corresponding to each
          # node in the subgraph.
          # Existing/context nodes: values are 0
          # Newly added/prediction nodse: values are 1, ..., K
          if jj == 0:
            att_idx += [np.arange(1, K + 1).astype(np.uint8)]
          else:
            att_idx += [
                np.concatenate([
                    np.zeros(jj).astype(np.uint8),
                    np.arange(1, K + 1).astype(np.uint8)
                ])
            ]

          ### get node feature index for GNN input
          # use inf to indicate the newly added nodes where input feature is zero
          if jj == 0:
            node_idx_feat += [np.ones(K) * np.inf]
          else:
            node_idx_feat += [
                np.concatenate([np.arange(jj) + ii * N,
                                np.ones(K) * np.inf])
            ]

          ### get node index for GNN output
          idx_row_gnn, idx_col_gnn = np.meshgrid(
              np.arange(jj, jj + K), np.arange(jj + K))
          idx_row_gnn = idx_row_gnn.reshape(-1, 1)
          idx_col_gnn = idx_col_gnn.reshape(-1, 1)
          node_idx_gnn += [
              np.concatenate([idx_row_gnn, idx_col_gnn],
                             axis=1).astype(np.int64)
          ]

          ### get predict label
          label += [
              adj_full[idx_row_gnn, idx_col_gnn].flatten().astype(np.uint8)
          ]

          subgraph_size += [jj + K]
          subgraph_idx += [
              np.ones_like(label[-1]).astype(np.int64) * subgraph_count
          ]
          subgraph_count += 1
        # Ends loop over subgraphs.
      # Ends loop over canonical orders.

      ### adjust index basis for the selected subgraphs
      cum_size = np.cumsum([0] + subgraph_size).astype(np.int64)
      for ii in range(len(edges)):
        edges[ii] = edges[ii] + cum_size[ii]
        node_idx_gnn[ii] = node_idx_gnn[ii] + cum_size[ii]

      ### pack tensors
      data = {}
      data['adj'] = np.tril(np.stack(adj_list, axis=0), k=-1)  # [canonical_order_num, max_node_num, max_node_num]
      data['edges'] = torch.cat(edges, dim=1).t().long()  # [sum(subgraph_context_edge_nums), 2]
      data['node_idx_gnn'] = np.concatenate(node_idx_gnn)  # [sum(subgraph_predict_edge_nums), 2]
      data['node_idx_feat'] = np.concatenate(node_idx_feat)  # [sum(subgraph_node_nums)]
      data['label'] = np.concatenate(label)  # [sum(subgraph_predict_edge_nums)]
      data['att_idx'] = np.concatenate(att_idx)  # [sum(subgraph_node_nums)]
      # The index of the subgraph that the predicted edge belongs to.
      data['subgraph_idx'] = np.concatenate(subgraph_idx)  # [sum(subgraph_predict_edge_nums)]
      # The number of subgraphs in this data item.
      data['subgraph_count'] = subgraph_count  # Scalar
      # The number of nodes in the very original graph for this data item (one graph per item).
      data['num_nodes'] = num_nodes  # Scalar
      # The number of (context and predict) nodes in each subgraph of this data item.
      data['subgraph_size'] = subgraph_size  # [item_subgraph_num]
      # The total number of nodes (from each subgraph) in this data item.
      data['num_count'] = sum(subgraph_size)  # Scalar
      data_batch += [data]
    # Ends loop over "fwd passes".

    end_time = time.time()

    return data_batch

  def __len__(self):
    return self.num_graphs

  def collate_fn(self, batch):
    assert isinstance(batch, list)
    start_time = time.time()
    batch_size = len(batch)
    N = self.max_num_nodes
    C = self.num_canonical_order
    batch_data = []

    # Since num_fwd_pass = 1 for all configs, this loop can be ignored.
    for ff in range(self.num_fwd_pass):
      data = {}
      batch_pass = []
      for bb in batch:
        batch_pass += [bb[ff]]

      pad_size = [self.max_num_nodes - bb['num_nodes'] for bb in batch_pass]
      subgraph_idx_base = np.array([0] +
                                   [bb['subgraph_count'] for bb in batch_pass])
      subgraph_idx_base = np.cumsum(subgraph_idx_base)
      # [B] When putting all subgraphs from a batch in a list, this is starting index of each data item (original graph).
      data['subgraph_idx_base'] = torch.from_numpy(
        subgraph_idx_base)
      # [B] Number of nodes in each original graph.
      data['num_nodes_gt'] = torch.from_numpy(
          np.array([bb['num_nodes'] for bb in batch_pass])).long().view(-1)
      # [B, C, N, N] 0-padded subgraph adj-matrics, to [max_node_num, max_node_num]
      data['adj'] = torch.from_numpy(
          np.stack(
              [
                  np.pad(
                      bb['adj'], ((0, 0), (0, pad_size[ii]), (0, pad_size[ii])),
                      'constant',
                      constant_values=0.0) for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).float()
      # Similar to subgraph_idx_base, but for nodes. When putting all nodes in the batch to a list, this is the starting index of each data item (original graph).
      idx_base = np.array([0] + [bb['num_count'] for bb in batch_pass])
      idx_base = np.cumsum(idx_base)
      # [batch_context_edge_num, 2]
      data['edges'] = torch.cat(
          [bb['edges'] + idx_base[ii] for ii, bb in enumerate(batch_pass)],
          dim=0).long()
      # [batch_predict_edge_num, 2]
      data['node_idx_gnn'] = torch.from_numpy(
          np.concatenate(
              [
                  bb['node_idx_gnn'] + idx_base[ii]
                  for ii, bb in enumerate(batch_pass)
              ],
              axis=0)).long()
      # [batch_node_num]
      data['att_idx'] = torch.from_numpy(
          np.concatenate([bb['att_idx'] for bb in batch_pass], axis=0)).long()

      # [batch_node_num] shift one position for padding 0-th row feature in the model
      node_idx_feat = np.concatenate(
          [
              bb['node_idx_feat'] + ii * C * N
              for ii, bb in enumerate(batch_pass)
          ],
          axis=0) + 1
      node_idx_feat[np.isinf(node_idx_feat)] = 0
      node_idx_feat = node_idx_feat.astype(np.int64)
      # Notice, from now on 0 stands for nodes being predicted.
      data['node_idx_feat'] = torch.from_numpy(node_idx_feat).long()
      # [batch_predict_edge_num]
      data['label'] = torch.from_numpy(
          np.concatenate([bb['label'] for bb in batch_pass])).float()
      # [batch_predict_edge_num], the subgraph index of each predicted edge.
      data['subgraph_idx'] = torch.from_numpy(
          np.concatenate([
              bb['subgraph_idx'] + subgraph_idx_base[ii]
              for ii, bb in enumerate(batch_pass)
          ])).long()

      batch_data += [data]

    end_time = time.time()
    # print('collate time = {}'.format(end_time - start_time))

    return batch_data
