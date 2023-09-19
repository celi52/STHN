from tqdm import tqdm
import torch
import time
import copy
import numpy as np
from torch_sparse import SparseTensor
from data_process_utils import pre_compute_subgraphs, get_random_inds, get_subgraph_sampler
from construct_subgraph import construct_mini_batch_giant_graph
from utils import row_norm
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from sklearn.preprocessing import MinMaxScaler

def run(model, optimizer, args, subgraphs, df, node_feats, edge_feats, MLAUROC, MLAUPRC, mode):
    time_epoch = 0
    ###################################################
    # setup modes
    if mode == 'train':
        model.train()
        cur_df = df[:args.train_edge_end]
        neg_samples = args.neg_samples
        cached_neg_samples = args.extra_neg_samples
        cur_inds = 0

    elif mode == 'valid':
        model.eval()
        cur_df = df[args.train_edge_end:args.val_edge_end]
        neg_samples = 1
        cached_neg_samples = 1
        cur_inds = args.train_edge_end

    elif mode == 'test':
        model.eval()
        cur_df = df[args.val_edge_end:]
        neg_samples = 1
        cached_neg_samples = 1
        cur_inds = args.val_edge_end

    train_loader = cur_df.groupby(cur_df.index // args.batch_size)
    pbar = tqdm(total=len(train_loader))
    pbar.set_description('%s mode with negative samples %d ...'%(mode, neg_samples))        
        
    ###################################################
    # compute + training + fetch all scores
    subgraphs, elabel = subgraphs
    loss_lst = []
    MLAUROC.reset()
    MLAUPRC.reset()
    scaler = MinMaxScaler()
    
    for ind in range(len(train_loader)):
        ###################################################
        if args.use_cached_subgraph == False and mode == 'train':
            subgraph_data_list = subgraphs.all_root_nodes[ind]
            mini_batch_inds = get_random_inds(len(subgraph_data_list), cached_neg_samples, neg_samples)
            subgraph_data = subgraphs.mini_batch(ind, mini_batch_inds)
        else: # valid + test
            subgraph_data_list = subgraphs[ind]
            mini_batch_inds = get_random_inds(len(subgraph_data_list), cached_neg_samples, neg_samples)
            subgraph_data = [subgraph_data_list[i] for i in mini_batch_inds]      
        subgraph_data = construct_mini_batch_giant_graph(subgraph_data, args.max_edges)

        # raw edge feats 
        subgraph_edge_feats = edge_feats[subgraph_data['eid']]
        subgraph_edts = torch.from_numpy(subgraph_data['edts']).float()
        if args.use_graph_structure and node_feats:
            num_of_df_links = len(subgraph_data_list) //  (cached_neg_samples+2)   
            subgraph_node_feats = compute_sign_feats(node_feats, df, cur_inds, num_of_df_links, subgraph_data['root_nodes'], args)
            cur_inds += num_of_df_links
        else:
            subgraph_node_feats = None
        # scale
        scaler.fit(subgraph_edts.reshape(-1,1))
        subgraph_edts = scaler.transform(subgraph_edts.reshape(-1,1)).ravel().astype(np.float32) * 1000
        subgraph_edts = torch.from_numpy(subgraph_edts)
        
        # get mini-batch inds
        all_inds, has_temporal_neighbors = [], []

        # ignore an edge pair if (src_node, dst_node) does not have temporal neighbors
        all_edge_indptr = subgraph_data['all_edge_indptr']
        
        for i in range(len(all_edge_indptr)-1):
            num_edges = all_edge_indptr[i+1] - all_edge_indptr[i]
            all_inds.extend([(args.max_edges * i + j) for j in range(num_edges)])
            has_temporal_neighbors.append(num_edges>0)
            
        if not args.predict_class:
            inputs = [
                subgraph_edge_feats.to(args.device), 
                subgraph_edts.to(args.device), 
                len(has_temporal_neighbors), 
                torch.tensor(all_inds).long() 
            ]
        else:
            subgraph_edge_type = elabel[ind]
            inputs = [
                subgraph_edge_feats.to(args.device), 
                subgraph_edts.to(args.device), 
                len(has_temporal_neighbors), 
                torch.tensor(all_inds).long(),  
                torch.from_numpy(subgraph_edge_type).to(args.device)
            ]
        
        start_time = time.time()
        loss, pred, edge_label = model(inputs, neg_samples, subgraph_node_feats)
        if mode == 'train' and optimizer != None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time_epoch += (time.time() - start_time)
        
        batch_auroc = MLAUROC.update(pred, edge_label)
        batch_auprc = MLAUPRC.update(pred, edge_label)
        loss_lst.append(float(loss))

        pbar.update(1)
    pbar.close()    
    total_auroc = MLAUROC.compute()
    total_auprc = MLAUPRC.compute()
    print('%s mode with time %.4f, AUROC %.4f, AUPRC %.4f, loss %.4f'%(mode, time_epoch, total_auroc, total_auprc, loss.item()))
    return_loss = np.mean(loss_lst)
    return total_auroc, total_auprc, return_loss, time_epoch


def link_pred_train(model, args, g, df, node_feats, edge_feats):
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ###################################################
    # get cached data
    if args.use_cached_subgraph:
        train_subgraphs = pre_compute_subgraphs(args, g, df, mode='train')
    else:
        train_subgraphs = get_subgraph_sampler(args, g, df, mode='train')
        
    valid_subgraphs = pre_compute_subgraphs(args, g, df, mode='valid')
    test_subgraphs  = pre_compute_subgraphs(args, g, df, mode='test' )
          
    ###################################################
    all_results = {
        'train_ap': [],
        'valid_ap': [],
        'test_ap' : [],
        'train_auc': [],
        'valid_auc': [],
        'test_auc' : [],
        'train_loss': [],
        'valid_loss': [],
        'test_loss': [],
    }

    low_loss = 100000
    user_train_total_time = 0
    user_epoch_num = 0
    if args.predict_class:
        num_classes = args.num_edgeType+1
        train_AUROC = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        valid_AUROC = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        test_AUROC = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        train_AUPRC = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
        valid_AUPRC = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
        test_AUPRC = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
    else:
        train_AUROC = BinaryAUROC(thresholds=None)
        valid_AUROC = BinaryAUROC(thresholds=None)
        test_AUROC = BinaryAUROC(thresholds=None)
        train_AUPRC = BinaryAveragePrecision(thresholds=None)
        valid_AUPRC = BinaryAveragePrecision(thresholds=None)
        test_AUPRC = BinaryAveragePrecision(thresholds=None)
        
    for epoch in range(args.epochs):
        print('>>> Epoch ', epoch+1)
        train_auc, train_ap, train_loss, time_train = run(model, optimizer, args, train_subgraphs, df, 
                                              node_feats, edge_feats, train_AUROC, train_AUPRC, mode='train')
        with torch.no_grad():
            # second variable (optimizer) is only required for training
            valid_auc, valid_ap, valid_loss, time_valid = run(copy.deepcopy(model), None, args, valid_subgraphs, df, 
                                                  node_feats, edge_feats, valid_AUROC, valid_AUPRC, mode='valid')
            # second variable (optimizer) is only required for training
            test_auc,  test_ap,  test_loss, time_test = run(copy.deepcopy(model), None, args, test_subgraphs,  df, 
                                                  node_feats, edge_feats, test_AUROC, test_AUPRC, mode='test')  

        if valid_loss < low_loss:
            best_auc_model = copy.deepcopy(model).cpu() 
            # best_auc = valid_auc
            low_loss = valid_loss
            best_epoch = epoch
            best_test_auc, best_test_ap = test_auc, test_ap
        
        user_train_total_time += time_train + time_valid
        user_epoch_num += 1
        if epoch > best_epoch + 20:
            break
        
        all_results['train_ap'].append(train_ap)
        all_results['valid_ap'].append(valid_ap)
        all_results['test_ap'].append(test_ap)
        
        all_results['valid_auc'].append(valid_auc)
        all_results['train_auc'].append(train_auc)
        all_results['test_auc'].append(test_auc)
        
        all_results['train_loss'].append(train_loss)
        all_results['valid_loss'].append(valid_loss)
        all_results['test_loss'].append(test_loss)        
        
    print('auroc %.4f, auprc score %.4f'%(best_test_auc, best_test_ap))     
    return best_auc_model


def compute_sign_feats(node_feats, df, start_i, num_links, root_nodes, args):
    num_duplicate = len(root_nodes) // num_links 
    num_nodes = args.num_nodes

    root_inds = torch.arange(len(root_nodes)).view(num_duplicate, -1)
    root_inds = [arr.flatten() for arr in root_inds.chunk(1, dim=1)]

    output_feats = torch.zeros((len(root_nodes), node_feats.size(1))).to(args.device)
    i = start_i

    for _root_ind in root_inds:

        if i == 0 or args.structure_hops == 0:
            sign_feats = node_feats.clone()
        else:
            prev_i = max(0, i - args.structure_time_gap)
            cur_df = df[prev_i: i] # get adj's row, col indices (as undirected)
            src = torch.from_numpy(cur_df.src.values)
            dst = torch.from_numpy(cur_df.dst.values)
            edge_index = torch.stack([
                torch.cat([src, dst]), 
                torch.cat([dst, src])
            ])
            edge_index, edge_cnt = torch.unique(edge_index, dim=1, return_counts=True) 
            mask = edge_index[0]!=edge_index[1] # ignore self-loops
            adj = SparseTensor(
                value = torch.ones_like(edge_cnt[mask]).float(),
                row = edge_index[0][mask].long(),
                col = edge_index[1][mask].long(),
                sparse_sizes=(num_nodes, num_nodes)
            )
            adj_norm = row_norm(adj).to(args.device)
            sign_feats = [node_feats]
            for _ in range(args.structure_hops):
                sign_feats.append(adj_norm@sign_feats[-1])
            sign_feats = torch.sum(torch.stack(sign_feats), dim=0)

        output_feats[_root_ind] = sign_feats[root_nodes[_root_ind]]

        i += len(_root_ind) // num_duplicate

    return output_feats
