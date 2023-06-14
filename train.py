import torch
import numpy as np
import argparse
from utils import set_seed, load_feat, load_graph
from data_process_utils import check_data_leakage


####################################################################
####################################################################
####################################################################

def print_model_info(model):
    print(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters])
    print('Trainable Parameters: %d' % parameters)

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='movie')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=600)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--max_edges', type=int, default=50)
    parser.add_argument('--num_edgeType', type=int, default=0, help='num of edgeType')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--predict_class', action='store_true')
    # FFN layer
    parser.add_argument('--channel_expansion_factor', type=int, default=2)
    # model
    parser.add_argument('--window_size', type=int, default=5) # 10
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='sthn') 
    parser.add_argument('--neg_samples', type=int, default=1)
    parser.add_argument('--extra_neg_samples', type=int, default=5)
    parser.add_argument('--num_neighbors', type=int, default=50)
    parser.add_argument('--sampled_num_hops', type=int, default=1)
    parser.add_argument('--time_dims', type=int, default=100)
    parser.add_argument('--hidden_dims', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--regen_models', action='store_true')
    parser.add_argument('--check_data_leakage', action='store_true')
    
    parser.add_argument('--ignore_node_feats', action='store_true')
    parser.add_argument('--node_feats_as_edge_feats', action='store_true')
    parser.add_argument('--ignore_edge_feats', action='store_true')
    parser.add_argument('--use_onehot_node_feats', action='store_true')
    parser.add_argument('--use_type_feats', action='store_true')

    parser.add_argument('--use_graph_structure', action='store_true')
    parser.add_argument('--structure_time_gap', type=int, default=2000)
    parser.add_argument('--structure_hops', type=int, default=1) 

    parser.add_argument('--use_node_cls', action='store_true')
    parser.add_argument('--use_cached_subgraph', action='store_true')
    return parser.parse_args()


def load_all_data(args):

    # load graph
    g, df = load_graph(args.data)
    
    args.train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    args.val_edge_end   = df[df['ext_roll'].gt(1)].index[ 0]
    args.num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
    args.num_edges = len(df)
    print('Train %d, Valid %d, Test %d'%(args.train_edge_end, 
                                         args.val_edge_end-args.train_edge_end,
                                         len(df)-args.val_edge_end))
    print('Num nodes %d, num edges %d'%(args.num_nodes, args.num_edges))

    # load feats 
    node_feats, edge_feats = load_feat(args.data)
    node_feat_dims = 0 if node_feats is None else node_feats.shape[1]
    edge_feat_dims = 0 if edge_feats is None else edge_feats.shape[1]

    # feature pre-processing
    if args.use_onehot_node_feats:
        print('>>> Use one-hot node features')
        node_feats = torch.eye(args.num_nodes)
        node_feat_dims = node_feats.size(1)

    if args.ignore_node_feats:
        print('>>> Ignore node features')
        # node_feats = None
        node_feats = torch.zeros(args.num_nodes, 1)
        node_feat_dims = 0

    if args.use_type_feats:
        edge_type = df.label.values
        args.num_edgeType = len(set(edge_type.tolist()))
        edge_feats = torch.nn.functional.one_hot(torch.from_numpy(edge_type-1), 
                                                 num_classes = args.num_edgeType)
        edge_feat_dims = edge_feats.size(1)
        
    print('Node feature dim %d, edge feature dim %d'%(node_feat_dims, edge_feat_dims))
    
    # double check (if data leakage then cannot continue the code)
    if args.check_data_leakage:
        check_data_leakage(args, g, df)

    args.node_feat_dims = node_feat_dims
    args.edge_feat_dims = edge_feat_dims
    
    if node_feats != None:
        node_feats = node_feats.to(args.device) # here we only move node feats to cuda, not edges because too many edges
    if edge_feats != None:
        edge_feats = edge_feats.to(args.device)
    
    return node_feats, edge_feats, g, df, args


def load_model(args):

    # get model 
    
    edge_predictor_configs = {
        'dim_in_time': args.time_dims,
        'dim_in_node': args.node_feat_dims,
        'predict_class': 1 if not args.predict_class else args.num_edgeType+1,
    }
    if args.model == 'sthn':
        if args.predict_class:
            from model import Mixer_multiclass as Mixer_per_node
        else:
            from model import Mixer_per_node
        from link_pred_train_utils import link_pred_train

        mixer_configs = {
            'per_graph_size'  : args.max_edges,
            'time_channels'   : args.time_dims, 
            'input_channels'  : args.edge_feat_dims, 
            'hidden_channels' : args.hidden_dims, 
            'out_channels'    : args.hidden_dims,
            'num_layers'      : args.num_layers,
            'dropout'         : args.dropout,
            'channel_expansion_factor': args.channel_expansion_factor,
            'window_size'     : args.window_size,
            'use_single_layer' : False
        }  
        
    else:
        NotImplementedError()

    model = Mixer_per_node(mixer_configs, edge_predictor_configs)
    for k, v in model.named_parameters():
        print(k, v.requires_grad)

    print_model_info(model)

    return model, args, link_pred_train
        
####################################################################
####################################################################
####################################################################

if __name__ == "__main__":
    args = get_args()

    args.regen_models = True
    args.use_graph_structure = True
    args.ignore_node_feats = True
    args.use_type_feats = True
    
    args.use_cached_subgraph = True
    args.num_neighbors = args.max_edges
    print(args)
    
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.device = torch.device(args.device)
    set_seed(0)
    
    ###################################################

    # load feats + graph
    node_feats, edge_feats, g, df, args = load_all_data(args)
        
    ###################################################
    # get model 
    model, args, link_pred_train = load_model(args)

    ###################################################
    # Link prediction
    print('Train link prediction task from scratch ...')
    model = link_pred_train(model.to(args.device), args, g, df, node_feats, edge_feats)
