import torch
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix
import time

from src.Utils import load_our_data, get_model
from src.args import get_citation_args
from src.recommendation_evaluate import recommendation_model

from sklearn.metrics import precision_score, recall_score, f1_score

args = get_citation_args()


#Retail_Rocket
args.dataset = 'Retail_Rocket'
eval_name = r'Retail_Rocket'
net_path = r'data/Retail_Rocket/Retail_Rocket.mat'
savepath = r'data/Retail_Rocket'
eval_name = r'Retail_Rocket'
encode_path=r'data/Retail_Rocket/Retail_Rocket_encode.txt'
file_name = r'data/Retail_Rocket'
eval_type = 'all'

adj_path=r'data/Retail_Rocket/Retail_Rocket_new_edge.pt'
graph_path = r'data/Retail_Rocket/Retail_Rocket.mat'


# #imdb_small
# args.dataset = 'imdb_small'
# eval_name = r'imdb_small'
# net_path = r'data/imdb_small/imdb_small.mat'
# savepath = r'data/imdb_small'
# eval_name = r'imdb_small'
# encode_path=r'data/imdb_small/imdb_small_encode.pt'
# adj_path=r'data/dblp_small/dblp_small_new_edge.pt'
# graph_path = r'data/small_alibaba_1_10/alibaba_multi.mat'
# file_name = r'data/imdb_small'
# eval_type = 'all'


#alibaba_small
# args.dataset = 'alibaba_small'
# eval_name = r'alibaba_small'
# net_path = r'data/alibaba_small/alibaba_small.mat'
# savepath = r'data/alibaba_small'
# eval_name = r'alibaba_small'
# encode_path=r'data/alibaba_small/alibaba_small_encode.txt'
# # adj_path=r'data/dblp_small/dblp_small_new_edge.pt'
# # graph_path = r'data/small_alibaba_1_10/alibaba_multi.mat'
# file_name = r'data/alibaba_small'
# eval_type = 'all'


# IMDB
# args.dataset = 'imdb_1_10'
# eval_name = r'data/imdb_1_10'
# net_path = r"data/IMDB/imdb_1_10.mat"
# savepath = r'data/imdb_embedding_1_10'
# eval_name = r'imdb_1_10'
# file_name = r'data/IMDB'
# eval_type = 'all'


# # DBLP
# args.dataset = 'DBLP'
# net_path = r"data/dblp/DBLP.mat"
# savepath = r'data/DBLP_embedding'
# eval_name = r'DBLP'
# file_name = r'data/dblp'
# eval_type = 'all'


# #dblp_large
# args.dataset = 'dblp_large'
# eval_name = r'dblp_large'
# net_path = r'data/dblp_large/dblp_large.mat'
# savepath = r'data/dblp_large'
# eval_name = r'dblp_large'
# encode_path=r'data/dblp_large/dblp_large_encode.pt'
# adj_path=r'data/dblp_small/dblp_small_new_edge.pt'
# graph_path = r'data/small_alibaba_1_10/alibaba_multi.mat'
# file_name = r'data/dblp_large'
# eval_type = 'all'


# Aminer
# args.dataset = 'Aminer_10k_4class'
# eval_name = r'Aminer_10k_4class'
# net_path = r'../data/Aminer_1_13/Aminer_10k_4class.mat'
# savepath = r'embedding/Aminer_10k_4class_aminer_embedding_'
# file_name = r'../data/Aminer_1_13'
# eval_type = 'all'


# alibaba
# args.dataset = 'small_alibaba_1_10'
# eval_name = r'small_alibaba_1_10'
# net_path = r'data/small_alibaba_1_10/small_alibaba_1_10.mat'
# savepath = r'data/alibaba_embedding_'
# file_name = r'data/small_alibaba_1_10'
# eval_type = 'all'


# amazon
# args.dataset = 'amazon'
# eval_name = r'amazon'
# net_path = r'data/amazon/amazon.mat'
# savepath = r'data/amazon_embedding_'
# file_name = r'data/amazon'
# eval_type = 'all'


# mat = loadmat(net_path)
# 
# try:
#     train = mat['A']
# except:
#     try:
#         train = mat['train']+mat['valid']+mat['test']
#     except:
#         try:
#             train = mat['train_full']+mat['valid_full']+mat['test_full']
#         except:
#             try:
#                 train = mat['edges']
#             except:
#                 train = np.vstack((mat['edge1'],mat['edge2']))
# 
# try:
#     feature = mat['full_feature']
# except:
#     try:
#         feature = mat['feature']
#     except:
#         try:
#             feature = mat['features']
#         except:
#             feature = mat['node_feature']
# 
# feature = csc_matrix(feature) if type(feature) != csc_matrix else feature
# 
# if net_path == 'imdb_1_10.mat':
#     A = train[0]
# elif args.dataset == 'Aminer_10k_4class':
#     A = [[mat['PAP'], mat['PCP'], mat['PTP'] ]]
#     feature = mat['node_feature']
#     feature = csc_matrix(feature) if type(feature) != csc_matrix else feature
# else:
#     A = train

print('start')
# new_adj = torch.load(adj_path)
mat = loadmat(net_path)
encode=np.loadtxt(encode_path)
encode=torch.tensor(encode)
print('end')
try:
    train = mat['A']
except:
    try:
        train = mat['train']+mat['valid']+mat['test']
    except:
        try:
            train = mat['train_full']+mat['valid_full']+mat['test_full']
        except:
            try:
                train = mat['edges']
            except:
                train = mat['edge']

try:
    feature = mat['full_feature']
except:
    try:
        feature = mat['feature']
    except:
        try:
            feature = mat['features']
        except:
            feature = mat['node_feature']
A = train[0][0]
feature = csc_matrix(feature) if type(feature) != csc_matrix else feature

if net_path == 'imdb_1_10.mat':
    A = train[0]
elif args.dataset == 'Aminer_10k_4class':
    A = [[mat['PAP'], mat['PCP'], mat['PTP'] ]]
    feature = mat['node_feature']
    feature = csc_matrix(feature) if type(feature) != csc_matrix else feature
else:
    A = train
if args.dataset == 'small_alibaba_1_10':
    mat=loadmat(graph_path)
    # mat=loadmat(net_path)
    A=mat['edge']
if args.dataset == 'dblp_small':
    mat=loadmat(net_path)
    A=mat['edge']
if args.dataset == 'imdb_small':
    mat=loadmat(net_path)
    A=mat['edge']
if args.dataset == 'alibaba_large':
    mat=loadmat(net_path)
    A=mat['edge']
if args.dataset == 'alibaba_small':
    mat=loadmat(net_path)
    A=mat['edge']
if args.dataset == 'dblp_large':
    mat=loadmat(net_path)
    A=mat['edge']

node_matching = False

# adj, features, labels, idx_train, idx_val, idx_test = load_our_data(args.dataset, False)
# model = get_model(args.model, features.size(1), labels.max().item()+1, A, args.hidden, args.out, args.dropout, False)
adj, features, labels, idx_train, idx_val, idx_test = load_our_data(args.dataset, False)
model = get_model(args.model, features.size(1), labels.max().item()+1, A, args.hidden, args.out, args.dropout, False)

starttime=time.time()
precision, recall, f1 = recommendation_model(model, file_name, features, A, encode, eval_type, node_matching)
endtime=time.time()

print('Test ROC: {:.10f},  PR: {:.10f}'.format(precision, recall, f1))
