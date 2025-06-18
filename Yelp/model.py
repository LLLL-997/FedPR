import torch
import torch.nn as nn
import math
import config
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from sklearn.manifold import TSNE

seed = config.Seed
g = torch.Generator()
g.manual_seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
device = torch.device("cuda")
device_ids = [0, 1]
class OneGraphAttention(nn.Module):
    def __init__(self, in_size, hidden_size = 32):
        super(OneGraphAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1,)
        )
    def forward(self, z):
        # 元路径下每个元路径对应的注意力权重
        w = self.project(z)
        # 元路径注意力系数
        # softmax表示权值重要程度
        beta = torch.softmax(w, dim = 0)
        # beta = beta.expand((z.shape[0],) + beta.shape)
        # (N, M, 1)
        embed = torch.mm(beta.T, z)
        # 返回最终经过语义处理的每个节点的embed
        return embed

class NN(nn.Module):
    def __init__(self, edge_event, event_feature, in_channels, out_channels, user_len, business_len, num_embed, w_out):
        super(NN, self).__init__()

        # PoI邻接矩阵， v_u_matrix
        self.edge_event = edge_event

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.user_len = user_len
        self.business_len = business_len
        self.num_user = user_len
        self.num_business = business_len
        self.num_feature_embed = 20
        self.num_embed = num_embed
        self.w_out = w_out
        # 初始化embedding
        #self.event_embed = nn.Embedding(self.num_business, self.num_user)
        # 初始化用户事件交互的embedding

        self.v_u_stars_embed = nn.Embedding((int(edge_event[0].max() + 1)), num_embed)
        self.v_u_date_embed = nn.Embedding((int(edge_event[1].max() + 1)), num_embed)
        self.v_u_review_count_embed = nn.Embedding((int(edge_event[2].max() + 1)), num_embed)

        self.conv_event = GTConv(4, 1, num_embed, self.edge_event.shape[1], config.neighbor_num)

        '''
        self.gcn1_user = GCNConv(num_embed, config.n_hidden)
        self.gcn2_user = GCNConv(config.n_hidden, config.n_out)
        self.gcn1_business = GCNConv(num_embed, config.n_hidden)
        self.gcn2_business = GCNConv(config.n_hidden, config.n_out)
        '''
        self.feature_linear_layer = nn.Linear(event_feature.shape[1],self.num_feature_embed)
        self.combined_linear_layer = nn.Linear(self.num_feature_embed + num_embed, num_embed)

        self.gat1_preference = GATConv(config.n_emd, config.n_hidden, heads = config.heads)
        self.gat2_preference = GATConv(config.n_hidden * config.heads, config.n_out)
        self.gat1_spatial = GATConv(config.n_emd, config.n_hidden, heads=config.heads)
        self.gat2_spatial = GATConv(config.n_hidden * config.heads, config.n_out)

        self.gru = nn.GRU(input_size = 50, hidden_size = config.n_emd, num_layers = 2, batch_first = False)

        self.oneGraphAttention = OneGraphAttention(num_embed)


        self.dense1 = nn.Linear(config.n_out * 2, config.dense_hidden1)
        self.dense2 = nn.Linear(config.dense_hidden1, config.dense_hidden2)
        self.out = nn.Linear(config.dense_hidden2, config.dense_out)
        #self.dense3 = nn.Linear(config.dense_hidden2, config.dense_hidden3)
        #self.out = nn.Linear(config.dense_hidden3, config.dense_out)


    def compute_embedding(self, words_embed, edge_event, v_v_dict):
        edge_event = edge_event.clone().detach()

        v_u_stars_embed = torch.reshape(self.v_u_stars_embed(torch.reshape(edge_event[0].long(), (1, -1))),
                                      (1, edge_event.shape[1], edge_event.shape[2], self.num_embed))
        v_u_data_embed = torch.reshape(self.v_u_date_embed(torch.reshape(edge_event[1].long(), (1, -1))),
                                       (1, edge_event.shape[1], edge_event.shape[2], self.num_embed))
        v_u_review_count_embed = torch.reshape(self.v_u_review_count_embed(torch.reshape(edge_event[2].long(), (1, -1))),
                                               (1, edge_event.shape[1], edge_event.shape[2], self.num_embed))
        # v_category_embed = self.v_category_embed(torch.reshape(event_feature, (1, -1)))

        v_u_words_embed = torch.zeros((edge_event.shape[1], edge_event.shape[2], words_embed.shape[1]))
        l1 = 0
        for i, k in enumerate(v_v_dict):
            l2 = 0
            for j in range(len(v_v_dict[k])):
                v_u_words_embed[k][j] = words_embed[j + l1]
                l2 += 1
            l1 += l2

        v_u_words_embed = v_u_words_embed.unsqueeze(0)
        v_u_words_embed = v_u_words_embed.to(device)
        #print(v_u_uid_embed.device, v_u_data_embed.device, v_u_review_count_embed.device, v_u_words_embed.device)

        event_user_embed = torch.cat((v_u_stars_embed, v_u_data_embed, v_u_review_count_embed, v_u_words_embed), dim = 0)

        return event_user_embed

    def forward(self, words_embed, v_u_matrix, event_feature, geo_v_v2, ug_v_v2, v_v_dict):
        # 获得语义级别的embed
        words_embed = words_embed.permute(1, 0, 2)
        out, (h, c) = self.gru(words_embed)
        words_embed = h

        event_user_embed = self.compute_embedding(words_embed, v_u_matrix, v_v_dict)
        event_user_embed = self.conv_event(event_user_embed)

        event_embed = torch.zeros((self.business_len, self.num_embed))
        event_embed = event_embed.to(device)

        # 初始化PoI embedding
        for i in range(event_user_embed.shape[0]):
            event_embed[i] = self.oneGraphAttention(event_user_embed[i])

        event_feature = event_feature.clone().detach().float()
        event_feature = event_feature.to(device)
        event_feature = self.feature_linear_layer(event_feature)
        event_embed = torch.cat((event_embed, event_feature), dim=1)
        # GAT
        event_embed = self.combined_linear_layer(event_embed)

        ug_v_v2 = ug_v_v2.clone().detach()
        x_1 = self.gat1_preference(event_embed, ug_v_v2.long())
        x_1 = F.relu(x_1)
        x_1 = F.dropout(x_1, p = 0.3)
        x_1 = self.gat2_preference(x_1, ug_v_v2.long())

        geo_v_v2 = geo_v_v2.clone().detach()
        x_2 = self.gat1_spatial(event_embed, geo_v_v2.long())
        x_2 = F.relu(x_2)
        x_2 = F.dropout(x_2, p = 0.3)
        x_2 = self.gat2_spatial(x_2, geo_v_v2.long())

        x = torch.cat((x_1, x_2), dim = 1)
        x = F.relu(self.dense1(x))
        x = F.dropout(x)
        x = F.relu(self.dense2(x))
        x = F.dropout(x, 0.2)
        #x = F.relu(self.dense3(x))
        x = self.out(x)

        return x
    def features(self, words_embed, v_u_matrix, event_feature, geo_v_v2, ug_v_v2, v_v_dict):
        # 获得语义级别的embed
        words_embed = words_embed.permute(1, 0, 2)
        out, (h, c) = self.gru(words_embed)
        words_embed = h

        event_user_embed = self.compute_embedding(words_embed, v_u_matrix, v_v_dict)
        event_user_embed = self.conv_event(event_user_embed)

        event_embed = torch.zeros((self.business_len, self.num_embed))
        event_embed = event_embed.to(device)

        # 初始化PoI embedding
        for i in range(event_user_embed.shape[0]):
            event_embed[i] = self.oneGraphAttention(event_user_embed[i])

        event_feature = event_feature.clone().detach().float()
        event_feature = event_feature.to(device)
        event_feature = self.feature_linear_layer(event_feature)
        event_embed = torch.cat((event_embed, event_feature), dim=1)
        # GAT
        event_embed = self.combined_linear_layer(event_embed)

        ug_v_v2 = ug_v_v2.clone().detach()
        x_1 = self.gat1_preference(event_embed, ug_v_v2.long())
        x_1 = F.relu(x_1)
        x_1 = F.dropout(x_1, p = 0.3)
        x_1 = self.gat2_preference(x_1, ug_v_v2.long())

        geo_v_v2 = geo_v_v2.clone().detach()
        x_2 = self.gat1_spatial(event_embed, geo_v_v2.long())
        x_2 = F.relu(x_2)
        x_2 = F.dropout(x_2, p = 0.3)
        x_2 = self.gat2_spatial(x_2, geo_v_v2.long())

        x = torch.cat((x_1, x_2), dim = 1)
        x = F.relu(self.dense1(x))
        x = F.dropout(x)
        x = F.relu(self.dense2(x))
        x = F.dropout(x, 0.2)
        #x = F.relu(self.dense3(x))

        return x

class GTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_dim, num_user, num_item):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_dim = num_dim
        self.num_user = num_user
        self.num_item = num_item

        self.conv_user = GTConv(in_channels, out_channels, num_dim, num_user, num_item)
        self.conv_event = GTConv(in_channels, out_channels, num_dim, num_item, num_user)

    def forward(self, A_user, A_event):
        A_user = self.conv_user(A_user)
        A_event = self.conv_event(A_event)

        return A_user, A_event

class GTConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_dim, num_user, num_item):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter((torch.Tensor(out_channels, in_channels, num_user, num_item, num_dim)))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = A * F.softmax(self.weight, dim = 1)
        A = torch.sum(torch.squeeze(A, dim = 0), dim = 0)

        return A
