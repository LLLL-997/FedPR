from Cython.Compiler.Errors import echo_file
import os

from model import NN
import numpy as np
import pandas as pd
import config
os.environ['CUDA_VISIBLE_DEVICES']=  '1'
import torch
from torch import nn
import torch.nn.functional as F
import json

from config import dataset_path, state_str, platform_str
device = torch.device("cuda")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]
participant_params = {
    'loss_function' : 'CrossEntropy',
    'optimizer_name' : 'Adam',
    'learning_rate' : 1e-3
}

def init_networks(participants_num, nets_name_list):
    nets_list = {net_i: None for net_i in range(participants_num)}
    for net_i in range(participants_num):
        net_name = nets_name_list[net_i]
        f_ground_truth_v_u_matrix = dataset_path + "yelp_v_u_matrix_GroundTruth.npy"
        v_u_matrix = np.load(f_ground_truth_v_u_matrix)
        v_u_matrix = torch.tensor(v_u_matrix).to(device)
        f_v_feature = config.dataset_path + "yelp_v_feature.npy"
        v_feature = np.load(f_v_feature)
        v_feature = torch.tensor(v_feature).to(device)
        f_params_dict = config.dataset_path + "params_dict_" + net_name + ".json"
        with open(f_params_dict, "r") as f_params:
            params_dict = json.load(f_params)
        in_channels = params_dict["in_channels"]
        out_channels = params_dict["out_channels"]
        user_len = params_dict["user_len"]
        business_len = params_dict["business_len"]
        num_embed = params_dict["num_embed"]
        w_out = params_dict["w_out"]
        net = NN(v_u_matrix, v_feature, in_channels, out_channels, user_len, business_len, num_embed, w_out)
        nets_list[net_i] = net
        del v_feature, v_u_matrix
        torch.cuda.empty_cache()
    return nets_list


def evaluate_network(network, net_name, dataloader):
    f_ground_truth_results_path = dataset_path + "yelp_ground_truth_results_" + state_str + ".json"
    gt_results = pd.read_json(f_ground_truth_results_path, orient="records", lines=True)
    f_word_embed = dataset_path + "yelp_word_embed_" + net_name + ".npy"
    word_embed = np.load(f_word_embed)
    word_embed = torch.tensor(word_embed, dtype = torch.float32).to(device)
    f_v_u_matrix = config.dataset_path + "yelp_v_u_matrix_" + net_name + ".npy"
    v_u_matrix = np.load(f_v_u_matrix)
    v_u_matrix = torch.tensor(v_u_matrix).to(device)
    f_v_feature = config.dataset_path + "yelp_v_feature.npy"
    v_feature = np.load(f_v_feature)
    v_feature = torch.tensor(v_feature).to(device)
    f_geo_v_v2 = dataset_path + "yelp_geo_v_v_" + state_str + ".npy"
    geo_v_v2 = np.load(f_geo_v_v2)
    geo_v_v2 = torch.tensor(geo_v_v2).to(device)
    f_ug_v_v2_path = dataset_path + "yelp_ug_v_v_" + net_name + ".npy"
    ug_v_v2 = np.load(f_ug_v_v2_path)
    ug_v_v2 = torch.tensor(ug_v_v2).to(device)
    f_v_v_dict_path = dataset_path + "yelp_v_v_dict_" + net_name + ".json"
    with open(f_v_v_dict_path, "r") as f_v_v_dict:
        v_v_dict = json.load(f_v_v_dict)
    v_v_dict = {int(k): v for k, v in v_v_dict.items()}
    v_v_dict = {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in v_v_dict.items()}
    network.eval()
    with torch.no_grad():
        cnt = np.zeros((3,), dtype = int)
        acc_cnt = np.zeros((3,), dtype = int)
        output = network(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)
        '''
        for i in dataloader:
            result = torch.argmax(output[i]).item()
            truth_result = gt_results[gt_results["business_id"] == i]["item_class"].item()
            if truth_result == result:
                cnt[gt_results[gt_results["business_id"] == i]["category"]] += 1
                acc_cnt[gt_results[gt_results["business_id"] == i]["category"]] += 1
            else:
                cnt[gt_results[gt_results["business_id"] == i]["category"]] += 1
        '''
        business_ids = dataloader
        predicted_classes = torch.argmax(output[dataloader], dim=1).cpu().numpy()

        # Create a dataframe from `business_id` and predicted classes
        predictions_df = pd.DataFrame({
            'business_id': business_ids,
            'predicted_class': predicted_classes
        })

        # Merge ground truth with predictions
        merged_df = pd.merge(predictions_df, gt_results[['business_id', 'item_class', 'category']], on='business_id',
                             how='left')

        # Loop through the merged dataframe to calculate accuracy
        for idx, row in merged_df.iterrows():
            result = row['predicted_class']
            truth_result = row['item_class']
            category = row['category']
            cnt[category] += 1
            if truth_result == result:
                acc_cnt[category] += 1

        print(cnt)
        print(acc_cnt)
        print(acc_cnt / cnt)
        r = acc_cnt.sum() / cnt.sum()
        print(r)
    del word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict
    torch.cuda.empty_cache()
    return acc_cnt / cnt


def get_dataloader(dataset, datapath):
    with open(datapath, "r") as f_data:
        mask_dict = json.load(f_data)
    train_index = mask_dict["train_mask"]
    test_index = mask_dict["test_mask"]
    return train_index, test_index

def update_model_after_collaboration(net_name, network, frozen_network, collaboration_network, temperature, wkd, private_epoch, train_data_list, lr):
    f_target_path = dataset_path + "yelp_target_" + state_str + "_" + net_name + ".json"
    gt_results = pd.read_json(f_target_path, orient="records", lines=True)
    target = gt_results["item_class"].to_numpy()
    target = torch.tensor(target, dtype = torch.long).to(device)
    f_word_embed = dataset_path + "yelp_word_embed_" + net_name + ".npy"
    word_embed = np.load(f_word_embed)
    word_embed = torch.tensor(word_embed, dtype=torch.float32).to(device)
    f_v_u_matrix = config.dataset_path + "yelp_v_u_matrix_" + net_name + ".npy"
    v_u_matrix = np.load(f_v_u_matrix)
    v_u_matrix = torch.tensor(v_u_matrix, dtype = torch.long).to(device)
    f_v_feature = config.dataset_path + "yelp_v_feature.npy"
    v_feature = np.load(f_v_feature)
    v_feature = torch.tensor(v_feature, dtype = torch.long).to(device)
    f_geo_v_v2 = dataset_path + "yelp_geo_v_v_" + state_str + ".npy"
    geo_v_v2 = np.load(f_geo_v_v2)
    geo_v_v2 = torch.tensor(geo_v_v2, dtype = torch.long).to(device)
    f_ug_v_v2_path = dataset_path + "yelp_ug_v_v_" + net_name + ".npy"
    ug_v_v2 = np.load(f_ug_v_v2_path)
    ug_v_v2 = torch.tensor(ug_v_v2, dtype = torch.long).to(device)
    f_v_v_dict_path = dataset_path + "yelp_v_v_dict_" + net_name + ".json"
    with open(f_v_v_dict_path, "r") as f_v_v_dict:
        v_v_dict = json.load(f_v_v_dict)
    v_v_dict = {int(k): v for k, v in v_v_dict.items()}
    v_v_dict = {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in v_v_dict.items()}
    train_data_list = torch.tensor(train_data_list, dtype=torch.long).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    criterion_hard = nn.CrossEntropyLoss()
    participant_local_loss_batch_list = []
    for epoch in range(private_epoch):
        shuffle_index = torch.randperm(len(train_data_list))
        train_mask = train_data_list[shuffle_index]
        output = network(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)
        output = output[train_mask]
        logsoft_output = F.log_softmax(output/temperature, dim=1)
        with torch.no_grad():
            frozen_output = frozen_network(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)
            frozen_soft_labels = F.softmax(frozen_output[train_mask]/temperature, dim=1)
        # frozen_loss_soft = criterion(logsoft_output, frozen_soft_labels)

        KL_per_sample = F.kl_div(logsoft_output, frozen_soft_labels, reduction = 'none').sum(dim = 1)
        threshold = 0.01
        mask = KL_per_sample > threshold

        if mask.sum() > 0:
            #print(mask.sum())
            selected_logsoft_output = logsoft_output[mask]
            selected_soft_labels = frozen_soft_labels[mask]
            frozen_loss_soft = criterion(selected_logsoft_output, selected_soft_labels)
        else:
            frozen_loss_soft = torch.tensor(0.0, device = device)


        loss_hard = criterion_hard(output, target[train_mask])
        #loss = loss_hard + frozen_loss_soft + collaboration_loss_soft
        loss = wkd * loss_hard + frozen_loss_soft
        optimizer.zero_grad()
        participant_local_loss_batch_list.append(loss.item())
        loss.backward()
        optimizer.step()
        #print("Private epoch : {}".format(epoch))
        #print(loss.item())
    del word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict, target
    torch.cuda.empty_cache()
    print(net_name)
    print(sum(participant_local_loss_batch_list)/len(participant_local_loss_batch_list))
    return network, participant_local_loss_batch_list

def update_model_via_private_data(net_name, network, private_epoch, train_data_list):
    f_target_path = dataset_path + "yelp_target_" + state_str + "_" + net_name + ".json"
    gt_results = pd.read_json(f_target_path, orient="records", lines=True)
    target = gt_results["item_class"].to_numpy()
    target = torch.tensor(target, dtype = torch.long).to(device)
    f_word_embed = dataset_path + "yelp_word_embed_" + net_name + ".npy"
    word_embed = np.load(f_word_embed)
    word_embed = torch.tensor(word_embed, dtype=torch.float32).to(device)
    f_v_u_matrix = config.dataset_path + "yelp_v_u_matrix_" + net_name + ".npy"
    v_u_matrix = np.load(f_v_u_matrix)
    v_u_matrix = torch.tensor(v_u_matrix, dtype = torch.long).to(device)
    f_v_feature = config.dataset_path + "yelp_v_feature.npy"
    v_feature = np.load(f_v_feature)
    v_feature = torch.tensor(v_feature, dtype = torch.long).to(device)
    f_geo_v_v2 = dataset_path + "yelp_geo_v_v_" + state_str + ".npy"
    geo_v_v2 = np.load(f_geo_v_v2)
    geo_v_v2 = torch.tensor(geo_v_v2, dtype = torch.long).to(device)
    f_ug_v_v2_path = dataset_path + "yelp_ug_v_v_" + net_name + ".npy"
    ug_v_v2 = np.load(f_ug_v_v2_path)
    ug_v_v2 = torch.tensor(ug_v_v2, dtype = torch.long).to(device)
    f_v_v_dict_path = dataset_path + "yelp_v_v_dict_" + net_name + ".json"
    with open(f_v_v_dict_path, "r") as f_v_v_dict:
        v_v_dict = json.load(f_v_v_dict)
    v_v_dict = {int(k): v for k, v in v_v_dict.items()}
    v_v_dict = {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in v_v_dict.items()}
    train_data_list = torch.tensor(train_data_list, dtype=torch.long).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    criterion_hard = nn.CrossEntropyLoss()
    participant_local_loss_batch_list = []
    for epoch in range(private_epoch):
        shuffle_index = torch.randperm(len(train_data_list))
        train_mask = train_data_list[shuffle_index]

        output = network(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)
        output = output[train_mask]
        loss = criterion(output, target[train_mask])
        optimizer.zero_grad()
        participant_local_loss_batch_list.append(loss.item())
        loss.backward()
        optimizer.step()
        print("Private epoch : {}".format(epoch))
        print(loss.item())
    del word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict, target
    torch.cuda.empty_cache()
    return network, participant_local_loss_batch_list

def update_model_after_collaboration_ablation(net_name, network, frozen_network, temperature, private_epoch, train_data_list):
    f_target_path = dataset_path + "yelp_target_" + state_str + "_" + platform_str + ".json"
    gt_results = pd.read_json(f_target_path, orient="records", lines=True)
    target = gt_results["item_class"].to_numpy()
    target = torch.tensor(target, dtype = torch.long).to(device)
    f_word_embed = dataset_path + "yelp_word_embed_" + net_name + ".npy"
    word_embed = np.load(f_word_embed)
    word_embed = torch.tensor(word_embed, dtype=torch.float32).to(device)
    f_v_u_matrix = config.dataset_path + "yelp_v_u_matrix_" + net_name + ".npy"
    v_u_matrix = np.load(f_v_u_matrix)
    v_u_matrix = torch.tensor(v_u_matrix, dtype = torch.long).to(device)
    f_v_feature = config.dataset_path + "yelp_v_feature.npy"
    v_feature = np.load(f_v_feature)
    v_feature = torch.tensor(v_feature, dtype = torch.long).to(device)
    f_geo_v_v2 = dataset_path + "yelp_geo_v_v_" + state_str + ".npy"
    geo_v_v2 = np.load(f_geo_v_v2)
    geo_v_v2 = torch.tensor(geo_v_v2, dtype = torch.long).to(device)
    f_ug_v_v2_path = dataset_path + "yelp_ug_v_v_" + net_name + ".npy"
    ug_v_v2 = np.load(f_ug_v_v2_path)
    ug_v_v2 = torch.tensor(ug_v_v2, dtype = torch.long).to(device)
    f_v_v_dict_path = dataset_path + "yelp_v_v_dict_" + net_name + ".json"
    with open(f_v_v_dict_path, "r") as f_v_v_dict:
        v_v_dict = json.load(f_v_v_dict)
    v_v_dict = {int(k): v for k, v in v_v_dict.items()}
    v_v_dict = {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in v_v_dict.items()}
    train_data_list = torch.tensor(train_data_list, dtype=torch.long).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    criterion_hard = nn.CrossEntropyLoss()
    participant_local_loss_batch_list = []
    for epoch in range(private_epoch):
        shuffle_index = torch.randperm(len(train_data_list))
        train_mask = train_data_list[shuffle_index]

        output = network(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)
        output = output[train_mask]
        logsoft_output = F.log_softmax(output/temperature, dim=1)
        with torch.no_grad():
            frozen_output = frozen_network(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)
            frozen_soft_labels = F.softmax(frozen_output[train_mask]/temperature, dim=1)
        frozen_loss_soft = criterion(logsoft_output, frozen_soft_labels)

        '''
        with torch.no_grad():
            collaboration_network_output = collaboration_network(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)
            collaboration_soft_labels = F.softmax(collaboration_network_output[train_mask]/temperature, dim=1)
        collaboration_loss_soft = criterion(logsoft_output, collaboration_soft_labels)
        '''


        loss_hard = criterion_hard(output, target[train_mask])
        #loss = loss_hard + frozen_loss_soft + collaboration_loss_soft
        loss = 0.5 * loss_hard + frozen_loss_soft
        optimizer.zero_grad()
        participant_local_loss_batch_list.append(loss.item())
        loss.backward()
        optimizer.step()
        print("Private epoch : {}".format(epoch))
        print(loss.item())
    del word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict, target
    torch.cuda.empty_cache()
    return network, participant_local_loss_batch_list

