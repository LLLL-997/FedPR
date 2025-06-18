from datetime import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES']=  '1'
import torch.optim as optim
import torch.nn as nn
from numpy import *
import numpy as np
import random
import torch
import copy
import argparse
import json

import config
from config import state_str
from model import NN
from utils_federated import evaluate_network,get_dataloader, init_networks, update_model_after_collaboration

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
method_name = "FedDQI"
participants_num = config.participants_num
communication_epochs = config.communication_epochs
train_batch_size = config.batch_size
test_batch_size = config.batch_size


dataset_path = config.dataset_path
model_path = config.model_save_path
clients_list = config.clients_list

private_dataset_list = config.private_dataset_list
# private_dataset_len_list = config.private_dataset_len_list
private_training_epochs = config.epochs
private_training_classes = config.private_training_classes
out_channels = config.out_channels

public_dataset_name = config.public_dataset_name
#public_dataset_len = config.public_dataset_len
public_dataset_path = config.dataset_path + config.public_dataset_name + ".json"
public_training_epochs = config.public_training_epochs

participant_params = {
    'loss_function' : 'CrossEntropy',
    'optimizer_name' : 'Adam',
    'learning_rate' : 1e-3
}
def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type = int, default = 1)
    parser.add_argument("--wfed", type = float, default =0.015)
    parser.add_argument("--wkd", type = float, default =0.5)
    return parser

def set_seed(seed):
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

def off_diagonal(x):
    n,m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

if __name__ == "__main__":
    seed = config.Seed
    args = set_parser().parse_args()
    temperature = args.temperature
    wfed = args.wfed
    wkd = args.wkd
    print("temperature = {}".format(temperature))
    print("wfed = {}".format(wfed))
    print("wkd = {}".format(wkd))

    set_seed(seed)
    print(method_name)
    # print("生成随机种子")

    device = torch.device("cuda")
    device_ids = [0, 1]

    print("加载数据")
    f_geo_v_v2 = dataset_path + "yelp_geo_v_v_" + state_str + ".npy"
    geo_v_v2 = np.load(f_geo_v_v2)
    geo_v_v2 = torch.tensor(geo_v_v2).to(device)
    f_v_feature = config.dataset_path + "yelp_v_feature.npy"
    v_feature = np.load(f_v_feature)
    v_feature = torch.tensor(v_feature).to(device)
    word_embed_list = []
    v_u_matrix_list = []
    geo_v_v2_list = []
    ug_v_v2_list = []
    v_v_dict_list = []
    for i in range(participants_num):
        net_name = private_dataset_list[i]
        f_word_embed = dataset_path + "yelp_word_embed_" + net_name + ".npy"
        word_embed = np.load(f_word_embed)
        word_embed = torch.tensor(word_embed,dtype = torch.float).to(device)
        word_embed_list.append(word_embed)
        f_v_u_matrix = config.dataset_path + "yelp_v_u_matrix_" + net_name + ".npy"
        v_u_matrix = np.load(f_v_u_matrix)
        v_u_matrix = torch.tensor(v_u_matrix).to(device)
        v_u_matrix_list.append(v_u_matrix)
        f_ug_v_v2_path = dataset_path + "yelp_ug_v_v_" + net_name + ".npy"
        ug_v_v2 = np.load(f_ug_v_v2_path)
        ug_v_v2 = torch.tensor(ug_v_v2).to(device)
        ug_v_v2_list.append(ug_v_v2)
        f_v_v_dict_path = dataset_path + "yelp_v_v_dict_" + net_name + ".json"
        with open(f_v_v_dict_path, "r") as f_v_v_dict:
            v_v_dict = json.load(f_v_v_dict)
        v_v_dict = {int(k): v for k, v in v_v_dict.items()}
        v_v_dict = {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in v_v_dict.items()}
        v_v_dict_list.append(v_v_dict)

    print("加载参与者模型")
    net_list = init_networks(participants_num = participants_num, nets_name_list = clients_list)
    for i in range(participants_num):
        network = net_list[i]
        network = network.to(device)
        netname = clients_list[i]
        private_dataset_name = private_dataset_list[i]
        private_model_path = model_path + private_dataset_name + "_original_model_.pth"
        network.load_state_dict(torch.load(private_model_path))

    print("加载预训练模型")
    frozen_net_list = init_networks(participants_num = participants_num, nets_name_list = clients_list)
    for i in range(participants_num):
        network = frozen_net_list[i]
        network = network.to(device)
        netname = clients_list[i]
        private_dataset_name = private_dataset_list[i]
        private_model_path = model_path + private_dataset_name + "_original_model_.pth"
        network.load_state_dict(torch.load(private_model_path))
        #network.cpu()
        torch.cuda.empty_cache()

    print("加载进阶模型")
    collaboration_net_list = init_networks(participants_num = participants_num, nets_name_list = clients_list)
    for i in range(participants_num):
        network = collaboration_net_list[i]
        network = network.to(device)
        netname = clients_list[i]
        private_dataset_name = private_dataset_list[i]
        private_model_path = model_path + private_dataset_name + "_original_model_.pth"
        network.load_state_dict(torch.load(private_model_path))
        #network.cpu()
        torch.cuda.empty_cache()

    # 加载公共数据
    public_train_data_path = dataset_path + "yelp_public_train_mask.npy"
    public_train_data = np.load(public_train_data_path).tolist()
    # 加载私有数据
    private_train_data_list = []
    private_test_data_list = []
    for i in range(participants_num):
        private_dataset_name = private_dataset_list[i]
        private_dataset_path = dataset_path + "yelp_mask_dict_" + private_dataset_name + ".json"
        train_index, test_index = get_dataloader(private_dataset_name, private_dataset_path)
        private_train_data_list.append(train_index)
        private_test_data_list.append(test_index)

    col_loss_list = []
    local_loss_list = []
    acc_list = []
    fed_lr = config.lr
    for epoch in range(communication_epochs):
        print("=====第{}次联邦训练=====".format(epoch + 1))
        print("评估模型")
        acc_epoch_list = []
        for participant_i in range(participants_num):
            netname = clients_list[participant_i]
            private_dataset_name = private_dataset_list[participant_i]
            private_dataset_path = dataset_path + private_dataset_name + ".json"
            print(netname + "_" + private_dataset_name + "_" + private_dataset_path)
            # 加载数据
            test_index = private_test_data_list[participant_i]
            network = net_list[participant_i]
            network = network.to(device)
            eval_result = evaluate_network(network = network, net_name = netname, dataloader = test_index)
            acc_epoch_list.append(eval_result)
            print(netname + " : " + str(eval_result))

        acc_list.append(acc_epoch_list)
        public_train_data = torch.tensor(public_train_data, dtype=torch.long).to(device)
        print("Federated Training")
        all_col_loss_list = [[], [], [], []]
        for _ in range(public_training_epochs):
            linear_output_list = []
            linear_output_target_list = []
            linear_output_collaboration_list = []
            col_loss_batch_list = []
            shuffle_index = torch.randperm(len(public_train_data))
            public_train_mask = public_train_data[shuffle_index]
            # 计算Linear output
            for participant_i in range(participants_num):
                network = net_list[participant_i]
                network = network.to(device)
                network.train()
                # 训练数据加载到GPU
                output = network(word_embed_list[participant_i], v_u_matrix_list[participant_i], v_feature, geo_v_v2, ug_v_v2_list[participant_i], v_v_dict_list[participant_i])
                linear_output = output[public_train_mask]
                linear_output_target_list.append(linear_output.clone().detach())
                linear_output_list.append(linear_output)

            # 计算合作Linear output
            for collaboration_participant_i in range(participants_num):
                collaboration_network = collaboration_net_list[collaboration_participant_i]
                collaboration_network = collaboration_network.to(device)
                collaboration_network.eval()
                with torch.no_grad():
                    output = collaboration_network(word_embed_list[collaboration_participant_i], v_u_matrix_list[collaboration_participant_i], v_feature, geo_v_v2, ug_v_v2_list[collaboration_participant_i], v_v_dict_list[collaboration_participant_i])
                    collaboration_linear_output = output[public_train_mask]
                    linear_output_collaboration_list.append(collaboration_linear_output)
                #collaboration_network.cpu()
                torch.cuda.empty_cache()

            # 根据col loss更新参与者模型
            for participant_i in range(participants_num):
                # 计算和合作方的loss
                network = net_list[participant_i]
                network = network.to(device)
                network.train()
                optimizer = optim.Adam(network.parameters(), fed_lr)
                optimizer.zero_grad()
                linear_output_target_avg_list = []
                for i in range(participants_num):
                    if i != participant_i:
                        linear_output_target_avg_list.append(linear_output_target_list[i])
                    if i == participant_i:
                        linear_output_target_avg_list.append(linear_output_collaboration_list[i])

                linear_output_target_avg = torch.mean(torch.stack(linear_output_target_avg_list), 0)
                linear_output = linear_output_list[participant_i]
                z_1_bn = (linear_output - linear_output.mean(0)) / linear_output.std(0)
                z_2_bn = (linear_output_target_avg - linear_output_target_avg.mean(0)) / linear_output_target_avg.std(0)
                c = z_1_bn.T @ z_2_bn
                c.div_(len(public_train_data))

                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).add_(1).pow_(2).sum()
                col_loss = on_diag + wfed * off_diag
                col_loss_batch_list.append(col_loss.item())
                all_col_loss_list[participant_i].append(col_loss.item())
                col_loss.backward()
                optimizer.step()
            col_loss_list.append(col_loss_batch_list)
        for participant_i in range(participants_num):
            avg_fed_loss = sum(all_col_loss_list[participant_i])/len(all_col_loss_list[participant_i])
            print(clients_list[participant_i])
            print(avg_fed_loss)

        print("Knowledge Distillation")
        # 更新隐私数据的参与者模型
        local_loss_batch_list = []
        for participant_i in range(participants_num):
            network = net_list[participant_i]
            network = network.to(device)
            network.train()

            forzen_network = frozen_net_list[participant_i]
            frozen_network = forzen_network.to(device)
            frozen_network.eval()

            collaboration_network = collaboration_net_list[participant_i]
            collaboration_network = collaboration_network.to(device)
            collaboration_network.eval()

            private_dataset_name = private_dataset_list[participant_i]
            private_dataset_path = dataset_path + private_dataset_name + ".json"

            private_epoch = 10


            network, private_loss_batch_list = update_model_after_collaboration(net_name = clients_list[participant_i], network = network, frozen_network = frozen_network, collaboration_network = collaboration_network,
                                                                                temperature = temperature, wkd = wkd, private_epoch = private_epoch, train_data_list = private_train_data_list[participant_i], lr = fed_lr)
            mean_private_loss_batch = mean(private_loss_batch_list)
            # print(clients_list[participant_i])
            the_loss = sum(all_col_loss_list[participant_i])/len(all_col_loss_list[participant_i]) + mean_private_loss_batch
            print(the_loss)
            local_loss_batch_list.append(mean_private_loss_batch)
            #frozen_network.cpu()
            #collaboration_network.cpu()
            torch.cuda.empty_cache()
        local_loss_list.append(local_loss_batch_list)

        for j in range(participants_num):
            collaboration_net_list[j] = copy.deepcopy(net_list[j])
        fed_lr = fed_lr * (1 - epoch / communication_epochs * 0.9)

        if epoch == communication_epochs - 1:
            acc_epoch_list = []
            print("最终评估模型表现")
            for participant_i in range(participants_num):
                netname = private_dataset_list[participant_i]
                private_dataset_name = private_dataset_list[participant_i]
                private_dataset_name = dataset_path + private_dataset_name + ".json"
                test_data = private_test_data_list[participant_i]
                network = net_list[participant_i]
                network = network.to(device)
                acc_epoch_list.append(evaluate_network(net_name = netname, network = network, dataloader = test_data))
            acc_list.append(acc_epoch_list)

        if epoch == communication_epochs - 1:
            for participant_i in range(participants_num):
                netname = clients_list[participant_i]
                private_dataset_name = private_dataset_list[participant_i]
                network = net_list[participant_i]
                network = network.to(device)
                torch.save(network.state_dict(), model_path + private_dataset_name + "_federated_model_.pth")
            np.save(dataset_path + method_name +"_acc_list.npy", acc_list)
