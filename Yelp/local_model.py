import os
os.environ['CUDA_VISIBLE_DEVICES']=  '1'
from torch import nn
import copy
import random
import torch.nn.functional as F
import pandas as pd
import json
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import resample
from collections import Counter

import config
from config import platform_str, model_save_path, dataset_path, state_str, batch_size
from model import NN
from dataloader import getData, trainData
from utils import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

def balance_sampling(train_mask, target, method = "oversample"):
    unique_classes, class_counts = torch.unique(target[train_mask], return_counts=True)
    max_count = class_counts.max().item()
    min_count = class_counts.min().item()

    balanced_mask = []

    for cls in unique_classes:
        cls_index = [idx for idx in train_mask if target[idx].item() == cls.item()]

        if method == "oversample":
            resample_cls_index = resample(cls_index, replace = True, n_samples = max_count, random_state = 42)
        elif method == "undersample":
            resample_cls_index = resample(cls_index, replace = False, n_samples = min_count, random_state = 42)
        else:
            raise ValueError("Invalid Method.")
        balanced_mask.extend(resample_cls_index)
    return balanced_mask

def train_groundTruth_model(word_embed, v_u_matrix, v_feature, function_weight, geo_v_v2, ug_v_v2, v_v_dict, params_dict, train_mask, test_mask, target, all_business, n_split = 5):
    in_channels = params_dict['in_channels']
    out_channels = params_dict['out_channels']
    user_len = params_dict['user_len']
    business_len = params_dict['business_len']
    num_embed = params_dict['num_embed']
    w_out = params_dict['w_out']
    # 五折交叉训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = target.cpu()
    fold_index = get_fold_indexed(train_mask, target[train_mask], n_split = 5)
    total_best_epoch = []
    for fold, index in enumerate(fold_index):
        print("========第{}折========".format(fold + 1))
        fold_train_mask = index["train"]
        fold_train_mask = balance_sampling(fold_train_mask, target, method = "oversample")
        fold_val_mask = index["val"]
        best_acc = 0
        best_f1 = 0
        _f1_score = 0
        best_epoch = 0
        patience = 15
        no_improve = 0
        model = NN(v_u_matrix, v_feature, in_channels, out_channels, user_len, business_len, num_embed, w_out)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        model = model.to(device)
        target = target.to(device)
        criterion = nn.CrossEntropyLoss(reduction = 'mean')
        for epoch in range(epochs):
            model.train()
            shuffle_index = torch.randperm(len(fold_train_mask))
            fold_train_mask = torch.tensor(fold_train_mask)
            fold_train_mask = fold_train_mask[shuffle_index]
            batch_num = len(fold_train_mask) // batch_size + 1
            for batch in range(batch_num):
                batch_mask = fold_train_mask[batch * batch_size:(batch + 1) * batch_size]
                predict = model(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)
                loss = criterion(predict[batch_mask], target[batch_mask])
                #weighted_loss = function_weight[batch_mask] * base_loss

                #loss = weighted_loss.mean()

                print("=====第{}.{}次迭代=====".format(epoch + 1, batch))
                print("CEL Loss:{}".format(loss.item()))
                print("=======******=======")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                output = model(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)
                preds = torch.argmax(output, dim=1)

                acc = accuracy_score(target[fold_val_mask].cpu().numpy(), preds[fold_val_mask].cpu().numpy())
                f1 = f1_score(target[fold_val_mask].cpu().numpy(), preds[fold_val_mask].cpu().numpy(), average='macro')

                print(f"\nFold {fold + 1} Results:")
                print(f"Accuracy: {acc:.4f}")
                print(f"F1-score: {f1:.4f}")
                if f1 > best_f1:
                    best_acc = acc
                    _f1_score = f1
                    best_f1 = _f1_score
                    best_epoch = epoch
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {best_epoch}")
                    print(f"\nFold {fold + 1} Results:")
                    print(f"Best Accuracy: {best_acc:.4f}")
                    print(f"Best F1-score: {_f1_score:.4f}")
                    break
        total_best_epoch.append(best_epoch)
    optimal_epochs = int(np.mean(total_best_epoch))

    # 训练正式模型
    print("=======Final Model Training=======")
    final_model = NN(v_u_matrix, v_feature, in_channels, out_channels, user_len, business_len, num_embed, w_out)
    final_criterion = nn.CrossEntropyLoss(reduction='mean')
    final_model = final_model.to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=config.lr)
    target = target.cpu()
    train_mask = balance_sampling(train_mask, target, method = "oversample")
    target = target.to(device)
    for epoch in range(optimal_epochs):
        final_model.train()
        shuffle_index = torch.randperm(len(train_mask))
        train_mask = torch.tensor(train_mask)
        train_mask = train_mask[shuffle_index]
        batch_num = len(train_mask) // batch_size + 1
        for batch in range(batch_num):
            batch_mask = train_mask[batch * batch_size:(batch + 1) * batch_size]
            predict = final_model(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)

            loss = final_criterion(predict[batch_mask], target[batch_mask])
            #weighted_loss = function_weight[batch_mask] * base_loss

            #loss = weighted_loss.mean()

            print("=====第{}.{}次迭代=====".format(epoch + 1, batch))
            print("CEL Loss:{}".format(loss.item()))
            print("=======******=======")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    final_model.eval()
    with torch.no_grad():
        f_ground_truth_results_path = dataset_path + "yelp_ground_truth_results_" + state_str + ".json"
        output = final_model(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)
        for i in test_mask:
            result = torch.argmax(output[i])
            all_business.loc[all_business["business_id"] == i, "item_class"] = torch.argmax(output[i]).item()
            print(i, result)
    if not os.path.exists(f_ground_truth_results_path):
        all_business.to_json(f_ground_truth_results_path, orient="records", lines=True)
    final_model_path = model_save_path + platform_str + "_original_model_.pth"
    torch.save(final_model.state_dict(), final_model_path)
    loaded_model = NN(v_u_matrix, v_feature, in_channels, out_channels, user_len, business_len, num_embed, w_out)
    loaded_model.load_state_dict(torch.load(final_model_path), strict = True)
    loaded_model = loaded_model.to(device)
    loaded_model.eval()
    with torch.no_grad():
        loaded_output = loaded_model(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)
        for i in test_mask:
            result = torch.argmax(loaded_output[i])
            result_training = torch.argmax(output[i])
            if result != result_training:
                print(i, result, result_training)



def train_platform_model(word_embed, v_u_matrix, v_feature, function_weight, geo_v_v2, ug_v_v2, v_v_dict, params_dict, train_mask, test_mask, target):
    in_channels = params_dict['in_channels']
    out_channels = params_dict['out_channels']
    user_len = params_dict['user_len']
    business_len = params_dict['business_len']
    num_embed = params_dict['num_embed']
    w_out = params_dict['w_out']
    f_ground_truth_v_u_matrix = dataset_path + "yelp_v_u_matrix_GroundTruth.npy"
    g_v_u_matrix = np.load(f_ground_truth_v_u_matrix)
    final_model = NN(g_v_u_matrix, v_feature, in_channels, out_channels, user_len, business_len, num_embed, w_out)
    train_mask = balance_sampling(train_mask, target, method="oversample")
    final_model_path = model_save_path + platform_str + "_original_model_.pth"
    final_criterion = nn.CrossEntropyLoss(reduction='mean')
    # 训练正式模型
    print("=======Final Model Training=======")
    final_model = final_model.to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=config.lr)
    max_acc = 0
    patience = 20
    patience_cnt = 0
    for epoch in range(epochs):
        final_model.train()
        shuffle_index = torch.randperm(len(train_mask))
        train_mask = torch.tensor(train_mask)
        train_mask = train_mask[shuffle_index]
        batch_num = len(train_mask) // batch_size + 1
        for batch in range(batch_num):
            batch_mask = train_mask[batch * batch_size:(batch + 1) * batch_size]
            predict = final_model(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)

            loss = final_criterion(predict[batch_mask], target[batch_mask])
            #weighted_loss = function_weight[batch_mask] * base_loss

            ##loss = base_loss.mean()

            print("=====第{}.{}次迭代=====".format(epoch + 1, batch))
            print("CEL Loss:{}".format(loss.item()))
            print("=======******=======")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final_model.eval()
        with torch.no_grad():
            f_ground_truth_results_path = dataset_path + "yelp_ground_truth_results_" + state_str + ".json"
            output = final_model(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)
            gt_results = pd.read_json(f_ground_truth_results_path, orient="records", lines=True)
            cnt = np.zeros((3,), dtype=int)
            acc_cnt = np.zeros((3,), dtype=int)
            for i in test_mask:
                result = torch.argmax(output[i]).item()
                truth_result = gt_results[gt_results["business_id"] == i]["item_class"].item()
                if truth_result == result:
                    cnt[gt_results[gt_results["business_id"] == i]["category"]] += 1
                    acc_cnt[gt_results[gt_results["business_id"] == i]["category"]] += 1
                else:
                    cnt[gt_results[gt_results["business_id"] == i]["category"]] += 1
        print(cnt)
        print(acc_cnt)
        print(acc_cnt / cnt)
        target_function = config.clients_target[platform_str]
        target_function_len = len(target_function)
        score = 0
        sum_score = 0
        for i in range(target_function_len):
            score += acc_cnt[target_function[i]]
            sum_score += cnt[target_function[i]]
        r = score / sum_score
        print(acc_cnt.sum() / cnt.sum())
        final_acc.append(r)
        if r > max_acc:
            max_acc = r
            best_point = copy.deepcopy(final_model.state_dict())
            patience_cnt = 0
            print("best", epoch)
        else:
            patience_cnt += 1
        if patience_cnt >= patience:
            break
        print(max_acc)
    if not os.path.exists(final_model_path):
        torch.save(final_model.state_dict(), final_model_path)

if __name__ == "__main__":
    seed = config.Seed
    set_seed(seed)
    # 读取数据和存储数据
    dataset, train_data, test_data, train_mask, test_mask, sparse_mask, ug_v_v2, geo_v_v2 = getData()
    v_u_matrix, v_feature, target, function_target, words_list, v_v_dict = trainData(dataset, train_data)
    ug_v_v2 = np.array(ug_v_v2)
    geo_v_v2 = np.array(geo_v_v2)

    _words = delete_stopwords(words_list)
    _words = cut_words_key(_words, config.cut_len)

    dict_word = get_glove(config.glove_path)

    word_embed = charToEmbed(_words, dict_word)

    f_v_feature_path = dataset_path + "yelp_v_feature.npy"
    if not os.path.exists(f_v_feature_path):
        np.save(f_v_feature_path, v_feature)
    f_v_u_matrix_path = dataset_path + "yelp_v_u_matrix_" + platform_str + ".npy"
    if not os.path.exists(f_v_u_matrix_path):
        np.save(f_v_u_matrix_path, v_u_matrix)
    if platform_str != "GroundTruth":
        f_ug_v_v2_path = dataset_path + "yelp_ug_v_v_" + platform_str + ".npy"
        if not os.path.exists(f_ug_v_v2_path):
            np.save(f_ug_v_v2_path, ug_v_v2)
        f_v_v_dict_path = dataset_path + "yelp_v_v_dict_" + platform_str + ".json"
        if not os.path.exists(f_v_v_dict_path):
            v_v_dict_str = {str(k): v for k, v in v_v_dict.items()}
            with open(f_v_v_dict_path, "w") as f:
                json.dump(v_v_dict_str, f)
        f_word_embed_path = dataset_path + "yelp_word_embed_" + platform_str + ".npy"
        if not os.path.exists(f_word_embed_path):
            np.save(f_word_embed_path, word_embed)
        f_mask_dict = dataset_path + "yelp_mask_dict_" + platform_str + ".json"
        if not os.path.exists(f_mask_dict):
            mask_dict = {
                "train_mask" : train_mask,
                "test_mask" : test_mask,
                "sparse_mask" : sparse_mask
            }
            with open(f_mask_dict, "w") as f:
                json.dump(mask_dict, f)

    # 指定gpu并且把数据加载到gpu上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #train_data = train_data.to(device)
    #test_data = test_data.to(device)
    ug_v_v2 = torch.tensor(ug_v_v2, dtype=torch.long)
    ug_v_v2 = ug_v_v2.to(device)
    geo_v_v2 = torch.tensor(geo_v_v2, dtype=torch.long)
    geo_v_v2 = geo_v_v2.to(device)
    v_u_matrix = torch.tensor(v_u_matrix, dtype=torch.long)
    v_u_matrix = v_u_matrix.to(device)
    v_feature = torch.tensor(v_feature, dtype=torch.long)
    v_feature = v_feature.to(device)

    # 计算功能分类权重
    platform_target = function_target.copy()
    function_target = torch.tensor(function_target, dtype=torch.long)
    platform_target = torch.tensor(platform_target, dtype=torch.long)
    function_target = function_target.to(device)

    function_counts = Counter(function_target.tolist())
    total_samples = len(function_target)

    function_weight = torch.tensor([
        total_samples / function_counts[i] for i in range(len(function_counts))
    ])
    function_normalized_weight = function_weight / function_weight.max()

    sample_weights = function_normalized_weight[function_target]
    sample_weights = sample_weights.to(device)

    platform_weights = platform_target
    if platform_str != "GroundTruth":
        if platform_str == "A":
            custom_weights = {0:1, 1:1, 2:1}
        elif platform_str == "B":
            custom_weights = {0:1, 1:1, 2:1}
        elif platform_str == "C":
            custom_weights = {0:1, 1:1, 2:1}
        elif platform_str == "D":
            custom_weights = {0:1, 1:1, 2:1}
        map_function = np.vectorize(lambda x: custom_weights.get(x,0))
        platform_weights = map_function(platform_target)
    platform_weights = torch.tensor(platform_weights, dtype=torch.float)
    platform_weights = platform_weights.to(device)

    v_v_dict = {k:torch.tensor(v, dtype=torch.long).to(device) for k,v in v_v_dict.items()}

    word_embed = torch.Tensor(np.array(word_embed, dtype=np.float32))

    # 设置参数
    epochs = config.epochs

    in_channels = config.in_channels
    out_channels = config.out_channels

    w_out = config.w_out

    num_embed = config.n_emd

    f_dataset_path = dataset_path + "yelp_dataset.json"
    with open(f_dataset_path, "r") as f_dataset:
        all_dataset = pd.read_json(f_dataset, lines = True)
        all_business = all_dataset[["business_id", "latitude", "longitude", "category", "item_count", "item_score", "item_class"]].drop_duplicates()
    user_len = len(dataset["user_id"].unique())
    business_len = len(all_dataset["business_id"].unique())
    f_ground_truth_v_u_matrix = dataset_path + "yelp_v_u_matrix_GroundTruth.npy"
    g_v_u_matrix = np.load(f_ground_truth_v_u_matrix)
    final_model = NN(g_v_u_matrix, v_feature, in_channels, out_channels, user_len, business_len, num_embed, w_out)
    final_model = final_model.to(device)

    word_embed = word_embed.to(device)

    params_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "user_len": user_len,
        "business_len": business_len,
        "num_embed": num_embed,
        "w_out": w_out
    }

    f_params_dict_path = dataset_path + "params_dict_" + platform_str + ".json"
    if not os.path.exists(f_params_dict_path):
        with open(f_params_dict_path, "w") as f:
            json.dump(params_dict, f)

    final_acc = []
    max_acc = 0
    final_model_path = model_save_path + platform_str + "_original_model_.pth"
    final_criterion = nn.CrossEntropyLoss(reduction='mean')

    target = torch.tensor(target, dtype=torch.long)
    assert target.min() >= 0 and target.max() < 3, "标签超出范围"
    target = target.to(device)

    # 训练并保存模型
    if not os.path.exists(final_model_path):
        if platform_str == "GroundTruth":
            train_groundTruth_model(word_embed, v_u_matrix, v_feature, sample_weights, geo_v_v2, ug_v_v2, v_v_dict, params_dict, train_mask, test_mask, target, all_business, 5)
        else:
            train_platform_model(word_embed, v_u_matrix, v_feature, platform_weights, geo_v_v2, ug_v_v2, v_v_dict, params_dict, train_mask, test_mask, target)

    # 训练结束后保存groundtruth预测结果
    f_ground_truth_results_path = dataset_path + "yelp_ground_truth_results_" + state_str + ".json"
    if platform_str == "GroundTruth":
        cnt = np.zeros((3,) ,dtype = int)
        acc_cnt = np.zeros((3,) ,dtype = int)
        gt_results = pd.read_json(f_ground_truth_results_path, orient="records", lines=True)
        final_model.eval()
        final_model.load_state_dict(torch.load(final_model_path, map_location=lambda storage, loc: storage), strict=True)
        with torch.no_grad():
            output = final_model(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)
            for i in test_mask:
                result = torch.argmax(output[i]).item()
                truth_result = gt_results[gt_results["business_id"] == i]["item_class"].item()
                if truth_result == result:
                    cnt[gt_results[gt_results["business_id"] == i]["category"]] += 1
                    acc_cnt[gt_results[gt_results["business_id"] == i]["category"]] += 1
                else:
                    print(i, result, truth_result)
                    cnt[gt_results[gt_results["business_id"] == i]["category"]] += 1
        print(cnt)
        print(acc_cnt)
        print(acc_cnt / cnt)
        r = acc_cnt.sum() / cnt.sum()
        print(acc_cnt.sum() / cnt.sum())
    else:
        final_model.eval()
        final_model.load_state_dict(torch.load(final_model_path, map_location=lambda storage, loc: storage))
        with torch.no_grad():
            output = final_model(word_embed, v_u_matrix, v_feature, geo_v_v2, ug_v_v2, v_v_dict)
            gt_results = pd.read_json(f_ground_truth_results_path, orient = "records", lines = True)
            cnt = np.zeros((3,), dtype = int)
            acc_cnt = np.zeros((3,), dtype = int)
            for i in test_mask:
                result = torch.argmax(output[i]).item()
                truth_result = gt_results[gt_results["business_id"] == i]["item_class"].item()
                if truth_result == result:
                    cnt[gt_results[gt_results["business_id"] == i]["category"]] += 1
                    acc_cnt[gt_results[gt_results["business_id"] == i]["category"]] += 1
                else:
                    cnt[gt_results[gt_results["business_id"] == i]["category"]] += 1
        print(cnt)
        print(acc_cnt)
        print(acc_cnt / cnt)
        r = acc_cnt.sum() / cnt.sum()
        print(acc_cnt.sum() / cnt.sum())