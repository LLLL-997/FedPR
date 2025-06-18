from typing import TextIO
from urllib.parse import uses_relative

import numpy as np
import pandas as pd
import json
from collections import defaultdict
import time
import math
from pandas.tests.io.formats.test_format import filepath_or_buffer_id
from geopy.distance import geodesic
import os

import config
from config import dataset_path, state_str, platform_str, neighbor_num, preference_neighbor_num
from datapreprocessing import splitDataset, busDataPreprocessing



def getData():
    if platform_str == "GroundTruth":
        f_dataset = dataset_path + "yelp_dataset.json"
    else:
        f_dataset = dataset_path + "yelp_dataset_" + platform_str + ".json"
    if not os.path.exists(f_dataset):
        # 读取文件
        path_bus = dataset_path + "yelp_academic_dataset_business_" + state_str + ".json"
        path_review = dataset_path + "yelp_academic_dataset_review_" + state_str + ".json"
        path_user = dataset_path + "yelp_academic_dataset_user_" + state_str + ".json"
        if not os.path.exists(path_review) or not os.path.exists(path_user):
            splitDataset()

        f_bus = open(path_bus, encoding="utf-8")
        f_review = open(path_review, encoding="utf-8")
        f_user = open(path_user, encoding="utf-8")

        # 读入json
        js_bus, js_review, js_user = [], [], []
        for line_bus in f_bus:
            js_bus.append(json.loads(line_bus))
        for line_review in f_review:
            js_review.append(json.loads(line_review))
        for line_user in f_user:
            js_user.append(json.loads(line_user))

        f_bus.close()
        f_review.close()
        f_user.close()

        # 添加功能分类信息
        if config.func_Categories_str not in js_bus[0].keys():
            js_bus = busDataPreprocessing(js_bus)


        # 转为df处理
        # 提取数据名
        business_list = ["business_id", "latitude", "longitude", "stars", "review_count", "Foods", "Entertainments", "Function", "category", "categories"]
        review_list = ["business_id", "user_id", "text", "date", "stars"]
        user_list = ["user_id", "yelping_since", "review_count"]

        # features
        df_bus = pd.DataFrame(js_bus)
        df_review = pd.DataFrame(js_review)
        df_user = pd.DataFrame(js_user)

        fea_bus = df_bus[business_list]
        fea_review = df_review[review_list]
        fea_user = df_user[user_list]

        bus_rename = ["business_id", "latitude", "longitude", "item_stars", "item_review_count", "Foods", "Entertainments", "Function", "category", "categories"]
        fea_bus.columns = bus_rename
        user_rename = ["user_id", "yelping_since", "user_review_count"]
        fea_user.columns = user_rename

        # 特征集
        fea_user_review = pd.merge(fea_user, fea_review, on=["user_id"])
        dataset = pd.merge(fea_bus, fea_user_review, on=["business_id"])

        # 数据处理
        users = dataset["user_id"].unique()
        items = dataset["business_id"].unique()

        item_count = dataset["business_id"].value_counts()
        item_count.name = "item_count"

        dataset = dataset.join(item_count, on="business_id")

        user_count = dataset["user_id"].value_counts()
        user_count.name = "user_count"

        dataset = dataset.join(user_count, on="user_id")

        # 没想明白要不要去除稀疏对象
        dataset = dataset[(dataset["user_count"] >= 2) & (dataset["item_count"] >= 2)]
        users = dataset["user_id"].unique()
        items = dataset["business_id"].unique()

        del dataset["user_count"]
        del dataset["item_count"]



        item_count = dataset["business_id"].value_counts()
        item_count.name = "item_count"

        dataset = dataset.join(item_count, on="business_id")

        dataset = dataset[dataset["item_count"] >= 10]
        dataset.reset_index(drop=True, inplace=True)
        # 计算scores
        item_score = dataset.groupby("business_id")["stars"].mean().reset_index()
        item_score.rename(columns={"stars": "item_score"}, inplace=True)
        #item_score["item_score"] = item_score["item_score"].apply(lambda x: min(max(int(np.ceil(x)), 0), 5))
        item_score["item_score"] = item_score["item_score"].apply(lambda x: min(max(int(np.floor(x)), 0), 5))

        dataset = pd.merge(dataset, item_score, on="business_id")
        # 对数据集进行id替换
        business_uni = dataset["business_id"].unique()
        user_uni = dataset["user_id"].unique()
        bus_id_invmap = {id_: i for i, id_ in enumerate(business_uni)}
        user_id_invmap = {id_: i for i, id_ in enumerate(user_uni)}
        dataset["business_id"].replace(bus_id_invmap, inplace=True)
        dataset["user_id"].replace(user_id_invmap, inplace=True)


        print(dataset.head())
        dataset.to_json(f_dataset, orient="records", lines=True)

    dataset = pd.read_json(f_dataset, lines = True)
    dataset["categories"] = dataset["categories"].apply(tuple)
    users = dataset["user_id"].unique()
    items = dataset["business_id"].unique()


    # 数据统计部分
    #dataset.to_csv("../data/yelp/state/FL/yelp_academic_dataset.csv")

    # 输出poi数量，用户数量和总评价数量
    print(len(items),len(users),len(dataset))
    print(dataset["date"])
    # 输出不同poi的数量
    business_dataset = dataset[["business_id", "Foods", "Entertainments", "Function"]].drop_duplicates()
    food_count = (business_dataset["Foods"] == 1).sum()
    enterments_count = (business_dataset["Entertainments"] == 1).sum()
    function_count = (business_dataset["Function"] == 1).sum()
    print(food_count, enterments_count, function_count)
    
    # 输出不同poi的评价数量
    food_count = (dataset["Foods"] == 1).sum()
    enterments_count = (dataset["Entertainments"] == 1).sum()
    function_count = (dataset["Function"] == 1).sum()
    print(food_count, enterments_count, function_count)



    v_v_geo_neighbor_dict = {}
    geo_dict_path = dataset_path + "yelp_geo_dict_" + state_str + ".json"
    if os.path.exists(geo_dict_path):
        with open(geo_dict_path, "r", encoding="utf-8") as f_geo_dict:
            v_v_geo_neighbor_dict = json.load(f_geo_dict)
        for k in v_v_geo_neighbor_dict:
            v_v_geo_neighbor_dict[k] = set(v_v_geo_neighbor_dict[k])
    else:
        # 计算
        location = dataset[["business_id", "latitude", "longitude"]].drop_duplicates(subset="business_id", keep='first')
        poi_location = list(zip(location["latitude"], location["longitude"]))
        poi_len = len(poi_location)
        print(poi_len)
        v_v_distance_matrix = np.zeros((poi_len, poi_len))
        # 建立PoI距离矩阵
        print("begin")
        for i in range(poi_len):
            for j in range(poi_len):
                if i > j:
                    continue
                elif i == j:
                    v_v_distance_matrix[i, j] = 0
                else:
                    v_v_distance_matrix[i, j] = geodesic(poi_location[i], poi_location[j]).km
                    v_v_distance_matrix[j, i] = v_v_distance_matrix[i, j]
            print(i)
        for k in range(poi_len):
            neighbor_set = set()
            indexed_list = list(enumerate(v_v_distance_matrix[k]))
            sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=False)
            index = [index for index, value in sorted_list]
            value = [value for index, value in sorted_list]
            for i, v in zip(index, value):
                if v <= 2.5:
                    if i != k:
                        neighbor_set.add(i)
                        if len(neighbor_set) >= neighbor_num:
                            break
                    else:
                        continue
                else:
                    break
            v_v_geo_neighbor_dict[k] = neighbor_set
        for k in range(poi_len):
            if k in v_v_geo_neighbor_dict:
                for n in v_v_geo_neighbor_dict[k]:
                    if k not in v_v_geo_neighbor_dict[n]:
                        v_v_geo_neighbor_dict[n].add(k)
            v_v_geo_neighbor_dict[k] = set(filter(lambda x: x > k, v_v_geo_neighbor_dict[k]))
        for k in range(poi_len):
            v_v_geo_neighbor_dict[k] = list(v_v_geo_neighbor_dict[k])
        with open(geo_dict_path, "w", encoding="utf-8") as f_geo_dict:
            json.dump(v_v_geo_neighbor_dict, f_geo_dict)

    geo_v_v2 = []
    geo_v_v_path = dataset_path + "yelp_geo_v_v_" + state_str + ".npy"
    if os.path.exists(geo_v_v_path):
        geo_v_v2 = np.load(geo_v_v_path).tolist()
    else:
        geo_v_v = []
        geo_v_v_reverse = []
        for k in v_v_geo_neighbor_dict:
            for v in v_v_geo_neighbor_dict[k]:
                geo_v_v.append(int(k))
                geo_v_v.append(int(v))
                geo_v_v_reverse.append(int(v))
                geo_v_v_reverse.append(int(k))


        for k in items:
            geo_v_v.append(int(k))
            geo_v_v_reverse.append(int(k))
        geo_v_v2.append(geo_v_v)
        geo_v_v2.append(geo_v_v_reverse)
        np.save(geo_v_v_path, np.array(geo_v_v2))


    # 对数据集字段进行处理
    # 第一个阶段 7-9 第二个阶段 9-12 第三个阶段 12-18 第四个阶段 18-7
    time_phase_list = []
    print(dataset["date"])
    for i in range(dataset.shape[0]):
        date = dataset["date"].iloc[i].hour
        if 9 > date >= 7:
            time_phase_list.append(3)
        elif 12 > date >= 9:
            time_phase_list.append(2)
        elif 18 > date >= 12:
            time_phase_list.append(1)
        else:
            time_phase_list.append(4)

    df_date = pd.DataFrame(time_phase_list, columns=["date"])
    dataset["date"] = df_date["date"]

    # 二分类用户评分数据（可选，可以先实现了到时候和评分分类对比看看效果）
    dataset["stars_mean"] = dataset["stars"] / dataset["stars"].mean()
    like_list = []
    for i in range(dataset.shape[0]):
        if dataset.iloc[i]["stars"] >= 4:
            like_list.append(1)
        else:
            like_list.append(0)
    dataset["islike"] = pd.DataFrame(like_list)

    ug_v_v2 = []
    ug_v_v2_path = dataset_path + "yelp_ug_v_v_" + platform_str + ".npy"
    if os.path.exists(ug_v_v2_path):
        ug_v_v2 = np.load(ug_v_v2_path).tolist()
    else:
        v_v_neighbor_num_dict = {}
        # PoI-用户-PoI事件邻接矩阵
        v_v_dict = {}
        ug_v_v = []
        ug_v_v_reverse = []
        for business in dataset["business_id"].unique():
            business_list = []
            for user in dataset[dataset["business_id"] == business]["user_id"]:
                business_list.append(user)
            v_v_dict[business] = business_list
        key_list = []
        for k in v_v_dict.keys():
            key_list.append(k)

        user_business_likes = defaultdict(dict)
        for (usr, biz), group in dataset.groupby(["user_id", "business_id"]):
            user_business_likes[usr][biz] = set(group["islike"].values)

        v_v_dict_sets = {k: set(v) for k, v in v_v_dict.items()}
        # i是索引，k是key
        for i, k in enumerate(v_v_dict):
            k_set = v_v_dict_sets[k]
            overlap_counts = []
            for i2, k2 in enumerate(v_v_dict):
                if i2 == i:
                    continue
                k2_set = v_v_dict_sets[k2]
                interset = k_set & k2_set
                if not interset:
                    continue
                if len(interset) > 0:
                    preference_neighbor_cnt = 0
                    for usr in interset:
                        sub_interset = user_business_likes[usr][k] & user_business_likes[usr][k2]
                        if not sub_interset:
                            continue
                        else:
                            preference_neighbor_cnt += 1
                    overlap_counts.append((k2, preference_neighbor_cnt))
            overlap_counts.sort(key=lambda x: x[1], reverse=True)
            top_neighbor_list = overlap_counts[:preference_neighbor_num]
            for neighbor, _ in top_neighbor_list:
                if (neighbor > k):
                    ug_v_v.append(k)
                    ug_v_v.append(neighbor)
                    ug_v_v_reverse.append(neighbor)
                    ug_v_v_reverse.append(k)
        for i in items:
            ug_v_v.append(int(i))
            ug_v_v_reverse.append(int(i))
        ug_v_v2.append(ug_v_v)
        ug_v_v2.append(ug_v_v_reverse)
        np.save(ug_v_v2_path, np.array(ug_v_v2))




    '''
    # 输出PoI评价数量查看情况
    count_dataset = dataset[["business_id", "item_count", "Foods", "Entertainments", "Function", "Accommodation"]].drop_duplicates()
    count_dataset.to_csv(dataset_path + "count.csv", index = False)
    print(count_dataset["item_count"].mean())
    '''
    # 训练集测试集
    if platform_str == "GroundTruth":
        train_mask_threshold = config.gt_train_mask_threshold
        test_mask_threshold = config.gt_test_mask_threshold
    else:
        train_mask_threshold = config.platform_train_mask_threshold
        test_mask_threshold = config.platform_test_mask_threshold

    business_data = dataset[["business_id", "latitude", "longitude", "item_stars", "item_review_count", "Foods", "Entertainments", "Function", "category", "categories", "item_count", "item_score", "item_class"]].drop_duplicates()
    train_data = business_data[business_data["item_count"] >= train_mask_threshold]
    #test_data = business_data[(business_data["item_count"] < train_mask_threshold) &
     #                         (business_data["item_count"] >= test_mask_threshold)]

    test_data = business_data[((business_data["Entertainments"] != 1) & (business_data["item_count"] < train_mask_threshold) & (business_data["item_count"] >= test_mask_threshold)) |
                              ((business_data["Entertainments"] == 1) & (business_data["item_count"] < train_mask_threshold) & (business_data["item_count"] >= 4))]
    sparse_data = business_data[business_data["item_count"] < test_mask_threshold]


    train_data.reset_index(drop=True, inplace=True)
    print(len(train_data))
    test_data.reset_index(drop=True, inplace=True)
    print(len(test_data))
    print(len(sparse_data))

    print(business_data.head())
    train_mask = train_data["business_id"].tolist()
    test_mask = test_data["business_id"].tolist()
    sparse_mask = sparse_data["business_id"].tolist()

    return dataset, train_data, test_data, train_mask, test_mask, sparse_mask, ug_v_v2, geo_v_v2
    # 得到了数据集、用户-PoI-用户矩阵、PoI-用户-PoI矩阵（Preference矩阵）、PoI距离矩阵




def trainData(dataset, train_data):
    f_category_path = dataset_path + "yelp_category_dict.json"
    f_target_path = ""
    if platform_str == "GroundTruth":
        f_target_path = dataset_path + "yelp_ground_truth_results_" + state_str + ".json"
    else:
        f_target_path = dataset_path + "yelp_target_" + state_str + "_" + platform_str + ".json"
    with open(f_category_path, "r") as f_category:
        categoryDict = json.load(f_category)
    with open(dataset_path + "yelp_dataset.json", "r") as f_data:
        allData = pd.read_json(f_data, lines = True)
        allData["categories"] = allData["categories"].apply(tuple)

    user_len = len(allData["user_id"].unique())
    business_len = len(allData["business_id"].unique())
    category_len = len(categoryDict)

    # PoI-PoI 交互字典
    v_v_dict = {}
    for cur_business in dataset["business_id"].unique():
        business_set = set()
        for cur_user in dataset[dataset["business_id"] == cur_business]["user_id"]:
            business_set.add(cur_user)
        v_v_dict[cur_business] = business_set

    for i, k in enumerate(v_v_dict):
        if len(v_v_dict[k]) > config.neighbor_num:
            v_v_dict[k] = list(v_v_dict[k])[:config.neighbor_num]
        else:
            v_v_dict[k] = list(v_v_dict[k])

    # PoI-用户交互矩阵
    # 评论时间矩阵
    v_u_date = np.zeros((business_len, config.neighbor_num))
    # 用户评价次数矩阵
    v_u_review_count = np.zeros((business_len, config.neighbor_num))
    # PoI-用户评分
    v_u_stars = np.zeros((business_len, config.neighbor_num))

    words_list = []
    for i, k in enumerate(v_v_dict):
        for j in range(len(v_v_dict[k])):
            v_u_stars[k][j] = \
                dataset[(dataset["business_id"] == k) & (dataset["user_id"] == v_v_dict[k][j])]["stars"].unique()[0]
            v_u_date[k][j] = \
                dataset[(dataset["business_id"] == k) & (dataset["user_id"] == v_v_dict[k][j])]["date"].unique()[0]
            v_u_review_count[k][j] = \
                dataset[(dataset["business_id"] == k) & (dataset["user_id"] == v_v_dict[k][j])]["user_review_count"].unique()[0]
            words_list.append(
                dataset[(dataset["business_id"] == k) & (dataset["user_id"] == v_v_dict[k][j])]["text"].unique()[0]
            )

    business_data = allData[["business_id", "latitude", "longitude", "item_stars", "item_review_count", \
                             "Foods", "Entertainments", "Function", "category", "categories", "item_count", "item_score", "item_class"]].drop_duplicates()

    f_v_feature_path = dataset_path + "yelp_v_feature.npy"
    if os.path.exists(f_v_feature_path):
        v_feature = np.load(f_v_feature_path)
    else:
        # 大功能分类
        v_category = np.zeros((business_len, 4))
        # 细分功能分类
        v_categories_subdivided = np.zeros((business_len, category_len))
        # PoI位置
        for cur_business in business_data["business_id"].unique():
            index = business_data[business_data["business_id"] == cur_business]["category"].values[0] - 1
            v_category[cur_business][index] = 1
            for i,k in enumerate(business_data[business_data["business_id"] == cur_business]["categories"].tolist()):
                for item in k:
                    v_categories_subdivided[cur_business][item] = 1
        v_feature = np.concatenate((v_category, v_categories_subdivided), axis=1)
        #v_feature = v_category

    v_u_matrix = np.stack((v_u_stars, v_u_date, v_u_review_count), axis=0)

    with open(f_target_path, "r") as f_target:
        target_dataset = pd.read_json(f_target, lines = True)

    target = target_dataset["item_class"].to_numpy()
    function_target = business_data["category"].to_numpy()


    return v_u_matrix, v_feature, target, function_target, words_list, v_v_dict


if __name__ == '__main__':
    getData()
