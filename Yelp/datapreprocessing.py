import numpy as np
import pandas as pd
import json
import os
import random
import config

from numpy import record

from config import func_Categories_str, func_Categories, dataset_path, state_str, original_path, clients_list


def busDataPreprocessing(j_bus):
    # category的分类标注
    f_category_path = dataset_path + "yelp_category_dict.json"
    # 若不存在则新建dict文件
    if not os.path.exists(f_category_path):
        categorySet = set()
        for j in j_bus:
            if j["categories"] is None:
                continue
            else:
                minerList = j["categories"].split(",")
                for categoryStr in minerList:
                    categorySet.add(categoryStr.strip())

        categoryDict = {key:"" for key in categorySet}
        f_category = open(f_category_path, "w")
        json.dump(categoryDict, f_category, indent = 4)
        f_category.close()
    
    # 读入词典
    with open(f_category_path, "r") as f_category:
        categoryDict = json.load(f_category)
    if -1 in categoryDict.values():
        changeCategoryDict()
        with open(f_category_path, "r") as f_category:
            categoryDict = json.load(f_category)

    # 对每条business数据进行处理
    deleteIndexList = []
    for bus_index in range(len(j_bus)):
        # 初始化全为0
        for i in range(1, 4):
            j_bus[bus_index][func_Categories[i]] = 0
        # 如果没有tag就分为实用
        if j_bus[bus_index]["categories"] is None:
            deleteIndexList.append(bus_index)
        else:
            # cnt范围[0, 3]
            cnt = [0] * 4
            categoryList = j_bus[bus_index]["categories"].split(",")
            # 得到的是categories tags
            # 对不同分类tag计数
            for categoryStr in categoryList:
                category = categoryStr.strip()
                cnt[categoryDict[category] - 1] += 1
            max_category = max(cnt)
            max_count = cnt.count(max_category)
            if max_count == 1:
                if cnt.index(max_category) == 3:
                    j_bus[bus_index][func_Categories[cnt.index(max_category)]] = 1
                    j_bus[bus_index]["category"] = cnt.index(max_category) - 1
                else:
                    j_bus[bus_index][func_Categories[cnt.index(max_category) + 1]] = 1
                    j_bus[bus_index]["category"] = cnt.index(max_category)
            else:
                if 1 < max_count < 5:
                    deleteIndexList.append(bus_index)
                else:
                    print("error")
    j_bus = [j_bus[i] for i in range(len(j_bus)) if i not in deleteIndexList]

    # 替换Index
    categoryIndexDict = {key: i for i, key in enumerate(categoryDict.keys())}
    for j in j_bus:
        if j["categories"] is None:
            continue
        else:
            categorySet = set()
            categoryList = []
            minerList = j["categories"].split(",")
            for categoryStr in minerList:
                categorySet.add(categoryStr.strip())
            for item in categorySet:
                categoryList.append(categoryIndexDict[item])
            j["categories"] = tuple(categoryList)
    return j_bus


def changeCategoryDict():
    f_category_pre_path = "../data/yelp/yelp_category_dict.json"
    f_category_path = dataset_path + "yelp_category_dict.json"

    with open(f_category_pre_path, "r") as f_category_pre:
        category_dict_pre = json.load(f_category_pre)
    with open(f_category_path, "r") as f_category:
        category_dict = json.load(f_category)

    for i,k in enumerate(category_dict):
        if k in category_dict_pre:
            if(category_dict[k] == -1):
                category_dict[k] = category_dict_pre[k]

    f_category_new_path = dataset_path + "yelp_category_dict.json"
    with open(f_category_new_path, "w") as f_category_new:
        json.dump(category_dict, f_category_new, indent = 4)


def splitDataset():
    # split business data by the state
    '''
    f_bus = open("../data/yelp/yelp_academic_dataset_business.json", encoding="utf-8")
    js_bus = []
    for line in f_bus:
        js_bus.append(json.loads(line))
    f_bus.close()

    bus_line = ["state"]
    df_bus = pd.DataFrame(js_bus)
    state_list = df_bus["state"].unique()
    print(state_list)

    for state in state_list:
        tmp_json = [item for item in js_bus if item["state"] == state]
        file_str = "../data/yelp/yelp_academic_dataset_business_" + state + ".json"
        with open(file_str, "w") as f_tmp:
            for item in tmp_json:
                json.dump(item, f_tmp, separators=(',', ':'), indent = None)
                f_tmp.write("\n")

    '''



    f_bus = open(dataset_path + "yelp_academic_dataset_business_" + state_str + ".json", encoding="utf-8")
    f_review = open(original_path +"yelp_academic_dataset_review.json", encoding="utf-8")
    f_user = open(original_path + "yelp_academic_dataset_user.json", encoding="utf-8")

    js_bus, js_review, js_user = [], [], []
    for line_bus in f_bus:
        js_bus.append(json.loads(line_bus))
    for line_review in f_review:
        js_review.append(json.loads(line_review))
    for line_user in f_user:
        js_user.append(json.loads(line_user))

    business_list = ["business_id", "latitude", "longitude", "stars", "review_count"]
    review_list = ["business_id", "user_id", "text", "date", "stars"]
    user_list = ["user_id", "yelping_since", "review_count"]

    df_bus = pd.DataFrame(js_bus)
    df_review = pd.DataFrame(js_review)
    df_user = pd.DataFrame(js_user)
    fea_bus = df_bus[business_list]
    fea_review = df_review[review_list]
    fea_user = df_user[user_list]

    bus_rename = ["business_id", "latitude", "longitude", "item_stars", "item_review_count"]
    fea_bus.columns = bus_rename
    user_rename = ["user_id", "yelping_since", "user_review_count"]
    fea_user.columns = user_rename

    user_review = pd.merge(fea_user, fea_review, on=["user_id"])
    dataset = pd.merge(fea_bus, user_review, on=["business_id"])

    user = dataset["user_id"].unique()
    items = dataset["business_id"].unique()

    print(len(fea_bus), len(user), len(dataset))

    restore_user_review = dataset[["business_id","user_id", "yelping_since", "user_review_count", "text", "date", "stars"]]
    restore_user = restore_user_review[["user_id", "yelping_since", "user_review_count"]].drop_duplicates()
    restore_review = restore_user_review[["business_id", "user_id", "text", "date", "stars"]]

    restore_user_list = ["user_id", "yelping_since", "review_count"]
    restore_user.columns = restore_user_list
    review_file_str = dataset_path + "yelp_academic_dataset_review_" + state_str + ".json"
    user_file_str = dataset_path + "yelp_academic_dataset_user_" + state_str + ".json"

    restore_user.to_json(user_file_str, orient="records", lines = True)
    restore_review.to_json(review_file_str, orient="records", lines = True)


def splitPlatform():
    f_dataset = dataset_path + "yelp_dataset.json"
    dataset = pd.read_json(f_dataset, lines = True)
    business = dataset["business_id"].unique()
    users = dataset["user_id"].unique()
    platformA = []
    platformB = []
    platformC = []
    platformD = []
    rateAB = 0.5
    rateC = 0.8
    rateC_1 = 0.05
    rateD = 0.5
    rateD_1 = 0.2
    result_f1 = dataset.groupby("user_id")["Foods"].sum()
    result_f2 = dataset.groupby("user_id")["Entertainments"].sum()
    result_f3 = dataset.groupby("user_id")["Function"].sum()
    result_f4 = dataset.groupby("user_id")["Accommodation"].sum()
    for user in users:
        num_f1 = result_f1[user]
        num_f2 = result_f2[user]
        num_f3 = result_f3[user]
        num_f4 = result_f4[user]
        sum_f = num_f1 + num_f2 + num_f3 + num_f4
        rate1 = num_f1 / sum_f
        rate2 = (num_f2 + num_f1) / sum_f
        rate3 = (num_f3 + num_f4) / sum_f
        rate2_1 = num_f2 / (num_f1 + num_f2)
        rate3_1 = num_f4 / sum_f

        #print(rate1, rate2, rate3)
        user = user.item()
        if rate3 >= rateD or rate3_1 >= rateD_1:
            platformD.append(user)
        elif rate3 < rateD and rate2 >= rateC:
            if rate2_1 >= rateC_1:
                platformC.append(user)
            else:
                x = random.random()
                if x >= 0.5:
                    platformA.append(user)
                else:
                    platformB.append(user)
        elif rate2 < rateC and rate1 >= rateAB:
            x = random.random()
            if x >= 0.5:
                platformA.append(user)
            else:
                platformB.append(user)

        else:
            x = random.random()
            if x < 0.25:
                platformA.append(user)
            elif x < 0.5:
                platformB.append(user)
            elif x < 0.75:
                platformC.append(user)
            else:
                platformD.append(user)

    platformDict = {"A": platformA, "B": platformB, "C": platformC, "D":platformD}
    print(len(platformDict["A"]))
    print(len(platformDict["B"]))
    print(len(platformDict["C"]))
    print(len(platformDict["D"]))
    platform_path = dataset_path + "yelp_platform_" + state_str + ".json"
    with open(platform_path, "w") as f:
        json.dump(platformDict, f)

def generateNewDataset():
    '''
    每个功能挑出来1000个POI
    :return:
    '''
    f_dataset = dataset_path + "yelp_dataset.json"
    dataset = pd.read_json(f_dataset, lines=True)
    business_data = dataset[["business_id", "latitude", "longitude", "item_stars", "item_review_count",
                               "Foods", "Entertainments", "Function", "category",
                               "item_score"]].drop_duplicates()
    business = business_data["business_id"].unique()
    f_list = []
    e_list = []
    u_list = []

    for busi in business:
        print(business_data[business_data["business_id"] == busi]["Foods"].item(), business_data[business_data["business_id"] == busi]["Entertainments"].item(), business_data[business_data["business_id"] == busi]["Function"].item())
        if business_data[business_data["business_id"] == busi]["Foods"].item() == 1 and len(f_list) < 1000:
            f_list.append(busi.item())
        elif business_data[business_data["business_id"] == busi]["Entertainments"].item() == 1 and len(e_list) < 1000:
            e_list.append(busi.item())
        elif business_data[business_data["business_id"] == busi]["Function"].item() == 1 and len(u_list) < 1000:
            u_list.append(busi.item())
        else:
            continue
        if len(f_list) >= 1000 and len(e_list) >= 1000 and len(u_list) >= 1000:
            break
    merged_list = f_list + e_list + u_list
    new_dataset = dataset[dataset["business_id"].isin(merged_list)]
    new_dataset.to_json(f_dataset, orient="records", lines = True)

def generateNewPlatform():
    f_dataset_path = dataset_path + "yelp_dataset.json"
    dataset = pd.read_json(f_dataset_path, lines=True)
    users = dataset["user_id"].unique()
    f_list = []
    e_list = []
    u_list = []
    result_f = dataset.groupby("user_id")["Foods"].sum()
    result_e = dataset.groupby("user_id")["Entertainments"].sum()
    result_u = dataset.groupby("user_id")["Function"].sum()
    for user in users:
        user = user.item()
        f_count = result_f[user]
        e_count = result_e[user]
        u_count = result_u[user]
        if e_count >= 1:
            x = random.random()
            if x >= 0.5:
                e_list.append(user)
            else:
                y = random.random()
                if y >= 0.5:
                    f_list.append(user)
                else:
                    u_list.append(user)
        elif u_count >= 1:
            u_list.append(user)
        elif f_count >= 1:
            f_list.append(user)
    platformDict = {"A": f_list, "B": e_list, "C": u_list}
    print(len(platformDict["A"]))
    print(len(platformDict["B"]))
    print(len(platformDict["C"]))
    platform_path = dataset_path + "yelp_platform_" + state_str + ".json"
    with open(platform_path, "w") as f:
        json.dump(platformDict, f)


def checkNewDataset():
    f_dataset = dataset_path + "yelp_dataset.json"
    datasetA = pd.read_json(f_dataset, lines=True)
    business_dataA = datasetA[["business_id", "latitude", "longitude", "item_stars", "item_review_count",
                               "Foods", "Entertainments", "Function", "category",
                               "item_score"]].drop_duplicates()
    food_countA = (business_dataA["Foods"] == 1).sum()
    enterments_countA = (business_dataA["Entertainments"] == 1).sum()
    function_countA = (business_dataA["Function"] == 1).sum()
    food_review_countA = (datasetA["Foods"] == 1).sum()
    enterments_review_countA = (datasetA["Entertainments"] == 1).sum()
    function_review_countA = (datasetA["Function"] == 1).sum()

    print("A:", len(datasetA), len(datasetA["business_id"].unique()))
    print("PoI count", food_countA, enterments_countA, function_countA)
    print("review count:", food_review_countA, enterments_review_countA, function_review_countA)


def checkPlatform():
    f_platform_dict = dataset_path + "yelp_platform_" + state_str + ".json"
    with open(f_platform_dict, "r") as f_platform:
        platform_dict = json.load(f_platform)
    f_dataset = dataset_path + "yelp_dataset.json"
    dataset = pd.read_json(f_dataset, lines=True)
    datasetA = dataset[dataset["user_id"].isin(platform_dict["A"])]
    datasetB = dataset[dataset["user_id"].isin(platform_dict["B"])]
    datasetC = dataset[dataset["user_id"].isin(platform_dict["C"])]
    datasetD = dataset[dataset["user_id"].isin(platform_dict["D"])]

    business_dataA = datasetA[["business_id", "latitude", "longitude", "item_stars", "item_review_count",
                              "Foods", "Entertainments", "Function", "Accommodation", "category", "item_score"]].drop_duplicates()
    food_countA = (business_dataA["Foods"] == 1).sum()
    enterments_countA = (business_dataA["Entertainments"] == 1).sum()
    function_countA = (business_dataA["Function"] == 1).sum()
    accommodation_countA = (business_dataA["Accommodation"] == 1).sum()
    food_review_countA = (datasetA["Foods"] == 1).sum()
    enterments_review_countA = (datasetA["Entertainments"] == 1).sum()
    function_review_countA = (datasetA["Function"] == 1).sum()
    accommodation_review_countA = (datasetA["Accommodation"] == 1).sum()
    business_dataB = datasetB[["business_id", "latitude", "longitude", "item_stars", "item_review_count",
                              "Foods", "Entertainments", "Function", "Accommodation", "category", "item_score"]].drop_duplicates()
    food_countB = (business_dataB["Foods"] == 1).sum()
    enterments_countB = (business_dataB["Entertainments"] == 1).sum()
    function_countB = (business_dataB["Function"] == 1).sum()
    accommodation_countB = (business_dataB["Accommodation"] == 1).sum()
    food_review_countB = (datasetB["Foods"] == 1).sum()
    enterments_review_countB = (datasetB["Entertainments"] == 1).sum()
    function_review_countB = (datasetB["Function"] == 1).sum()
    accommodation_review_countB = (datasetB["Accommodation"] == 1).sum()
    business_dataC = datasetC[["business_id", "latitude", "longitude", "item_stars", "item_review_count",
                               "Foods", "Entertainments", "Function", "Accommodation", "category", "item_score"]].drop_duplicates()
    food_countC = (business_dataC["Foods"] == 1).sum()
    enterments_countC = (business_dataC["Entertainments"] == 1).sum()
    function_countC = (business_dataC["Function"] == 1).sum()
    accommodation_countC = (business_dataC["Accommodation"] == 1).sum()
    food_review_countC = (datasetC["Foods"] == 1).sum()
    enterments_review_countC = (datasetC["Entertainments"] == 1).sum()
    function_review_countC = (datasetC["Function"] == 1).sum()
    accommodation_review_countC = (datasetC["Accommodation"] == 1).sum()
    business_dataD = datasetD[["business_id", "latitude", "longitude", "item_stars", "item_review_count",
                               "Foods", "Entertainments", "Function", "Accommodation", "category", "item_score"]].drop_duplicates()
    food_countD = (business_dataD["Foods"] == 1).sum()
    enterments_countD = (business_dataD["Entertainments"] == 1).sum()
    function_countD = (business_dataD["Function"] == 1).sum()
    accommodation_countD = (business_dataD["Accommodation"] == 1).sum()
    food_review_countD = (datasetD["Foods"] == 1).sum()
    enterments_review_countD = (datasetD["Entertainments"] == 1).sum()
    function_review_countD = (datasetD["Function"] == 1).sum()
    accommodation_review_countD = (datasetD["Accommodation"] == 1).sum()

    print("A:", len(datasetA), len(datasetA["business_id"].unique()))
    print("PoI count", food_countA, enterments_countA, function_countA, accommodation_countA)
    print("review count:", food_review_countA, enterments_review_countA, function_review_countA, accommodation_review_countA)
    print("B:", len(datasetB), len(datasetB["business_id"].unique()))
    print("PoI count", food_countB, enterments_countB, function_countB, accommodation_countB)
    print("review count:", food_review_countB, enterments_review_countB, function_review_countB,
          accommodation_review_countB)
    print("C:", len(datasetC), len(datasetC["business_id"].unique()))
    print("PoI count", food_countC, enterments_countC, function_countC, accommodation_countC)
    print("review count:", food_review_countC, enterments_review_countC, function_review_countC,
          accommodation_review_countC)
    print("D:", len(datasetD), len(datasetD["business_id"].unique()))
    print("PoI count", food_countD, enterments_countD, function_countD, accommodation_countD)
    print("review count:", food_review_countD, enterments_review_countD, function_review_countD,
          accommodation_review_countD)

def checkNewPlatform():
    f_platform_dict = dataset_path + "yelp_platform_" + state_str + ".json"
    with open(f_platform_dict, "r") as f_platform:
        platform_dict = json.load(f_platform)
    f_dataset = dataset_path + "yelp_dataset.json"
    dataset = pd.read_json(f_dataset, lines=True)
    datasetA = dataset[dataset["user_id"].isin(platform_dict["A"])]
    datasetB = dataset[dataset["user_id"].isin(platform_dict["B"])]
    datasetC = dataset[dataset["user_id"].isin(platform_dict["C"])]

    business_dataA = datasetA[["business_id", "latitude", "longitude", "item_stars", "item_review_count",
                               "Foods", "Entertainments", "Function", "category",
                               "item_score"]].drop_duplicates()
    food_countA = (business_dataA["Foods"] == 1).sum()
    enterments_countA = (business_dataA["Entertainments"] == 1).sum()
    function_countA = (business_dataA["Function"] == 1).sum()
    food_review_countA = (datasetA["Foods"] == 1).sum()
    enterments_review_countA = (datasetA["Entertainments"] == 1).sum()
    function_review_countA = (datasetA["Function"] == 1).sum()
    business_dataB = datasetB[["business_id", "latitude", "longitude", "item_stars", "item_review_count",
                               "Foods", "Entertainments", "Function", "category",
                               "item_score"]].drop_duplicates()
    food_countB = (business_dataB["Foods"] == 1).sum()
    enterments_countB = (business_dataB["Entertainments"] == 1).sum()
    function_countB = (business_dataB["Function"] == 1).sum()
    food_review_countB = (datasetB["Foods"] == 1).sum()
    enterments_review_countB = (datasetB["Entertainments"] == 1).sum()
    function_review_countB = (datasetB["Function"] == 1).sum()
    business_dataC = datasetC[["business_id", "latitude", "longitude", "item_stars", "item_review_count",
                               "Foods", "Entertainments", "Function", "category",
                               "item_score"]].drop_duplicates()
    food_countC = (business_dataC["Foods"] == 1).sum()
    enterments_countC = (business_dataC["Entertainments"] == 1).sum()
    function_countC = (business_dataC["Function"] == 1).sum()
    food_review_countC = (datasetC["Foods"] == 1).sum()
    enterments_review_countC = (datasetC["Entertainments"] == 1).sum()
    function_review_countC = (datasetC["Function"] == 1).sum()



    print("A:", len(datasetA), len(datasetA["business_id"].unique()))
    print("PoI count", food_countA, enterments_countA, function_countA)
    print("review count:", food_review_countA, enterments_review_countA, function_review_countA)
    print("B:", len(datasetB), len(datasetB["business_id"].unique()))
    print("PoI count", food_countB, enterments_countB, function_countB)
    print("review count:", food_review_countB, enterments_review_countB, function_review_countB)
    print("C:", len(datasetC), len(datasetC["business_id"].unique()))
    print("PoI count", food_countC, enterments_countC, function_countC)
    print("review count:", food_review_countC, enterments_review_countC, function_review_countC)

def checkNewPlatformAfterDelete():
    f_dataset_A = dataset_path + "yelp_dataset_A.json"
    f_dataset_B = dataset_path + "yelp_dataset_B.json"
    f_dataset_C = dataset_path + "yelp_dataset_C.json"
    datasetA = pd.read_json(f_dataset_A, lines=True)
    datasetB = pd.read_json(f_dataset_B, lines=True)
    datasetC = pd.read_json(f_dataset_C, lines=True)

    business_dataA = datasetA[["business_id", "latitude", "longitude", "item_stars", "item_review_count",
                               "Foods", "Entertainments", "Function", "category",
                               "item_score"]].drop_duplicates()
    food_countA = (business_dataA["Foods"] == 1).sum()
    enterments_countA = (business_dataA["Entertainments"] == 1).sum()
    function_countA = (business_dataA["Function"] == 1).sum()
    food_review_countA = (datasetA["Foods"] == 1).sum()
    enterments_review_countA = (datasetA["Entertainments"] == 1).sum()
    function_review_countA = (datasetA["Function"] == 1).sum()
    business_dataB = datasetB[["business_id", "latitude", "longitude", "item_stars", "item_review_count",
                               "Foods", "Entertainments", "Function", "category",
                               "item_score"]].drop_duplicates()
    food_countB = (business_dataB["Foods"] == 1).sum()
    enterments_countB = (business_dataB["Entertainments"] == 1).sum()
    function_countB = (business_dataB["Function"] == 1).sum()
    food_review_countB = (datasetB["Foods"] == 1).sum()
    enterments_review_countB = (datasetB["Entertainments"] == 1).sum()
    function_review_countB = (datasetB["Function"] == 1).sum()
    business_dataC = datasetC[["business_id", "latitude", "longitude", "item_stars", "item_review_count",
                               "Foods", "Entertainments", "Function", "category",
                               "item_score"]].drop_duplicates()
    food_countC = (business_dataC["Foods"] == 1).sum()
    enterments_countC = (business_dataC["Entertainments"] == 1).sum()
    function_countC = (business_dataC["Function"] == 1).sum()
    food_review_countC = (datasetC["Foods"] == 1).sum()
    enterments_review_countC = (datasetC["Entertainments"] == 1).sum()
    function_review_countC = (datasetC["Function"] == 1).sum()



    print("A:", len(datasetA), len(datasetA["business_id"].unique()))
    print("PoI count", food_countA, enterments_countA, function_countA)
    print("review count:", food_review_countA, enterments_review_countA, function_review_countA)
    print("B:", len(datasetB), len(datasetB["business_id"].unique()))
    print("PoI count", food_countB, enterments_countB, function_countB)
    print("review count:", food_review_countB, enterments_review_countB, function_review_countB)
    print("C:", len(datasetC), len(datasetC["business_id"].unique()))
    print("PoI count", food_countC, enterments_countC, function_countC)
    print("review count:", food_review_countC, enterments_review_countC, function_review_countC)

def generatePlatformDataset():
    f_platform_dict = dataset_path + "yelp_platform_" + state_str + ".json"
    with open(f_platform_dict, "r") as f_platform:
        platform_dict = json.load(f_platform)
    f_dataset = dataset_path + "yelp_dataset.json"
    dataset = pd.read_json(f_dataset, lines=True)
    for i,client in enumerate(clients_list):
        f_subdataset = dataset_path + "yelp_dataset_" + client + ".json"
        sub_dataset = dataset[dataset["user_id"].isin(platform_dict[client])].reset_index(drop=True)
        sub_dataset.to_json(f_subdataset, orient = "records", lines = True)

def checkDataset():
    p_str = "D"
    results_path = dataset_path + "yelp_ground_truth_results_" + state_str + ".json"
    f_dataset_A_path = dataset_path + "yelp_dataset_" + p_str + ".json"

    with open(results_path, "r") as results_file:
        busi = pd.read_json(results_file, lines=True)

    with open(f_dataset_A_path, "r") as f_dataset_A:
        dataset_A = pd.read_json(f_dataset_A, lines=True)
        dataset_A["categories"] = dataset_A["categories"].apply(tuple)


    train_data = busi[busi["item_count"] >= config.gt_train_mask_threshold]

    busi_A = dataset_A[["business_id", "category", "item_count", "item_score"]].drop_duplicates()
    train_data_A = busi_A[busi_A["item_count"] >= config.platform_train_mask_threshold]
    test_data_A = busi_A[(busi_A["item_count"] < config.platform_train_mask_threshold) &
                         (busi_A["item_count"] >= config.platform_test_mask_threshold)]
    sparse_data_A = busi_A[busi_A["item_count"] < config.platform_test_mask_threshold]

    print(len(train_data_A), len(test_data_A), len(sparse_data_A))
    '''
    f_mask_dict_path = dataset_path + "yelp_mask_dict_" + p_str +".json"
    with open(f_mask_dict_path, "r") as f_mask_dict:
        mask_dict = json.load(f_mask_dict)
    train_mask = mask_dict["train_mask"]
    '''

    train_mask = train_data_A["business_id"].tolist()

    cnt = np.zeros((4,), dtype=int)
    same_cnt = np.zeros((4,), dtype=int)
    for i in train_mask:
        if i in train_data["business_id"]:
            x_A = busi_A[busi_A["business_id"] == i]["item_score"]
            score_A = x_A.item()
            x = train_data[train_data["business_id"] == i]["item_score"]
            score = x.item()
            category = busi_A[busi_A["business_id"] == i]["category"].item()
            if score_A == score:
                cnt[category] += 1
                same_cnt[category] += 1
            else:
                cnt[category] += 1
    print(cnt)
    print(same_cnt)
    print(same_cnt / cnt)


def generateUserWeights():
    f_dataset_path = dataset_path + "yelp_dataset.json"
    with open(f_dataset_path, "r") as f_dataset:
        dataset = pd.read_json(f_dataset, lines = True)
    rateAB = 0.5
    rateC = 0.8
    rateC_1 = 0.05
    rateD = 0.4

    users = dataset["user_id"].unique()

    result_f1 = dataset.groupby("user_id")["Foods"].sum()
    result_f2 = dataset.groupby("user_id")["Entertainments"].sum()
    result_f3 = dataset.groupby("user_id")["Function"].sum()
    list_id = []
    list_food = []
    list_entertainments = []
    list_function = []
    for user in users:
        num_f1 = result_f1[user]
        num_f2 = result_f2[user]
        num_f3 = result_f3[user]

        sum_f = num_f1 + num_f2 + num_f3
        rate1 = num_f1 / sum_f
        rate2 = (num_f2 + num_f1) / sum_f
        rate3 = num_f3 / sum_f
        rate2_1 = num_f2 / (num_f1 + num_f2) if (num_f1 + num_f2) > 0 else 0
        if rate1 >= rateAB:
            score_1 = 1
        else:
            score_1 = rate1/rateAB

        if rate2 >= rateC or rate2_1 >= rateC_1:
            score_2 = 1
        else:
            score_2 = rate2/rateC

        if rate3 >= rateD:
            score_3 = 1
        else:
            score_3 = rate3/rateD


        list_id.append(user.item())
        list_food.append(score_1)
        list_entertainments.append(score_2)
        list_function.append(score_3)

    userWeight_df = pd.DataFrame(list(zip(list_id, list_food, list_entertainments, list_function)), columns=["user_id", "Foods", "Entertainments", "Function"])
    userWeight_path = dataset_path + "yelp_UserWeight.json"
    userWeight_df.to_json(userWeight_path, orient="records", lines=True)

def changeItemScoreAverage():
    #p_str = "D"
    p_str = "GroundTruth"
    if p_str == "GroundTruth":
        f_dataset_path = dataset_path + "yelp_dataset.json"
    else:
        f_dataset_path = dataset_path + "yelp_dataset_" + p_str + ".json"
    with open(f_dataset_path, "r") as f_dataset:
        dataset = pd.read_json(f_dataset, lines=True)

    del dataset["item_count"]
    del dataset["item_score"]
    item_count = dataset["business_id"].value_counts()
    item_count.name = "item_count"

    dataset = dataset.join(item_count, on="business_id")
    item_score = dataset.groupby("business_id")["stars"].mean().reset_index()
    item_score.rename(columns={"stars": "item_score"}, inplace=True)
    item_score["item_score"] = item_score["item_score"].apply(lambda x: min(max(int(np.ceil(x)), 0), 5))

    dataset = pd.merge(dataset, item_score, on="business_id")
    dataset.reset_index(drop = True, inplace = True)
    dataset.to_json(f_dataset_path, orient = "records", lines = True)
    print(dataset.head())



def changeItemScoreWeighted():
    p_str = "C"
    #p_str = "GroundTruth"
    if p_str == "GroundTruth":
        f_dataset_path = dataset_path + "yelp_dataset.json"
    else:
        f_dataset_path = dataset_path + "yelp_dataset_" + p_str + ".json"
    with open(f_dataset_path, "r") as f_dataset:
        dataset = pd.read_json(f_dataset, lines=True)

    f_userweight_path = dataset_path + "yelp_UserWeight.json"
    with open(f_userweight_path, "r") as f_userweight:
        userweight = pd.read_json(f_userweight, lines=True)

    userweight.set_index("user_id", inplace=True, drop=False)

    del dataset["item_count"]
    del dataset["item_score"]
    item_count = dataset["business_id"].value_counts()
    item_count.name = "item_count"

    dataset = dataset.join(item_count, on="business_id")

    dataset["user_weight"] = dataset.apply(
        lambda row: userweight.loc[row["user_id"], config.func_Categories[row["category"] + 1]],
        #lambda row: print(row["user_id"], config.func_Categories[row["category"] + 1], "KeyError") or None,
        axis = 1
    )


    dataset["weighted_score"] = dataset["stars"] * dataset["user_weight"]
    has_zero = (dataset['user_weight'] == 0).any()
    print("是否有 0 值:", has_zero)
    rows_with_zero = dataset[dataset['user_weight'] == 0]
    print("包含 0 值的行:")
    print(rows_with_zero)
    item_score = dataset.groupby("business_id").apply(
        lambda x: x["weighted_score"].sum() / x["user_weight"].sum()
    ).reset_index(name = "item_score")
    #item_score["item_score"] = item_score["item_score"].apply(lambda x: min(max(int(np.ceil(x)), 0), 5))
    item_score["item_score"] = item_score["item_score"].apply(lambda x: min(max(int(np.floor(x)), 0), 5))
    del dataset["weighted_score"]
    del dataset["user_weight"]

    dataset = pd.merge(dataset, item_score, on="business_id")
    dataset.reset_index(drop = True, inplace = True)
    dataset.to_json(f_dataset_path, orient = "records", lines = True)
    print(dataset.head())
    print(dataset["item_score"].value_counts())

def changeItemFunction():
    p_str = "D"
    #p_str = "GroundTruth"
    if p_str == "GroundTruth":
        f_dataset_path = dataset_path + "yelp_dataset.json"
    else:
        f_dataset_path = dataset_path + "yelp_dataset_" + p_str + ".json"
    with open(f_dataset_path, "r") as f_dataset:
        dataset = pd.read_json(f_dataset, lines=True)

    dataset.loc[dataset["category"] == 3, ["category", "Function"]] = [2, 1]
    del dataset["Accommodation"]
    dataset.to_json(f_dataset_path, orient = "records", lines = True)

def changeItemClass():
    p_str = "C"
    #p_str = "GroundTruth"
    if p_str == "GroundTruth":
        f_dataset_path = dataset_path + "yelp_dataset.json"
    else:
        f_dataset_path = dataset_path + "yelp_dataset_" + p_str + ".json"
    with open(f_dataset_path, "r") as f_dataset:
        dataset = pd.read_json(f_dataset, lines=True)

    conditions = [
        dataset["item_score"].isin([1, 2]),
        dataset["item_score"] == 3,
        dataset["item_score"].isin([4, 5]),
    ]
    choices = [0, 1, 2]
    dataset["item_class"] = np.select(conditions, choices, default=0)

    dataset.to_json(f_dataset_path, orient = "records", lines = True)

def deleteReview():
    p_str = "C"
    f_dataset_path = dataset_path + "yelp_dataset_" + p_str + ".json"
    with open(f_dataset_path, "r") as f_dataset:
        dataset = pd.read_json(f_dataset, lines=True)

    business_f = dataset[dataset["Foods"] == 1]["business_id"].unique()
    business_e = dataset[dataset["Entertainments"] == 1]["business_id"].unique()
    business_u = dataset[dataset["Function"] == 1]["business_id"].unique()
    limit_ids_f = np.random.choice(business_f, size = int(len(business_f) * 0.8), replace = False)
    limits_set_f = set(limit_ids_f)
    keep_id_set_f = set(business_f) - limits_set_f
    limits_id_e = np.random.choice(business_e, size = int(len(business_e) * 0.8), replace = False)
    limits_set_e = set(limits_id_e)
    keep_id_set_e = set(business_e) - limits_set_e
    limit_id_u = np.random.choice(business_u, size = int(len(business_u) * 0.8), replace = False)
    limits_set_u = set(limit_id_u)
    keep_id_set_u = set(business_u) - limits_set_u

    df_limited_f = dataset[dataset["business_id"].isin(limits_set_f)].groupby("business_id").head(5)
    df_limited_e = dataset[dataset["business_id"].isin(limits_set_e)].groupby("business_id").head(5)
    #df_limited_u = dataset[dataset["business_id"].isin(limits_set_u)].groupby("business_id").head(5)

    df_keep_f = dataset[dataset["business_id"].isin(keep_id_set_f)]
    df_keep_e = dataset[dataset["business_id"].isin(keep_id_set_e)]
    #df_keep_u = dataset[dataset["business_id"].isin(keep_id_set_u)]

    #df_all_f = dataset[dataset["business_id"].isin(business_f)]
    #df_all_e = dataset[dataset["business_id"].isin(business_e)]
    df_all_u = dataset[dataset["business_id"].isin(business_u)]


    df_result = pd.concat([df_all_u, df_keep_f, df_limited_f, df_keep_e, df_limited_e])
    df_result.to_json(f_dataset_path, orient = "records", lines = True)

def datasetIDsMap():
    f_dataset_path = dataset_path + "yelp_dataset.json"
    with open(f_dataset_path, "r") as f_dataset:
        dataset = pd.read_json(f_dataset, lines=True)
    business_ids = dataset["business_id"].unique()
    bus_id_invmp = {id_: i for i, id_ in enumerate(business_ids)}

    for i_str in clients_list:
        f_sub_dataset_path = dataset_path + "yelp_dataset_" + i_str + ".json"
        with open(f_sub_dataset_path, "r") as f_sub_dataset:
            sub_dataset = pd.read_json(f_sub_dataset, lines=True)
        sub_dataset["business_id"].replace(bus_id_invmp, inplace=True)
        sub_dataset.reset_index(drop = True, inplace = True)
        sub_dataset.to_json(f_sub_dataset_path, orient = "records", lines = True)
    dataset["business_id"].replace(bus_id_invmp, inplace=True)
    dataset.reset_index(drop = True, inplace = True)
    dataset.to_json(f_dataset_path, orient = "records", lines = True)



if __name__ == '__main__':
    #splitDataset()
    #changeCategoryDict()
    #splitPlatform()
    #checkPlatform()
    #generatePlatformDataset()
    #checkDataset()
    #changeItemScoreAverage()
    #generateUserWeights()
    #changeItemScoreWeighted()
    #changeItemFunction()
    #changeItemClass()
    #generateNewDataset()
    #checkNewDataset()
    #generateNewPlatform()
    #checkNewPlatform()
    #deleteReview()
    checkNewPlatformAfterDelete()
    #datasetIDsMap()