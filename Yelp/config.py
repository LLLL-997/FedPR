# Training settings

batch_size = 8192
lr = 1e-3
epochs = 20
gru_epochs = 10

Seed = 42


# Hyperparameters
n_emd = 50
n_hidden = 32
n_out = 50
dense_hidden1 = 64
dense_hidden2 = 32
dense_hidden3 = 16
dense_out = 3
multi_processing = False
window_size = 5

in_channels = 1
out_channels = 3

w_out = 50

func_Categories = [" ", "Foods", "Entertainments", "Function"]
func_Categories_str = "function_categories"

# Path settings
save_path = "../data/words_embed.txt"
glove_path = "../data/glove.6B.50d.txt"
#state_str = "CA"
#state_str = "INN"
state_str = "FL"
#state_str = "NJ"
distance_matrix_path = "../data/yelp/state/" + state_str + "/distance_matrix.csv"
dataset_path = "../data/yelp/state/" + state_str + "/"
original_path = "../data/yelp/"
model_save_path = "../Yelp/model/" + state_str + "/"
platform_str = "GroundTruth"
#platform_str = "C"
# Cut length
cut_len = 20
gt_train_mask_threshold = 10
gt_test_mask_threshold = 5

platform_train_mask_threshold = 10
platform_test_mask_threshold = 5

heads = 8

# Graph
neighbor_num = 10
preference_neighbor_num = 5

# Federated Settings
#participants_num = 4
participants_num = 3
communication_epochs = 20
public_training_epochs = 2
#clients_list = ["A", "B", "C", "D"]
clients_list = ["A", "B", "C"]
#private_dataset_list = ["A", "B", "C", "D"]
private_dataset_list = ["A", "B", "C"]
private_training_classes = [0, 1, 2]
# clients_target = {"A": [0], "B":[0], "C":[0, 1], "D": [2]}
clients_target = {"A": [0], "B":[1], "C":[2]}

public_dataset_name = "public"