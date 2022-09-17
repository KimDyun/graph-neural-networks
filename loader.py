from scipy import io
import os
import pandas as pd
import numpy as np
import random

import torch
from torch_geometric.data import Data, DataLoader

# EEG_band : delta, theta, alpha, beta, gamma, all = 1,2,3,4,5,None
# Feature_name : de_LDS, psd_LDS, etc.
def load_data(data_dir_path, label_file_path, label_var_name, trial, feature_name, EEG_band=None):
    subject_data_list = []
    file_list = os.listdir(data_dir_path)

    for idx, file in enumerate(file_list):
        trial_list = []
        data = io.loadmat(data_dir_path + file)

        if EEG_band is None:
            for trial_idx in range(1, trial + 1):
                trial_list.append(data[feature_name + str(trial_idx)][:, :, :])  # EEG all bands

        else:
            for trial_idx in range(1, trial + 1):
                trial_list.append(data[feature_name + str(trial_idx)][:, :, EEG_band])

        session_list.append(trial_list)

        if (idx + 1) % 3 == 0:
            subject_data_list.append(session_list)
            session_list = []

    label = io.loadmat(label_file_path)[label_var_name]
    label = [1 + label[i] for i in range(trial)]
    # label {negative : -1 , neutral : 0, positive : 1} --> [-1, 1] --> [0, 2]

    # subject_data_list : num_subjects(15) × num_sessions(3) × num_trials(15)
    return subject_data_list, label


# The edge weights are computed by using gaussian kernel function based on the distance between all pairs of EEG channels
# edge attributes are composed of edge weights.
def load_edge_information(channels_dir_path, file_name, theta, tau):
    def gaussian_kernel_edge_weight(i, j, theta, tau):
        gaussian_distance = 0
        for k in range(1, 4):
            gaussian_distance += np.linalg.norm(i[k] - j[k]) ** 2

        edge_weight = 0
        if gaussian_distance <= tau ** 2:
            edge_weight = np.exp(-gaussian_distance / ((theta ** 2) * 2))

        return edge_weight

    edge_index_list, edge_attr_list = [], []

    channel_file = os.path.join(channels_dir_path, file_name)
    lines = pd.read_excel(channel_file, sheet_name='Sheet1')
    channel_placement = lines.values[3:-6]
    channel_num = len(channel_file)

    for i in range(channel_num):
        for j in range(channel_num):
            edge_index_list.append([i, j])
            edge_attr = gaussian_kernel_edge_weight(channel_placement[i], channel_placement[j], theta, tau)
            if i != j:
                edge_attr_list.append(edge_attr)
            else:
                edge_attr_list.append(0)

    return edge_index_list, edge_attr_list


# Graph Representation
def GraphDataLoader(subject_data, subject_label, edge_index_list, edge_attr_list, num_train_trials , batch_size):
    edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    train_loader, test_loader = [], []

    num_subjects = len(subject_data)
    num_sessions = len(subject_data[0])
    num_trials = len(subject_data[0][0])

    for subject in range(num_subjects):
        data_list, train_dataset, test_dataset = [], [], []
        for session in range(num_sessions):
            for trial in range(num_trials):
                trial_data = subject_data[subject][session][trial] # trial_data = [62][about 240][1or5] = [nodes][trial time][EEG band(s)]
                blocks = len(trial_data[1]) # for about 240(240 sec, 4minutes), the number of blocks is the same as trial time
                for block_idx in range(blocks):
                    # if using all frequency bands,
                    # node_feature = [delta, theta, alpha, beta, gamma]
                    data_sample = torch.tensor(trial_data[:, block_idx, :], dtype=torch.float)#data_sample = [62][5] = [nodes, node_features(EEG band(s)]
                    data_label = torch.tensor(subject_label[0][trial], dtype=torch.long) # data_label [0,2]

                    data_list.append(Data(x=data_sample, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=data_label))

                if trial < num_train_trials: train_dataset.extend(data_list)
                else: test_dataset.extend(data_list)

        random.shuffle(train_dataset)
        random.shuffle(test_dataset)

        batch_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        batch_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_loader.append(batch_train_loader)
        test_loader.append(batch_test_loader)

    return train_loader, test_loader