import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

def pkload(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class adni2(Dataset):
    def __init__(self, data, split, mode='train'):
        # 路径拼接
        data_path = os.path.join('Data', data + '_5split_' + mode + '_' + str(split) + '_data.npy')
        label_path = os.path.join('Data', data + '_5split_' + mode + '_' + str(split) + '_label.pkl')

        self.mode = mode
        if os.path.exists(label_path) and os.path.exists(data_path):
            self.names, self.labels = pkload(label_path)
            self.datas = np.load(data_path)
            if mode == 'train':
                print(f"[{mode}] Loaded data shape: {self.datas.shape}")
        else:
            # 假数据模式 (防止没有文件时报错，方便调试代码逻辑)
            print(f"[Warning] Data not found at {data_path}. Using DUMMY data for debugging!")
            self.names = [str(i) for i in range(100)]
            self.labels = np.random.randint(0, 2, 100)
            self.datas = np.random.randn(100, 1, 128, 90, 1)

    def __getitem__(self, item):
        label = torch.tensor(self.labels[item]).long()
        raw_data = self.datas[item]
        data = torch.from_numpy(raw_data).float()

        # 适配形状 -> (90, 128, 1)
        if len(data.shape) == 4 and data.shape[0] == 1:
            data = data.squeeze(0)  # (128, 90, 1)

        if data.shape[0] == 128 and data.shape[1] == 90:
            data = data.permute(1, 0, 2)  # (90, 128, 1)

        return data, label

    def __len__(self):
        return len(self.names)

    def get_num_class(self):
        return len(np.unique(self.labels))