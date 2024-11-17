import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pickle
from torch.utils.data import random_split


class MyDataset(Dataset):
    def __init__(self, seq_list):
        self.x = []
        self.y = []
        for dic in seq_list:
            self.y.append(dic['key'])
            self.x.append(dic['value'])

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        return torch.Tensor(np.array(data)), torch.tensor(np.array(int(label)))
        # return torch.Tensor(np.array(data)), torch.tensor(np.array(int(label)), dtype=torch.long)

    # RuntimeError: expected scalar type Long but found Int -> 이 오류때문에 dtype 변경

    def __len__(self):
        return len(self.x)

if __name__ == "__main__":
# 저장된 데이터셋 파일에서 불러오기
    with open('test.pkl', 'rb') as f:
        dataset = pickle.load(f)
    print(len(dataset))

    split_ratio = [0.8, 0.1, 0.1]
    train_len = int(len(dataset) * split_ratio[0])
    val_len = int(len(dataset) * split_ratio[1])
    test_len = len(dataset) - train_len - val_len
    print('{}, {}, {}'.format(train_len, val_len, test_len))


    train_dataset = MyDataset(dataset)
    train_data, valid_data, test_data = random_split(train_dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_data, batch_size=8)
    val_loader = DataLoader(valid_data, batch_size=8)
    test_loader = DataLoader(test_data, batch_size=8)
