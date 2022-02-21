import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle as pkl
import numpy as np
import hickle as hkl
import pandas as pd
import torch
import math

class BaseDataset(Dataset):
    def __init__(self, path):
        logger.info(f"Load train indices")
        self.indices = pkl.load(open(f"../audio-retrieval/data/AudioCaps/structured-symlinks/train_index_search.pkl", "rb"))
        self.num_samples = self.indices.shape[0]

        self.train_emb = []
        for i in range(10):
            logger.info(f"Load train dataset {i}")
            try:
                self.train_emb.append(hkl.load(f"../audio-retrieval/data/AudioCaps/structured-symlinks/final_data/train_{i}.hkl"))

            except:
                self.train_emb.append(
                    pkl.load(open(f"../audio-retrieval/data/AudioCaps/structured-symlinks/final_data/train_{i}.pkl", "rb")))
        self.index_corrects = np.array(range(self.num_samples)).reshape(self.num_samples, 1)
        self.values = self.indices == self.index_corrects
        self.sum_values = self.values.sum(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < self.num_samples:

            rank_pos = []
            experts = []
            captions = []
            query_index = idx
            exist = self.sum_values[idx]
            ind_df = np.int(idx / 5000)
            ind_doc = np.int(idx % 5000)
            captions.append(self.train_emb[ind_df]["raw_captions"][ind_doc])

            for pos, index in enumerate(self.indices[idx]):
                ind_df = np.int(index / 5000)
                ind_doc = np.int(index % 5000)

                experts.append(self.train_emb[ind_df]["raw_experts"][ind_doc].cpu().detach().numpy())
                rank_pos.append(pos + 1)

            experts = torch.from_numpy(np.array(experts))

            index_label = np.argwhere(self.indices[idx] == idx)
            if index_label.size == 0:
                index_label = None
                label = None

            else:
                index_label = index_label[0][0]
                label = np.zeros(len(rank_pos))
                label[index_label] = 1

            result = {
                "query_index": idx,
                "expert": experts,
                "captions": captions,
                "label_index": index_label,
                "label": label,
            }
            return result

    def collate_data(self, data):
        return data


class EvaluateDataset(Dataset):
    def __init__(self, path):
        logger.info(f"Load test indices")
        self.indices = pkl.load(open(f"../audio-retrieval/data/AudioCaps/structured-symlinks/test_index_search.pkl", "rb"))
        self.num_samples = self.indices.shape[0]

        logger.info(f"Load test dataset")

        self.test_emb = hkl.load("../audio-retrieval/data/AudioCaps/structured-symlinks/final_data/test.hkl")

        self.index_corrects_test = np.array(range(self.num_samples)).reshape(self.num_samples, 1)
        self.values = self.num_samples == self.index_corrects_test
        self.sum_values = self.values.sum(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < self.num_samples:

            rank_pos = []
            experts = []
            captions = []
            query_index = idx
            exist = self.sum_values[idx]
            captions.append(self.test_emb["raw_captions"][idx])

            for pos, index in enumerate(self.indices[idx]):

                experts.append(self.test_emb["raw_experts"][index].cpu().detach().numpy())
                rank_pos.append(pos + 1)

            experts = torch.from_numpy(np.array(experts))

            index_label = np.argwhere(self.indices[idx] == math.floor(idx/5))
            if index_label.size == 0:
                index_label = None
                label = None

            else:
                index_label = index_label[0][0]
                label = np.zeros(len(rank_pos))
                label[index_label] = 1

            result = {
                "query_index": idx,
                "expert": experts,
                "captions": captions,
                "label_index": index_label,
                "label": label,
            }
            return result

    def collate_data(self, data):
        return data

