
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch

from torch import cuda
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from model import CrossModalBERT

from dataloader import BaseDataset, EvaluateDataset


def evaluate(model, test_dataloader, tokenizer, window_size):
    recall = 0
    model.eval()
    index_corrects = np.repeat(np.arange(0, test_dataloader.dataset.__len__() / 5), 5).reshape(
        test_dataloader.dataset.__len__(), 1)
    with torch.no_grad():
        classname = {0: 'Irrelevant', 1: 'Relevant'}
        correct_pred = defaultdict(lambda: 0)
        total_pred = defaultdict(lambda: 0)

        for inputs in test_dataloader:
            y_pred = []

            expert = inputs[0]["expert"]
            data = tokenizer(
                inputs[0]["captions"],
                truncation=True,
                return_tensors="pt",
                max_length=150,
                padding='max_length')
            index_true = inputs[0]["label_index"]

            labels = inputs[0]["label"]

            if labels is not None and index_true < window_size:
                labels = torch.tensor(
                    labels,
                    dtype=torch.long)

                input_ids = data["input_ids"]
                input_ids = input_ids.repeat(expert.shape[0], 1)

                attention_mask = data["attention_mask"]
                attention_mask = attention_mask.repeat(expert.shape[0], 1)

                torch_zeros = torch.zeros((window_size, 1, 145, 145))
                torch_zeros[:, 0, :128, :] = expert[:window_size].reshape((window_size, 128, 145))
                for minibatch in range(int(window_size / 100)):
                    range_index = list(range(100 * minibatch, 100 * (minibatch + 1)))
                    target = labels[range_index]

                    output = model(
                        input_ids[range_index],
                        attention_mask[range_index],
                        torch_zeros[range_index])

                    y_pred += list(output[:, 1].to('cpu'))

                top_10_rerank = np.argsort(y_pred)[:10] == index_true
                recall_consult = top_10_rerank.sum()
            else:
                recall_consult = 0

            recall += recall_consult / test_dataloader.dataset.__len__()
        return recall



def main(n_epochs):
    window_size = 300

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load data (train/test)
    evaluate_dataset = EvaluateDataset("AA")
    evaluate_dataloader = DataLoader(
        dataset=evaluate_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        collate_fn=evaluate_dataset.collate_data,
    )
    train_dataset = BaseDataset("AA")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        collate_fn=dl.collate_data,
    )


    #load model
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    loss = torch.nn.CrossEntropyLoss()

    model = CrossModalBERT()
    model.to('cpu')
    model.train()
    for epoch in range(n_epochs):

        for step, batch in enumerate(train_dataloader):
            expert = batch[0]["expert"]
            data = tokenizer(
                batch[0]["captions"],
                truncation=True,
                return_tensors="pt",
                max_length=150,
                padding='max_length')
            index_true = batch[0]["label_index"]
            labels = batch[0]["label"]
            if labels is not None and index_true < window_size:
                labels = torch.tensor(
                    labels,
                    dtype=torch.long)

                input_ids = data["input_ids"]
                input_ids = input_ids.repeat(expert.shape[0], 1)

                attention_mask = data["attention_mask"]
                attention_mask = attention_mask.repeat(expert.shape[0], 1)

                torch_zeros = torch.zeros((window_size, 1, 145, 145))
                torch_zeros[:, 0, :128, :] = expert[:window_size].reshape((window_size, 128, 145))
                for minibatch in range(int(window_size / 50)):

                    range_index = list(range(50 * (minibatch), 50 * (minibatch + 1)))
                    range_index.append(index_true)

                    target = labels[range_index]

                    output = model(
                        input_ids[range_index],
                        attention_mask[range_index],
                        torch_zeros[range_index])

                    optimizer.zero_grad()
                    l = loss(output, target)
                    l.backward()
                    optimizer.step()

                    if step % 10 == 0:
                        print(f'Epoch: {epoch}, {step}/{len(dataloader)}, Loss:  {l.item()}')

            recall = evaluate(model, evaluate_dataloader, tokenizer, 100)
            print(f"Epoch:{epoch}, evaluate recall: {recall}")

if __name__ == "__main__":
    n_epochs = 100
    main(n_epochs)