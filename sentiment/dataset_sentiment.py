from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from preprocess import preprocess_fn
import yaml
# from pandas_profiling import ProfileReport


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if 'token_type_ids' in encoding:
            # token_type_ids
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
            token_type_ids = encoding["token_type_ids"].squeeze()

            return {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "labels": torch.tensor(label),
            }

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label),
        }


class Data:
    def __init__(self, tokenizer=None, batch_size=16, max_length=256):

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        with open('dataset.yaml', 'r') as file:
            cfs = yaml.safe_load(file)

        self.dataTrain = pd.read_csv(cfs['train'])
        self.dataVal = pd.read_csv(cfs['val'])
        self.dataTest = pd.read_csv(cfs['test'])

        self.key = cfs['key']

    def getBatchDataTrain(self):
        X_train = self.dataTrain['Review'].apply(
            lambda review: preprocess_fn(review, preprocess=True))
        y_train = self.dataTrain[self.key[1:]]
        train_dataset = TextDataset(
            texts=X_train.values,
            labels=y_train.values,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )

        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True), len(train_dataset)

    def getBatchDataVal(self):
        X_val = self.dataVal['Review'].apply(
            lambda review: preprocess_fn(review, preprocess=True))
        y_val = self.dataVal[self.key[1:]]
        val_dataset = TextDataset(
            texts=X_val.values,
            labels=y_val.values,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )

        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True), len(val_dataset)

    def getBatchDataTest(self):
        X_test = self.dataTest['Review'].apply(
            lambda review: preprocess_fn(review, preprocess=True))
        y_test = self.dataTest[self.key[1:]]
        test_dataset = TextDataset(
            texts=X_test.values,
            labels=y_test.values,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True), len(test_dataset)

    def getStrData(self, str):
        str = preprocess_fn(str, preprocess=True)
        str_dataset = TextDataset(
            texts=[str],
            labels=[0],
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        return DataLoader(str_dataset, batch_size=1, shuffle=False), len(str_dataset)
