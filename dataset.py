from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from preprocess import preprocess_fn
# from pandas_profiling import ProfileReport


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
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


class Data:
    def __init__(self, type='Restaurant', batch_size=32, tokenizer=None, max_length=512, key=['Review', 'AMBIENCE', 'QUALITY', 'PRICES', 'LOCATION', 'SERVICE']):

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        if (type == 'Restaurant'):
            self.dataTrain = pd.read_csv(
                'relabel/train_merge.csv')
            self.dataVal = pd.read_csv(
                'relabel/dev_merge.csv')
            self.dataTest = pd.read_csv(
                'relabel/test_merge.csv')
        elif (type == 'Hotel'):
            self.dataTrain = pd.read_csv(r'relabel\hotel\Hotel-train.csv')
            self.dataVal = pd.read_csv(r'relabel\hotel\Hotel-dev.csv')
            self.dataTest = pd.read_csv(r'relabel\hotel\Hotel-test.csv')

        # profile = ProfileReport(
        #     self.dataTrain, title="Profiling Report", explorative=True)
        # profile.to_file("x_train_profiling.html")

        self.key = key

    def getBatchDataTrain(self):
        X_train = self.dataTrain['Review'].apply(preprocess_fn)
        y_train = self.dataTrain[self.key[1:]]
        train_dataset = TextDataset(
            texts=X_train.values,
            labels=y_train.values,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True), len(train_dataset)

    def getBatchDataVal(self):
        X_val = self.dataVal['Review'].apply(preprocess_fn)
        y_val = self.dataVal[self.key[1:]]
        val_dataset = TextDataset(
            texts=X_val.values,
            labels=y_val.values,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True), len(val_dataset)

    def getBatchDataTest(self):
        X_test = self.dataTest['Review'].apply(preprocess_fn)
        y_test = self.dataTest[self.key[1:]]
        test_dataset = TextDataset(
            texts=X_test.values,
            labels=y_test.values,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        return DataLoader(test_dataset, batch_size=self.batch_size * 2, shuffle=True), len(test_dataset)

    def getStrData(self, str):
        str = preprocess_fn(str)
        str_dataset = TextDataset(
            texts=[str],
            labels=[0],
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        return DataLoader(str_dataset, batch_size=1, shuffle=False), len(str_dataset)
