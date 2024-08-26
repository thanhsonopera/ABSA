from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from preprocess import preprocess_fn
import yaml
from vncorenlp import VnCoreNLP
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextDataset(Dataset):
    def __init__(self, texts, labels, max_length=256):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_padding = pad_sequences([text], maxlen=self.max_length,
                                     dtype="long", value=0, truncating="post", padding="post")[0]
        input_ids = torch.tensor(text_padding, dtype=torch.long)
        mask = [int(token_id > 0) for token_id in text_padding]
        attention_mask = torch.tensor(mask, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label),
        }


class BPEConfig:
    def __init__(self, bpe_codes):
        self.bpe_codes = bpe_codes


def convert_vncore(text, rdrsegmenter, bpe, vocab):
    text = preprocess_fn(text, preprocess=False)
    text = rdrsegmenter.tokenize(text)
    text = ' '.join([' '.join(x) for x in text])
    subwords = '<s> ' + bpe.encode(text) + ' </s>'
    encoded_sent = vocab.encode_line(
        subwords, append_eos=True, add_if_not_exist=False).long().tolist()
    return encoded_sent


class DataVNCoreNLP:
    def __init__(self, batch_size=16, max_length=256):

        self.batch_size = batch_size
        self.max_length = max_length
        with open('sentiment/dataset.yaml', 'r') as file:
            cfs = yaml.safe_load(file)

        self.dataTrain = pd.read_csv('sentiment/' + cfs['train'])
        self.dataVal = pd.read_csv('sentiment/' + cfs['val'])
        self.dataTest = pd.read_csv('sentiment/' + cfs['test'])

        self.key = cfs['key']
        self.rdrsegmenter = VnCoreNLP(
            "vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

        cfg = BPEConfig(bpe_codes='sentiment/bpe.codes')

        self.bpe = fastBPE(cfg)
        self.vocab = Dictionary()
        self.vocab.add_from_file('sentiment/vocab.txt')

    def getBatchDataTrain(self):
        X_train = self.dataTrain['Review'].apply(lambda review: convert_vncore(
            review, self.rdrsegmenter, self.bpe, self.vocab))

        y_train = self.dataTrain[self.key[1:]]

        train_dataset = TextDataset(
            texts=X_train.values,
            labels=y_train.values,
            max_length=self.max_length
        )

        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True), len(train_dataset)

    def getBatchDataVal(self):
        X_val = self.dataVal['Review'].apply(lambda review: convert_vncore(
            review, self.rdrsegmenter, self.bpe, self.vocab))
        y_val = self.dataVal[self.key[1:]]
        val_dataset = TextDataset(
            texts=X_val.values,
            labels=y_val.values,
            max_length=self.max_length
        )

        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True), len(val_dataset)

    def getBatchDataTest(self):
        X_test = self.dataTest['Review'].apply(lambda review: convert_vncore(
            review, self.rdrsegmenter, self.bpe, self.vocab))
        y_test = self.dataTest[self.key[1:]]
        test_dataset = TextDataset(
            texts=X_test.values,
            labels=y_test.values,
            max_length=self.max_length
        )
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True), len(test_dataset)

    def getStrData(self, str):
        str = convert_vncore(
            str, self.rdrsegmenter, self.bpe, self.vocab)
        str_dataset = TextDataset(
            texts=[str],
            labels=[0],
            max_length=self.max_length
        )
        return DataLoader(str_dataset, batch_size=1, shuffle=False), len(str_dataset)
