from vncorenlp import VnCoreNLP
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
from preprocess import preprocess_fn
import pandas as pd
from dataset_vncorenlp import Data


from tqdm import tqdm

# class BPEConfig:
#     def __init__(self, bpe_codes):
#         self.bpe_codes = bpe_codes


# rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar",
#                          annotators="wseg", max_heap_size='-Xmx500m')

# text = "Đại học Bách Khoa Hà Nội."

# word_segmented_text = rdrsegmenter.tokenize(text)
# # print(word_segmented_text)
# cfg = BPEConfig(bpe_codes='sentiment/bpe.codes')
# bpe = fastBPE(cfg)

# vocab = Dictionary()
# vocab.add_from_file('sentiment/vocab.txt')

# print(bpe.encode('Hôm_nay trời nóng quá nên tôi ở nhà viết Viblo!'))
# print(vocab.encode_line(
#     '<s> ' + 'Hôm_nay trời nóng quá nên tôi ở nhà viết Vi@@ blo@@ !' + ' </s>'))

# data_train = pd.read_csv('sentiment/relabel/all/vn/train_sentiment.csv')
# train_text = []
# text = data_train['Review'][0]
# text = preprocess_fn(text, preprocess=False)
# print(text)
# text = rdrsegmenter.tokenize(text)
# text = ' '.join([' '.join(x) for x in text])
# print(text)
# subwords = '<s> ' + bpe.encode(text) + ' </s>'
# encoded_sent = vocab.encode_line(
#     subwords, append_eos=True, add_if_not_exist=False).long().tolist()
# print(encoded_sent)

# print(torch.cuda.is_available())
# MAX_LEN = 256
# train_ids = pad_sequences([encoded_sent], maxlen=MAX_LEN,
#                           dtype="long", value=0, truncating="post", padding="post")
# print(train_ids)
# train_masks = []
# for sent in train_ids:
#     mask = [int(token_id > 0) for token_id in sent]
#     train_masks.append(mask)
# print(train_masks)


def convert_vncore(text, rdrsegmenter, bpe, vocab):
    text = preprocess_fn(text, preprocess=False)
    text = rdrsegmenter.tokenize(text)
    text = ' '.join([' '.join(x) for x in text])
    subwords = '<s> ' + bpe.encode(text) + ' </s>'
    encoded_sent = vocab.encode_line(
        subwords, append_eos=True, add_if_not_exist=False).long().tolist()
    return encoded_sent


# X = data_train['Review'].apply(
#     lambda x: convert_vncore(x, rdrsegmenter, bpe, vocab))
# # print(X.values)
# t = torch.tensor(pad_sequences([X.values[0]], maxlen=256,
#                                dtype="long", value=0, truncating="post", padding="post"), dtype=torch.long).squeeze(0)
# print(t)
# print(pad_sequences([X.values[0]], maxlen=256,
#                     dtype="long", value=0, truncating="post", padding="post"))


# x = Data()
# train_loader, len_train_data = x.getBatchDataTrain()
# for data in tqdm(train_loader):
#     print(data['input_ids'], data['attention_mask'], data['labels'])
#     break
