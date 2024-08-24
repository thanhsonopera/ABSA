from vncorenlp import VnCoreNLP
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import fastBPE


class BPEConfig:
    def __init__(self, bpe_codes):
        self.bpe_codes = bpe_codes
# rdrsegmenter = VnCoreNLP("../vncorenlp/VnCoreNLP-1.1.1.jar",
#                          annotators="wseg", max_heap_size='-Xmx500m')

# text = "Đại học Bách Khoa Hà Nội."


# word_segmented_text = rdrsegmenter.tokenize(text)
# print(word_segmented_text)
cfg = BPEConfig(bpe_codes='sentiment/bpe.codes')
bpe = fastBPE(cfg)

# vocab = Dictionary()
# vocab.add_from_file('vocab.txt')
