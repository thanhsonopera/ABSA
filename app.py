from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE
from vncorenlp import VnCoreNLP
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from model import BertClassifierVer3
from sentiment.model_sentiment import SentimentClassifier
from sentiment.evaluate import PolarityMapping
from flask_cors import CORS
from transformers import AutoTokenizer
import torch
from preprocess import preprocess_fn
app = Flask(__name__)
CORS(app)


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


@app.before_request
def load_model():
    rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar",
                             annotators="wseg", max_heap_size='-Xmx500m')
    cfg = BPEConfig(bpe_codes='sentiment/bpe.codes')
    bpe = fastBPE(cfg)

    vocab = Dictionary()
    vocab.add_from_file('sentiment/vocab.txt')
    text = 'Hello world'
    a_t = convert_vncore(text, rdrsegmenter, bpe, vocab)
    a_t = pad_sequences([a_t], maxlen=256, dtype="long",
                        value=0, truncating="post", padding="post")
    print(a_t)
    global model_for_aspects, tokenizer_for_aspects, config_for_aspects, key, config_for_sentiment, model_for_sentiment, tokenizer_for_sentiment

    key = ['AMBIENCE', 'QUALITY', 'PRICES', 'LOCATION', 'SERVICE']
    seed = 32
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config_for_aspects = {
        'name_model': 'uitnlp/visobert',
        'num_classes': 5,
        'drop_rate': 0.5,
        'device': 'cpu'
    }
    model_for_aspects = BertClassifierVer3(
        config_for_aspects).to(config_for_aspects['device'])

    model_for_aspects.load_state_dict(torch.load(
        'checkpoint/cp6/model.pth', weights_only=False, map_location=torch.device(config_for_aspects['device'])))

    tokenizer_for_aspects = AutoTokenizer.from_pretrained(
        config_for_aspects['name_model'])

    model_for_aspects.eval()

    config_for_sentiment = {
        'name_model': 'vinai/phobert-base',
        'num_classes': 5,
        'drop_rate': [0.2, 0.3],
        'device': 'cpu',
        'losses': 2,
        'model': 2,
        'layer_norm': True
    }
    model_for_sentiment = SentimentClassifier(config_for_sentiment).to(
        config_for_sentiment['device'])
    model_for_sentiment.load_state_dict(torch.load(
        'sentiment/checkpoint/good/9/model.pth', weights_only=False, map_location=torch.device(config_for_sentiment['device'])))
    tokenizer_for_sentiment = AutoTokenizer.from_pretrained(
        config_for_sentiment['name_model'])
    model_for_sentiment.eval()


@app.route('/')
def home():
    return "Hello world!"


@app.route('/predict/aspects', methods=['POST'])
def predict_aspects():

    try:
        data = request.get_json()
        value = []
        input_ids = None
        attention_masks = None

        for k in data:
            text = preprocess_fn(data[k])
            encoding = tokenizer_for_aspects.encode_plus(
                text,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            if (input_ids is None):
                input_ids = encoding["input_ids"].to(
                    config_for_aspects['device'])
            else:
                input_ids = torch.cat(
                    (input_ids, encoding["input_ids"].to(config_for_aspects['device'])), 0)

            if (attention_masks is None):
                attention_masks = encoding["attention_mask"].to(
                    config_for_aspects['device'])
            else:
                attention_masks = torch.cat(
                    (attention_masks, encoding["attention_mask"].to(config_for_aspects['device'])), 0)

        with torch.no_grad():
            outputs = model_for_aspects(input_ids, attention_masks)
            outputs = torch.sigmoid(outputs).tolist()

            # outputs = [int(torch.sigmoid(output).item() > 0.5)
            #            for output in outputs]

        for v in outputs:
            value.append([key[i] for i in range(len(v)) if v[i] > 0.5])

        return jsonify({'result': value}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/sentiment', methods=['POST'])
def predict_sentiment():
    try:
        data = request.get_json()
        value = []
        for k in data:
            text = preprocess_fn(data[k])
            value.append(text)
        encode_result = tokenizer_for_sentiment.batch_encode_plus(value, max_length=256, padding="max_length",
                                                                  truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model_for_sentiment(encode_result['input_ids'].to(
                config_for_sentiment['device']), encode_result['token_type_ids'].to(config_for_sentiment['device']), encode_result['attention_mask'].to(config_for_sentiment['device']))
            if config_for_sentiment['losses'] == 2:
                outputs = [torch.nn.Softmax(dim=1)(outputs[i].squeeze(-1))
                           for i in range(config_for_sentiment['num_classes'])]

            predict = [torch.max(outputs[i].squeeze(-1), dim=1).indices
                       for i in range(config_for_sentiment['num_classes'])]
        result = []
        for j in range(len(value)):
            for i in range(config_for_sentiment['num_classes']):
                if predict[i][j].item() != 0:
                    result.append(
                        {key[i]: PolarityMapping.INDEX_TO_POLARITY[predict[i][j].item()]})

        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
