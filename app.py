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
import pandas as pd
import numpy as np
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


def exportCsv(data):
    number_comment = 0
    columns = ['user_name', 'user_timec', 'user_rating',
               'user_titlec', 'user_comment', 'nameUserLikeText', 'numberCmtOfPost']
    comment_csv = []
    comment_csv.append(columns)

    for comment_user in data:
        row = []
        for i, key in enumerate(comment_user.keys()):
            if (i <= 1) or (i >= 7 and i <= 11):
                continue

            if key == 'user_CmtOfCmt':
                for c_key in comment_user[key].keys():
                    if c_key == 'allCmtInPost':
                        continue
                    if comment_user[key][c_key] == '':
                        row.append(np.nan)
                    else:
                        row.append(comment_user[key][c_key])
            else:
                row.append(comment_user[key])
        if i < 8:
            row.append(np.nan)
            row.append(0)
        number_comment += 1
        comment_csv.append(row)

    if len(comment_csv) > 1:
        df = pd.DataFrame(data=comment_csv[1:], columns=comment_csv[0])
        df['user_timec'] = pd.to_datetime(df['user_timec'])

        # df_sort_date = df.sort_values(by='user_timec')
        max_rating = df[df['user_rating'] == 10.0]

        df_sort_name = max_rating.sort_values(by=['user_name', 'user_timec'])

        # Tìm các tên trùng nhau có ngày cách nhau dưới 2
        duplicate_names = df_sort_name[df_sort_name.duplicated(
            subset=['user_name'], keep=False)]

        # Tính khoảng cách giữa các lần xuất hiện của cùng một tên
        duplicate_names['time_diff'] = duplicate_names.groupby(
            'user_name')['user_timec'].diff().dt.days.abs()

        # Lọc ra các bản ghi có ngày cách nhau dưới 2
        filter_seeder = duplicate_names[duplicate_names['time_diff'] < 2]

        filter_seeder = filter_seeder.drop(columns=['time_diff'])

        df_remove_seeder = df[~df['user_name'].isin(
            filter_seeder['user_name'])]

        number_comment_seeder = df.shape[0] - df_remove_seeder.shape[0]
        # print(df_remove_seeder, number_comment_seeder)

        ####
        max_rating = df_remove_seeder[df_remove_seeder['user_rating'] == 10.0]
        filter_comment_min15 = max_rating[max_rating['user_comment'].str.len(
        ) <= 15]
        df_remove_comment_min15 = df_remove_seeder[~df_remove_seeder['user_name'].isin(
            filter_comment_min15['user_name'])]

        number_comment_min15 = df_remove_seeder.shape[0] - \
            df_remove_comment_min15.shape[0]

        # print(df_remove_comment_min15, number_comment_min15)

        ####
        max_rating = df_remove_comment_min15[df_remove_comment_min15['user_rating'] == 10.0]
        filter_title_as_comment = max_rating[max_rating['user_titlec']
                                             == max_rating['user_comment']]
        df_remove_title_same_comment = df_remove_comment_min15[
            ~df_remove_comment_min15['user_name'].isin(filter_title_as_comment['user_name'])]

        number_title_same_comment = df_remove_comment_min15.shape[0] - \
            df_remove_title_same_comment.shape[0]

        # print(filter_title_as_comment, df_remove_title_same_comment,
        #       number_title_same_comment)

        total_comment_remove = number_comment_seeder + \
            number_comment_min15 + number_title_same_comment

        # print(total_comment_remove)

        new_data = []
        user_name = df_remove_title_same_comment['user_name'].unique()

        for comment_user in data:
            if comment_user['user_name'] in user_name:
                new_data.append(comment_user)
        return new_data, number_comment_seeder, number_comment_min15, number_title_same_comment, total_comment_remove

    return [], 0, 0, 0, 0


@app.before_request
def load_model():
    rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar",
                             annotators="wseg", max_heap_size='-Xmx500m')
    cfg = BPEConfig(bpe_codes='sentiment/bpe.codes')
    bpe = fastBPE(cfg)

    vocab = Dictionary()
    vocab.add_from_file('sentiment/vocab.txt')
    # text = 'Hello world'
    # a_t = convert_vncore(text, rdrsegmenter, bpe, vocab)
    # a_t = pad_sequences([a_t], maxlen=256, dtype="long",
    #                     value=0, truncating="post", padding="post")
    # print(a_t)
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


@app.route('/normalize', methods=['POST'])
def normalize():
    try:
        data = request.get_json()
        new_data, number_comment_seeder, number_comment_min15, number_title_same_comment, total_comment_remove = exportCsv(
            data)
        return jsonify({'newdata': new_data,
                        'number_comment_seeder': number_comment_seeder,
                        'number_comment_min15': number_comment_min15,
                        'number_title_same_comment': number_title_same_comment,
                        'total_comment_remove': total_comment_remove}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
