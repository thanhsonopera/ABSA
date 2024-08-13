from flask import Flask, request, jsonify
from model import BertClassifier, BertClassifierVer3
from transformers import AutoTokenizer
import torch
from preprocess import preprocess_fn
app = Flask(__name__)


@app.before_request
def load_model():
    # config = {
    #     'name_model': 'vinai/phobert-base',
    #     'num_classes': 5,
    #     'drop_rate': 0.2,
    #     'device': 'cpu'
    # }

    global model, tokenizer, key, config
    config = {
        'name_model': 'uitnlp/visobert',
        'num_classes': 5,
        'drop_rate': 0.5,
        'device': 'cuda'
    }
    model = BertClassifierVer3(config).to(config['device'])
    model.load_state_dict(torch.load(
        'checkpoint/cp6/model.pth', weights_only=False))
    tokenizer = AutoTokenizer.from_pretrained(config['name_model'])
    model.eval()
    key = ['AMBIENCE', 'QUALITY', 'PRICES', 'LOCATION', 'SERVICE']


@app.route('/')
def home():
    return "Hello world!"


@app.route('/predict/aspects', methods=['POST'])
def predict():

    try:
        data = request.get_json()
        value = []
        input_ids = None
        attention_masks = None

        for k in data:
            text = preprocess_fn(data[k])
            encoding = tokenizer(
                text,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            if (input_ids is None):
                input_ids = encoding["input_ids"].to(config['device'])
            else:
                input_ids = torch.cat(
                    (input_ids, encoding["input_ids"].to(config['device'])), 0)

            if (attention_masks is None):
                attention_masks = encoding["attention_mask"].to(
                    config['device'])
            else:
                attention_masks = torch.cat(
                    (attention_masks, encoding["attention_mask"].to(config['device'])), 0)

        with torch.no_grad():
            outputs = model(input_ids, attention_masks)
            outputs = torch.sigmoid(outputs).tolist()

            # outputs = [int(torch.sigmoid(output).item() > 0.5)
            #            for output in outputs]
        for v in outputs:
            value.append([key[i] for i in range(len(v)) if v[i] > 0.5])

        return jsonify({'result': value}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
