from flask import Flask, request, jsonify
from model import BertClassifier, BertClassifierVer3
from transformers import AutoTokenizer
import torch
from preprocess import preprocess_fn
app = Flask(__name__)


@app.before_request
def load_model():
    config = {
        'name_model': 'vinai/phobert-base',
        'num_classes': 5,
        'drop_rate': 0.2,
        'device': 'cpu'
    }
    config = {
        'name_model': 'uitnlp/visobert',
        'num_classes': 5,
        'drop_rate': 0.5,
        'device': 'cpu'
    }
    global model, tokenizer, key
    model = BertClassifierVer3(config)
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
        for k in data:
            text = preprocess_fn(data[k])
            encoding = tokenizer(
                text,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to('cpu')
            attention_mask = encoding["attention_mask"].to('cpu')

            # token_type_ids = encoding["token_type_ids"].to('cpu')
            with torch.no_grad():
                # outputs = model(input_ids, token_type_ids, attention_mask)
                outputs = model(input_ids, attention_mask)
                outputs = torch.sigmoid(outputs).squeeze().tolist()

                # outputs = [int(torch.sigmoid(output).item() > 0.5)
                #            for output in outputs]
                outputs = [int(output > 0.5) for output in outputs]
                print(outputs)

                value.append([key[i]
                             for i, val in enumerate(outputs) if val == 1])

        return jsonify({'result': value}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':

    app.run(debug=True)
