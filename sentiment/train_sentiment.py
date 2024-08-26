
from dataset_sentiment import Data
from dataset_vncorenlp import DataVNCoreNLP
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import tqdm
from model_sentiment import SentimentClassifier
from evaluate import *
import yaml
import torch
import numpy as np
import json
import random
import os
from torch.utils.tensorboard import SummaryWriter
# from lion_pytorch import Lion
# from schedule import WarmupCosineDecay


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Instructor:
    def __init__(self, modelId, root=''):
        with open(root + 'configModel/model_{}.yaml'.format(modelId), 'r') as file:
            config = yaml.safe_load(file)
        self.root = root
        self.config = config
        set_seed(config['seed'])
        tokenizer = AutoTokenizer.from_pretrained(self.config['name_model'])
        if self.config['isVnCore']:
            self.data = DataVNCoreNLP(
                batch_size=self.config['batch_size'], max_length=self.config['max_length'])
        else:
            self.data = Data(tokenizer=tokenizer,
                             batch_size=self.config['batch_size'], max_length=self.config['max_length'])

        self.model = SentimentClassifier(config).to(self.config['device'])

    def train(self):
        if not self.config['isKaggle']:
            if not os.path.exists(self.root + 'result/logs'):
                os.makedirs('result/logs')
            self.writer = SummaryWriter(self.root + 'result/logs')

        train_loader, len_train_data = self.data.getBatchDataTrain()

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config['weight_decay']},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        if self.config['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                optimizer_parameters, lr=self.config['lr'])
        else:
            self.optimizer = AdamW(
                optimizer_parameters, lr=self.config['lr'], correct_bias=False)
            # self.optimizer = Lion(
            #     self.model.parameters(), lr=self.config['lr'])

        if (self.config['losses'] == 1):
            self.losses = [torch.nn.BCELoss()
                           for _ in range(self.config['num_classes'])]
        else:
            self.losses = [torch.nn.CrossEntropyLoss()
                           for _ in range(self.config['num_classes'])]
        num_training_steps = int(len_train_data / self.config['batch_size'] *
                                 self.config['epochs'])

        if self.config['model'] == 3:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=100,
                                                             num_training_steps=num_training_steps)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['T_max'],  # ~ 2896 / 16 * 25 * 30%
                eta_min=self.config['eta_min']
            )
            # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     self.optimizer, mode='min', factor=0.01, patience=1, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

        for epoch in range(self.config['epochs']):
            self.model.train()
            print('Epoch : {} / {}'.format(epoch+1, self.config['epochs']))
            totol_loss = [0 for _ in range(self.config['num_classes'])]
            all_prediction = None
            all_target = None

            for data in tqdm.tqdm(train_loader):
                y_train = data['labels'].to(
                    device=self.config['device'], dtype=torch.float)

                if (self.config['model'] == 3) or (self.config['model'] == 4) or (self.config['isVnCore']):
                    X_train = {
                        'input_ids': data['input_ids'].to(self.config['device']),
                        'token_type_ids': torch.tensor([]).to(self.config['device']),
                        'attention_mask': data['attention_mask'].to(self.config['device'])
                    }
                else:
                    X_train = {
                        'input_ids': data['input_ids'].to(self.config['device']),
                        'token_type_ids': data['token_type_ids'].to(self.config['device']),
                        'attention_mask': data['attention_mask'].to(self.config['device'])
                    }

                self.optimizer.zero_grad()

                outputs = self.model(**X_train)

                if (self.config['losses'] == 1):
                    prediction = [torch.max(outputs[i].squeeze(-1), dim=1).indices
                                  for i in range(self.config['num_classes'])]
                    losses = [self.losses[i](outputs[i].squeeze(-1),
                                             torch.tensor([PolarityMapping.INDEX_TO_ONEHOT[v.item()]
                                                          for v in y_train[:, i]]).to(self.config['device'], dtype=torch.float))
                              for i in range(self.config['num_classes'])]
                else:
                    losses = [self.losses[i](outputs[i].squeeze(-1), y_train[:, i].long())
                              for i in range(self.config['num_classes'])]
                    outputs = [torch.nn.Softmax(dim=1)(outputs[i].squeeze(-1))
                               for i in range(self.config['num_classes'])]
                    prediction = [torch.max(outputs[i].squeeze(-1), dim=1).indices
                                  for i in range(self.config['num_classes'])]

                for i in range(self.config['num_classes']):
                    totol_loss[i] += losses[i].item() * \
                        self.config['batch_size']

                all_loss = sum(losses)
                # all_loss = 0
                # for i in range(self.config['num_classes']):
                #     all_loss += self.config['weights'][i] * losses[i]

                all_loss.backward()
                if self.config['clip_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0)

                self.optimizer.step()
                if self.config['scheduler']:
                    self.scheduler.step()

                if all_prediction is None:
                    all_prediction = prediction
                    all_target = y_train
                else:
                    all_prediction = [torch.cat((all_prediction[i], prediction[i]))
                                      for i in range(self.config['num_classes'])]
                    all_target = torch.cat((all_target, y_train))

            print('Len Train Data : ', len_train_data)
            for i in range(self.config['num_classes']):
                print('Losses Train {} : {}'.format(
                    i, totol_loss[i] / len_train_data))

                if not self.config['isKaggle']:
                    self.writer.add_scalar(
                        f'training_loss_class_{i}', totol_loss[i] / len_train_data, epoch)

            all_prediction = np.array([all_prediction[i].detach().cpu().numpy()
                                       for i in range(self.config['num_classes'])])

            all_prediction = np.transpose(all_prediction, (1, 0))
            all_target = all_target.cpu().numpy().astype(int)

            if not self.config['isKaggle']:
                Evaluator(all_target, all_prediction,
                          self.config['key'], type='train', num=epoch, root=self.root)
            self.validate(epoch)

        if not self.config['isKaggle']:
            self.writer.close()

    def validate(self, epoch):
        val_loader, len_val_data = self.data.getBatchDataVal()
        self.model.eval()
        # best_loss = self.load_checkpoint()
        # print('Best Loss : ', best_loss)
        save_pred = False
        with torch.no_grad():
            totol_loss = [0 for _ in range(self.config['num_classes'])]
            all_prediction = None
            all_target = None
            for data in tqdm.tqdm(val_loader):
                y_val = data['labels'].to(
                    device=self.config['device'], dtype=torch.float)
                if (self.config['model'] == 3) or (self.config['model'] == 4) or (self.config['isVnCore']):
                    X_val = {
                        'input_ids': data['input_ids'].to(self.config['device']),
                        'token_type_ids': torch.tensor([]).to(self.config['device']),
                        'attention_mask': data['attention_mask'].to(self.config['device'])
                    }
                else:
                    X_val = {
                        'input_ids': data['input_ids'].to(self.config['device']),
                        'token_type_ids': data['token_type_ids'].to(self.config['device']),
                        'attention_mask': data['attention_mask'].to(self.config['device'])
                    }

                outputs = self.model(**X_val)

                if (self.config['losses'] == 1):
                    prediction = [torch.max(outputs[i].squeeze(-1), dim=1).indices
                                  for i in range(self.config['num_classes'])]
                    losses = [self.losses[i](outputs[i].squeeze(-1),
                                             torch.tensor([PolarityMapping.INDEX_TO_ONEHOT[v.item()]
                                                          for v in y_val[:, i]]).to(self.config['device'], dtype=torch.float))
                              for i in range(self.config['num_classes'])]
                else:
                    losses = [self.losses[i](outputs[i].squeeze(-1), y_val[:, i].long())
                              for i in range(self.config['num_classes'])]
                    outputs = [torch.nn.Softmax(dim=1)(outputs[i].squeeze(-1))
                               for i in range(self.config['num_classes'])]
                    prediction = [torch.max(outputs[i].squeeze(-1), dim=1).indices
                                  for i in range(self.config['num_classes'])]

                if all_prediction is None:
                    all_prediction = prediction
                    all_target = y_val
                else:
                    all_prediction = [torch.cat((all_prediction[i], prediction[i]))
                                      for i in range(self.config['num_classes'])]
                    all_target = torch.cat((all_target, y_val))

                for i in range(self.config['num_classes']):
                    totol_loss[i] += losses[i].item() * \
                        self.config['batch_size']

            print('Len Val Data : ', len_val_data)
            for i in range(self.config['num_classes']):
                print('Losses Validate {} : {}'.format(
                    i, totol_loss[i] / len_val_data))

                totol_loss[i] /= len_val_data

                if not self.config['isKaggle']:
                    self.writer.add_scalar(
                        f'validation_loss_class_{i}', totol_loss[i], epoch)

            # if self.config['scheduler']:
            #     self.scheduler.step(sum(totol_loss))

            all_prediction = np.array([all_prediction[i].detach().cpu().numpy()
                                       for i in range(self.config['num_classes'])])

            all_prediction = np.transpose(all_prediction, (1, 0))
            all_target = all_target.cpu().numpy().astype(int)

            print('Param optimizer : ')
            for param_group in self.optimizer.param_groups:
                print('Learning rate', param_group['lr'])
                print('Beta', param_group['betas'])
                print('Weight decay', param_group['weight_decay'])

            # if ((sum(totol_loss) < sum(best_loss)) and (not self.config['isKaggle'])):
            #     best_loss = totol_loss
            #     self.save_checkpoint(best_loss)
            #     save_pred = True
            if not self.config['isKaggle']:
                path = self.root + 'checkpoint' + '/' + \
                    str(self.config['model']) + '_' + str(epoch)

                self.save_checkpoint(totol_loss, path)
                save_pred = True

                aspect_cate_polar_report = Evaluator(all_target, all_prediction,
                                                     self.config['key'], type='val', num=epoch, root=self.root).aspect_cate_polar_report
                if save_pred:
                    with open(path + '/aspect_cate_polar_report.json', 'w') as f:
                        json.dump(aspect_cate_polar_report, f)

    def prediction(self, str, type=False):
        evalData, _ = self.data.getStrData(str)
        if not type:
            _ = self.load_checkpoint(isPred=True)

        for data in evalData:
            X = {
                'input_ids': data['input_ids'].to(self.config['device']),
                'token_type_ids': data['token_type_ids'].to(self.config['device']),
                'attention_mask': data['attention_mask'].to(self.config['device'])
            }
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**X)
                prediction = [torch.max(outputs[i].squeeze(-1), dim=1).indices
                              for i in range(self.config['num_classes'])]
                prediction = np.transpose(prediction, (1, 0))
                print(prediction)

    def load_checkpoint(self, isPred=False):
        path_losses = 'checkpoint/losses.json'
        checkpoint_path = 'checkpoint/model.pth'

        if not os.path.exists(path_losses):
            return [999 for _ in range(self.config['num_classes'])]

        with open(path_losses, 'r') as f:
            losses = json.load(f)

        if isPred:
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(
                    checkpoint_path, map_location=self.config['device']))
                print("Model checkpoint loaded successfully.")
            else:
                print("No model checkpoint found at 'checkpoint/model.pth'.")
        return losses

    def save_checkpoint(self, totol_loss, path):

        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), path + '/model.pth')

        with open(path + '/losses.json', 'w') as f:
            json.dump(totol_loss, f)
