
from dataset import Data
from transformers import AutoTokenizer
import tqdm
from model import BertClassifier, BertClassifierVer2
import torch
import numpy as np
from preprocess import preprocess_fn
from eval3 import aspect_eval
import json
import os


class Instructor:
    def __init__(self, config):

        self.name_model = config['name_model']
        self.max_length = config['max_length']
        self.drop_rate = config['drop_rate']
        self.num_classes = config['num_classes']
        self.batch_size = config['batch_size']
        self.epochs = config['num_epochs']
        self.device = config['device']
        self.weights = config['weights']
        self.config = config

        tokenizer = AutoTokenizer.from_pretrained(self.name_model)

        self.data = Data(type=config['type'], tokenizer=tokenizer,
                         batch_size=self.batch_size, max_length=self.max_length)

    def train(self):
        train_loader, len_train_data = self.data.getBatchDataTrain()

        self.model = BertClassifier(
            self.config).to(self.device)

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-5, weight_decay=0.01)

        self.losses = [torch.nn.BCEWithLogitsLoss()
                       for _ in range(self.num_classes)]

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(len_train_data / self.batch_size *
                      self.epochs * 0.3),  # 30% of total steps
            eta_min=1e-5  # Minimum learning rate
        )

        for epoch in range(self.epochs):
            self.model.train()
            print('Epoch : {} / {}'.format(epoch+1, self.epochs))
            totol_loss = [0 for _ in range(self.num_classes)]
            for data in tqdm.tqdm(train_loader):

                y_train = data['labels'].to(
                    device=self.device, dtype=torch.float)
                X_train = {
                    'input_ids': data['input_ids'].to(self.device),
                    'token_type_ids': data['token_type_ids'].to(self.device),
                    'attention_mask': data['attention_mask'].to(self.device)
                }
                # print('Shape', y_train.shape, X_train['input_ids'].shape,
                #       X_train['token_type_ids'].shape, X_train['attention_mask'].shape)

                self.optimizer.zero_grad()
                # Không có **X_train -> unhashable type: 'slice'
                outputs = self.model(**X_train)
                # print(y_train.shape)
                # print('Output : ')
                # for i in range(self.num_classes):
                #     print(outputs[i].shape, outputs[i].squeeze(-1))

                losses = [self.losses[i](outputs[i].squeeze(-1), y_train[:, i])
                          for i in range(self.num_classes)]
                for i in range(self.num_classes):
                    totol_loss[i] += losses[i].item() * self.batch_size

                loss = sum(losses)
                # loss = 0
                # for i in range(self.num_classes):
                #     loss += self.weights[i] * losses[i]
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                # self.scheduler.step()

            print('Len Train Data : ', len_train_data)
            for i in range(self.num_classes):
                print('Losses Train {} : {}'.format(
                    i, totol_loss[i] / len_train_data))

            self.validate(epoch)

    def validate(self, epoch):
        val_loader, len_val_data = self.data.getBatchDataVal()
        self.model.eval()
        best_loss = self.load_checkpoint()
        save_pred = False
        with torch.no_grad():
            totol_loss = [0 for _ in range(self.num_classes)]
            totol_pred = None
            totol_label = None
            for data in tqdm.tqdm(val_loader):
                y_val = data['labels'].to(
                    device=self.device, dtype=torch.float)
                X_val = {
                    'input_ids': data['input_ids'].to(self.device),
                    'token_type_ids': data['token_type_ids'].to(self.device),
                    'attention_mask': data['attention_mask'].to(self.device)
                }

                outputs = self.model(**X_val)

                losses = [self.losses[i](outputs[i].squeeze(-1), y_val[:, i])
                          for i in range(self.num_classes)]

                pred = [torch.sigmoid(outputs[i].squeeze(-1))
                        for i in range(self.num_classes)]
                # print(y_val.shape)
                # for i in range(self.num_classes):
                #     print('Predict : ', pred[i].shape, pred[i])

                if totol_pred is None:
                    totol_pred = pred
                    totol_label = y_val
                else:
                    totol_pred = [torch.cat((totol_pred[i], pred[i]))
                                  for i in range(self.num_classes)]
                    totol_label = torch.cat((totol_label, y_val))

                for i in range(self.num_classes):
                    totol_loss[i] += losses[i].item() * self.batch_size

            print('Len Val Data : ', len_val_data)
            for i in range(self.num_classes):
                print('Losses Validate {} : {}'.format(
                    i, totol_loss[i] / len_val_data))
                totol_loss[i] /= len_val_data

            totol_pred = np.array([totol_pred[i].cpu().numpy()
                                   for i in range(self.num_classes)])
            totol_pred = np.transpose(totol_pred, (1, 0))
            totol_pred = (totol_pred > 0.5).astype(int)
            totol_label = totol_label.cpu().numpy().astype(int)

            if (totol_loss[0] < best_loss[0]) and (totol_loss[3] < best_loss[3]):
                best_loss = totol_loss
                self.save_checkpoint(best_loss)
                save_pred = True

            aspect_eval(totol_label, totol_pred, 3 + epoch, save_pred)

            # print('Predict : ', totol_pred.shape, totol_label.shape)
            # print(totol_pred[0], totol_label[0])

    def prediction(self, str):
        evalData, len_pred = self.data.getStrData(str)
        key = ['AMBIENCE', 'QUALITY', 'PRICES', 'LOCATION', 'SERVICE']
        for data in evalData:
            X = {
                'input_ids': data['input_ids'].to(self.device),
                'token_type_ids': data['token_type_ids'].to(self.device),
                'attention_mask': data['attention_mask'].to(self.device)
            }
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**X)
                pred = [torch.sigmoid(outputs[i].squeeze(-1))
                        for i in range(self.num_classes)]
                pred = np.array([pred[i].cpu().numpy()
                                for i in range(self.num_classes)])
                pred = np.transpose(pred, (1, 0))
                pred = (pred > 0.5).astype(int)
                print(pred)

    def load_checkpoint(self):
        path_losses = 'checkpoint/losses.json'

        if not os.path.exists(path_losses):
            return [999 for _ in range(self.num_classes)]

        with open(path_losses, 'r') as f:
            losses = json.load(f)

        return losses

    def save_checkpoint(self, totol_loss):
        torch.save(self.model.state_dict(), 'checkpoint/model.pth')

        with open('checkpoint/losses.json', 'w') as f:
            json.dump(totol_loss, f)


class InstructorVer2:
    def __init__(self, config):

        self.name_model = config['name_model']
        self.max_length = config['max_length']
        self.drop_rate = config['drop_rate']
        self.num_classes = config['num_classes']
        self.batch_size = config['batch_size']
        self.epochs = config['num_epochs']
        self.device = config['device']
        self.weights = config['weights']
        self.config = config

        tokenizer = AutoTokenizer.from_pretrained(self.name_model)

        self.data = Data(type=config['type'], tokenizer=tokenizer,
                         batch_size=self.batch_size, max_length=self.max_length)

    def train(self):
        train_loader, len_train_data = self.data.getBatchDataTrain()

        self.model = BertClassifierVer2(
            self.config).to(self.device)

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-5, weight_decay=0.01)
        # weight=torch.tensor(self.weights).to(self.device)
        self.losses = torch.nn.BCEWithLogitsLoss()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(len_train_data / self.batch_size *
                      self.epochs * 0.3),
            eta_min=1e-6
        )

        for epoch in range(self.epochs):
            self.model.train()
            print('Epoch : {} / {}'.format(epoch+1, self.epochs))
            # totol_loss = [0 for _ in range(self.num_classes)]
            totol_loss = 0
            for data in tqdm.tqdm(train_loader):

                y_train = data['labels'].to(
                    device=self.device, dtype=torch.float)
                X_train = {
                    'input_ids': data['input_ids'].to(self.device),
                    'token_type_ids': data['token_type_ids'].to(self.device),
                    'attention_mask': data['attention_mask'].to(self.device)
                }

                self.optimizer.zero_grad()

                outputs = self.model(**X_train)

                loss = self.losses(outputs, y_train)

                totol_loss += loss.item() * self.batch_size
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                # self.scheduler.step()

            print('Len Train Data : ', len_train_data)

            print('Losses Train : {}'.format(
                totol_loss / len_train_data))

            self.validate(epoch)

    def validate(self, epoch):
        val_loader, len_val_data = self.data.getBatchDataVal()
        self.model.eval()
        best_loss = self.load_checkpoint()
        save_pred = False
        with torch.no_grad():
            # totol_loss = [0 for _ in range(self.num_classes)]
            totol_loss = 0
            totol_pred = None
            totol_label = None
            for data in tqdm.tqdm(val_loader):
                y_val = data['labels'].to(
                    device=self.device, dtype=torch.float)
                X_val = {
                    'input_ids': data['input_ids'].to(self.device),
                    'token_type_ids': data['token_type_ids'].to(self.device),
                    'attention_mask': data['attention_mask'].to(self.device)
                }

                outputs = self.model(**X_val)

                loss = self.losses(outputs, y_val)

                pred = torch.sigmoid(outputs)

                if totol_pred is None:
                    totol_pred = pred
                    totol_label = y_val
                else:
                    totol_pred = torch.cat((totol_pred, pred))
                    totol_label = torch.cat((totol_label, y_val))

                totol_loss += loss.item() * self.batch_size

            print('Len Val Data : ', len_val_data)

            print('Losses Validate : {}'.format(
                totol_loss / len_val_data))

            totol_loss /= len_val_data

            totol_pred = totol_pred.cpu().numpy()

            totol_pred = (totol_pred > 0.5).astype(int)

            totol_label = totol_label.cpu().numpy().astype(int)

            # if (totol_loss[0] < best_loss[0]) and (totol_loss[3] < best_loss[3]):
            #     best_loss = totol_loss
            #     self.save_checkpoint(best_loss)
            #     save_pred = True

            print('Predict : ', totol_pred.shape, totol_label.shape)
            print(totol_pred[0], totol_label[0])
            aspect_eval(totol_label, totol_pred, 3 + epoch, save_pred)

    def test(self, str):
        self.data.getStrData(str)

    def load_checkpoint(self):
        path_losses = 'checkpoint/losses.json'

        if not os.path.exists(path_losses):
            return [999 for _ in range(self.num_classes)]

        with open(path_losses, 'r') as f:
            losses = json.load(f)

        return losses

    def save_checkpoint(self, totol_loss):
        torch.save(self.model.state_dict(), 'checkpoint/model.pth')

        with open('checkpoint/losses.json', 'w') as f:
            json.dump(totol_loss, f)
