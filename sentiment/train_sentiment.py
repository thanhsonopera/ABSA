
from dataset_sentiment import Data
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
from lion_pytorch import Lion


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
    def __init__(self):
        with open('configModel/model.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.name_model = config['name_model']
        self.max_length = config['max_length']
        self.drop_rate = config['drop_rate']
        self.num_classes = config['num_classes']
        self.batch_size = config['batch_size']
        self.epochs = config['num_epochs']
        self.device = config['device']
        self.weights = config['weights']
        self.config = config
        set_seed(config['seed'])
        tokenizer = AutoTokenizer.from_pretrained(self.name_model)

        self.data = Data(tokenizer=tokenizer,
                         batch_size=self.batch_size, max_length=self.max_length)

        self.model = SentimentClassifier(config).to(self.device)

    def train(self):
        if not self.config['isKaggle']:
            if not os.path.exists('result/logs'):
                os.makedirs('result/logs')
            self.writer = SummaryWriter('result/logs')

        train_loader, len_train_data = self.data.getBatchDataTrain()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.losses = [torch.nn.BCELoss()
                       for _ in range(self.num_classes)]

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(len_train_data / self.batch_size *
                      self.epochs * 0.3),
            eta_min=1e-5  # Minimum learning rate
        )
        for epoch in range(self.epochs):
            self.model.train()
            print('Epoch : {} / {}'.format(epoch+1, self.epochs))
            totol_loss = [0 for _ in range(self.num_classes)]
            all_prediction = None
            all_target = None
            cnt = 0
            for data in tqdm.tqdm(train_loader):
                y_train = data['labels'].to(
                    device=self.device, dtype=torch.float)
                X_train = {
                    'input_ids': data['input_ids'].to(self.device),
                    'token_type_ids': data['token_type_ids'].to(self.device),
                    'attention_mask': data['attention_mask'].to(self.device)
                }
                cnt += 1
                self.optimizer.zero_grad()

                outputs = self.model(**X_train)

                prediction = [torch.max(outputs[i].squeeze(-1), dim=1).indices
                              for i in range(self.num_classes)]

                losses = [self.losses[i](outputs[i].squeeze(-1),
                                         torch.tensor([PolarityMapping.INDEX_TO_ONEHOT[v.item()]
                                                       for v in y_train[:, i]]).to(self.device, dtype=torch.float))
                          for i in range(self.num_classes)]

                for i in range(self.num_classes):
                    totol_loss[i] += losses[i].item() * self.batch_size

                all_loss = sum(losses)
                # all_loss = 0
                # for i in range(self.num_classes):
                #     all_loss += self.weights[i] * losses[i]

                all_loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                # self.scheduler.step()

                if all_prediction is None:
                    all_prediction = prediction
                    all_target = y_train
                else:
                    all_prediction = [torch.cat((all_prediction[i], prediction[i]))
                                      for i in range(self.num_classes)]
                    all_target = torch.cat((all_target, y_train))

            print('Len Train Data : ', len_train_data)
            for i in range(self.num_classes):
                print('Losses Train {} : {}'.format(
                    i, totol_loss[i] / len_train_data))

                if not self.config['isKaggle']:
                    self.writer.add_scalar(
                        f'training_loss_class_{i}', totol_loss[i] / len_train_data, epoch)

            all_prediction = np.array([all_prediction[i].detach().cpu().numpy()
                                       for i in range(self.num_classes)])

            all_prediction = np.transpose(all_prediction, (1, 0))
            all_target = all_target.cpu().numpy().astype(int)

            if not self.config['isKaggle']:
                Evaluator(all_target, all_prediction,
                          self.config['key'], type='train', num=epoch, save_pred=False)
            self.validate(epoch)

        if not self.config['isKaggle']:
            self.writer.close()

    def validate(self, epoch):
        val_loader, len_val_data = self.data.getBatchDataVal()
        self.model.eval()
        best_loss = self.load_checkpoint()
        print('Best Loss : ', best_loss)
        save_pred = False
        with torch.no_grad():
            totol_loss = [0 for _ in range(self.num_classes)]
            all_prediction = None
            all_target = None
            for data in tqdm.tqdm(val_loader):
                y_val = data['labels'].to(
                    device=self.device, dtype=torch.float)
                X_val = {
                    'input_ids': data['input_ids'].to(self.device),
                    'token_type_ids': data['token_type_ids'].to(self.device),
                    'attention_mask': data['attention_mask'].to(self.device)
                }

                outputs = self.model(**X_val)

                prediction = [torch.max(outputs[i].squeeze(-1), dim=1).indices
                              for i in range(self.num_classes)]

                losses = [self.losses[i](outputs[i].squeeze(-1),
                                         torch.tensor([PolarityMapping.INDEX_TO_ONEHOT[v.item()]
                                                       for v in y_val[:, i]]).to(self.device, dtype=torch.float))
                          for i in range(self.num_classes)]

                if all_prediction is None:
                    all_prediction = prediction
                    all_target = y_val
                else:
                    all_prediction = [torch.cat((all_prediction[i], prediction[i]))
                                      for i in range(self.num_classes)]
                    all_target = torch.cat((all_target, y_val))

                for i in range(self.num_classes):
                    totol_loss[i] += losses[i].item() * self.batch_size

            print('Len Val Data : ', len_val_data)
            for i in range(self.num_classes):
                print('Losses Validate {} : {}'.format(
                    i, totol_loss[i] / len_val_data))

                totol_loss[i] /= len_val_data

                if not self.config['isKaggle']:
                    self.writer.add_scalar(
                        f'validation_loss_class_{i}', totol_loss[i], epoch)

            all_prediction = np.array([all_prediction[i].detach().cpu().numpy()
                                       for i in range(self.num_classes)])

            all_prediction = np.transpose(all_prediction, (1, 0))
            all_target = all_target.cpu().numpy().astype(int)

            print('Param optimizer : ')
            for param_group in self.optimizer.param_groups:
                print('Learning rate', param_group['lr'])
                print('Beta', param_group['betas'])
                print('Eps', param_group['eps'])
                print('Weight decay', param_group['weight_decay'])

            if ((sum(totol_loss) < sum(best_loss)) and (not self.config['isKaggle'])):
                best_loss = totol_loss
                self.save_checkpoint(best_loss)
                save_pred = True
            Evaluator(all_target, all_prediction,
                      self.config['key'], type='val', num=epoch, save_pred=save_pred)

    def prediction(self, str, type=False):
        evalData, _ = self.data.getStrData(str)
        if not type:
            _ = self.load_checkpoint(isPred=True)
        for data in evalData:
            X = {
                'input_ids': data['input_ids'].to(self.device),
                'token_type_ids': data['token_type_ids'].to(self.device),
                'attention_mask': data['attention_mask'].to(self.device)
            }
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**X)
                prediction = [torch.max(outputs[i].squeeze(-1), dim=1).indices
                              for i in range(self.num_classes)]
                prediction = np.transpose(prediction, (1, 0))
                print(prediction)

    def load_checkpoint(self, isPred=False):
        path_losses = 'checkpoint/losses.json'
        checkpoint_path = 'checkpoint/model.pth'

        if not os.path.exists(path_losses):
            return [999 for _ in range(self.num_classes)]

        with open(path_losses, 'r') as f:
            losses = json.load(f)

        if isPred:
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(
                    checkpoint_path, map_location=self.device))
                print("Model checkpoint loaded successfully.")
            else:
                print("No model checkpoint found at 'checkpoint/model.pth'.")
        return losses

    def save_checkpoint(self, totol_loss):
        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')
        torch.save(self.model.state_dict(), 'checkpoint/model.pth')

        with open('checkpoint/losses.json', 'w') as f:
            json.dump(totol_loss, f)
