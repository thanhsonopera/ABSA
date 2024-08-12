
from dataset import Data
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import tqdm
from model import BertClassifier, BertClassifierVer2, BertClassifierVer3
from losses import AsymmetricLoss
from torch.amp import autocast, GradScaler
import torch
import numpy as np
from eval3 import aspect_eval, cus_confusion_matrix
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
        set_seed(config['seed'])
        tokenizer = AutoTokenizer.from_pretrained(self.name_model)

        self.data = Data(type=config['type'], tokenizer=tokenizer,
                         batch_size=self.batch_size, max_length=self.max_length)

        self.model = BertClassifier(
            self.config).to(self.device)

    def train(self):
        if not self.config['isKaggle']:
            self.writer = SummaryWriter('result/logs')
        train_loader, len_train_data = self.data.getBatchDataTrain()

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-08)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-5, weight_decay=0.01)

        self.losses = [torch.nn.BCEWithLogitsLoss()
                       for _ in range(self.num_classes)]

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(len_train_data / self.batch_size *
                      self.epochs * 0.3),
            eta_min=5e-5  # Minimum learning rate
        )

        for epoch in range(self.epochs):
            self.model.train()
            print('Epoch : {} / {}'.format(epoch+1, self.epochs))
            totol_loss = [0 for _ in range(self.num_classes)]
            totol_pred = None
            totol_label = None
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
                pred = [torch.sigmoid(outputs[i].squeeze(-1))
                        for i in range(self.num_classes)]
                losses = [self.losses[i](outputs[i].squeeze(-1), y_train[:, i])
                          for i in range(self.num_classes)]

                for i in range(self.num_classes):
                    totol_loss[i] += losses[i].item() * self.batch_size

                # loss = sum(losses)
                loss = 0
                for i in range(self.num_classes):
                    loss += self.weights[i] * losses[i]

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                # self.scheduler.step()

                if totol_pred is None:
                    totol_pred = pred
                    totol_label = y_train
                else:
                    totol_pred = [torch.cat((totol_pred[i], pred[i]))
                                  for i in range(self.num_classes)]
                    totol_label = torch.cat((totol_label, y_train))

            print('Len Train Data : ', len_train_data)
            for i in range(self.num_classes):
                print('Losses Train {} : {}'.format(
                    i, totol_loss[i] / len_train_data))
                if not self.config['isKaggle']:
                    self.writer.add_scalar(
                        f'training_loss_class_{i}', totol_loss[i] / len_train_data, epoch)

            totol_pred = np.array([totol_pred[i].detach().cpu().numpy()
                                   for i in range(self.num_classes)])

            totol_pred = np.transpose(totol_pred, (1, 0))
            totol_pred = (totol_pred > 0.5).astype(int)
            totol_label = totol_label.cpu().numpy().astype(int)
            if not self.config['isKaggle']:
                aspect_eval(totol_label, totol_pred, epoch,
                            save_pred=False, type='train')
                cus_confusion_matrix(
                    totol_label, totol_pred, epoch, save_pred=False, type='train')

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
                if not self.config['isKaggle']:
                    self.writer.add_scalar(
                        f'validation_loss_class_{i}', totol_loss[i], epoch)

            totol_pred = np.array([totol_pred[i].cpu().numpy()
                                   for i in range(self.num_classes)])
            totol_pred = np.transpose(totol_pred, (1, 0))
            totol_pred = (totol_pred > 0.5).astype(int)
            totol_label = totol_label.cpu().numpy().astype(int)

            print('Param optimizer : ')
            for param_group in self.optimizer.param_groups:
                print('Learning rate', param_group['lr'])
                print('Beta', param_group['betas'])
                print('Eps', param_group['eps'])
                print('Weight decay', param_group['weight_decay'])
                print('Initial lr', param_group['initial_lr'])
                print('AdamGrad', param_group['amsgrad'])

            if ((sum(totol_loss) < sum(best_loss)) and (not self.config['isKaggle'])):
                best_loss = totol_loss
                self.save_checkpoint(best_loss)
                save_pred = True
            if not self.config['isKaggle']:
                aspect_eval(totol_label, totol_pred,
                            epoch, save_pred, type='val')

                cus_confusion_matrix(totol_label, totol_pred,
                                     epoch, save_pred, type='val')
            # print('Predict : ', totol_pred.shape, totol_label.shape)
            # print(totol_pred[0], totol_label[0])

    def prediction(self, str, type=False):
        evalData, len_pred = self.data.getStrData(str)
        if not type:
            _ = self.load_checkpoint(isPred=True)
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

        self.model = BertClassifierVer2(
            self.config).to(self.device)

    def train(self):
        if not self.config['isKaggle']:
            self.writer = SummaryWriter('result/logs')
        train_loader, len_train_data = self.data.getBatchDataTrain()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-5, weight_decay=0.01)

        # self.optimizer = torch.optim.AdamW(
        #     self.model.parameters(), lr=1e-5, weight_decay=0.01)

        # weight=torch.tensor(self.weights).to(self.device)
        # self.losses = torch.nn.BCEWithLogitsLoss()

        self.losses = AsymmetricLoss(
            gamma_neg=4, gamma_pos=1, clip=0.05, disable_torch_grad_focal_loss=True)

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer,
        #     T_max=int(len_train_data / self.batch_size *
        #               self.epochs * 0.3),
        #     eta_min=1e-6
        # )
        scaler = GradScaler('cuda')
        for epoch in range(self.epochs):
            self.model.train()
            print('Epoch : {} / {}'.format(epoch+1, self.epochs))
            # totol_loss = [0 for _ in range(self.num_classes)]
            totol_loss = 0
            totol_pred = None
            totol_label = None
            for data in tqdm.tqdm(train_loader):

                y_train = data['labels'].to(
                    device=self.device, dtype=torch.float)
                X_train = {
                    'input_ids': data['input_ids'].to(self.device),
                    'token_type_ids': data['token_type_ids'].to(self.device),
                    'attention_mask': data['attention_mask'].to(self.device)
                }

                self.optimizer.zero_grad()
                with autocast('cuda'):
                    outputs = self.model(**X_train).float()

                loss = self.losses(outputs, y_train)

                totol_loss += loss.item() * self.batch_size

                pred = torch.sigmoid(outputs)
                if totol_pred is None:
                    totol_pred = pred
                    totol_label = y_train
                else:
                    totol_pred = torch.cat((totol_pred, pred))
                    totol_label = torch.cat((totol_label, y_train))

                scaler.scale(loss).backward()

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
                # self.optimizer.step()
                # self.scheduler.step()

            print('Len Train Data : ', len_train_data)

            print('Losses Train : {}'.format(
                totol_loss / len_train_data))
            if not self.config['isKaggle']:
                self.writer.add_scalar(
                    f'train_loss', totol_loss, epoch)

            totol_pred = totol_pred.detach().cpu().numpy()

            totol_pred = (totol_pred > 0.5).astype(int)

            totol_label = totol_label.cpu().numpy().astype(int)

            aspect_eval(totol_label, totol_pred, epoch,
                        save_pred=False, type='train')
            cus_confusion_matrix(totol_label, totol_pred,
                                 epoch, save_pred=False, type='train')

            self.validate(epoch)

    def validate(self, epoch):
        val_loader, len_val_data = self.data.getBatchDataVal()
        self.model.eval()
        best_loss = self.load_checkpoint()
        save_pred = False

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
            with torch.no_grad():
                with autocast('cuda'):
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
        if (totol_loss < best_loss[0]):
            best_loss = [totol_loss]
            self.save_checkpoint(best_loss)
            save_pred = True

        if not self.config['isKaggle']:
            self.writer.add_scalar(
                f'validation_loss', totol_loss, epoch)

        aspect_eval(totol_label, totol_pred, epoch, save_pred, type='val')
        cus_confusion_matrix(totol_label, totol_pred,
                             epoch, save_pred, type='val')

    def prediction(self, str, type=False):
        evalData, len_pred = self.data.getStrData(str)
        if not type:
            _ = self.load_checkpoint(isPred=True)
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
                pred = torch.sigmoid(outputs)
                pred = pred.cpu().numpy()
                pred = (pred > 0.5).astype(int)
                print(pred)

    def load_checkpoint(self, isPred=False):
        path_losses = 'checkpoint/losses.json'
        checkpoint_path = 'checkpoint/model.pth'
        if not os.path.exists(path_losses):
            return [999]

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
        torch.save(self.model.state_dict(), 'checkpoint/model.pth')

        with open('checkpoint/losses.json', 'w') as f:
            json.dump(totol_loss, f)


class InstructorVer3:
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

        self.model = BertClassifierVer3(
            self.config).to(self.device)

    def train(self):
        if not self.config['isKaggle']:
            self.writer = SummaryWriter('result/logs')

        train_loader, len_train_data = self.data.getBatchDataTrain()
        num_training_steps = int(len_train_data / self.batch_size *
                                 self.epochs)
        # self.optimizer = AdamW(self.model.parameters(),
        #                        lr=1e-5, correct_bias=False)
        self.optimizer = Lion(self.model.parameters(),
                              lr=1e-5)
        # self.optimizer = Lion(self.model.parameters(),
        #                       lr=1e-5)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=10,
                                                         num_training_steps=num_training_steps)
        self.losses = torch.nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            self.model.train()
            print('Epoch : {} / {}'.format(epoch+1, self.epochs))
            totol_loss = 0
            totol_pred = None
            totol_label = None
            batch_count = 0
            for data in tqdm.tqdm(train_loader):

                y_train = data['labels'].to(
                    device=self.device, dtype=torch.float)
                X_train = {
                    'input_ids': data['input_ids'].to(self.device),
                    'attention_mask': data['attention_mask'].to(self.device)
                }

                self.optimizer.zero_grad()

                outputs = self.model(**X_train)

                loss = self.losses(outputs, y_train)

                totol_loss += loss.item() * self.batch_size

                loss.backward()
                batch_count += 1
                if batch_count % 1 == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()
                    self.scheduler.step()

                pred = torch.sigmoid(outputs)
                if totol_pred is None:
                    totol_pred = pred
                    totol_label = y_train
                else:
                    totol_pred = torch.cat((totol_pred, pred))
                    totol_label = torch.cat((totol_label, y_train))

            print('Len Train Data : ', len_train_data)

            print('Losses Train : {}'.format(
                totol_loss / len_train_data))
            if not self.config['isKaggle']:
                self.writer.add_scalar(
                    f'train_loss', totol_loss, epoch)

            if batch_count % 1 != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

            totol_pred = totol_pred.detach().cpu().numpy()

            totol_pred = (totol_pred > 0.5).astype(int)

            totol_label = totol_label.cpu().numpy().astype(int)

            aspect_eval(totol_label, totol_pred, epoch,
                        save_pred=False, type='train')
            cus_confusion_matrix(totol_label, totol_pred,
                                 epoch, save_pred=False, type='train')

            self.validate(epoch)

    def validate(self, epoch):
        val_loader, len_val_data = self.data.getBatchDataVal()
        self.model.eval()
        best_loss = self.load_checkpoint()
        save_pred = False

        totol_loss = 0
        totol_pred = None
        totol_label = None

        for data in tqdm.tqdm(val_loader):
            y_val = data['labels'].to(
                device=self.device, dtype=torch.float)
            X_val = {
                'input_ids': data['input_ids'].to(self.device),
                'attention_mask': data['attention_mask'].to(self.device)
            }
            with torch.no_grad():
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
        if (totol_loss < best_loss[0]):
            best_loss = [totol_loss]
            self.save_checkpoint(best_loss)
            save_pred = True

        if not self.config['isKaggle']:
            self.writer.add_scalar(
                f'validation_loss', totol_loss, epoch)

        aspect_eval(totol_label, totol_pred, epoch, save_pred, type='val')
        cus_confusion_matrix(totol_label, totol_pred,
                             epoch, save_pred, type='val')

    def prediction(self, str, type=False):
        evalData, len_pred = self.data.getStrData(str)
        if not type:
            _ = self.load_checkpoint(isPred=True)
        key = ['AMBIENCE', 'QUALITY', 'PRICES', 'LOCATION', 'SERVICE']
        for data in evalData:
            X = {
                'input_ids': data['input_ids'].to(self.device),
                'attention_mask': data['attention_mask'].to(self.device)
            }
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**X)
                pred = torch.sigmoid(outputs)
                pred = pred.cpu().numpy()
                pred = (pred > 0.5).astype(int)
                print(pred)

    def load_checkpoint(self, isPred=False):
        path_losses = 'checkpoint/losses.json'
        checkpoint_path = 'checkpoint/model.pth'
        if not os.path.exists(path_losses):
            return [999]

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
        torch.save(self.model.state_dict(), 'checkpoint/model.pth')

        with open('checkpoint/losses.json', 'w') as f:
            json.dump(totol_loss, f)
