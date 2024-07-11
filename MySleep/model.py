import torch
import torch.nn as nn
import os
import timeit
import numpy as np
import sklearn.metrics as skmetrics
from network import TinySleepNet
from torch.optim import Adam
from tensorboardX import SummaryWriter
import logging
import math
from random import random
logger = logging.getLogger("default_log")
import torch.nn.functional as F

# 支持多分类和二分类
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """
 
    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
 
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
 
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
 
    def forward(self, input, target):
        logit = F.softmax(input, dim=1) #这里看情况选择，如果之前softmax了，后续就不用了
 
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
 
        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
 
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
 
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
 
        gamma = self.gamma
 
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
 
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class Model:
    def __init__(self, config=None, output_dir="./output", use_rnn=False, testing=False, use_best=False, device=None):
        self.tsn = TinySleepNet(config)
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")
        self.device = device
        self.tsn.to(device)
        self.T0=100
        self.T=self.T0
        self.Tf=0.01
        self.alpha=0.8


        self.optimizer_all = Adam(self.tsn.parameters(),
            lr=config['learning_rate'], betas=(config["adam_beta_1"], config["adam_beta_2"]),
            eps=config["adam_epsilon"])
        # self.CE_loss = nn.CrossEntropyLoss(reduce=False)
        self.CE_loss = FocalLoss(self.config["n_classes"])

        self.train_writer = SummaryWriter(os.path.join(self.log_dir, "train"))
        self.train_writer.add_graph(self.tsn, input_to_model=(torch.rand(size=(self.config['batch_size']*self.config['seq_length'], 1, self.config['input_size'])).to(device), (torch.zeros(size=(1, self.config['batch_size'], 128)).to(device), torch.zeros(size=(1, self.config['batch_size'], 128)).to(device))))

        self.global_epoch = 0
        self.global_step = 0

        if testing and use_best:  # load form best checkpoint
            best_ckpt_path = os.path.join(self.best_ckpt_path, "best_model.ckpt")
            self.tsn.load_state_dict(torch.load(best_ckpt_path))
            logger.info(f'load best model from {best_ckpt_path}')






    def Metrospolis(self, f, f_new):   #Metropolis准则
        if f_new <= f:
            return 1
        else:
            p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0


    def get_current_epoch(self):
        return self.global_epoch

    def pass_one_epoch(self):
        self.global_epoch = self.global_epoch + 1

    def train_with_dataloader(self, minibatches,logger):
        self.tsn.train()

        start = timeit.default_timer()
        preds, trues, losses, outputs = ([], [], [], {})
        # 用来模拟退火的变量
        loss_=0
        num=0
        for x, y, w, sl, re in minibatches:
            # w is used to mark whether the sample is true, if the sample is filled with zero, w == 0
            # while calculating loss, multiply with w
            x = torch.from_numpy(x).view(self.config['batch_size'] * self.config['seq_length'], 1, self.config['input_size'])  # shape(batch_size* seq_length, in_channels, input_length)
            y = torch.from_numpy(y).long()
            w = torch.from_numpy(w)
            if re:  # Initialize state of RNN
                state = (torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])),
                         torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])))
                state = (state[0].to(self.device), state[1].to(self.device))
            self.optimizer_all.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            w = w.to(self.device)
            y_pred, state = self.tsn.forward(x, state)
            state = (state[0].detach(), state[1].detach())
            loss = self.CE_loss(y_pred, y)
            # weight by sample
            loss = torch.mul(loss, w)
            # Weight by class
            one_hot = torch.zeros(len(y), self.config["n_classes"]).to(self.device).scatter_(1, y.unsqueeze(dim=1), 1)
            sample_weight = torch.mm(one_hot, torch.Tensor(self.config["class_weights"]).to(self.device).unsqueeze(dim=1)).view(-1)  # (300, 5) * (5,) = (300,)
            loss = torch.mul(loss, sample_weight).sum() / w.sum()

            cnn_weights = [parm for name, parm in self.tsn.cnn.named_parameters() if 'conv' in name]
            # EEGNet
            # cnn_weights = [parm for name, parm in self.tsn.EEGNetLayer.named_parameters() if 'conv' in name]
            reg_loss = 0
            for p in cnn_weights:
                reg_loss += torch.sum(p ** 2) / 2
            reg_loss = self.config["l2_weight_decay"] * reg_loss
            ce_loss = loss
            # print(f"ce loss {ce_loss:.2f}, reg loss {reg_loss:.2f}")
            loss = loss + reg_loss
            # logger.info("loss is {}".format(loss))
            tmp_preds = np.reshape(np.argmax(y_pred.cpu().detach().numpy(), axis=1),
                                   (self.config["batch_size"], self.config["seq_length"]))
            tmp_trues = np.reshape(y.cpu().detach().numpy(), (self.config["batch_size"], self.config["seq_length"]))
            # acc = skmetrics.accuracy_score(y_true=tmp_trues[i, :sl[i]],y_pred=tmp_preds[i, :sl[i]])
            # logger.info("acc is {}".format(acc))
            for i in range(self.config["batch_size"]):
                preds.extend(tmp_preds[i, :sl[i]])
                trues.extend(tmp_trues[i, :sl[i]])
            acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)

            # logger.info("acc is {}".format(acc))
            if(self.Metrospolis(loss_,loss)):
                # loss变小了，说明走的方向正确，才应该反向传播
                if(self.global_epoch>89):
                    name = "model"+str(num)+".pt";
                    torch.save(self.tsn,name)
                    # logger.info("loss is {}".format(loss))
                    # logger.info("acc is {}".format(acc))
                loss.backward()

                nn.utils.clip_grad_norm_(self.tsn.parameters(), max_norm=self.config["clip_grad_value"], norm_type=2)
                self.optimizer_all.step()
                losses.append(loss.detach().cpu().numpy())
            
            
            num=num+1
            self.global_step += 1

        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        all_loss = np.array(losses).mean()
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs.update({
            "global_step": self.global_step,
            "train/trues": trues,
            "train/preds": preds,
            "train/accuracy": acc,
            "train/loss": all_loss,
            "train/f1_score": f1_score,
            "train/cm": cm,
            "train/duration": duration,

        })
        self.global_epoch += 1
        return outputs

    # without 模拟退火
    def train_with_dataloader_NSA(self, minibatches,logger):
        self.tsn.train()

        start = timeit.default_timer()
        preds, trues, losses, outputs = ([], [], [], {})
        for x, y, w, sl, re in minibatches:
            # w is used to mark whether the sample is true, if the sample is filled with zero, w == 0
            # while calculating loss, multiply with w
            x = torch.from_numpy(x).view(self.config['batch_size'] * self.config['seq_length'], 1, self.config['input_size'])  # shape(batch_size* seq_length, in_channels, input_length)
            y = torch.from_numpy(y).long()
            w = torch.from_numpy(w)
            if re:  # Initialize state of RNN
                state = (torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])),
                         torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])))
                state = (state[0].to(self.device), state[1].to(self.device))
            self.optimizer_all.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            w = w.to(self.device)
            y_pred, state = self.tsn.forward(x, state)
            state = (state[0].detach(), state[1].detach())
            loss = self.CE_loss(y_pred, y)
            # weight by sample
            loss = torch.mul(loss, w)
            # Weight by class
            one_hot = torch.zeros(len(y), self.config["n_classes"]).to(self.device).scatter_(1, y.unsqueeze(dim=1), 1)
            sample_weight = torch.mm(one_hot, torch.Tensor(self.config["class_weights"]).to(self.device).unsqueeze(dim=1)).view(-1)  # (300, 5) * (5,) = (300,)
            loss = torch.mul(loss, sample_weight).sum() / w.sum()

            cnn_weights = [parm for name, parm in self.tsn.cnn.named_parameters() if 'conv' in name]
            reg_loss = 0
            for p in cnn_weights:
                reg_loss += torch.sum(p ** 2) / 2
            reg_loss = self.config["l2_weight_decay"] * reg_loss
            ce_loss = loss
            # print(f"ce loss {ce_loss:.2f}, reg loss {reg_loss:.2f}")
            loss = loss + reg_loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.tsn.parameters(), max_norm=self.config["clip_grad_value"], norm_type=2)
            self.optimizer_all.step()
            losses.append(loss.detach().cpu().numpy())
            self.global_step += 1
            tmp_preds = np.reshape(np.argmax(y_pred.cpu().detach().numpy(), axis=1), (self.config["batch_size"], self.config["seq_length"]))
            tmp_trues = np.reshape(y.cpu().detach().numpy(), (self.config["batch_size"], self.config["seq_length"]))
            for i in range(self.config["batch_size"]):
                preds.extend(tmp_preds[i, :sl[i]])
                trues.extend(tmp_trues[i, :sl[i]])

        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        all_loss = np.array(losses).mean()
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs.update({
            "global_step": self.global_step,
            "train/trues": trues,
            "train/preds": preds,
            "train/accuracy": acc,
            "train/loss": all_loss,
            "train/f1_score": f1_score,
            "train/cm": cm,
            "train/duration": duration,

        })
        self.global_epoch += 1
        return outputs

    def evaluate_with_dataloader(self, minibatches):
        self.tsn.eval()
        start = timeit.default_timer()
        preds, trues, losses, outputs = ([], [], [], {})
        with torch.no_grad():
            for x, y, w, sl, re in minibatches:
                x = torch.from_numpy(x).view(self.config['batch_size'] * self.config['seq_length'], 1,
                                             self.config['input_size'])  # shape(batch_size* seq_length, in_channels, input_length)
                y = torch.from_numpy(y).long()
                w = torch.from_numpy(w)

                if re:
                    state = (torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])),
                             torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])))
                    state = (state[0].to(self.device), state[1].to(self.device))

                # Carry the states from the previous batches through time  # 在测试时,将上一批样本的lstm状态带入下一批样本

                x = x.to(self.device)
                y = y.to(self.device)
                w = w.to(self.device)

                # summary(self.tsn, x, state)
                # exit(0)
                y_pred, state = self.tsn.forward(x, state)
                state = (state[0].detach(), state[1].detach())
                loss = self.CE_loss(y_pred, y)
                # weight by sample
                loss = torch.mul(loss, w)
                # Weight by class
                one_hot = torch.zeros(len(y), self.config["n_classes"]).to(self.device).scatter_(1, y.unsqueeze(dim=1),
                                                                                                 1)
                sample_weight = torch.mm(one_hot, torch.Tensor(self.config["class_weights"]).to(self.device).unsqueeze(
                    dim=1)).view(-1)  # (300, 5) * (5,) = (300,)
                loss = torch.mul(loss, sample_weight).sum() / w.sum()

                losses.append(loss.detach().cpu().numpy())
                tmp_preds = np.reshape(np.argmax(y_pred.cpu().detach().numpy(), axis=1),
                                       (self.config["batch_size"], self.config["seq_length"]))
                tmp_trues = np.reshape(y.cpu().detach().numpy(), (self.config["batch_size"], self.config["seq_length"]))
                for i in range(self.config["batch_size"]):
                    preds.extend(tmp_preds[i, :sl[i]])
                    trues.extend(tmp_trues[i, :sl[i]])
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        all_loss = np.array(losses).mean()
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs = {
            "test/trues": trues,
            "test/preds": preds,
            "test/loss": all_loss,
            "test/accuracy": acc,
            "test/f1_score": f1_score,
            "test/cm": cm,
            "test/duration": duration,
        }
        return outputs

    def save_best_checkpoint(self, name):
        if not os.path.exists(self.best_ckpt_path):
            os.makedirs(self.best_ckpt_path)
        save_path = os.path.join(self.best_ckpt_path, "{}.ckpt".format(name))
        torch.save(self.tsn.state_dict(), save_path)
        logger.info("Saved best checkpoint to {}".format(save_path))


if __name__ == '__main__':
    from torchsummaryX import summary
    from config.sleepedf import train
    model = TinySleepNet(config=train)
    summary(model, torch.randn(size=(2, 1, 3000)))



