# -*- coding: utf-8 -*-
'''
@file: mytrain.py
@author: fanc
@time: 2024/5/10 17:46
'''
from accelerate import optimizer
import matplotlib.pyplot as plt
from mydata.nets import PatchNet, mresnet50, mresnet18, mresnet34, mresnet101
from mydata.dataloader import PatchDataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import os
import time
from mydata.loss import FocalLoss
from mydata.utils import plot_curve
import logging
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
# criterion = FocalLoss(alpha=[1-0.62, 1-0.17, 1-0.16, 1-0.045], device=device)

def setup_logger(log_directory, log_filename='training.log'):
    # 确保日志目录存在，如果不存在则创建
    # if not os.path.exists(log_directory):
    #     os.makedirs(log_directory)

    # 创建一个logger
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)  # 设置日志记录的最低级别

    # 创建一个handler，用于写入日志文件，指定路径
    file_path = os.path.join(log_directory, log_filename)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)

    # 创建一个handler，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
class Trainer:
    def __init__(self, model, optimizer, scheduler, epochs, train_loader, val_loader, save_path, device, weights_path=None):
        self.model = model
        self.num_classes = 5
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.floss = FocalLoss(alpha=[1-0.03, 1-0.12, 1-0.007, 1-0.028, 1-0.82], device=device)
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
        self.output_prob = []
        self.true_label = []
        self.precision_history = []
        self.recall_history = []
        self.auc_history = []
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_acc = 0
        self.eval_best = 0
        self.save_path = save_path
        self.device = device
        self.clsses_acc = [0, 0, 0, 0, 0]
        self.epoch = 0
        # self.phase = 'val'
        if weights_path:
            self.load_model(weights_path)

    def load_model(self, weights_path):
        self.model = torch.load(weights_path)
        print("Loaded model from {}".format(weights_path))
    def train(self):
        # self.phase = 'train'
        self.logger = setup_logger(self.save_path)
        self.logger.info("Trainer has been set up.")
        # self.model.train()
        for epoch in range(self.epochs):
            self.epoch = epoch+1
            self.train_epoch()
            if self.epoch % 5 == 0:
                self.scheduler.step()
                self.logger.info(f"lr: {self.optimizer.param_groups[0]['lr']}, 'params' {sum(p.numel() for p in model.parameters())}")
                self.eval()
            plot_curve(self.train_loss_history,
                       self.train_acc_history,
                       self.val_loss_history,
                       self.val_acc_history,
                       self.save_path,
                       validation_interval=5
                       )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = list(0. for i in range(self.num_classes))
        class_total = list(0. for i in range(self.num_classes))
        for img, text, label in self.train_loader:
            img = img.to(self.device)
            text = text.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model([img, text])
            # self.logger.info(output)
            loss = self.floss(output, label)
            loss.backward()
            self.optimizer.step()
            self.update_metrics(output, label)
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            c = (predicted == label).squeeze()
            for i in range(len(label)):
                lbl = label[i]
                class_correct[lbl] += c[i].item()
                class_total[lbl] += 1

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        self.save()
        if self.best_acc < epoch_acc:
            self.best_acc = epoch_acc
            self.save('best')
        self.train_loss_history.append(epoch_loss)
        self.train_acc_history.append(epoch_acc)
        class_accuracies = [class_correct[i] / class_total[i] if class_total[i] != 0 else 0 for i in
                            range(self.num_classes)]
        self.clsses_acc.append(class_accuracies)
        self.auc_history.append(roc_auc_score)
        self.compute_metrics()
        self.logger.info(
                f'Epoch Train: {self.epoch}/{self.epochs}, '
                f'Loss: {epoch_loss:.4f}, '
                f'Accuracy: {epoch_acc:.4f}, '
                f'Class Acc: {class_accuracies}, '
                f'p: {self.precision_history[-1]}, '
                f'r: {self.recall_history[-1]}, '
                f'auc: {self.auc:.4f}')

    def eval(self):
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = list(0. for i in range(self.num_classes))
        class_total = list(0. for i in range(self.num_classes))
        with torch.no_grad():
            self.model.eval()
            for img, text, label in self.val_loader:
                img = img.to(self.device)
                text = text.to(self.device)
                label = label.to(self.device)
                output = self.model([img, text])

                self.update_metrics(output, label)
                loss = self.floss(output, label)
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)

                total += label.size(0)
                correct += (predicted == label).sum().item()
                c = (predicted == label).squeeze()
                for i in range(len(label)):
                    lbl = label[i]
                    class_correct[lbl] += c[i].item()
                    class_total[lbl] += 1
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total
        if self.phase == 'train':
            if self.eval_best < epoch_acc:
                self.eval_best = epoch_acc
                self.save('val_best')

        class_accuracies = [class_correct[i] / class_total[i] if class_total[i] != 0 else 0 for i in
                            range(self.num_classes)]
        self.val_loss_history.append(epoch_loss)
        self.val_acc_history.append(epoch_acc)
        self.clsses_acc.append(class_accuracies)
        self.compute_metrics()
        try:
            self.logger.info(
                f'Epoch Val: {self.epoch}/{self.epochs}, '
                f'Loss: {epoch_loss:.4f}, '
                f'Accuracy: {epoch_acc:.4f}, '
                f'Class Acc: {class_accuracies}, '
                f'p: {self.precision_history[-1]}, '
                f'r: {self.recall_history[-1]}, '
                f'auc: {self.auc:.4f}')
        except:
            print(
                f'Epoch Train: {self.epoch}/{self.epochs}, '
                f'Loss: {epoch_loss:.4f}, '
                f'Accuracy: {epoch_acc:.4f}, '
                f'Class Acc: {class_accuracies}, '
                f'p: {self.precision_history[-1]}, '
                f'r: {self.recall_history[-1]}, '
                f'auc: {self.auc:.4f}')
    def save(self, mode='last'):
        torch.save(self.model, os.path.join(self.save_path, f'model_{mode}.pt'))
        if mode == 'best':
            self.logger.info(f'Model saved to {os.path.join(self.save_path, f"model_{mode}.pt")}')

    def update_metrics(self, output, label):
        _, predicted = torch.max(output.data, 1)
        self.output_prob.append(torch.softmax(output, dim=1))
        self.true_label.append(label)
        for i in range(self.num_classes):
            tp_mask = (predicted == i) & (label == i)
            fp_mask = (predicted == i) & (label != i)
            fn_mask = (predicted != i) & (label == i)

            self.tp[i] += tp_mask.sum().item()
            self.fp[i] += fp_mask.sum().item()
            self.fn[i] += fn_mask.sum().item()

    def compute_metrics(self):
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        self.precision_history.append(precision.mean().item())
        self.recall_history.append(recall.mean().item())
        self.output_prob = torch.cat(self.output_prob, dim=0).detach().cpu().numpy()
        self.true_label = torch.cat(self.true_label, dim=0).cpu().numpy()
        if np.isnan(self.output_prob).any():

            self.logger.error("Warning: NaN found in output_prob")
        if np.isnan(self.true_label).any():
            self.logger.error("Warning: NaN found in true_label")
        try:
            self.auc = roc_auc_score(self.true_label, self.output_prob, multi_class='ovr', average='macro')
        except ValueError as e:
            self.logger.error(e)
            self.auc = 0
        self.plot_roc_curve()

        self.output_prob = []
        self.true_label = []
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)


    def plot_roc_curve(self):
        # 假设 self.true_label 和 self.output_prob 是保存的真实标签和预测概率
        # self.true_label = np.concatenate(self.true_label)
        # self.output_prob = np.concatenate(self.output_prob)

        # Binarize the true labels
        y_true_binarized = label_binarize(self.true_label, classes=range(self.num_classes))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], self.output_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), self.output_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.num_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= self.num_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)

        for i in range(self.num_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        # plt.savefig(os.path.join(self.save_path, 'roc_curve.png'))
        plt.savefig(os.path.join('./results/roc_curve.png'))
        # plt.show()



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    epochs = 50
    num_classes = 5
    phase = 'train'
    weights_path = None
    # weights_path = '/home/zcd/codes/YOLOv8/ultralytics/results/202405141842/model_best.pt' # ours2
    # weights_path = '/home/zcd/codes/YOLOv8/ultralytics/results/202405140905/model_last.pt' # 101
    # weights_path = '/home/zcd/codes/YOLOv8/ultralytics/results/202405121355/model_last.pt' # 50
    # weights_path = '/home/zcd/codes/YOLOv8/ultralytics/results/202405141529/model_last.pt' # 34
    print(device)
    # model = PatchNet(num_classes=num_classes, device=device)
    model = mresnet50(num_classes=num_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001, betas=(0.9, 0.99))
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    datetime = time.strftime("%Y%m%d%H%M", time.localtime())
    save_dir = os.path.join(f'./results/{datetime}')
    print(save_dir)
    if phase == 'train':
        os.makedirs(save_dir, exist_ok=True)
    img_dir = '/home/zcd/datasets/DENTEX/training_data/quadrant_enumeration_disease/infer_cropped'
    text_dir = '/home/zcd/datasets/DENTEX/training_data/quadrant_enumeration_disease/infer_cropped_texts'
    dataset = PatchDataset(img_dir, text_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    img_dir = '/home/zcd/datasets/DENTEX/training_data/quadrant_enumeration_disease/infer_cropped_val'
    text_dir = '/home/zcd/datasets/DENTEX/training_data/quadrant_enumeration_disease/infer_cropped_texts_val'
    dataset = PatchDataset(img_dir, text_dir)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    trainer = Trainer(model, optimizer, scheduler, epochs, train_loader, val_loader, save_dir, device, weights_path)
    trainer.phase = phase
    if phase == 'train':
        trainer.train()
    trainer.eval()