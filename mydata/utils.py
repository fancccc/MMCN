# -*- coding: utf-8 -*-
'''
@file: utils.py
@author: fanc
@time: 2024/5/10 19:49
'''
import matplotlib.pyplot as plt
import os
def plot_curve(train_loss_history, train_acc_history, val_loss_history, val_acc_history, save_path, validation_interval=5):
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    epochs = len(train_loss_history)
    validation_epochs = range(validation_interval - 1, epochs, validation_interval)
    # 确保验证数据点的数量与validation_epochs的数量相匹配
    val_len = min(len(val_loss_history), len(validation_epochs))
    validation_epochs = list(validation_epochs)[:val_len]
    val_loss_history = val_loss_history[:val_len]
    val_acc_history = val_acc_history[:val_len]

    plt.close('all')
    fig, ax1 = plt.subplots()
    # 绘制训练损失和准确率
    ax1.plot(range(epochs), train_loss_history, 'r-', label='Training Loss')
    ax1.plot(range(epochs), train_acc_history, 'b-', label='Training Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Metrics')
    ax1.tick_params(axis='y')
    # 实例化第二个y轴
    ax2 = ax1.twinx()
    # 绘制验证损失和准确率
    if val_loss_history:
        ax2.plot(validation_epochs, val_loss_history, 'r--', label='Validation Loss')
    if val_acc_history:
        ax2.plot(validation_epochs, val_acc_history, 'b--', label='Validation Accuracy')
    ax2.set_ylabel('Validation Metrics')
    ax2.tick_params(axis='y')
    # 图例和网格
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    ax1.grid(True)
    plt.title('Training and Validation Metrics')
    plt.savefig(os.path.join(save_path, 'Metrics.png'))

