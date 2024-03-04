import numpy as np
import torch


# 计算准确率
def accuracy(output, label):
    output = torch.nn.functional.softmax(output, dim=-1)
    output = output.data.cpu().numpy()
    output = np.argmax(output, axis=1)
    label = label.data.cpu().numpy()
    acc = np.mean((output == label).astype(int))
    return acc

def accuracy_recall(output, label):
    output = torch.nn.functional.softmax(output, dim=-1)
    output = output.data.cpu().numpy()
    output = np.argmax(output, axis=1)
    label = label.data.cpu().numpy()

    # 计算准确率
    acc = np.mean((output == label).astype(int))

    # 初始化用于计算召回率的变量
    unique_labels = np.unique(label)
    recall_sum = 0

    # 对每个类别计算召回率
    for lbl in unique_labels:
        true_positives = np.sum((output == lbl) & (label == lbl))
        actual_positives = np.sum(label == lbl)
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        recall_sum += recall

    # 计算平均召回率
    average_recall = recall_sum / len(unique_labels)

    return acc, average_recall