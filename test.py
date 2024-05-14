import os

import torch
import random
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, accuracy_score
import csv
from config import get_args


def add_res(model_opt, x_test, y_test, n=None,
            xtest=None, ytest=None, device=None):
    if xtest is None:
       # xtest = x_test.clone().detach().cpu()
       # ytest = y_test.clone().detach().cpu().requires_grad_(True)
        xtest = x_test
        ytest = y_test

    #model_opt = model_opt.cpu()
    output = model_opt(xtest.float().cuda())

    ground_truth = ytest
    output = torch.sigmoid(output)
    # mse = mean_squared_error(ground_truth.detach().cpu().numpy(), output.detach().cpu().numpy())

    # Convert output to binary predictions (for binary classification)
    # Calculate the average and variance of the output probabilities
    #average_prob = output.mean(dim=1)
    #variance = output.var(dim=1)

    # # Threshold calculation
    # threshold = average_prob + variance/2  # Adjust this formula based on your requirements
    # # Thresholding to get binary predictions
    # predictions = (output >= threshold[:, None]).float()  # Unsqueeze to match dimensions for broadcasting
    # accuracy = accuracy_score(ground_truth.detach().cpu().numpy(), predictions.detach().cpu().numpy())
    # f1 = f1_score(ground_truth.detach().cpu().numpy(), predictions.detach().cpu().numpy(), average='macro')

    precision_at_1 = precision_at_k(output.detach().cpu(), ground_truth.detach().cpu(), 1)
    precision_at_3 = precision_at_k(output.detach().cpu(), ground_truth.detach().cpu(), 3)
    precision_at_5 = precision_at_k(output.detach().cpu(), ground_truth.detach().cpu(), 5)

    macro_auc = roc_auc_score(ground_truth.detach().cpu().numpy(), output.detach().cpu().numpy(), average='macro')
    micro_auc = roc_auc_score(ground_truth.detach().cpu().numpy(), output.detach().cpu().numpy(), average='micro')

    file_path = './result/data.csv'

    with open(file_path, 'a', newline='') as f:
        writer_obj = csv.writer(f)
        writer_obj.writerow(output.detach().cpu().numpy()[1])

    for i, j in zip(ground_truth.detach().cpu().numpy(), output.detach().cpu().numpy()):
        a = []
        for element_gt, element_output in zip(i, j):
            if element_gt == 1:
                a += [element_gt]
                a += [element_output]
        with open(file_path, 'a', newline='') as f:
            writer_obj = csv.writer(f)
            writer_obj.writerow(a)

    return precision_at_1, precision_at_3, precision_at_5, micro_auc, macro_auc


def precision_at_k(output, target, k):
    """
    Compute precision at K for multilabel classification.

    Args:
    - output (torch.Tensor): Predicted probabilities (batch_size, num_classes).
    - target (torch.Tensor): Ground truth labels (batch_size, num_classes).
    - k (int): Number of top predictions to consider.

    Returns:
    - precision (float): Precision at K.
    """

    # Sort predicted probabilities for each sample
    _, indices = torch.topk(output, k, dim=1)

    # gather the corresponding ground truth labels for the top K predictions.
    top_k_predictions = torch.gather(target, 1, indices)

    # Calculate precision at K for each sample
    batch_size = output.size(0)
    precisions = torch.sum(top_k_predictions, dim=1).float() / k

    # Average precision across all samples
    precision = torch.sum(precisions) / batch_size

    return precision.item()

# Example usage:
# output = torch.tensor([[0.8, 0.3, 0.6], [0.2, 0.7, 0.9]])
# target = torch.tensor([[1, 0, 1], [0, 1, 1]])
# k = 2
# precision = precision_at_k(output, target, k)
# print("Precision at {}:".format(k), precision)