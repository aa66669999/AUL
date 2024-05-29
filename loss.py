import numpy as np
import torch
from torch import nn, tensor
import torch.nn.functional as F
from util import get_device


# def pairwise_ranking_loss(y_true, y_pred):
#     # 根据模型输出排序
#     sorted_indices = torch.argsort(y_pred, dim=0, descending=True)
#     sorted_labels = y_true[sorted_indices]
#
#     # 找到正样本和负样本的索引
#     positive_indices = torch.where(sorted_labels == 1)[0]
#     negative_indices = torch.where(sorted_labels == 0)[0]
#
#     # 确保正样本和负样本索引都不为空
#     if len(positive_indices) == 0 or len(negative_indices) == 0:
#         return torch.tensor(0.0)  # 如果没有找到正样本或负样本，返回零损失
#
#     # 计算 pairwise ranking loss
#     y_pred_positive = y_pred[positive_indices][:, None]
#     y_pred_negative = y_pred[negative_indices][None, :]
#     loss = torch.sum(torch.maximum(torch.tensor(0.0), 1.0 - y_pred_negative + y_pred_positive))
#
#     return loss


def pairwise_ranking_loss(y_true, y_pred):
    """
    Compute pairwise ranking loss.

    Parameters:
        y_true (tensor): True labels (binary, 0 or 1).
        y_pred (tensor): Predicted scores.

    Returns:
        loss (tensor): Pairwise ranking loss.
    """
    # Compute differences between pairs of predicted scores
    pairwise_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)

    # Create mask to filter out same pairs
    mask = torch.eye(y_true.size(0), dtype=torch.bool)

    # Compute pairwise ranking loss
    loss = torch.sum(torch.relu(1 - pairwise_diff[mask])) / torch.sum(mask.float())

    return loss


def ml_nn_loss(y, outputs, model, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    # non_mse = nn.MSELoss()
    # Multilabelsoftmarginloss:
    # non_mse = nn.MultiLabelSoftMarginLoss()
    # Non Multilabel Loss:
    # non_mse = nn.BCELoss()  # Binary Cross-Entropy Loss
    crossentropy_loss = nn.CrossEntropyLoss()
    surrogate_auc_loss_u2 = SurrogateAUCLossNatural()
    loss = crossentropy_loss(outputs, y)/50 + surrogate_auc_loss_u2(y, outputs)
    # loss = pairwise_ranking_loss(y, outputs)
    return loss


def get_auc_loss_u2(y, output, num_l, device=None):
    if not device:
        device = get_device()
    y = y.to(device)

    negative_elements_list = []
    positive_elements_list = []
    # use transposition for calculating every label.
    # and fill negative_elements_list and positive_elements_list.
    for row, mask_row in zip(output.t(), y.t()):
        positive_elements = [float(element) for element, mask in zip(row, mask_row) if mask == 1]
        positive_elements_list.append(positive_elements)
        negative_elements = [float(element) for element, mask in zip(row, mask_row) if mask == 0]
        negative_elements_list.append(negative_elements)

    def custom_max(x):
        return torch.max(torch.tensor(0), torch.tensor(1) - x)

    def hinge_loss(y_true, y_pred):
        return torch.max(torch.tensor(0.0), torch.tensor(1.0) - y_true * y_pred)

    sum_auc = 0
    for pos_list, neg_list in zip(positive_elements_list, negative_elements_list):
        for pos in pos_list:
            if len(pos_list) != 0:
                sum_auc += hinge_loss(1, pos) / len(pos_list)
        for neg in neg_list:
            if len(neg_list) != 0:
                sum_auc += hinge_loss(-1, -neg) / len(neg_list)
    auc_loss = (sum_auc / num_l if num_l != 0 else torch.tensor(0.0)).clone().detach().requires_grad_(True)

    return auc_loss


class SurrogateAUCLossDynamicWeighted(nn.Module):
    def __init__(self):
        super(SurrogateAUCLossDynamicWeighted, self).__init__()

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_true = y_true.view(batch_size, -1)  # Reshape the tensor to have a batch size and inferred second dimension
        y_pred = y_pred.view(batch_size, -1)

        loss = 0.0  # Initialize loss

        for label in range(y_true.size(1)):
            label_mask = (y_true[:, label] == 1)  # Mask for samples with the current label
            num_positives = torch.sum(label_mask)  # Count the number of positives for the current label
            num_negatives = torch.sum(~label_mask)  # Count the number of negatives for the current label

            if num_positives > 0 and num_negatives > 0:
                pos_scores = y_pred[label_mask]  # Predicted scores for positive samples
                neg_scores = y_pred[~label_mask]  # Predicted scores for negative samples

                # Ensure pos_scores and neg_scores have compatible shapes for broadcasting
                pos_scores = pos_scores.view(-1, 1)  # Reshape to column vector
                neg_scores = neg_scores.view(1, -1)  # Reshape to row vector

                # Calculate hinge loss
                hinge_loss = torch.mean(torch.clamp(1 - pos_scores + neg_scores, min=0))

                # Calculate label-specific weight
                weight = 1.0 / (num_positives * num_negatives)

                # Apply label-specific weight to the loss
                weighted_loss = weight * hinge_loss

                loss += weighted_loss

        return loss


class SurrogateAUCLossU1(nn.Module):
    def __init__(self):
        super(SurrogateAUCLossU1, self).__init__()

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_true = y_true.view(batch_size, -1)  # Reshaping the tensor to have a batch size and inferred second dimension
        y_pred = y_pred.view(batch_size, -1)


        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)

        pos_scores = y_pred[pos_mask]
        neg_scores = y_pred[neg_mask]

        # Reshaping pos_scores into a column vector and expanding to match neg_scores at second dim
        pos_scores = pos_scores.view(-1, 1).expand(-1, neg_scores.size(0))
        # Reshaping neg_scores into a row vector. copying the elements and expanding to match pos_scores at first dim
        neg_scores = neg_scores.view(1, -1).expand(pos_scores.size(0), -1)

        # The torch.mean() function calculates the mean value of all the elements in the tensor,
        # regardless of its dimension.
        hinge_loss = torch.mean(torch.clamp(1 - pos_scores + neg_scores, min=0))  # clamps all negative values to zero
        return hinge_loss


class SurrogateAUCLossNatural(nn.Module):
    def __init__(self):
        super(SurrogateAUCLossNatural, self).__init__()

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_true = y_true.view(batch_size, -1)  # Reshaping the tensor to have a batch size and inferred second dimension
        y_pred = y_pred.view(batch_size, -1)
        # print('y_true:', y_true.shape)
        # print('y_pred', y_pred.shape)

        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)

        pos_scores = y_pred[pos_mask]
        neg_scores = y_pred[neg_mask]

        # Reshaping pos_scores into a column vector and expanding to match neg_scores at second dim
        pos_scores = pos_scores.view(-1, 1).expand(-1, neg_scores.size(0))
        # Reshaping neg_scores into a row vector. copying the elements and expanding to match pos_scores at first dim
        neg_scores = neg_scores.view(1, -1).expand(pos_scores.size(0), -1)

        # The torch.mean() function calculates the mean value of all the elements in the tensor,
        # regardless of its dimension.
        hinge_loss = torch.mean(torch.clamp(1 - pos_scores + neg_scores, min=0))  # clamps all negative values to zero
        # lsep_pre = torch.mean(torch.log(1 + torch.exp(-pos_scores + neg_scores)))

        return hinge_loss


class SurrogateAUCLossCVPR(nn.Module):
    def __init__(self):
        super(SurrogateAUCLossCVPR, self).__init__()

    def forward(self, y_true, y_pred):
        batch_size = y_true.size(0)
        y_true = y_true.view(batch_size, -1)  # Reshaping the tensor to have a batch size and inferred second dimension
        y_pred = y_pred.view(batch_size, -1)
        # print('y_true:', y_true.shape)
        # print('y_pred', y_pred.shape)

        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)

        pos_scores = y_pred[pos_mask]
        neg_scores = y_pred[neg_mask]

        # Reshaping pos_scores into a column vector and expanding to match neg_scores at second dim
        pos_scores = pos_scores.view(-1, 1).expand(-1, neg_scores.size(0))
        # Reshaping neg_scores into a row vector. copying the elements and expanding to match pos_scores at first dim
        neg_scores = neg_scores.view(1, -1).expand(pos_scores.size(0), -1)

        # The torch.mean() function calculates the mean value of all the elements in the tensor,
        # regardless of its dimension.
        # hinge_loss = torch.mean(torch.clamp(1 - pos_scores + neg_scores, min=0))  # clamps all negative values to zero
        lsep_pre = torch.mean(torch.log(1 + torch.exp(-pos_scores + neg_scores)))

        return lsep_pre

class MacroAUCLossNatural(nn.Module):
    def __init__(self):
        super(MacroAUCLossNatural, self).__init__()

    def forward(self, y_true, y_pred):

        batch_size, num_labels = y_true.size()

        # Initialize a list to store the rank loss for each label
        rank_losses = []

        # Loop over each label
        for label in range(num_labels):
            y_true_label = y_true[:, label]
            y_pred_label = y_pred[:, label]

            pos_mask = (y_true_label == 1)
            neg_mask = (y_true_label == 0)

            pos_scores = y_pred_label[pos_mask]
            neg_scores = y_pred_label[neg_mask]

            if pos_scores.size(0) == 0 or neg_scores.size(0) == 0:
                # If there are no positive or negative examples for this label, skip the calculation
                continue

            # Reshaping pos_scores into a column vector and expanding to match neg_scores at the second dimension
            pos_scores = pos_scores.view(-1, 1).expand(-1, neg_scores.size(0))
            # Reshaping neg_scores into a row vector and expanding to match pos_scores at the first dimension
            neg_scores = neg_scores.view(1, -1).expand(pos_scores.size(0), -1)

            # Calculate the hinge loss for the current label
            # a = 1 - pos_scores + neg_scores
            # hinge_loss = torch.mean(torch.clamp(a, min=0))  # Clamps all negative values to zero

            # # Lu2:
            # a = torch.clamp(1 - pos_scores, min=0) + torch.clamp(1 + neg_scores, min=0)
            # hinge_loss = torch.mean(a) # Clamps all negative values to zero

            # log-sum-exp pairwise loss:
            lsep_pre = torch.mean(torch.log(1 + torch.exp(-pos_scores + neg_scores)))

            # def tanh_function(x):
            #     return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
            # tanh_loss = torch.mean(tanh_function(pos_scores) + tanh_function(-neg_scores))

            # def logistic_function(x):
            #     # Logistic function
            #     logistic_value = (1 + torch.exp(-x))
            #     # Apply base-2 logarithm
            #     log_base_2_value = torch.log2(logistic_value)
            #     return torch.mean(log_base_2_value)
            # logistic_loss = logistic_function(pos_scores) + logistic_function(-neg_scores)

            # Append the hinge loss for the current label to the list
            rank_losses.append(lsep_pre)
            # rank_losses.append(logistic_loss)

        # Calculate the mean rank loss across all labels
        if rank_losses:
            final_loss = torch.mean(torch.stack(rank_losses))
        else:
            final_loss = torch.tensor(0.0)

        return final_loss

def ml_nn_loss2(targets, outputs, model, device=None):
    # if not device:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # targets = targets.to(device)
    # outputs = outputs.to(device)

    # Apply sigmoid activation if not already applied in the model
    if not isinstance(model, nn.Sequential) or not isinstance(model[-1], nn.Sigmoid):
        outputs = torch.sigmoid(outputs)
    # bce_logits_loss = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss

    # soft_margin_loss = nn.MultiLabelSoftMarginLoss()
    # loss = soft_margin_loss(outputs, targets)

    # gpt_surrogate_auc_loss = SurrogateAUCLossDynamicWeighted()
    # loss = gpt_surrogate_auc_loss(targets, outputs)

    surrogate_auc_loss_u2 = SurrogateAUCLossCVPR()
    # surrogate_auc_loss_u2 = SurrogateAUCLossNatural()
    # surrogate_auc_loss_u2 = MacroAUCLossNatural()
    loss = surrogate_auc_loss_u2(targets, outputs)

    # label_length = targets.size(1)
    # auc_loss = get_auc_loss_u2(targets, outputs, label_length, device)
    # loss = auc_loss
    # loss += 0.2*auc_loss

    return loss


def ml_nn_loss1(targets, outputs, model, device=None):
    if not device:
        device = get_device()
    targets = targets.to(device)
    alpha = 0.1  # You can adjust the weight of the surrogate loss
    # loss = nn.BCEWithLogitsLoss(outputs, y)  # Binary Cross-Entropy Loss

    soft_margin_loss = nn.MultiLabelSoftMarginLoss()
    loss = soft_margin_loss(outputs, targets)
    return loss


def ml_nn_loss_regularization(y, outputs, model, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    # BCE_logits_loss = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss

    # Apply sigmoid activation if not already applied in the model
    if not isinstance(model, nn.Sequential) or not isinstance(model[-1], nn.Sigmoid):
        outputs = torch.sigmoid(outputs)

    soft_margin_loss = nn.MultiLabelSoftMarginLoss()
    loss = soft_margin_loss(outputs, y)  # Add L2 regularization term
    l2_regularization = torch.tensor(0., device=device)
    for param in model.parameters():
        l2_regularization += torch.norm(param, p=2) ** 2
    weight_decay = 0.01
    loss += 0.01 * weight_decay * l2_regularization
    return loss


def ml_nn_loss_surrogate_auc(y, outputs, model, device=None):
    if not device:
        device = get_device()

    y = y.to(device)

    # Calculate the Hinge loss
    loss = F.hinge_embedding_loss(outputs, y.float(), margin=1.0)

    # Add L2 regularization term
    l2_regularization = torch.tensor(0., device=device)
    for param in model.parameters():
        l2_regularization += torch.norm(param, p=2) ** 2

    weight_decay = 0.01
    loss += 0.5 * weight_decay * l2_regularization

    return loss