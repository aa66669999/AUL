import copy
import csv
import time

from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from config import get_args
from util import *
from architecture import *


def train(model,
          dataloaders,
          criterion,
          optimizer,
          scheduler=None,
          num_epochs=None,
          device_train=None,
          num_l=None,
          fname=None,
          ):
    since = time.time()
    print('Using {} device'.format(device_train))
    # if not device_train:
    #     device_train = get_device()

    best_model_wts = copy.deepcopy(model.state_dict())
    loss_list = []
    micro_auc_list = []
    macro_auc_list = []

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)
            for phase in ["train"]:  # , "val"]:
                if phase == "train":
                    print("Training...")
                    model.train()  # Set model to training mode
                else:
                    print("Validating...")
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                all_labels = []
                all_probs = []

                for i, (inputs, labels) in enumerate(dataloaders[phase]):
                    # inputs = preprocess_data(inputs)
                    inputs = inputs
                    labels = labels

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        y = labels
                        # print('inputs',inputs)
                        # print('labels',labels)
                        # inputs = inputs.to(torch.int64).to(device_train)
                        batch_outputs = model(inputs)
                        # print('outputs',outputs)
                        loss = criterion(y.float(), batch_outputs, model, device=device_train)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item()  # * inputs.size(0)
                    probs = torch.sigmoid(batch_outputs)
                    all_labels.append(labels.cpu().numpy())
                    all_probs.append(probs.cpu().detach().numpy())
                # preds = predict(model,inputs)
                # match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                # acc = torch.mean(match)
                # print(acc)
                all_labels = np.concatenate(all_labels)
                all_probs = np.concatenate(all_probs)

                # Compute micro and macro AUC
                micro_auc = roc_auc_score(all_labels, all_probs, average='micro')
                macro_auc = roc_auc_score(all_labels, all_probs, average='macro')
                print('Micro AUC:', micro_auc)
                print('Macro AUC:', macro_auc)
                micro_auc_list.append(micro_auc)
                macro_auc_list.append(macro_auc)

                if scheduler is not None:
                    if phase == "train":
                        scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase]) # .dataset)
                # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                print('epoch loss', epoch_loss)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f'Parameter: {name}, Gradient: {param.grad.mean()}')
                    else:
                        print(f'Parameter: {name} has no gradient')

                loss_list.append(epoch_loss)
            print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    # # Calculate average AUC scores
    # avg_micro_auc = sum(micro_auc_list) / len(micro_auc_list)
    # avg_macro_auc = sum(macro_auc_list) / len(macro_auc_list)

    # Write results to CSV file
    # auc_result = [avg_micro_auc, avg_macro_auc]
    # with open('./result/' + fname, 'a') as f:
    #     writer_obj = csv.writer(f)
    #     writer_obj.writerow(auc_result)

    # metrics = (losses, accuracy)
    return model, loss_list

