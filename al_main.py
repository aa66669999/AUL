import time

from data import get_data
from loss import *
import csv
import os
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import torch
from config import get_args
from loss import *
from test import add_res
from pretrain import train
from util import *
from architecture import *
from data import get_data
from active_train import ActiveLearning

args = get_args()

check_path = './result/data.csv'
if os.path.exists(check_path):
    os.remove(check_path)

fname = args.data_name
fnamesub = fname + '.csv'
header = ['K@1', 'K@3', 'K@5', 'micro_AUC', 'macro_AUC']

with open('./result/' + fnamesub, 'w') as f:
    writer_obj = csv.writer(f)
    writer_obj.writerow(header)

# get GPU or CPU
device = get_device()
# device = 'cpu'

# for i in range(10, 0, -1):
results = []

# get Dataloader
train_data, pool_data, test_data, train_origin, pool_origin = get_data(train_ratio=args.train, pool_ratio=args.pool,test_ratio=args.test)

num_labels = test_data.label_length()
# print(test_label_length)
out_size = num_labels
num_features = train_data.x.size(1)
def get_new_resnet18():
    active_model = ResNet18(in_size=num_features, hidden_size=args.m_hidden, out_size=out_size, embed=args.m_embed,
                            drop_p=args.m_drop_p, activation=args.m_activation)
    return active_model

#linear_model = ResNet18(in_size=num_features, hidden_size=args.m_hidden, out_size=out_size, embed=args.m_embed,
#                       drop_p=args.m_drop_p, activation=args.m_activation).to(device)

#train_model = linear_model
#
# # train_data.change_x_data(check_cv_and_preprocess_data(train_model, train_data))
#
# train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
# dataloaders = {
#     "train": train_dataloader,
#     "val": test_dataloader,
# }
#
# # check total parameters of this model
# pytorch_total_params = sum(p.numel() for p in train_model.parameters())
# print(" Number of Parameters: ", pytorch_total_params)
#
# optimizer = optim.Adam(train_model.parameters(), lr=args.lr, weight_decay=args.wd)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
# criterion = ml_nn_loss2  # ml_nn_loss2
#
# pretrain_model, loss_opt = train(train_model,
#                                  dataloaders,
#                                  criterion=criterion,
#                                  optimizer=optimizer,
#                                  scheduler=scheduler,
#                                  num_epochs=10,
#                                  device_train=device,
#                                  num_l=num_labels,
#                                  fname=fnamesub
#                                  )
#
# pretrain_model.eval()
# results += add_res(pretrain_model, test_data.get_x(), test_data.get_y(), device=device)
# print(results)
#
# if results:
#     with open('./result/' + fnamesub, 'a', newline='') as f:  # Add newline='' to avoid extra empty lines
#         writer_obj = csv.writer(f)
#         writer_obj.writerow(results)

active_learning = ActiveLearning(train_origin,pool_origin,train_data, pool_data, device_al_train=device)
#active_model = get_new_resnet18().to(device)
for i in range(args.active_rounds, 0, -1):
    since = time.time()
    active_model = get_new_resnet18().to(device)

    active_learning.select_instances_entropy(active_model, n_instances=args.active_instances)

    active_learning.train_model(active_model,args)

    al_results = add_res(active_model, test_data.get_x(), test_data.get_y(), device=device)
    time_elapsed = time.time() - since
    print(
        'There are ', i - 1, "round of active learning left. This round complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print(al_results)

    if al_results:
        with open('./result/' + fnamesub, 'a', newline='') as f:  # Add newline='' to avoid extra empty lines
            writer_obj = csv.writer(f)
            writer_obj.writerow(al_results)
