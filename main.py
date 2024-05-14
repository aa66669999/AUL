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

args = get_args()

check_path = './result/data.csv'
if os.path.exists(check_path):
    os.remove(check_path)

fname = args.data_name
fnamesub = fname + '.csv'
header = ['K@1', 'K@2', 'micro_AUC', 'macro_AUC']

with open('./result/' + fnamesub, 'w') as f:
    writer_obj = csv.writer(f)
    writer_obj.writerow(header)

# get GPU or CPU
device = get_device()
results = []

# get Dataloader
train_data, pool_data, test_data = get_data(train_ratio=args.train, pool_ratio=args.pool, test_ratio=args.test)

num_labels = test_data.label_length()
# print(test_label_length)
out_size = num_labels
num_features = train_data.x.size(1)

# model 1
# linear_model = LinearNN1(in_size=num_features, hidden_size=args.m_hidden, out_size=out_size, embed=args.m_embed,
#                      drop_p=args.m_drop_p, activation=args.m_activation).to(device)


# Bernoulli model
# linear_model = BernoulliMixture(in_size=num_features, hidden_size=32, out_size=out_size,
#                                 embed_length=args.m_embed, drop_p=args.m_drop_p, activation=nn.ELU()).to(device)

# Bernoulli Resnet18
# linear_model = BernoulliResnet18(in_size=num_features, hidden_size=32, out_size=out_size,
#                                  embed_length=args.m_embed, drop_p=args.m_drop_p, activation=nn.ELU()).to(device)

linear_model = ResNet18(in_size=num_features, hidden_size=args.m_hidden, out_size=out_size, embed=args.m_embed,
                        drop_p=args.m_drop_p, activation=args.m_activation).to(device)

train_model = linear_model

# train_data.change_x_data(check_cv_and_preprocess_data(train_model, train_data))

train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
dataloaders = {
    "train": train_dataloader,
    "val": test_dataloader,
}

# check total parameters of this model
pytorch_total_params = sum(p.numel() for p in train_model.parameters())
print(" Number of Parameters: ", pytorch_total_params)

optimizer = optim.Adam(train_model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
criterion = ml_nn_loss

model_opt, loss_opt = train(train_model,
                            dataloaders,
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            num_epochs=args.pretrain_epochs,
                            device_train=None,
                            num_l=num_labels,
                            fname=fnamesub
                            )

model_opt.eval()
results += add_res(model_opt, test_data.get_x(), test_data.get_y(), device=device)
print(results)

# documentation
with open('./result/' + fnamesub, 'a') as f:
    writer_obj = csv.writer(f)
    writer_obj.writerow(results)

