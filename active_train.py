import copy
import csv
import time

import torch
from sklearn.metrics import roc_auc_score
from torch import optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import jaccard_score
import select1
from config import get_args
from loss import ml_nn_loss2
from util import *
from architecture import *
from data import get_data, MyDataset
import numpy as np
import torch.nn.functional as F
# get GPU or CPU
#args = get_args()
#train_data, pool_data, test_data,train_origin,pool_origin= get_data(train_ratio=args.train, pool_ratio=args.pool,test_ratio=args.test)

#num_labels = test_data.label_length()
# print(test_label_length)
#out_size = num_labels
#num_features = train_data.x.size(1)






class ActiveLearning:
    def __init__(self, train_origin,pool_origin,training_data: MyDataset, pooling_data: MyDataset, device_al_train):
        self.dataset = pooling_data
        self.new_dataset = training_data
        self.device_al_train = device_al_train
        self.train_origin = train_origin
        self.pool_origin = pool_origin
        M = pooling_data.y
        M = np.array(M)
        df = pd.DataFrame(data=M)
        # Compute Jaccard similarity for each pair of labels
        jaccard_similarity_matrix = np.zeros((df.shape[1], df.shape[1]))
        for i in range(df.shape[1]):
            for j in range(df.shape[1]):
                if i != j:
                    jaccard_similarity_matrix[i, j] = jaccard_score(df.iloc[:, i], df.iloc[:, j])

        temp1=torch.tensor(jaccard_similarity_matrix)
        temp2 = torch.sum(temp1, dim=0)
        value, row1=torch.topk(temp2,k=80,largest=False)
        self.covalue=value.cuda()
        self.coindex=row1.cuda()
        #print("Jaccard Similarity Matrix:\n", jaccard_similarity_matrix)



    def select_instances_dropout(self, model, n_instances, n_samples=10):
        # List to store predictions for each instance
        all_predictions = []

        # Perform inference with dropout enabled multiple times
        for _ in range(n_samples):
            with torch.no_grad():
                # Enable dropout during inference
                model.train()

                # Forward pass to get predictions
                logits = model(self.dataset.x.to(self.device_al_train))
                probs = torch.sigmoid(logits)

            all_predictions.append(probs.unsqueeze(0))  # Store predictions for each sample

        print(len(all_predictions))

        # Concatenate predictions along the sample dimension
        all_predictions = torch.cat(all_predictions, dim=0)

        print(all_predictions.shape)

        # Calculate uncertainty as the variance across samples for each instance
        uncertainty = torch.var(all_predictions, dim=0).sum(dim=1)

        print('uncertainty', uncertainty.shape)

        # Select the instances with the highest uncertainty
        _, selected_indices = torch.topk(uncertainty, n_instances)

        # Get the corresponding instances and labels
        new_training_data = self.dataset.x[selected_indices]
        new_training_labels = self.dataset.y[selected_indices]

        # Append the new instances to the new dataset
        self.new_dataset.x = torch.cat([self.new_dataset.x, new_training_data])
        self.new_dataset.y = torch.cat([self.new_dataset.y, new_training_labels])

        # Remove the selected instances from the pool dataset
        indices_to_keep = [i for i in range(len(self.dataset)) if i not in selected_indices]
        self.dataset.x = torch.index_select(self.dataset.x, 0, torch.tensor(indices_to_keep))
        self.dataset.y = torch.index_select(self.dataset.y, 0, torch.tensor(indices_to_keep))



    def select_instances_entropy(self,model, n_instances,):
        # Calculate the entropy of predictions for all instances in the pool dataset
        with torch.no_grad():
           # model = model.cpu()
           # logits = model(self.dataset.x.cpu())
            model=model.cuda()

            tempdata=self.dataset.x.cuda()
            logits = model(tempdata)
            # print("Logits shape:", logits.shape)
            probs = torch.sigmoid(logits)  # Using sigmoid instead of softmax
            # probs = F.softmax(logits, dim=1)

        # # margin is not suitable for multi-lable classification
        # margins, _ = torch.topk(probs, 2, dim=1)
        # margin = margins[:, 0] - margins[:, 1]  # Difference between top two probabilities
        # _, selected_indices = torch.topk(margin, n_instances)

        # multiple margin
        # multiple margin
       # selected_indices=self.mutilabel_margin(self, probs, n_instances)

        # _, selected_indices = torch.topk(margin, n_instances, largest=True)

            # confidences, _ = torch.max(probs, dim=1)
        #     least_confidence = 1 - confidences  # Confidence is inversely related to uncertainty
        # _, selected_indices = torch.topk(least_confidence, n_instances)

        #     # entropy select
        #     entropy = -torch.sum(probs * torch.log(probs), dim=1)
        # _, selected_indices = torch.topk(entropy, n_instances)  # , largest=False)


        selected_indices=select1.testsum(self,probs, n_instances)
        #selected_indices=self.random(n_instances)

        # Get the corresponding instances and labels
        new_training_data = self.dataset.x[selected_indices]
        new_training_labels = self.dataset.y[selected_indices]
        new_training_origin_data=self.pool_origin[selected_indices]

        # Append the new instances to the new dataset
        self.new_dataset.x = torch.cat([self.new_dataset.x, new_training_data])
        self.new_dataset.y = torch.cat([self.new_dataset.y, new_training_labels])
        self.train_origin =torch.cat([self.train_origin, new_training_origin_data])

        # Remove the selected instances from the pool dataset
        indices_to_keep = [i for i in range(len(self.dataset)) if i not in selected_indices]
        self.dataset.x = torch.index_select(self.dataset.x, 0, torch.tensor(indices_to_keep))
        self.dataset.y = torch.index_select(self.dataset.y, 0, torch.tensor(indices_to_keep))
        self.pool_origin = torch.index_select(self.pool_origin,0, torch.tensor(indices_to_keep))

        print(self.new_dataset.__len__())
        print(self.dataset.__len__())

    def train_model(self, model, args):
        model = model.to(self.device_al_train)
        criterion = ml_nn_loss2
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        al_data_loader = DataLoader(self.new_dataset, batch_size=args.active_batch_size, shuffle=True)
        for epoch in range(args.active_epochs):
            running_loss = 0.0
            for (inputs, labels) in al_data_loader:
                inputs = inputs.to(self.device_al_train)
                labels = labels.to(self.device_al_train)
                optimizer.zero_grad()
                batch_outputs = model(inputs)
                loss = criterion(labels, batch_outputs, model, device=self.device_al_train)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(al_data_loader)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

