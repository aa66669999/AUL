import copy
import csv
import time
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from config import get_args
from loss import ml_nn_loss2
from util import *
from architecture import *
from data import get_data, MyDataset
import numpy as np
import torch.nn.functional as F
np.random.seed(0)
# get GPU or CPU
args = get_args()
train_data, pool_data, test_data = get_data(train_ratio=args.train, pool_ratio=args.pool,test_ratio=args.test)

num_labels = test_data.label_length()
# print(test_label_length)
out_size = num_labels
num_features = train_data.x.size(1)


def get_new_resnet18():
    active_model = ResNet18(in_size=num_features, hidden_size=args.m_hidden, out_size=out_size, embed=args.m_embed,
                            drop_p=args.m_drop_p, activation=args.m_activation)
    return active_model
class dpmeans:

    def __init__(self, X):
        # Initialize parameters for DP means

        self.K = 1
        self.K_init = 10
        self.d = X.shape[1]
        self.z = np.mod(np.random.permutation(X.shape[0]), self.K) + 1
        self.mu = np.random.standard_normal((self.K, self.d))
        self.sigma = 1
        self.nk = np.zeros(self.K)
        self.pik = np.ones(self.K) / self.K

        # init mu
        self.mu = np.array([np.mean(X, 0)])

        # init lambda
        self.Lambda = self.kpp_init(X, self.K_init)

        self.max_iter = 100
        self.obj = np.zeros(self.max_iter)
        self.em_time = np.zeros(self.max_iter)

    def kpp_init(self, X, k):
        # k++ init
        # lambda is max distance to k++ means

        [n, d] = np.shape(X)
        mu = np.zeros((k, d))
        dist = np.inf * np.ones(n)

        mu[0, :] = X[np.ceil(np.random.rand() * n - 1).astype(int), :]
        for i in range(1, k):
            D = X - np.tile(mu[i - 1, :], (n, 1))
            dist = np.minimum(dist, np.sum(D * D, 1))
            idx = np.where(np.random.rand() < np.cumsum(dist / float(sum(dist))))
            mu[i, :] = X[idx[0][0], :]
            Lambda = np.max(dist)

        return Lambda

    def fit(self, X):

        obj_tol = 1e-3
        max_iter = self.max_iter
        [n, d] = np.shape(X)

        obj = np.zeros(max_iter)
        em_time = np.zeros(max_iter)
        print('running dpmeans...')

        for iter in range(max_iter):
            tic = time.time()
            dist = np.zeros((n, self.K))

            # assignment step
            for kk in range(self.K):
                Xm = X - np.tile(self.mu[kk, :], (n, 1))
                dist[:, kk] = np.sum(Xm * Xm, 1)

            # update labels
            dmin = np.min(dist, 1)
            self.z = np.argmin(dist, 1)
            idx = np.where(dmin > self.Lambda)

            if (np.size(idx) > 0):
                self.K = self.K + 1
                self.z[idx[0]] = self.K - 1  # cluster labels in [0,...,K-1]
                self.mu = np.vstack([self.mu, np.mean(X[idx[0], :], 0)])
                Xm = X - np.tile(self.mu[self.K - 1, :], (n, 1))
                dist = np.hstack([dist, np.array([np.sum(Xm * Xm, 1)]).T])

            # update step
            self.nk = np.zeros(self.K)
            for kk in range(self.K):
                self.nk[kk] = self.z.tolist().count(kk)
                idx = np.where(self.z == kk)
                self.mu[kk, :] = np.mean(X[idx[0], :], 0)

            self.pik = self.nk / float(np.sum(self.nk))

            # compute objective
            for kk in range(self.K):
                idx = np.where(self.z == kk)
                obj[iter] = obj[iter] + np.sum(dist[idx[0], kk], 0)
            obj[iter] = obj[iter] + self.Lambda * self.K

            # check convergence
            if (iter > 0 and np.abs(obj[iter] - obj[iter - 1]) < obj_tol * obj[iter]):
                print('converged in %d iterations\n' % iter)
                break
            em_time[iter] = time.time() - tic
        # end for
        self.obj = obj
        self.em_time = em_time
        return self.z, obj, em_time


class ActiveLearning:
    def __init__(self, training_data: MyDataset, pooling_data: MyDataset, device_al_train):
        self.dataset = pooling_data
        self.new_dataset = training_data
        self.device_al_train = device_al_train

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
    def mutilabel_margin(self,probs,n_instances):
        # Calculate the number of positive labels in new_training_data
        avg_num_pos_labels = torch.mean(torch.sum(self.new_dataset.y, dim=1)).item()
        print(avg_num_pos_labels)
        # Calculate the top m margins for each instance0
        margins, _ = torch.topk(probs, int(avg_num_pos_labels), dim=1)
        margins_diff = margins[:, :-1] - margins[:, 1:]
        # Calculate the mean margin across labels for each instance
        xi = int(avg_num_pos_labels/2)
        # Create a tensor representing the indices of margins_diff
        indices = torch.arange(margins_diff.size(1), dtype=torch.float).unsqueeze(0)
        # Compute the power term for each element
        power_term = xi - indices
        # Compute margins_diff multiplied by e to the power of (xi - index)
        margins_diff_exp = margins_diff * torch.exp(power_term)
        print('margin_diff_exp.shape', margins_diff_exp.shape)
        sum_margin = torch.sum(margins_diff_exp, dim=1)
        print('mean_margin', sum_margin.shape)
        # Select the instances with the highest mean margin
        _, selected_indices = torch.topk(sum_margin, n_instances, largest=False)
        return selected_indices
    def random(self,n_instances):
    # # random select
        total_instances = len(self.dataset)
        indices = list(range(total_instances))
        random.shuffle(indices)
        selected_indices = indices[:n_instances]
        return selected_indices

    def test1(self, probs, n_instances):
        temp1=np.square(probs.data-0.5)
        temp2 = torch.sum(temp1,dim=0)
        collum=torch.argmin(temp2)
        temp3 = torch.sum(temp1,dim=1)
        _,row=torch.sort(temp1[:,collum],dim=0)
        row1=row[:n_instances]
        return row1

    def test2(self, probs, n_instances):
        temp1=np.square(probs.data-0.5)
        temp2 = torch.sum(temp1,dim=0)
        _,collum=torch.topk(temp2,k=2,largest=False)
        temp3 = torch.sum(temp1,dim=1)
        t1=temp3.data.numpy()
        c=collum.indices.data.numpy()
        c1=c[0]
        c2=c[1]
        _,row1=torch.sort(temp1[:,c1],dim=0)
        _,row2 = torch.sort(temp1[:,c2], dim=0)
        d1=int(np.floor(n_instances/2))
        e1=row1[:d1].numpy()
        e2=row2[:n_instances-d1].numpy()
        row3=np.hstack((e1,e2))
        print("cc")
        return row3

    def test3(self, probs, n_instances):
        temp1 = np.square(probs.data - 0.5)
        temp2 = torch.sum(temp1, dim=0)
        _,collum = torch.topk(temp2, k=3,largest=False)
        temp3 = torch.sum(temp1, dim=1)
        t1 = temp3.data.numpy()
        c = collum.indices.data.numpy()
        c1 = c[0]
        c2 = c[1]
        c3 = c[2]
        _, row1 = torch.sort(temp1[:, c1], dim=0)
        _, row2 = torch.sort(temp1[:, c2], dim=0)
        _, row3 = torch.sort(temp1[:, c3], dim=0)
        d1 = int(np.floor(n_instances / 3))
        e1 = row1[:d1].numpy()
        e2 = row2[:d1].numpy()
        e3 = row2[:n_instances -2*d1].numpy()
        row3 = np.hstack((e1,e2,e3))
        print("cc")
        return row3
    def test5(self, probs, n_instances):
        temp1 = torch.square(probs.data - 0.5)
        temp2 = torch.sum(temp1, dim=0)
        _,collum = torch.topk(temp2, k=3,largest=False)
       # temp3 = torch.sum(temp1, dim=1)
       # t1 = temp3.data.numpy()
       # c = collum.indices.data.numpy()
        c1 = collum[0]
        c2 = collum[1]
        c3 = collum[2]
        temp4=temp1[c1]+temp1[c2]+temp1[c3]
        _,row5=torch.topk(temp4, k=n_instances, largest=False)
        row6=row5.cpu()
        row6=row6.numpy()
        print("cc")
        return row6
    def test6(self, probs, n_instances):
        temp1=torch.square(probs.data-0.5)
        temp2 = torch.sum(temp1,dim=0)
        _,collum=torch.topk(temp2,k=2,largest=False)
        temp3 = torch.sum(temp1,dim=1)
        t1=temp3.data
        t1=t1.cpu()
        t1=t1.numpy()
        c=collum
        c=c.cpu()
        c=c.numpy()
        c1=c[0]
        c2=c[1]
        _,row1=torch.sort(temp1[:,c1],dim=0)
        _,row2 = torch.sort(temp1[:,c2], dim=0)
        d1=int(n_instances/2)
        e1=row1[:d1]
        e1=e1.cpu()
        e1=e1.numpy()
        e2=row2[:n_instances-d1]
        e2=e2.cpu()
        e2=e2.numpy()
        row3=np.hstack((e1,e2))
        print("cc")
        return row3
    def test7(self, probs, n_instances):
        temp1=torch.square(probs.data-0.5)
        temp2 = torch.sum(temp1,dim=0)
        temp3 = torch.sum(temp1, dim=1)
        a=temp3.size()
        a=a[0]

        temp2=temp2/a
        _,collum=torch.topk(temp2,k=1,largest=False)
        temp3 = torch.sum(temp1,dim=1)
        t1=temp3.data
        t1=t1.cpu()
        t1=t1.numpy()
        c=collum
        c=c.cpu()
        c=c.numpy()
        c1=c[0]

        _,row1=torch.sort(temp1[:,c1],dim=0)

        e1=row1[:n_instances]
        e1=e1.cpu()
        e1=e1.numpy()


        print("cc")
        return e1
    def test8(self, probs, n_instances):
        temp1=torch.square(probs.data-0.5)
        temp2 = torch.sum(temp1,dim=0)
        temp31 = torch.sum(temp1, dim=1)
        a=temp31.size()
        a=a[0]
        temp21=temp2/a
        testmin=torch.argmin(temp21)
        print(f"Min index in pooling {testmin}")
        #temp21=temp21.cpu()
       # temp2 = temp2.numpy()

        a1 = np.array(temp21)
        a22 = np.expand_dims(a1, 0)
        a2=a22.T
        dp=dpmeans(a2)
        dp.fit(a2)
        k=dp.K
        print(f"K means size {k}")
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(a2)
        cc1=kmeans.cluster_centers_
        labels=kmeans.labels_
        labelmin=np.argmin(cc1)
        index2=np.where(labels == labelmin)[0]
        print(f"K means index {index2}")
        size11=index2.size
        temptotal=0
        for i in range(size11):
            d1=temp1[:,index2[i]]
            temptotal=temptotal+d1
        _,collum=torch.topk(temptotal,k=n_instances,largest=False)

        c=collum
        c=c.cpu()
        c=c.numpy()
        print("cc")
        return c
    def select_instances_entropy(self, model, n_instances):
        # Calculate the entropy of predictions for all instances in the pool dataset
        with torch.no_grad():
            model = model.cpu()
            logits = model(self.dataset.x.cpu())
          #  model=model.cuda()

          #  tempdata=self.dataset.x.cuda()
          #  logits = model(tempdata)
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

        selected_indices=self.test8(probs, n_instances)
        #selected_indices=self.random(n_instances)

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

        print(self.new_dataset.__len__())
        print(self.dataset.__len__())

    def train_model(self, model, epochs=args.active_epochs):
        model = model.to(self.device_al_train)
        criterion = ml_nn_loss2
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        al_data_loader = DataLoader(self.new_dataset, batch_size=args.active_batch_size, shuffle=True)
        for epoch in range(epochs):
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

