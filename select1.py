import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel as RBF
from sklearn.cluster import KMeans
import random
import time
import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.metrics import jaccard_score
from scipy import stats
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
    temp1 = np.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    collum = torch.argmin(temp2)
    temp3 = torch.sum(temp1, dim=1)
    _, row = torch.sort(temp1[:, collum], dim=0)
    row1 = row[:n_instances]
    return row1


def test2(self, probs, n_instances):
    temp1 = np.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    _, collum = torch.topk(temp2, k=2, largest=False)
    temp3 = torch.sum(temp1, dim=1)
    t1 = temp3.data.numpy()
    c = collum.indices.data.numpy()
    c1 = c[0]
    c2 = c[1]
    _, row1 = torch.sort(temp1[:, c1], dim=0)
    _, row2 = torch.sort(temp1[:, c2], dim=0)
    d1 = int(np.floor(n_instances / 2))
    e1 = row1[:d1].numpy()
    e2 = row2[:n_instances - d1].numpy()
    row3 = np.hstack((e1, e2))
    print("cc")
    return row3


def test3(self, probs, n_instances):
    temp1 = np.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    _, collum = torch.topk(temp2, k=3, largest=False)
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
    e3 = row2[:n_instances - 2 * d1].numpy()
    row3 = np.hstack((e1, e2, e3))
    print("cc")
    return row3


def test5(self, probs, n_instances):
    temp1 = torch.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    _, collum = torch.topk(temp2, k=3, largest=False)
    # temp3 = torch.sum(temp1, dim=1)
    # t1 = temp3.data.numpy()
    # c = collum.indices.data.numpy()
    c1 = collum[0]
    c2 = collum[1]
    c3 = collum[2]
    temp4 = temp1[c1] + temp1[c2] + temp1[c3]
    _, row5 = torch.topk(temp4, k=n_instances, largest=False)
    row6 = row5.cpu()
    row6 = row6.numpy()
    print("cc")
    return row6


def test6(self, probs, n_instances):
    temp1 = torch.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    _, collum = torch.topk(temp2, k=2, largest=False)
    temp3 = torch.sum(temp1, dim=1)
    t1 = temp3.data
    t1 = t1.cpu()
    t1 = t1.numpy()
    c = collum
    c = c.cpu()
    c = c.numpy()
    c1 = c[0]
    c2 = c[1]
    _, row1 = torch.sort(temp1[:, c1], dim=0)
    _, row2 = torch.sort(temp1[:, c2], dim=0)
    d1 = int(n_instances / 2)
    e1 = row1[:d1]
    e1 = e1.cpu()
    e1 = e1.numpy()
    e2 = row2[:n_instances - d1]
    e2 = e2.cpu()
    e2 = e2.numpy()
    row3 = np.hstack((e1, e2))
    print("cc")
    return row3


def test7(self, probs, n_instances):
    temp1 = torch.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    temp3 = torch.sum(temp1, dim=1)
    a = temp3.size()
    a = a[0]

    temp2 = temp2 / a
    _, collum = torch.topk(temp2, k=1, largest=False)
    temp3 = torch.sum(temp1, dim=1)
    t1 = temp3.data
    t1 = t1.cpu()
    t1 = t1.numpy()
    c = collum
    c = c.cpu()
    c = c.numpy()
    c1 = c[0]

    _, row1 = torch.sort(temp1[:, c1], dim=0)

    e1 = row1[:n_instances]
    e1 = e1.cpu()
    e1 = e1.numpy()

    print("cc")
    return e1


def test8(self, probs, n_instances):
    temp1 = torch.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    temp31 = torch.sum(temp1, dim=1)
    a = temp31.size()
    a = a[0]
    temp21 = temp2 / a
    testmin = torch.argmin(temp21)
    print(f"Min index in pooling {testmin}")
    # temp21=temp21.cpu()
    # temp2 = temp2.numpy()

    a1 = np.array(temp21)
    a22 = np.expand_dims(a1, 0)
    a2 = a22.T
    dp = dpmeans(a2)
    dp.fit(a2)
    k = dp.K
    print(f"K means size {k}")
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(a2)
    cc1 = kmeans.cluster_centers_
    labels = kmeans.labels_
    labelmin = np.argmin(cc1)
    index2 = np.where(labels == labelmin)[0]
    print(f"K means index {index2}")
    size11 = index2.size
    temptotal = 0
    for i in range(size11):
        d1 = temp1[:, index2[i]]
        temptotal = temptotal + d1
    _, collum = torch.topk(temptotal, k=n_instances, largest=False)

    c = collum
    c = c.cpu()
    c = c.numpy()
    print("cc")
    return c

def pmi(x, y, total):
    px = np.sum(x) / total
    py = np.sum(y) / total
    x = x.astype(int)
    y = y.astype(int)
    c=(x & y).any()
    pxy = c / total
    return math.log2(pxy / (px * py)) if pxy > 0 else 0
def test9(self, probs, n_instances):


    # Sample multi-label data: binary matrix (n instances x k labels)
    M=self.dataset.y
    M=np.array(M)
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data=M)


    # Co-occurrence analysis
    co_occurrence_matrix = df.T.dot(df)
    print("\nCo-occurrence matrix:")
    print(co_occurrence_matrix)

    # Visualization
    #sns.heatmap(co_occurrence_matrix, annot=True, fmt="f")
    #plt.title("Label Co-Occurrence Matrix")
    #plt.show()

    # Function to compute PMI


    # Compute PMI matrix
    total_instances = df.shape[0]
    pmi_matrix = np.zeros((df.shape[1], df.shape[1]))
    for i in range(df.shape[1]):
        for j in range(df.shape[1]):
            if i != j:
                pmi_matrix[i, j] = pmi(df.iloc[:, i].values, df.iloc[:, j].values, total_instances)

    print("PMI Matrix:\n", pmi_matrix)

    # Compute Jaccard similarity for each pair of labels
    jaccard_similarity_matrix = np.zeros((df.shape[1], df.shape[1]))
    for i in range(df.shape[1]):
        for j in range(df.shape[1]):
            if i != j:
                jaccard_similarity_matrix[i, j] = jaccard_score(df.iloc[:, i], df.iloc[:, j])

    print("Jaccard Similarity Matrix:\n", jaccard_similarity_matrix)

    ctraining = self.train_origin
    ctraining1 = np.array(ctraining)
    ctraining2 = ctraining1 * 10
    cpredit = self.pool_origin
    cpredit1 = np.array(cpredit)
    cpredit2 = cpredit1 * 10
    gamma = 5
    cresult = RBF(cpredit1, ctraining1, gamma)
    cresult2 = np.sum(cresult, axis=1)
    cresult3 = torch.tensor(cresult2)
    _, cresult5 = torch.topk(cresult3, k=n_instances, largest=False)
    cresult6 = np.array(cresult5)
    print("cc")
    return cresult6
def test10(self, probs, n_instances):



    ctraining = self.train_origin
    ctraining1 = np.array(ctraining)
    ctraining2 = ctraining1 * 10
    cpredit = self.pool_origin
    cpredit1 = np.array(cpredit)
    cpredit2 = cpredit1 * 10

    cresult = RBF(cpredit2, ctraining2)
    cresult2 = np.sum(cresult, axis=1)
    cresult3 = torch.tensor(cresult2)
    a = cresult3.size()
    a = a[0]
    value21, cresult5 = torch.topk(cresult3, k=int(a/20), largest=False)
    value22, cresult6 = torch.topk(cresult3, k=int(a /20), largest=True)
    interval1=value21[-1]
    interval1=interval1.numpy()
    interval2 = value22[-1]
    interval2=interval2.numpy()
    #cresult6 = np.array(cresult5)
    #variance,mean=torch.var_mean(cresult3, dim=0, keepdim=True)
    #interval = stats.norm.interval(0.8, mean, variance)
    temp1 = torch.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    #temp31 = torch.sum(temp1, dim=1)
    #a = temp31.size()
    #a = a[0]
    #temp21 = temp2 / a
    #testmin = torch.argmin(temp21)
    #print(f"Min index in pooling {testmin}")
    _, collum = torch.topk(temp2, k=1, largest=False)
    c = collum
    c = c.cpu()
    c = c.numpy()
    c1 = c[0]
    print(f"Min label in pooling {c1}")
    value, row1 = torch.sort(temp1[:, c1], dim=0)
    size2=row1.size()
    size2 = size2[0]
    list=[]
    for i in range(size2):
        indextemp1=row1[i]
        cvalue1=cresult3[indextemp1]
        cvalue2=cvalue1.numpy()
        if cvalue2 > interval1 and cvalue2 < interval2:
            indextemp2=indextemp1.cpu()
            indextemp3=indextemp2.numpy()
            list.append(indextemp3)
            if len(list)>=n_instances:
                print("selection breaking ")
                break
        else:
            print(f"not selected {indextemp1} value {cvalue2}")
    list1=np.array(list)
    print("cc")
    return list1
def test11(self, probs, n_instances):

    ctraining = self.train_origin
    ctraining1 = np.array(ctraining)
    ctraining2 = ctraining1 * 10
    cpredit = self.pool_origin
    cpredit1 = np.array(cpredit)
    cpredit2 = cpredit1 * 10

    cresult = RBF(cpredit2, ctraining2)
    cresult2 = np.sum(cresult, axis=1)
    cresult3 = torch.tensor(cresult2)
    a = cresult3.size()
    a = a[0]
    value21, cresult5 = torch.topk(cresult3, k=int(a/20), largest=False)
    value22, cresult6 = torch.topk(cresult3, k=int(a /20), largest=True)
    interval1=value21[-1]
    interval1=interval1.numpy()
    interval2 = value22[-1]
    interval2=interval2.numpy()
    #cresult6 = np.array(cresult5)
    #variance,mean=torch.var_mean(cresult3, dim=0, keepdim=True)
    #interval = stats.norm.interval(0.8, mean, variance)
    temp1 = torch.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    #temp31 = torch.sum(temp1, dim=1)
    #a = temp31.size()
    #a = a[0]
    #temp21 = temp2 / a
    #testmin = torch.argmin(temp21)
    #print(f"Min index in pooling {testmin}")
    _, collum = torch.topk(temp2, k=1, largest=False)
    c = collum
    c = c.cpu()
    c = c.numpy()
    c1 = c[0]
    print(f"Min label in pooling {c1}")
    value, row1 = torch.sort(temp1[:, c1], dim=0)
    size2=row1.size()
    size2 = size2[0]
    list=[]
    for i in range(size2):
        indextemp1=row1[i]
        cvalue1=cresult3[indextemp1]
        cvalue2=cvalue1.numpy()
        if cvalue2 > interval1 :
            indextemp2=indextemp1.cpu()
            indextemp3=indextemp2.numpy()
            list.append(indextemp3)
            if len(list)>=n_instances:
                print("selection breaking ")
                break
        else:
            print(f"not selected {indextemp1} value {cvalue2}")
    list1=np.array(list)
    print("cc")
    return list1


def test12(self, probs, n_instances):

    ctraining = self.train_origin
    ctraining1 = np.array(ctraining)
    ctraining2 = ctraining1 * 10
    cpredit = self.pool_origin
    cpredit1 = np.array(cpredit)
    cpredit2 = cpredit1 * 10

    cresult = RBF(cpredit2, ctraining2)
    cresult2 = np.sum(cresult, axis=1)
    cresult3 = torch.tensor(cresult2)
    a = cresult3.size()
    a = a[0]
    value21, cresult5 = torch.topk(cresult3, k=int(a/20), largest=False)
    value22, cresult6 = torch.topk(cresult3, k=int(a /20), largest=True)
    interval1=value21[-1]
    interval1=interval1.numpy()
    interval2 = value22[-1]
    interval2=interval2.numpy()
    #cresult6 = np.array(cresult5)
    #variance,mean=torch.var_mean(cresult3, dim=0, keepdim=True)
    #interval = stats.norm.interval(0.8, mean, variance)
    temp1 = torch.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    #temp31 = torch.sum(temp1, dim=1)
    #a = temp31.size()
    #a = a[0]
    #temp21 = temp2 / a
    #testmin = torch.argmin(temp21)
    #print(f"Min index in pooling {testmin}")
    value2, collum2 = torch.sort(temp2)
    _,collum=torch.topk(temp2, k=1, largest=False)
    for i in range(156):
        if collum2[i] in self.coindex:
            print(f"found and break {collum2[i]}")
            collum=collum2[i].cpu()
            break
        else:
            print(f"index not found {collum2[i]}")

    c = collum
    c = c.cpu()
    c = c.numpy()
    c1 = c
    print(f"Min label in pooling {c1}")
    value, row1 = torch.sort(temp1[:, c1], dim=0)
    size2=row1.size()
    size2 = size2[0]
    list=[]
    for i in range(size2):
        indextemp1=row1[i]
        cvalue1=cresult3[indextemp1]
        cvalue2=cvalue1.numpy()
        if cvalue2 > interval1 :
            indextemp2=indextemp1.cpu()
            indextemp3=indextemp2.numpy()
            list.append(indextemp3)
            if len(list)>=n_instances:
                print("selection breaking ")
                break
        else:
            print(f"not selected {indextemp1} value {cvalue2}")
    list1=np.array(list)
    print("cc")
    return list1
def test13(self, probs, n_instances):

    ctraining = self.train_origin
    ctraining1 = np.array(ctraining)
    ctraining2 = ctraining1 * 10
    cpredit = self.pool_origin
    cpredit1 = np.array(cpredit)
    cpredit2 = cpredit1 * 10

    cresult = RBF(cpredit2, ctraining2)
    cresult2 = np.sum(cresult, axis=1)
    cresult3 = torch.tensor(cresult2)
    a = cresult3.size()
    a = a[0]
    value21, cresult5 = torch.topk(cresult3, k=int(a/20), largest=True)
    value22, cresult6 = torch.topk(cresult3, k=int(a /20), largest=True)
    interval1=value21[-1]
    interval1=interval1.numpy()
    interval2 = value22[-1]
    interval2=interval2.numpy()
    #cresult6 = np.array(cresult5)
    #variance,mean=torch.var_mean(cresult3, dim=0, keepdim=True)
    #interval = stats.norm.interval(0.8, mean, variance)
    temp1 = torch.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    #temp31 = torch.sum(temp1, dim=1)
    #a = temp31.size()
    #a = a[0]
    #temp21 = temp2 / a
    #testmin = torch.argmin(temp21)
    #print(f"Min index in pooling {testmin}")
    _, collum = torch.topk(temp2, k=1, largest=False)
    c = collum
    c = c.cpu()
    c = c.numpy()
    c1 = c[0]
    print(f"Min label in pooling {c1}")
    value, row1 = torch.sort(temp1[:, c1], dim=0)
    size2=row1.size()
    size2 = size2[0]
    list=[]
    for i in range(size2):
        indextemp1=row1[i]
        cvalue1=cresult3[indextemp1]
        cvalue2=cvalue1.numpy()
        if cvalue2 < interval1 :
            indextemp2=indextemp1.cpu()
            indextemp3=indextemp2.numpy()
            list.append(indextemp3)
            if len(list)>=n_instances:
                print("selection breaking ")
                break
        else:
            print(f"not selected {indextemp1} value {cvalue2}")
    list1=np.array(list)
    print("cc")
    return list1
def test15(self, probs, n_instances):

    ctraining = self.train_origin
    ctraining1 = np.array(ctraining)
    ctraining2 = ctraining1 * 10
    cpredit = self.pool_origin
    cpredit1 = np.array(cpredit)
    cpredit2 = cpredit1 * 10

    cresult = RBF(cpredit2, ctraining2)
    cresult2 = np.max(cresult, axis=1)
    cresult3 = torch.tensor(cresult2)
    a = cresult3.size()
    a = a[0]
    value21, cresult5 = torch.topk(cresult3, k=int(a/20), largest=True)
    value22, cresult6 = torch.topk(cresult3, k=int(a /20), largest=True)
    interval1=value21[-1]
    interval1=interval1.numpy()
    interval2 = value22[-1]
    interval2=interval2.numpy()
    #cresult6 = np.array(cresult5)
    #variance,mean=torch.var_mean(cresult3, dim=0, keepdim=True)
    #interval = stats.norm.interval(0.8, mean, variance)
    temp1 = torch.square(probs.data - 0.5)
    temp2 = torch.sum(temp1, dim=0)
    #temp31 = torch.sum(temp1, dim=1)
    #a = temp31.size()
    #a = a[0]
    #temp21 = temp2 / a
    #testmin = torch.argmin(temp21)
    #print(f"Min index in pooling {testmin}")
    _, collum = torch.topk(temp2, k=1, largest=False)
    c = collum
    c = c.cpu()
    c = c.numpy()
    c1 = c[0]
    print(f"Min label in pooling {c1}")
    value, row1 = torch.sort(temp1[:, c1], dim=0)
    size2=row1.size()
    size2 = size2[0]
    list=[]
    for i in range(size2):
        indextemp1=row1[i]
        cvalue1=cresult3[indextemp1]
        cvalue2=cvalue1.numpy()
        if cvalue2 < interval1 :
            indextemp2=indextemp1.cpu()
            indextemp3=indextemp2.numpy()
            list.append(indextemp3)
            if len(list)>=n_instances:
                print("selection breaking ")
                break
        else:
            print(f"not selected {indextemp1} value {cvalue2}")
    list1=np.array(list)
    print("cc")
    return list1
def testsum(self, probs, n_instances):
    temp1=self.dataset.y
    temp2 = torch.sum(temp1, dim=1)
    value21, cresult5 = torch.topk(temp2, k=n_instances, largest=True)

    c = cresult5
    c = c.cpu()
    c = c.numpy()
    c1 = c
    print("cc")
    return c1





