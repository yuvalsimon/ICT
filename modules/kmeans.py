import torch.nn as nn
import torch.nn.functional as F
import torch
import sklearn
import sklearn.cluster

def cluster(data, k, temp, num_iter, init = None, cluster_temp=5):
    '''
    pytorch (differentiable) implementation of soft k-means clustering.
    '''
    #normalize x so it lies on the unit sphere
    data = torch.diag(1./torch.norm(data, p=2, dim=1)) @ data
    #use kmeans++ initialization if nothing is provided
    if init is None:
        data_np = data.detach().numpy()
        norm = (data_np**2).sum(axis=1)
        init = sklearn.cluster.k_means_._k_init(data_np, k, norm, sklearn.utils.check_random_state(None))
        init = torch.tensor(init, requires_grad=True)
        if num_iter == 0: return init
    mu = init.to('cuda') # TODO: change to some variable
    n = data.shape[0]
    d = data.shape[1]
#    data = torch.diag(1./torch.norm(data, dim=1, p=2))@data
    for t in range(num_iter):
        #get distances between all data points and cluster centers
#        dist = torch.cosine_similarity(data[:, None].expand(n, k, d).reshape((-1, d)), mu[None].expand(n, k, d).reshape((-1, d))).reshape((n, k))
        dist = data @ mu.t()
        #cluster responsibilities via softmax
        r = torch.softmax(cluster_temp*dist, 1)
        #total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        #mean of points in each cluster weighted by responsibility
        cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
        #update cluster means
        new_mu = torch.diag(1/cluster_r) @ cluster_mean
        mu = new_mu
    dist = data @ mu.t()
    r = torch.softmax(cluster_temp*dist, 1)
    return mu, r, dist

class KMeans(nn.Module):
    def __init__(self, in_features_num, K, cluster_temp=50, num_iter=1):
        super().__init__()
        self.K = K
        self.cluster_temp = cluster_temp
        self.init =  torch.rand(self.K, in_features_num)
        self.num_iter = num_iter

    def forward(self, x):
        mu_init, _, _ = cluster(x, self.K, 1, self.num_iter, cluster_temp = self.cluster_temp, init = self.init)
        mu, r, dist = cluster(x, self.K, 1, 1, cluster_temp = self.cluster_temp, init = mu_init.detach().clone())
        return mu, r, dist