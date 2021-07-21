from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from Losses import binarize_and_smooth_labels


class WEL(nn.Module):
    def __init__(self, model, dml_layers, params , device='cuda'):
        super(WEL, self).__init__()
        self.train_classes = params.train_classes
        self.c = len(self.train_classes)
        self.dml_layers = dml_layers
        m = len(self.dml_layers) + 1
        self.min_coeff2 = 1/(4*m)
        c = torch.from_numpy(params.prior / np.sum(params.prior))
        c = c - self.min_coeff2
        coeffs = torch.Tensor(torch.sqrt(c))
        self.coeffs = nn.Parameter(coeffs, requires_grad=True)
        self.eta = params.eta
        self.l_ma = np.zeros(m) # mean of losses obtained by exp moving average
        self.smooth_factor = params.smooth_factor
        self.num_batch = 0

        # create classification module
        d = params.d
        # d = params.embed_dim
        if (params.h_dim > 0):
            self.classifier = nn.Sequential(
                nn.Linear(d, params.h_dim), nn.ReLU(),
                nn.Linear(params.h_dim, len(self.train_classes))).to(device)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d, len(self.train_classes))).to(device)

        self.criterion = nn.CrossEntropyLoss()


    def forward(self, X, X_fea, y):
        coeff2 =  self.coeffs**2
        m = len(self.dml_layers) + 1
        err_coeff2 = torch.sum(coeff2)+ m*self.min_coeff2 - 1

        l = [None]*m
        losses = np.zeros(m)
        y_smooth =  binarize_and_smooth_labels(y, self.c, smoothing_const=self.smooth_factor)
        class_out = F.softmax(self.classifier(X_fea), dim=1)
        l[0] = torch.sum(-y_smooth * torch.log(class_out), -1).mean()
        for i in range(1, m):
            l[i] = self.dml_layers[i-1](X, y)

        for i in range(0, m):
            losses[i] = l[i].item()
            self.l_ma[i] = (losses[i] + self.num_batch*self.l_ma[i])/(self.num_batch+1)

        self.num_batch += 1
        l_mean = np.mean(self.l_ma)

        final_loss = (coeff2[0]+ self.min_coeff2) * (l_mean/self.l_ma[0]) * l[0]
        for i in range(1, m):
            final_loss += (coeff2[i]+ self.min_coeff2) * (l_mean/self.l_ma[i]) * l[i]

        # return final_loss + self.eta*err_coeff2 + 1000*self.eta*(err_coeff2)**2, losses, coeff2.cpu().data.numpy()
        return final_loss + 100*self.eta*(err_coeff2**2), losses, coeff2.cpu().data.numpy() + self.min_coeff2


class DistLoss(nn.Module):
    def __init__(self):
        super(DistLoss, self).__init__()

    def forward(self, xl):
        dist = 0
        m = len(xl)
        for i in range(m):
            for j in range(i+1,m):
                dist += torch.pairwise_distance(xl[i], xl[j]).mean()

        dist = 2/(m*(m-1))* dist
        dist_loss = torch.relu(2-dist)
        return dist_loss


class WEDL(nn.Module):
    def __init__(self, model, dml_layers, params , device='cuda'):
        super(WEDL, self).__init__()
        self.train_classes = params.train_classes
        self.c = len(self.train_classes)
        self.dml_layers = dml_layers

        # create classification module
        d = params.embed_dim
        if (params.h_dim > 0):
            self.classifier = nn.Sequential(
                nn.Linear(d, params.h_dim), nn.ReLU(),
                nn.Linear(params.h_dim, len(self.train_classes))).to(device)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d, len(self.train_classes))).to(device)

        self.dml_layers.insert(0, self.classifier)
        m = len(self.dml_layers)
        self.min_coeff2 = 1/(4*m)
        c = torch.from_numpy(params.prior / np.sum(params.prior))
        c = c - self.min_coeff2
        coeffs = torch.Tensor(torch.sqrt(c))
        self.coeffs = nn.Parameter(coeffs, requires_grad=True)
        self.eta = params.eta
        self.l_ma = np.zeros(m) # mean of losses obtained by exp moving average
        self.smooth_factor = params.smooth_factor
        self.num_batch = 0
        self.div_loss = DistLoss()
        self.lam = params.lam

    def forward(self, X, Xl, X_fea, y):
        coeff2 =  self.coeffs**2
        m = len(self.dml_layers)
        err_coeff2 = torch.sum(coeff2)+ m*self.min_coeff2 - 1

        l = [None]*m
        losses = np.zeros(m)
        y_smooth =  binarize_and_smooth_labels(y, self.c, smoothing_const=self.smooth_factor)
        class_out = F.softmax(self.classifier(Xl[0]), dim=1)
        l[0] = torch.sum(-y_smooth * torch.log(class_out), -1).mean()
        for i in range(1, m):
            l[i] = self.dml_layers[i](Xl[i], y)

        for i in range(0, m):
            losses[i] = l[i].item()
            self.l_ma[i] = (losses[i] + self.num_batch*self.l_ma[i])/(self.num_batch+1)

        self.num_batch += 1
        l_mean = np.mean(self.l_ma)

        loss_div = self.lam*self.div_loss(Xl)
        final_loss = (coeff2[0]+ self.min_coeff2) * (l_mean/self.l_ma[0]) * l[0]
        for i in range(1, m):
            final_loss += (coeff2[i]+ self.min_coeff2) * (l_mean/self.l_ma[i]) * l[i]

        final_loss += loss_div
        # return final_loss + self.eta*err_coeff2 + 1000*self.eta*(err_coeff2)**2, losses, coeff2.cpu().data.numpy()
        return final_loss + 100*self.eta*(err_coeff2**2), losses, coeff2.cpu().data.numpy() + self.min_coeff2, loss_div


class Last_Embedding(nn.Module):
    def __init__(self, m, embed_dim):
        super(Last_Embedding, self).__init__()
        self.embed_layer1 = nn.Linear(m*embed_dim, embed_dim*2)
        self.embed_layer = nn.Linear(2*embed_dim, embed_dim*2)


    def forward(self, X_cat):
        X = self.embed_layer1(X_cat)
        X = nn.Tanh()(X)
        X = self.embed_layer(X)
        return self.distance_loss(X, X_cat), X

    def distance_loss(self, X, X_cat):
        D1 = (torch.cdist(X, X) ** 2)
        D1 = D1 / D1.sum()
        D2 = torch.cdist(X_cat, X_cat) ** 2
        D2 = D2 / D2.sum()
        n = D1.numel()
        return torch.norm(D1-D2)*n


def main():
    data_size, input_dim = 1, 4
    # h_loss = HLoss()
    #
    # xl = []
    # xl.append(torch.tensor([1., 1, 0, 0], requires_grad=True).unsqueeze(0))
    # xl.append(torch.tensor([1., 1, 0, 0], requires_grad=True).unsqueeze(0))
    # xl.append(torch.tensor([1, .9, 0, .1], requires_grad=True).unsqueeze(0))
    # xl.append(torch.tensor([1, .8, .2, 1], requires_grad=True).unsqueeze(0))
    # xl.append(torch.tensor([1, .9, .1, 0], requires_grad=True).unsqueeze(0))
    #
    # div_loss = h_loss(xl)
    # print('div_loss=%2f'%(div_loss))
    #
    # xl = []
    # xl.append(torch.tensor([1., 0, 0, 0], requires_grad=True).unsqueeze(0))
    # xl.append(torch.tensor([0., 1, 0, 0], requires_grad=True).unsqueeze(0))
    # xl.append(torch.tensor([0., 0, 1., 0], requires_grad=True).unsqueeze(0))
    # xl.append(torch.tensor([0, 0, 0, 1.], requires_grad=True).unsqueeze(0))
    # div_loss = h_loss(xl)
    # print('div_loss=%2f' % (div_loss))

    l = list(range(0,3))
    target = torch.tensor(l)
    one_hot = torch.nn.functional.one_hot(target).float()
    n = len(one_hot)
    xl = []
    for i in range(n):
        xl.append(one_hot[i].unsqueeze(0).repeat(2,1))

    D = DistLoss()
    loss, dist = D(xl)

    print('div_loss=%2f'% (loss))

    print('finished!!!')


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
