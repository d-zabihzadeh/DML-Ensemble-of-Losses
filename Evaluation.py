from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.spatial.distance import cdist
import numpy as np
import torch
from statistics import mode
from commons.Struct import Struct
from commons.bcolors import bcolors
import models
import time


def evaluation(X, y, Kset=None, X_ref=None, y_ref=None, kNN_K = 3, sim_flag = False):
    if Kset is None:
        Kset = [1, 2, 4, 8]
    res = Struct()
    classN = len(np.unique(y))
    kmax = np.max(Kset)
    res.recallK = np.zeros(len(Kset))
    if X_ref is None:
        X_ref, y_ref = X, y

    n = len(X)
    #compute NMI
    kmeans = KMeans(n_clusters=classN).fit(X)
    res.nmi = normalized_mutual_info_score(y, kmeans.labels_, average_method='arithmetic') * 100

    #compute Recall@K
    if (sim_flag):
        mat = X.dot(X.T)
        indices = np.argsort(-mat, axis=1)[:, 1:kmax+1]
    else:
        mat = cdist(X, X_ref)
        indices = np.argsort(mat, axis=1)[:, 1:kmax+1]

    y_NN = y_ref[indices]
    y_pred = np.zeros(shape=(n), dtype=int)
    for j in range(0, n):
        y_pred[j] = mode(y_NN[j, :kNN_K])

    res.acc = np.sum(y_pred == y) / n * 100
    # res.map = average_precision_score(y_ref, y_pred)

    # Recall@k is the  proportion of relevant items found in the top - k retrievd items
    for i in range(0, len(Kset)):
        pos = 0.
        for j in range(0, n):
            if y[j] in y_NN[j, :Kset[i]]:
                pos += 1.
        res.recallK[i] = pos/n * 100
    return res, y_pred


def eval_model(test_loader, model, kSet=None, device='cuda', max_ins = 100000):
    # switch to evaluation mode
    if kSet is None:
        kSet = [1, 2, 4, 8]
    model.eval()
    testdata = torch.Tensor()
    testlabel = torch.LongTensor()
    with torch.no_grad():
        for i, J in enumerate(test_loader):
            input, target = J[0:2]
            if(device == 'cuda'):
                input = input.to(device)

            # compute output
            output = model(input)
            testdata = torch.cat((testdata, output.cpu()), 0)
            testlabel = torch.cat((testlabel, target))
            if(len(testlabel) >= max_ins):
                break

    res, y_pred = evaluation(testdata.numpy(), testlabel.numpy(), kSet)
    return res, testdata, y_pred, testlabel


def eval_print_res(model, test_loader, train_loader, epoch, epoch_time, kSet=None, device='cuda', train_max_ins = 1000):
    eval_time = time.time()
    with torch.no_grad():
        res, X_test_embed, y_pred, y_test = eval_model(test_loader, model, device=device)
        eval_time = time.time() - eval_time
        print(bcolors.HEADER +
              'epoch:{:d},  Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; '
              'NMI: {nmi:.3f}, Acc: {acc:.2f}, epoch time (seconds): {time:.2f}, eval time:{eval_time:.2f} \n'
              .format(epoch, recall=res.recallK, nmi=res.nmi, acc=res.acc, time=epoch_time, eval_time=eval_time) + bcolors.ENDC)

        res_train = eval_model(train_loader, model, device=device, max_ins=train_max_ins)[0]
        print(bcolors.OKBLUE +
              'Train Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f}, Acc: {acc:.2f} \n'
              .format(recall=res_train.recallK, nmi=res_train.nmi, acc=res_train.acc) + bcolors.ENDC)

    return res, res_train, X_test_embed, y_pred, y_test


def print_init_res(best_score, best_resInfo, hist, model, test_loader, train_loader):
    print(bcolors.HEADER + 'Best Test Recall@1 = %0.2f' % (best_score) + bcolors.ENDC)
    h = Struct()
    echo = False
    if best_score > 0 and hasattr(hist, 'acc_list'):
        print(bcolors.OKBLUE + 'Best Recall: ', np.round(best_resInfo.recallK, 2), bcolors.ENDC)
        print(bcolors.HEADER + 'Best NMI:%0.2f,  Best Acc: %0.2f' % (best_resInfo.nmi, best_resInfo.acc) + bcolors.ENDC)
        if not hasattr(hist, 'recallK_list'):
            hist.recallK_list = hist.recallK_list
        h.nmi_list, h.nmi_train_list = [hist.nmi_list[0]], [hist.nmi_train_list[0]]
        h.acc_list, h.acc_train_list = [hist.acc_list[0]], [hist.acc_train_list[0]]
        h.recallK_list, h.recallK_train_list = [hist.recallK_list[0]], [hist.recallK_train_list[0]]
    elif 'flowers102' in test_loader.dataset.name and isinstance(model, models.bn_inception.BNInception):
            h.nmi_list, h.nmi_train_list = [.65082*100], [0.607*100]
            h.recallK_list, h.recallK_train_list = [np.array([0.809, 0.879, 0.927, 0.956])*100], \
                                                   [np.array([0.618, 0.719, 0.812, 0.884])*100]
            h.acc_list, h.acc_train_list = [82.88], [62.89]
    elif 'flowers102' in test_loader.dataset.name and isinstance(model, models.bn_inception.BNInception_Attention):
        h.nmi_list, h.nmi_train_list = [.6308*100], [0.589*100]
        h.recallK_list, h.recallK_train_list = [np.array([0.778, 0.862, 0.916, 0.956])*100], \
                                               [np.array([0.599, 0.701, 0.796, 0.873])*100]
        h.acc_list, h.acc_train_list = [79.39], [61.82]
    else:
        res, res_train = eval_print_res(model, test_loader, train_loader, 0, 0)[:2]
        echo = True
        h.nmi_list, h.nmi_train_list = [res.nmi], [res_train.nmi]
        h.recallK_list, h.recallK_train_list = [res.recallK], [res_train.recallK]
        h.acc_list, h.acc_train_list = [res.acc], [res_train.acc]

    if(not echo):
        print(bcolors.HEADER +
              'epoch:{:d},  Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; '
              'NMI: {nmi:.3f}, Acc: {acc:.2f}\n'
              .format(0, recall=h.recallK_list[0], nmi=h.nmi_list[0], acc=h.acc_list[0]) + bcolors.ENDC)
        print(bcolors.OKBLUE +
              'Train Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; '
              'NMI: {nmi:.3f}, Acc:{acc:.2f}\n'
              .format(recall=h.recallK_train_list[0], nmi=h.nmi_train_list[0], acc=h.acc_train_list[0]) + bcolors.ENDC)


    return h


def print_res(score, resInfo):
    print(bcolors.HEADER + 'Best Test Recall@1 = %0.2f' % (score) + bcolors.ENDC)
    if score > 0:
        print(bcolors.OKBLUE + 'Best Recall: ', np.round(resInfo.recallK, 2), bcolors.ENDC)
        print(bcolors.FAIL + 'Best Test NMI = %0.2f, epoch = %d' % (resInfo.nmi, resInfo.epoch) + bcolors.ENDC)
        if hasattr(resInfo, 'acc'):
            print(bcolors.OKBLUE + 'Best Acc = %0.2f' % (resInfo.acc) + bcolors.ENDC)


def update_history(h, res, res_train=None):
    h.nmi_list.append(res.nmi), h.recallK_list.append(res.recallK)
    h.acc_list.append(res.acc)
    if(res_train is not None):
        h.acc_train_list.append(res_train.acc)
        h.nmi_train_list.append(res_train.nmi)
        h.recallK_train_list.append(res_train.recallK)

