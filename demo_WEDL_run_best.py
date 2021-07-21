from Evaluation import eval_print_res, print_init_res, update_history, evaluation
from commons.Struct import Struct
from commons.bcolors import bcolors
from commons.Result import print_params, get_bestRes
import numpy as np
import copy
import torch
from dataset import load_ds
import utils
from models.bn_inception_ens import BNInception_Ens_Attention
from Losses import ProxyNCA, Trip_hinge_loss, N_pair_loss, BinomialLoss
from WEL_DML import WEDL, Last_Embedding
import time
from utils import check_save


def override_best_params(params):
    params.epoch = 20
    return params


utils.reset_rng(seed=0)
np.set_printoptions(precision=3)
dml_alg_list = ['triplet_hinge', 'n_pairs', 'proxy_nca', 'soft_triplet', 'binomial']
alg, alg2 = 'WEDL', 'WEDL_C'
ds_name = 'flowers102'

best_score, params, best_resInfo, hist_best = get_bestRes(ds_name, alg)
best_score2 = get_bestRes(ds_name, alg2)[0]

if(best_score == 0):
    params = Struct()
    params.ds, params.alg = ds_name, alg
    params.epoch = 25
    params.model_lr = .0001
    params.weight_decay = 1e-4
    params.batch_size = 64
    params.embed_dim = 64
    params.h_dim = 0
    params.margin = 1
    params.embedding_lr = .001
    params.eta = 100
    params.smooth_factor = .15
    # params of NCA Proxy
    params.proxy_nca_lr = .015
    params.lam = .01
    params.scaling_x, params.scaling_p = 1, 3  # Scaling factor for the normalized proxies


print_params(params)
params = override_best_params(params)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device = torch.device('cpu')
config = utils.load_config()

# load data
train_loader, test_loader = load_ds(params, config)
c = len(params.train_classes)

trip_hinge_layer = Trip_hinge_loss(margin=params.margin, type_of_triplets='semihard').to(device)
n_pairs_layer = N_pair_loss().to(device)
proxy_nca_layer = ProxyNCA(nb_classes=c, sz_embedding= params.embed_dim).to(device)
proxy_nca_layer.scaling_p, proxy_nca_layer.scaling_x = params.scaling_p, params.scaling_x
bin_loss = BinomialLoss().to(device)
dml_layers = [trip_hinge_layer,bin_loss, proxy_nca_layer]

# create model
params.m = len(dml_layers)+1 # 1 for the classification
model = BNInception_Ens_Attention(m=params.m, embed_dim=params.embed_dim, is_pretrained=True).to(device)

params.prior = np.array([2, 1, 2, 3], dtype='float32')
ensemble_losses_dml = WEDL(model, dml_layers, params).to(device)
last_embed_module = Last_Embedding(params.m, params.embed_dim)

opt = torch.optim.Adam(
    [
        { # inception parameters, excluding embedding layer
            **{'params': list(
                set(
                    model.parameters()
                ).difference(
                    set(model.embed_layer_list.parameters())
                )
            )},
            **{"lr": params.model_lr}
        },
        { # embedding parameters
            **{'params': model.embed_layer_list.parameters()},
            **{'lr': params.embedding_lr}
        },
        { # proxy nca parameters
            **{'params': proxy_nca_layer.parameters()},
            **{'lr': params.proxy_nca_lr}
        },
        { # ensemble loss classifier
            **{'params': ensemble_losses_dml.classifier.parameters()},
            **{'lr': params.embedding_lr}
        },
        { # ensemble loss coefficients
            **{'params': ensemble_losses_dml.coeffs},
            **{'lr': params.model_lr}
        }
    ],
    eps=.01, weight_decay=params.weight_decay, lr=params.model_lr)

opt_dist = torch.optim.Adam([{"params": last_embed_module.parameters(), "lr": params.embedding_lr}],
                            eps=.01, weight_decay=params.weight_decay, lr=params.embedding_lr)

h = print_init_res(best_score, best_resInfo, hist_best, model, test_loader, train_loader)
print_params(params), print('')
h.kSet = [1, 2, 4, 8]
h2 = copy.deepcopy(h)

epoch = 1
while (epoch <= params.epoch):
    time_per_epoch = time.time()
    model.train()
    for batch_idx, (X, labels, _) in enumerate(train_loader):
        X, labels = X.to(device), labels.to(device)
        opt.zero_grad()
        X_cat, Xl, X_fea = model(X, return_fea = True)
        loss, losses, coeffs, loss_div = ensemble_losses_dml(X_cat, Xl, X_fea, labels)
        loss.backward()
        opt.step()

        opt.zero_grad()
        opt_dist.zero_grad()
        X_cat = model(X)
        loss_dist, _ = last_embed_module(X_cat.cpu())
        loss_dist.backward()
        opt_dist.step()

        if batch_idx % 20 == 0:
            print("Epoch {} Iteration {}: Loss = {}, losses: {}, coeffs: {}, loss div:{:0.3f}, loss dist:{:0.3f}, sum coeffs: {:0.2f}".
                  format(epoch, batch_idx, loss, np.array(losses), coeffs, loss_div, loss_dist, coeffs.sum()))

    time_per_epoch = time.time() - time_per_epoch
    res, res_train, X_test_embed, y_pred, y_test = eval_print_res(model, test_loader, train_loader, epoch, time_per_epoch)
    with torch.no_grad():
        loss_dist_test ,X_test_embed = last_embed_module(X_test_embed)
        res_dist,_ = evaluation(X_test_embed, y_test.numpy())
        print(bcolors.FAIL +
              'epoch:{:d},  Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; '
              'NMI: {nmi:.3f}, Acc: {acc:.2f}, loss_dist_test={loss_dist:.3f} \n'
              .format(epoch, recall=res_dist.recallK, nmi=res_dist.nmi, acc=res_dist.acc, loss_dist=loss_dist_test) + bcolors.ENDC)

    update_history(h, res, res_train), update_history(h2, res_dist)
    model.weight_embeddings = torch.from_numpy(np.sqrt(coeffs))
    epoch += 1


check_save(best_score,h,params)
check_save(best_score2,h2,params, alg=alg2, plot_train=False)

print('finished!!!')


