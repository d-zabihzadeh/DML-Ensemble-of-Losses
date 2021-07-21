from __future__ import print_function
from __future__ import division
import torch
import json
import random, numpy as np
from commons.Struct import Struct
from commons.Result import print_params, save_res
from Evaluation import print_res
from Visualization import plot_NMI_Recall


# __repr__ may contain `\n`, json replaces it by `\\n` + indent
json_dumps = lambda **kwargs: json.dumps(
    **kwargs
).replace('\\n', '\n    ')


class JSONEncoder(json.JSONEncoder):
    def default(self, x):
        # add encoding for other types if necessary
        if isinstance(x, range):
            return 'range({}, {})'.format(x.start, x.stop)
        if not isinstance(x, (int, str, list, float, bool)):
            return repr(x)
        return json.JSONEncoder.default(self, x)


def load_config(config_name = 'config.json'):
    config = json.load(open(config_name))
    def eval_json(config):
        for k in config:
            if type(config[k]) != dict:
                config[k] = eval(config[k])
            else:
                eval_json(config[k])
    eval_json(config)
    return config


def reset_rng(seed):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if (torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)  # set random seed for all gpus


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def check_save(best_score, h, params, alg=None, model=None, plot_train=True):
    recall1_list = np.array(h.recallK_list)[:, 0]
    best_epoch = np.argmax(recall1_list)
    best_recall1 = recall1_list[best_epoch]
    if alg is None:
        alg = params.alg

    if (best_recall1 > best_score):
        best_Res = Struct()
        best_Res.epoch, best_Res.nmi = best_epoch, h.nmi_list[best_epoch]
        best_Res.recallK, best_Res.acc = h.recallK_list[best_epoch], h.acc_list[best_epoch]
        save_res(params.ds, alg, best_recall1, params, best_Res, h)
        print_params(params)
        print_res(best_recall1, best_Res)
        plot_NMI_Recall(params.ds, h, save_flag=True, alg=alg, plot_train=plot_train)
        if model is not None:
            torch.save(model.state_dict(), './saved_models/%s__%s_%0.2f.mdl' % (alg, params.ds, best_recall1))

        return True, best_epoch
    else:
        return False, best_epoch


def main():
    print('finished!!!')


if __name__ == '__main__':
    main()
    print('Congratulations to you!')




