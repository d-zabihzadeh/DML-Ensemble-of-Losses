from os import path, listdir
import pickle
import textwrap

def get_bestRes(ds,alg,path_res=None):
    if(path_res is None):
        path_res = '.\\Res_'

    dir_name = path_res + ds
    file_names = listdir(dir_name)
    best_score, bestParams, best_resInfo, hist_best = 0, [], [], []
    for f in file_names:
        if(f.lower().startswith(alg.lower())):
            with open(path.join(dir_name, f), 'rb') as file:
                score, params, res_info, hist = pickle.load(file)
            if (score > best_score):
                best_score = score
                bestParams = params
                best_resInfo = res_info
                hist_best = hist

    return best_score, bestParams, best_resInfo, hist_best


def print_params(params):
    print('params:')
    print(textwrap.fill(str(params.__dict__), width=150))
    if hasattr(params, 'load_options'):
        print('load data options:')
        print(textwrap.fill(str(params.load_options.__dict__), width=150))


def save_res(ds, alg, best_score, params, res, hist):
    file_name = '.\\Res_%s\\%s__%s_%0.2f.res' % (ds, alg, ds, best_score)
    if hasattr(params,'a'):
        del params.a
    if hasattr(params, 'preds'):
        del params.preds

    with open(file_name, 'wb') as f:  # wb: open binary file for writing
        pickle.dump([best_score, params, res, hist], f)




