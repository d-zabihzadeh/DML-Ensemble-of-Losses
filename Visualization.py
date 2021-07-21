import matplotlib.pyplot as plt
import numpy as np
from commons.Result import get_bestRes


ds_names = {'cars': 'CARS 196', 'cub':'CUB-200-2011', 'flowers102':'Oxford 102 Flowers'}
font_size, title_size = 14, 16
colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y']


def show_annotate(text, loc='upper left'):
    from matplotlib.offsetbox import AnchoredText
    ax = plt.gca()
    at = AnchoredText(text,
                      prop=dict(size=15), frameon=True,
                      loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)


def plot_RecallK(recallK_list, title,  kSet, file_name=''):
    plt.figure()
    recallK_list = np.array(recallK_list)
    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y']
    epochs = range(1, len(recallK_list) + 1)

    for i in range(len(kSet)):
        plt.plot(epochs, recallK_list[:, i], color=colors[i], label='Test Recall@%d' % (kSet[i]))
    plt.title(title)
    plt.legend()
    if len(file_name) > 0:
        max_recall1 = np.max(recallK_list[:, 0])
        file_name = file_name +'%0.2f' % (max_recall1)
        plt.savefig(file_name + '.eps')
        plt.savefig(file_name + '.png')
    plt.show()


def plot_NMI_Recall(ds, hist, save_flag = False, plot_train = True, alg=''):
    epochs = range(1, len(hist.nmi_list) + 1)
    plt.plot(epochs, hist.nmi_list, color='blue', marker='*', label='Test NMI')
    plt.xlabel('Epochs')
    plt.ylabel('NMI')
    if plot_train:
        plt.plot(epochs, hist.nmi_train_list, color='red', marker='^', label='Train NMI')
        plt.title('Test and Train NMI on %s' % (ds))
    else:
        plt.title('Test NMI on %s' % (ds))

    plt.legend()
    if save_flag:
        max_nmi = np.max(hist.nmi_list)
        fname = './figs/%s_%s_nmi_%0.2f'%(alg,ds, max_nmi)
        plt.savefig(fname +'.eps')
        plt.savefig(fname +'.png')
        fname_recall_test = './figs/%s_%s_test_recall_' % (alg, ds)
        fname_recall_train = './figs/%s_%s_train_recall_' % (alg, ds)
    else:
        fname_recall_test, fname_recall_train = '', ''

    plt.show()
    plot_RecallK(hist.recallK_list, 'Test RecallK on %s' % ds, hist.kSet, file_name=fname_recall_test)

    if plot_train:
        plot_RecallK(hist.recallK_train_list, 'Train RecallK on %s' % ds, hist.kSet, file_name=fname_recall_train)


def plot_NMI_recall_best(ds_name, alg):
    best_score, params, best_resInfo, hist = get_bestRes(ds_name, alg)
    plot_NMI_Recall(ds_name, hist)


def plot_NMI_recall_dmls(ds, alg_list, num_epoch = 20, kSet=None, alg_display_names=None, save_flag = False,
                         annotate_loc = 'center', experiment_name=''):
    font_size, title_size = 14,16
    if kSet is None:
        kSet = [1, 2, 4, 8]
    if alg_display_names is None:
        alg_display_names = alg_list

    ds_name = ds_names[ds]
    hist_list = []
    nmi_bottom, recall1_bottom = 0, 0
    for alg in alg_list:
        best_score, params, best_resInfo, hist = get_bestRes(ds, alg)
        hist_list.append(hist)
        num_epoch = np.minimum(num_epoch, len(hist.nmi_list))
        nmi_bottom += hist.nmi_list[0]
        if not hasattr(hist, 'recallK_list'):
            hist.recallK_list = hist.recallK_ist

        hist.recallK_list = np.array(hist.recallK_list)
        recall1_bottom += hist.recallK_list[0, 0]


    nmi_bottom, recall1_bottom = nmi_bottom/ len(alg_list), recall1_bottom/len(alg_list)
    plt.figure()
    epochs = range(0, num_epoch)
    for i in range(len(alg_list)):
        plt.plot(epochs, hist_list[i].nmi_list[:num_epoch], color=colors[i], label='%s' % (alg_display_names[i]))


    plt.title('Test NMI on %s'% (ds_name), fontsize=title_size, fontweight='bold')
    plt.xticks(list(range(0,num_epoch,2)))
    plt.xlabel('Epochs', fontsize=font_size)
    plt.ylabel('NMI', fontsize=font_size)
    # plt.ylim(bottom=nmi_bottom)
    plt.legend(prop=dict(weight='bold'))
    # show_annotate(alg_display_names[-1], loc=annotate_loc)
    if save_flag:
        fname = './figs/test_nmi_%s_%s' % (ds, experiment_name)
        plt.savefig(fname + '.eps')
        plt.savefig(fname + '.png')

    plt.show()

    for j in range(len(kSet)):
        plt.figure()
        for i in range(len(alg_list)):
            plt.plot(epochs, hist_list[i].recallK_list[:num_epoch, j],
                     color=colors[i], label='%s'%(alg_display_names[i]))

        plt.title('Test Recall@%d on %s' % (kSet[j],ds_name), fontsize=14, fontweight='bold')
        plt.xticks(list(range(0, num_epoch, 2)))
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Recall@%d'%(kSet[j]), fontsize=12)
        # if(j==0):
        #     plt.ylim(bottom=recall1_bottom)
        plt.legend(prop=dict(weight='bold'))
        # show_annotate(alg_display_names[-1], loc=annotate_loc)
        if save_flag:
            fname = './figs/recall@%d_%s_%s' % (kSet[j], ds, experiment_name)
            plt.savefig(fname + '.eps')
            plt.savefig(fname + '.png')

        plt.show()



def main():
    print('Congratulations to you!')


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
