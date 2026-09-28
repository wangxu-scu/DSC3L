import sys
import numpy as np
from fxeval import fx_calc_map_label

def show_progressbar(rate, *args, **kwargs):
    '''
    :param rate: [current, total]
    :param args: other show
    '''
    inx = rate[0] + 1
    count = rate[1]
    bar_length = 30
    rate[0] = int(np.around(rate[0] * float(bar_length) / rate[1])) if rate[1] > bar_length else rate[0]
    rate[1] = bar_length if rate[1] > bar_length else rate[1]
    num = len(str(count))
    str_show = ('\r%' + str(num) + 'd / ' + '%' + str(num) + 'd  (%' + '3.2f%%) [') % (inx, count, float(inx) / count * 100)
    for i in range(rate[0]):
        str_show += '='

    if rate[0] < rate[1] - 1:
        str_show += '>'

    for i in range(rate[0], rate[1] - 1, 1):
        str_show += '.'
    str_show += '] '
    for l in args:
        str_show += ' ' + str(l)

    for key in kwargs:
        try:
            str_show += ' ' + key + ': %.4f' % kwargs[key]
        except Exception:
            str_show += ' ' + key + ': ' + str(kwargs[key])
    if inx == count:
        str_show += '\n'

    sys.stdout.write(str_show)
    sys.stdout.flush()

def cal_map_bi(img_feas, text_feas, labels, best_map_bi_50=0., best_map_bi_all=0., dataset_name=None):
    print('======================== Bi-modal Retrieval ==========================')
    print('======================================================================')
    result1_all = fx_calc_map_label(
        img_feas, text_feas, labels, k=0, dist_method='COS')
    result2_all = fx_calc_map_label(
        text_feas, img_feas, labels, k=0, dist_method='COS')
    # result1_50 = fx_calc_map_label(
    #     img_feas, text_feas, labels, k=50, dist_method='COS')
    # result2_50 = fx_calc_map_label(
    #     text_feas, img_feas, labels, k=50, dist_method='COS')
    print("MAP@All: Image query:{:.3f}, Text query:{:.3f}".format(result1_all, result2_all))
    # print("MAP@50: Image query:{:.3f}, Text query:{:.3f}".format(result1_50, result2_50))

    # mean_map_50 = (result1_50 + result2_50) / 2.0
    # if mean_map_50 > best_map_bi_50:
    #     best_map_bi_50 = mean_map_50
    #     print('best map@50 improves to: {:.3f}, detail: i2t: {:.3f}, t2i: {:.3f}'.format(best_map_bi_50, result1_50, result2_50))
    # else:
    #     print('best map@50 still is: {:.3f}'.format(best_map_bi_50))

    mean_map_all = (result1_all + result2_all) / 2.0
    if mean_map_all > best_map_bi_all:
        best_map_bi_all = mean_map_all
        if dataset_name is not None:
            import scipy.io as sio
            # sio.savemat(dataset_name+'_latent_xentropy.mat',{'XTe1proj':img_feas, 'XTe2proj':text_feas, 'testLabel':labels})
        print('best map@All improves to: {:.3f}, detail: i2t: {:.3f}, t2i: {:.3f}'.format(best_map_bi_all, result1_all, result2_all))
    else:
        print('best map@All still is: {:.3f}'.format(best_map_bi_all))
    return best_map_bi_50, best_map_bi_all

def cal_map_all(img_feas, text_feas, labels, best_map_all_50=0., best_map_all_all=0.):
    print('======================== All-modal Retrieval ==========================')
    print('======================================================================')
    result1_all = fx_calc_map_label(
        img_feas, np.concatenate(
            (img_feas, text_feas), 0), np.concatenate(
            (labels, labels), 0), k=0, dist_method='COS')
    result2_all = fx_calc_map_label(
        text_feas, np.concatenate(
            (img_feas, text_feas), 0), np.concatenate(
            (labels, labels), 0), k=0, dist_method='COS')
    result1_50 = fx_calc_map_label(
        img_feas, np.concatenate(
            (img_feas, text_feas), 0), np.concatenate(
            (labels, labels), 0), k=50, dist_method='COS')
    result2_50 = fx_calc_map_label(
        text_feas, np.concatenate(
            (img_feas, text_feas), 0), np.concatenate(
            (labels, labels), 0), k=50, dist_method='COS')
    print("MAP@All: Image query:{}, Text query:{:.3f}".format(result1_all, result2_all))
    print("MAP@50: Image query:{}, Text query:{:.3f}".format(result1_50, result2_50))
    mean_map_50 = (result1_50 + result2_50) / 2.0
    if mean_map_50 > best_map_all_50:
        best_map_all_50 = mean_map_50
        print('best map@50 improves to: {:.3f}, detail: i2t: {:.3f}, t2i: {:.3f}'.format(best_map_all_50, result1_50, result2_50))
    else:
        print('best map@50 still is: {:.3f}'.format(best_map_all_50))

    mean_map_all = (result1_all + result2_all) / 2.0
    if mean_map_all > best_map_all_all:
        best_map_all_all = mean_map_all
        print('best map@All improves to: {:.3f}, detail: i2t: {:.3f}, t2i: {:.3f}'.format(best_map_all_all, result1_all, result2_all))
    else:
        print('best map@All still is: {:.3f}'.format(best_map_all_all))
    return best_map_all_50, best_map_all_all


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def cal_knn_acc(v1, label_1, v2, label_2, best_acc=0., dataset_name=None):

    neigh = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    neigh.fit(v1, label_1)
    pred = neigh.predict(v2)
    acc12 = accuracy_score(label_2, pred)


    neigh = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    neigh.fit(v2, label_2)
    pred = neigh.predict(v1)
    acc21 = accuracy_score(label_1, pred)
    mean_acc = (acc12 + acc21) / 2.

    if mean_acc > best_acc:
        best_acc = mean_acc
        if dataset_name is not None:
            import scipy.io as sio
            # sio.savemat(dataset_name+'_latent_mcml.mat',{'XTe1proj':v1, 'XTe2proj':v2, 'testLabel':label_1})
        print('best acc improves to: {}, V1-V2:{}, V2-V1:{}'.format(best_acc, acc12, acc21))
    else:
        print('best acc still is: {}'.format(best_acc))

    return best_acc