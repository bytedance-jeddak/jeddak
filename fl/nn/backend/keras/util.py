import numpy as np

from sklearn.metrics import roc_auc_score


def loss_fn(y_true, y_pred):
    # return K.sum(y_true * y_pred)
    return y_true * y_pred


def get_prob(y_pred):
    res = []
    for pred in y_pred:
        if pred[0] > pred[1]:
            res.append(1 - pred[0])
        else:
            res.append(pred[1])
    return res


def auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def generate_random_nums(x, re_seed=False, lower=-2 ** 10, upper=2 * 10):
    r = np.zeros(x.shape)
    if x.ndim == 1:
        for i in range(len(x)):
            # if re_seed:
            #     np.random.seed(int(time.time()))
            r[i] = np.random.uniform(-1, 1)
    elif x.ndim == 2:
        for i in range(len(x)):
            for j in range(len(x[0])):
                # if re_seed:
                #     np.random.seed(int(time.time() * 1000 % 2 ** 32))
                r[i][j] = np.random.uniform(-1, 1)

    else:
        raise ValueError(f"un-support generate random for dim={x.ndim}, should be dim=1 or 2")

    return r
