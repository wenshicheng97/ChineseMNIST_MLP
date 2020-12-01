from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score


def evaluate(real, pred):
    precision = precision_score(real, pred, average='macro')
    recall = recall_score(real, pred, average='macro')
    f1_macro = f1_score(real, pred, average='macro')
    f1_micro = f1_score(real, pred, average='micro')
    kappa = cohen_kappa_score(real, pred, labels=[i for i in range(15)])
    print('precision:', precision)
    print('recall:', recall)
    print('f1_macro:', f1_macro)
    print('f1_micro:', f1_micro)
    print('kappa:', kappa)
    return precision, recall, f1_macro, f1_micro, kappa
