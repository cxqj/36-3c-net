import numpy as np

def getAP(conf,labels): # (1,20),(1,20)
    assert len(conf)==len(labels)
    sortind = np.argsort(-conf)  # 最大位置的索引
    tp = labels[sortind]==1; fp = labels[sortind]!=1
    npos = np.sum(labels);   # 多少个动作实例

    fp = np.cumsum(fp).astype('float32'); tp = np.cumsum(tp).astype('float32')
    rec=tp/npos
    prec=tp/(fp+tp)
    # 获取正样本所在的位置
    tmp = (labels[sortind]==1).astype('float32')

    return np.sum(tmp*prec)/npos  # 除以npos得到prec的均值

def getClassificationMAP(confidence,labels):  # (212,20), (212,20)
    ''' confidence and labels are of dimension n_samples x n_label '''

    AP = []
    for i in range(np.shape(labels)[1]):  # 212
       AP.append(getAP(confidence[:,i], labels[:,i]))
    return 100*sum(AP)/len(AP)
