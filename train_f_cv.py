import mat73
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
def get_pos_or_neg(data,position):
    arr = []
    index = 0
    for i in range(data.shape[0]):
        temp = []
        for j in position:
            for k in data[i][j[0]][j[1]]:
                temp.append(k)
        arr.append(temp)
    return np.array(arr)

def get_selected(data=None,freq = None, sec=None):
    if freq:
        data = np.delete(data,freq,axis=3)
    if sec == 0:
        return np.delete(data, 0, axis=4)
    elif sec == 1:
        return np.delete(data,1,axis=4)
    else:
        print('sec not specified')
        return data 

def male_cv(alpha=None):
    data_dict = mat73.loadmat("./data/Emotrans1_girl_data_preprocessed.mat", use_attrdict=True)
    arr = np.array(data_dict["All_Feature"])
    pos = [[0,0],[0,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3]]
    neg = [[0,2],[0,3],[1,0],[1,1],[3,0],[3,1],[3,2],[3,3]]
    pos_df = get_pos_or_neg(arr,pos)
    neg_df = get_pos_or_neg(arr,neg)
    pos_df=get_selected(pos_df,sec=0).reshape(15,96,128,8)
    neg_df = get_selected(neg_df,sec=0).reshape(15,96,128,8)
    raw_data = np.concatenate((pos_df,neg_df),axis=1).reshape(15,192,128*8)
    y = np.concatenate((np.ones((15,96)),np.zeros((15,96))),axis=1)
    # leave one out 
    train_scores=[]
    test_scores = []
    iter = 0
    alphas = [10,100,1000]
    coefs = []
    x = raw_data.reshape(15*192,1024)
    y = y.reshape(15*192)
    for a in alphas:
        # ,max_iter=int(1e6)
        model = LogisticRegression(C=a, penalty='l1',max_iter=5000,solver='saga')
        scores = cross_val_score(model,x,y,cv=5)
        with open('girl_with_acc_cv'+str(a)+".txt",'w') as f:
            f.write("accuracy - training")
            f.write(str(scores))
        print(scores)
def male_ex():
    #,0.001
    alphas = [0.1,0.01,0.001]
    male_cv()

male_ex() 