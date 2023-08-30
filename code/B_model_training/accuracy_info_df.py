from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

def accuracy_info_df(y_true, y_pred):
    """A fucntion to calculate accuracy metrics from confusion matrix and return them as a dataframe."""

    unique, counts = np.unique(y_true,return_counts=True)    
    N = counts[0]    
    P = counts[1]
    
    confmtx = confusion_matrix(y_true, y_pred)
    TN = confmtx[0,0]
    FP = confmtx[0,1]
    FN = confmtx[1,0]
    TP = confmtx[1,1]

    # P's  producer's accuracy (sensitivity) : TP/P
    PA_P =  np.round( TP/P  * 100, 2) 

    # N's producer's accuracy (specificity) : TN/N
    PA_N =  np.round( TN/N * 100, 2) 

    # P's user's accuracy (precision P) : TP/(TP+FP)
    UA_P = np.round( TP / (TP+FP) * 100, 2) 
    
    # N's user's accuracy (precision N) : TN/(TN+FN)
    UA_N = np.round( TN / (TN+FN) * 100, 2)
    
    # overal accuracy: (TP + TN)/(P + N)
    OA = np.round( (TP+TN)/y_true.shape[0]*100, 2) 
    
    D = {'acc':OA,
         'prod_acc_P':PA_P, 
         'prod_acc_N':PA_N,          
         'user_acc_P':UA_P,
         'user_acc_N':UA_N,         
         'TP':TP, 'TN':TN, 
         'FP':FP, 'FN':FN 
        }
    df = pd.DataFrame([D])
    return df