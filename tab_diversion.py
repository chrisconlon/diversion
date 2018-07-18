import pyblp
import numpy as np
import pandas as pd

def relative_error(z):
    med_abs = lambda x: np.median(np.abs(x))
    mean_abs = lambda x: np.mean(np.abs(x))
    st_abs = lambda x: np.std(np.abs(x))
    flist= [np.median,np.mean,med_abs,mean_abs,st_abs]
    return 100.0*np.array([f(z) for f in flist])

def compute_rel_table(mte_diversion,ate_diversion,logit_diversion,og_mask):
    col_list = ['med($y-x$)','mean($y-x$)', 'med($|y-x|$)','mean($|y-x|$)','std($|y-x|$)']
    row_list = ['$ATE$','$Logit$']*3

    rel_diff=np.log(ate_diversion/mte_diversion)
    log_diff=np.log(logit_diversion/mte_diversion)

    best_mte=np.amax(mte_diversion*(~og_mask),axis=1)
    best_ate=np.amax(ate_diversion*(~og_mask),axis=1)
    best_logit=np.amax(logit_diversion*(~og_mask),axis=1)

    rel_diff_b=np.log(best_ate/best_mte)
    log_diff_b=np.log(best_logit/best_mte)

    # compute each row of table
    a1=relative_error(rel_diff_b)
    a2=relative_error(log_diff_b)
    b1=relative_error(rel_diff[~og_mask])
    b2=relative_error(log_diff[~og_mask])
    c1=relative_error(rel_diff[og_mask])
    c2=relative_error(log_diff[og_mask])
    df=pd.DataFrame(np.vstack([a1,a2,b1,b2,c1,c2]),index=row_list)
    df.columns=col_list
    return df


def compute_avg_table(mte_diversion,ate_diversion,logit_diversion,og_mask):
    def do_one(z):
        best=np.amax(z*(~og_mask),axis=1)
        matches = np.mean(np.argmax(z*(~og_mask),axis=1)==best_ids)
        return np.array([np.median(best), np.mean(best),matches, np.median(z[og_mask]),np.mean(z[og_mask])])*100.0

    row_list = ['Med($D_{jk}$)','Mean($D_{jk}$)','\% Correct','Med($D_{j0}$)','Mean($D_{j0}$)']
    best_ids=np.argmax(mte_diversion*(~og_mask),axis=1)
    df=pd.DataFrame(np.vstack([do_one(mte_diversion),do_one(ate_diversion),do_one(logit_diversion)]).T,index=row_list)
    df.columns=['MTE','ATE','Logit']
    return df
