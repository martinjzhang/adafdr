import numpy as np
import scipy as sp
from scipy import stats
from sklearn.mixture import GaussianMixture
import torch
from torch.autograd import Variable
from adafdr.util import *
import multiprocessing as mp
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as tf
import time

np.set_printoptions(precision=4,suppress=True)

""" 
    preprocessing: standardize the hypothesis features 
    
    ----- input  -----
    x_: np_array, n*d, the hypothesis features.
    qt_norm: bool, if perform quantile normalization.
    return_metainfo: bool, if return the meta information regarding the features
    vis_dim: list, the dimensions to visualize. counting starts from 0, needed only when verbose is True
    verbose: bool, if generate ancillary information
    
    ----- output -----
    x: Processed feature stored as an n*d array. The discrete feature is reordered based on the alt/null ratio.
    meta_info: a d*2 array. The first dimensional corresponds to the type of the feature (continuous/discrete). The second 
               dimension is a list on the mapping (from small to large). For example, if the before reorder is [0.1,0.2,0.7].
               This list may be [0.7,0.1,0.2].
""" 

def feature_preprocess(x, qt_norm=True, continous_rank=True):
    """Feature preprocessing: 1. quantile normalization, 
        2. make the range [0,1] for each feature

    Args:
        x ((n,d) ndarray): The covaraites.
        qt_norm (bool): If perform quantile normalization.
        continous_rank (bool): Indicate if break the same value by randomization.
        
    Returns:
        x ((n,d) ndarray): The discrete feature is reordered based on the alt/null ratio.
    """
    if len(x.shape) == 1: 
        x = x.reshape([-1,1])
    n,d = x.shape     
    if qt_norm:
        x=rank(x, continous_rank=continous_rank)       
    # Rescale to be within [0,1].
    x = (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))        
    return x

def get_feature_type(x):
    """ Tell if a feature is continuous or discrete.
    """
    if np.unique(x).shape[0]<75:
        return 'discrete'
    else: 
        return 'continuous'

def get_order_discrete(p, x, x_val, n_full=None):
    """ Calculate the order of the discrete features according to the alt/null ratio
    Args:
        p ((n,) ndarray): The p-values.
        x ((n,) ndarray): The covaraites. The data is assumed to have been preprocessed.
        x_val ((n_val,) ndarray): All possible values for x, sorted in ascending order.
        n_full (int): Total number of hypotheses before filtering.    
        
    Returns:
        x_order ((d,) ndarray): the order (of x_val) from smallest alt/null ratio to 
            the largest.
    """
    n_val = x_val.shape[0]
    # Separate the null and the alt proportion.
    _, t_BH = bh_test(p, alpha=0.1, n_full=n_full)
    x_null, x_alt = x[p>0.75], x[p<t_BH]      
    # Calculate the alt/null ratio。
    cts_null = np.zeros([n_val], dtype=int)
    cts_alt  = np.zeros([n_val], dtype=int)
    for i,val in enumerate(x_val):
        cts_null[i] = np.sum(x_null==val)+1
        cts_alt[i] = np.sum(x_alt==val)+1
    p_null  = cts_null/np.sum(cts_null)
    p_alt   = cts_alt/np.sum(cts_alt)      
    p_ratio = p_alt/p_null        
    # Calculate the order of x_val based on the ratio.
    x_order = p_ratio.argsort()
    return x_order

def reorder_discrete(x, x_val, x_order):
    """ Reorder the covariates based on the new order.
    Args:
        x ((n,) ndarray): The covaraites.
        x_val ((n_val,) ndarray): All possible values for x, sorted in ascending order.
        x_order ((n_val,) ndarray): The new order of x_val.
        
    Returns:
        x_new ((d,) ndarray): the reordered covaraites.
    """
    x_new = np.copy(x)
    n_val = x_val.shape[0]
    for i in range(n_val):
        x_new[x == x_val[x_order[i]]] = x_val[i]
    return x_new

def adafdr_explore(p, x_input, alpha=0.1, n_full=None, vis_dim=None,\
                           cate_name={}, output_folder=None,\
                           title_list=None, h=None, figsize=[5,3]):
    """Provide a visualization of pi1/pi0 for each dimension,
    to visualize the amount of information carried in each dimension.
    
        Args:
        p ((n,) ndarray): the p-values.
        x_input ((n,d) ndarray): The covaraites. The discrete features should be coded by
            integers starting from 0.
        alpha (float): The nominal FDR level.
        n_full (int): Total number of hypotheses before filtering.
        vis_dim (list or ndarray): the dimensions to visualize
        cate_name (dic of dics): the names of discrete categories for each dimension.
            None when the category names are not available. An example is as follows:
            cate_name = {1: {0: 'name0', 1: 'name1'}}. Here, dimension 0 has
            no names. Dimension 1 is discrete and has two values 0,1, whose corresponding
            names are name0, name1
        output_folder (string): The output directory.
        title_list (list of strings): Titles for each figure. 
        h ((n,) ndarray): The ground truth (None if not available). 
    
        Returns:
    """
    def plot_feature_1d(x_margin, p, x_null, x_alt, meta_info,\
                        title='',cate_name=None, output_folder=None,\
                        h=None, figsize=[5,3]):
        # Some parameters for generating the figures.
        feature_type,cate_order,x_val_ = meta_info       
        x_min,x_max = np.percentile(x_margin, [1,99])
        x_range = x_max-x_min
        x_min -= 0.01*x_range
        x_max += 0.01*x_range
        bins = np.linspace(x_min,x_max, 51)
        bin_width = bins[1]-bins[0]
        # Ploting for the continuous case and the discrete case.
        if feature_type == 'continuous':         
            # Continuous: use kde to estimate the probability.
            n_bin = bins.shape[0]-1
            x_grid = (bins+bin_width/2)[0:-1]
            p_null,_ = np.histogram(x_null, bins=bins) 
            p_alt,_ = np.histogram(x_alt, bins=bins)   
            p_null = p_null+1
            p_alt = p_alt+1
            p_null = p_null/np.sum(p_null)*n_bin
            p_alt = p_alt/np.sum(p_alt)*n_bin            
            kde_null = stats.gaussian_kde(x_null).evaluate(x_grid)
            kde_alt = stats.gaussian_kde(x_alt).evaluate(x_grid)
            psuedo_density = np.min(kde_null[kde_null>0])/10
            psuedo_density = psuedo_density.clip(min=1e-20)
            p_ratio = (kde_alt+psuedo_density)/(kde_null+psuedo_density)
            
        else: 
            # Discrete: use the empirical counts.
            unique_null,cts_null = np.unique(x_null, return_counts=True)
            unique_alt,cts_alt = np.unique(x_alt, return_counts=True)            
            unique_val = np.array(list(set(list(unique_null)+list(unique_alt))))
            unique_val = np.sort(unique_val)            
            p_null,p_alt = np.zeros([unique_val.shape[0]]), np.zeros([unique_val.shape[0]])          
            for i,key in enumerate(unique_null): 
                p_null[unique_val==key] = cts_null[i]                
            for i,key in enumerate(unique_alt): 
                p_alt[unique_val==key] = cts_alt[i]           
            n_bin = unique_val.shape[0]           
            p_null = (p_null+1)/np.sum(p_null+1)*n_bin
            p_alt = (p_alt+1)/np.sum(p_alt+1)*n_bin            
            p_ratio = p_alt/p_null  
            x_grid = (np.arange(unique_val.shape[0])+1)/(unique_val.shape[0]+1)
            x_min,x_max,bin_width = 0,1,1/(unique_val.shape[0]+1)
            if cate_name is None: 
                cate_name_ = x_val_[cate_order]
            else:
                cate_name_ = []  
                for i in cate_order:
                    cate_name_.append(cate_name[x_val_[i]])
        # Generate the figure.
        rnd_idx=np.random.permutation(p.shape[0])[0:np.min([10000, p.shape[0]])]
        if feature_type == 'continuous':
            pts_jitter = 0*(np.random.rand(np.min([10000, p.shape[0]]))-0.5)
        else:
            pts_jitter = 0.75*(np.random.rand(np.min([10000, p.shape[0]]))-0.5)
        plt.figure(figsize=figsize)
        p = p[rnd_idx]
        x_margin = x_margin[rnd_idx]
        # Ground truth.
        if h is not None:
            h = h[rnd_idx]
            plt.scatter(x_margin[h==1]+pts_jitter[h==1],\
                        p[h==1], color='orange', alpha=0.3, s=4, label='alt')
            plt.scatter(x_margin[h==0]+pts_jitter[h==0],\
                        p[h==0], color='steelblue', alpha=0.3, s=4, label='null')
        else:
            plt.scatter(x_margin+pts_jitter,\
                        p, color='steelblue', alpha=0.3, s=4, label='alt')
        if x_max>1e4:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ylim([0, min((p[p<0.5].max()), 0.05)])
        plt.ylabel('p-value', fontsize=14)
        plt.xlabel('covariate $x$', fontsize=14)
        if feature_type == 'discrete':
            plt.xticks(x_val_, cate_name_, rotation=90, fontsize=12)
        plt.tight_layout()
        if output_folder is not None:
            plt.savefig(output_folder+'/explore_p_%s.png'%title)
            plt.savefig(output_folder+'/explore_p_%s.pdf'%title)
            plt.close()
        else:
            plt.show()
        plt.figure(figsize=figsize)
        plt.bar(x_grid, p_null, width=bin_width, color='steelblue', alpha=0.6, label='null')
        plt.bar(x_grid, p_alt, width=bin_width, color='orange', alpha=0.6, label='alt')
        plt.xlim([x_min, x_max])
        if feature_type=='discrete': 
            plt.xticks(x_grid, cate_name_, rotation=90, fontsize=12)
        elif x_max>1e4:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ylabel('null/alt proportion', fontsize=14)
        plt.legend(fontsize=12)
        # plt.subplot(313)
        # if feature_type == 'continuous':
        #     plt.plot(x_grid, p_ratio, color='seagreen', label='ratio', linewidth=4) 
        # else:
        #     plt.plot(x_grid, p_ratio, color='seagreen', marker='o', label='ratio', linewidth=4)
        #     plt.xticks(x_grid, cate_name_, rotation=45)
        # plt.xlim([x_min, x_max])
        # y_min,y_max = plt.ylim()
        # plt.ylim([0, y_max])
        # plt.ylabel('ratio')
        plt.xlabel('covariate $x$', fontsize=14)
        plt.tight_layout()
        if output_folder is not None:
            plt.savefig(output_folder+'/explore_ratio_%s.png'%title)
            plt.savefig(output_folder+'/explore_ratio_%s.pdf'%title)
            plt.close()
        else:
            plt.show()
    # Start of the main function. Make a copy of the data.
    np.random.seed(0)
    x = x_input.copy()
    if n_full is None: 
        n_full = p.shape[0]
    if len(x.shape) == 1: 
        x = x.reshape([-1, 1])
    _,d = x.shape   
    # Reorder the discrete features.
    meta_info = []        
    for i in range(d):
        feature_type = get_feature_type(x[:, i])
        if feature_type == 'discrete':
            x_val = np.sort(np.unique(x[:, i]))
            x_order = get_order_discrete(p, x[:, i], x_val, n_full=n_full)
            x[:, i] = reorder_discrete(x[:, i], x_val, x_order)
            meta_info.append([feature_type, x_order, x_val])
        else:
            meta_info.append([feature_type, None, None])    
    # Separate the null proportion and the alternative proportion.
    _,t_BH = bh_test(p, n_full=n_full, alpha=0.1)
    x_null,x_alt = x[p>0.75],x[p<t_BH]      
    # Generate figures.
    if vis_dim is None: 
        vis_dim = np.arange(min(5, d))    
    for i in vis_dim:
        x_margin = x[:, i]
        temp_null,temp_alt = x_null[:, i], x_alt[:, i]
        if i in cate_name.keys():
            temp_cate_name = cate_name[i]
        else: 
            temp_cate_name = None
        if title_list is not None:
            temp_title = title_list[i]
        else:
            temp_title = 'feature_%s'%str(i+1)
        plot_feature_1d(x_margin, p, temp_null, temp_alt,\
                        meta_info[i], title=temp_title,\
                        cate_name=temp_cate_name, output_folder=output_folder,\
                        h=h, figsize=figsize)            

def preprocess_two_fold(p1, x1, x2, n_full, f_write):
    """Data preprocessing two folds of data. Note that to prevent overfitting,
    the preprocessing procedure does not depend on the p-values from the second fold.
    
    Args:
        p1 ((n1,) ndarray): the p-value from the first fold.
        x1 ((n1,d) ndarray): the covariates from the first fold.
        x2 ((n1,d) ndarray): the covariates from the second fold.
        n_full (int): Total number of hypotheses before filtering.
    
    Returns:
        x1 ((n1,d) ndarray): the preprocessed covariates from the first fold.
        x2 ((n1,d) ndarray): the preprocessed covariates from the second fold.
    """
    if len(x1.shape) == 1: 
        x1 = x1.reshape([-1,1])
        x2 = x2.reshape([-1,1])
    n1,d = x1.shape
    n2,d = x2.shape
    n = n1+n2
    x1_new = x1.copy()
    x2_new = x2.copy()
    # Rearrange the discrete features.
    for i in range(d):
        temp_x = np.concatenate([x1[:, i], x2[:, i]])
        feature_type = get_feature_type(temp_x)
        if feature_type == 'discrete':
            x_val = np.sort(np.unique(temp_x))           
            x_order = get_order_discrete(p1, x1[:, i], x_val, n_full=n_full)
            x1_new[:, i] = reorder_discrete(x1[:, i], x_val, x_order)
            x2_new[:, i] = reorder_discrete(x2[:, i], x_val, x_order) 
            if f_write is not None:
                f_write.write('# dim %d, x_order=%s\n'%(i, x_order))
    # Rescale the data to [0,1] and perform quantile normalization.
    x = np.zeros([n, d])
    rand_idx = np.random.permutation(n1+n2)    
    x[rand_idx[0:n1], :] = x1_new
    x[rand_idx[n1:], :] = x2_new  
    x = feature_preprocess(x, qt_norm=True)
    x1_new = x[rand_idx[0:n1], :]
    x2_new = x[rand_idx[n1:], :]
    return x1_new, x2_new
    
def method_single_fold_wrapper(data):
    """ A wrapper for method_single_fold to be called by method_cv
    
    Args:
        data (list): data[0]: p1, x1, 
            data[1]: p2, x2,
            data[2]: K,alpha,n_full,n_itr,output_folder,random_state,
            data[3]: Fold number.
        
    Returns:
        n_rej (int): Number of rejected hypotheses.
        t2 ((n/2,) ndarray): The thresholds on the second fold of data.
        [a,b,w,mu,sigma,gamma]: Learned (reparametrized) parameters.
    """
    # Load the data.
    p1,x1 = data[0]
    p2,x2 = data[1]
    K, alpha, n_full,n_itr, output_folder, random_state, verbose, fast_mode = data[2]
    fold_number = data[3]
    # Start a write file.
    if output_folder is not None:
        fname = output_folder+'/record_fold_%d.txt'%fold_number
        f_write = open(fname,'w+')
        f_write.write('### record for fold_%d\n'%fold_number)
        f_write.write('# K=%d, alpha=%0.2f, n_full=%d, n_itr=%d, random_state=%d\n'\
                      %(K,alpha,n_full,n_itr,random_state))
    else:
        f_write = None
    np.random.seed(random_state+fold_number)
    # Preprocess the data.
    x1, x2 = preprocess_two_fold(p1, x1, x2, n_full, f_write)
    # Learn the threshold
    _,_,theta = method_single_fold(p1, x1, K=K, alpha=alpha, n_full=n_full, n_itr=n_itr,\
                                   verbose=verbose, output_folder=output_folder,\
                                   fold_number=fold_number, random_state=random_state,\
                                   f_write=f_write, fast_mode=fast_mode)
    a,b,w,mu,sigma,gamma = theta
    # Test on the second fold
    t2 = t_cal(x2,a,b,w,mu,sigma)
    gamma = rescale_mirror(t2,p2,alpha,f_write=f_write,title='cv',n_full=n_full)
    t2 = gamma*t2
    if f_write is not None:
        f_write.write('\n## Test result with method_cv fold_%d\n'%fold_number)
        result_summary(p2<t2, f_write=f_write, title='method_single_fold_wrapper_%d'%fold_number)
        f_write.close()
    return np.sum(p2<t2), t2, [a,b,w,mu,sigma,gamma]

def adafdr_test(p_input, x_input, K=5, alpha=0.1, n_full=None, n_itr=1500, qt_norm=True,\
                h=None, verbose=False, output_folder=None, random_state=0,\
                single_core=True, fast_mode=True):
    """Hypothesis testing with hypothesis splitting.

    Args:
        p_input ((n,) ndarray): The p-values.
        x_input ((n,d) ndarray): The covaraites. The discrete features should be coded by integers
            starting from 0.
        K (int): The number of bump components.
        alpha (float): The nominal FDR level.
        n_full (int): Total number of hypotheses before filtering.
        n_itr (int): The number of iterations for the optimization process.
        qt_norm (bool): If perform quantile normalization.
        h ((n,) ndarray): The ground truth (None if not available).        
        verbose (bool): Indicate if output the computation details.
        output_folder (string): The output directory.
        random_state (int): The random seed.
        single_core (bool): True: use single core. False: process two processes in parallel.
        fast_mode (bool): If True, go without optimization.
        
    Returns:
        n_rej (list): Number of rejected hypothesesfor the two folds.
        t ((n,) ndarray): The decision threshold.
        theta (list): Learned (reparametrized) parameters with the format [a,b,w,mu,sigma,gamma].
    """
    np.random.seed(random_state)    
    p = np.copy(p_input)
    x = np.copy(x_input)
    if fast_mode:
        single_core = False
    if n_full is None:
        n_full = p.shape[0]
    start_time=time.time()
    if len(x.shape) == 1: 
        x = x.reshape([-1,1])
    n_sample,d = x.shape
    n_sub = int(n_sample/2)
    # Randomly split the hypotheses into two folds.    
    rand_idx = np.random.permutation(n_sample)    
    fold_idx = np.zeros([n_sample],dtype=int)
    fold_idx[rand_idx[0:n_sub]] = 0
    fold_idx[rand_idx[n_sub:]] = 1
    # Construct the input data.
    args = [K, alpha, int(n_full/2), n_itr, output_folder, random_state, verbose, fast_mode]
    data_fold_1 = [p[fold_idx==0], x[fold_idx==0]]
    data_fold_2 = [p[fold_idx==1], x[fold_idx==1]]
    Y_input = []
    Y_input.append([data_fold_2, data_fold_1, args, 1])
    Y_input.append([data_fold_1, data_fold_2, args, 2])
    # Parallel processing the testing of the two folds.
    if single_core:
        res = list()
        res.append(method_single_fold_wrapper(Y_input[0]))
        res.append(method_single_fold_wrapper(Y_input[1]))
    else:
        po = mp.Pool(2)
        res = po.map(method_single_fold_wrapper, Y_input)
        po.close()
    # Summarize the result.
    n_rej = [res[0][0], res[1][0]]
    t = np.zeros([n_sample], dtype=float)
    t[fold_idx==0] = res[0][1]
    t[fold_idx==1] = res[1][1]
    theta = [res[0][2], res[1][2]]    
    if verbose:
        print('# total rejection: %d'%np.array(n_rej).sum(), n_rej)
        if h is not None:          
            print('# D=%d, FD=%d, FDP=%0.3f'\
                  %(np.sum(p<t), np.sum((p<t)*(h==0)),\
                    np.sum((p<t)*(h==0))/np.sum(p<t)))        
        # Visualize the learned threshold.
        if output_folder is not None:
            color_list = ['steelblue', 'orange']            
            n_figure = min(d,5)
            plt.figure(figsize=[8, 4+n_figure])
            for i_dim in range(n_figure):
                plt.subplot(str(n_figure)+'1'+str(i_dim+1))
                for i in range(2):
                    if h is not None:
                        plot_scatter_t(t[fold_idx==i], p[fold_idx==i], x[fold_idx==i, i_dim],\
                                       h[fold_idx==i], color=color_list[i], label='fold %d'%(i+1))
                    else:
                        plot_scatter_t(t[fold_idx==i], p[fold_idx==i], x[fold_idx==i, i_dim],\
                                       color=color_list[i], label='fold %d'%(i+1))
                plt.legend()
                plt.xlabel('x_%d'%(i_dim+1))
                plt.ylabel('p-value')
            plt.savefig(output_folder+'/learned_threshold.png')
            plt.close()
        print('#time total: %0.4fs'%(time.time()-start_time))
    res_adafdr = {'n_rej':n_rej, 'decision':p_input<=t, 'threshold':t, 'theta':theta}
    return res_adafdr
    # return (p_input<=t), t, theta

def method_single_fold(p_input, x_input, K=5, alpha=0.1, n_full=None,\
                       n_itr=1500, h=None, verbose=False,\
                       output_folder=None, fold_number=0, random_state=0, f_write=None,\
                       fast_mode=False):
    """Learn the decision threshold via optimization.

    Args:
        p_input ((n,) ndarray): The p-values.
        x_input ((n,d) ndarray): The covaraites. The data is assumed to have been preprocessed.
        K (int): The number of bump components.
        alpha (float): The nominal FDR level.
        n_full (int): Total number of hypotheses before filtering.
        n_itr (int): The number of iterations for the optimization process.
        h ((n,) ndarray): The ground truth (None if not available).        
        verbose (bool): Indicate if output the computation details.
        output_folder (string): The output directory.
        random_state (int): The random seed.
        fold_number (int(0,1)): The fold number.
        f_write (file handler (write mode)): The output file.
        fast_mode (bool): If True, go without optimization.
        
    Returns:
        n_rej (int): Number of rejected hypotheses.
        t ((n,) ndarray): The decision threshold.
        theta (list): Learned (reparametrized) parameters with the format [a,b,w,mu,sigma,gamma].
    """
    torch.manual_seed(random_state)
    p = np.copy(p_input)
    x = np.copy(x_input)
    if len(x.shape)==1:
        x = x.reshape([-1,1])
    d = x.shape[1]
    if f_write is not None:
        f_write.write('# n_sample=%d\n'%(x.shape[0]))
        for i in [1e-6,1e-5,1e-4,5e-4]:
            f_write.write('# p<%0.6f: %d\n'%(i,np.sum(p<i)))
        f_write.write('\n')    
    # Threshold initialization using method_init.
    a_init, mu_init, sigma_init, w_init = method_init(p, x, K, alpha=alpha,
                                                      n_full=n_full, verbose=verbose,\
                                                      output_folder=output_folder,\
                                                      random_state=random_state,\
                                                      fold_number=fold_number,\
                                                      f_write=f_write)
    # Reparametrization.
    a,b,w,mu,sigma = reparametrize(a_init, mu_init, sigma_init, w_init, d)
    # Rescale using mirror estiamtor. 
    t = t_cal(x,a,b,w,mu,sigma)    
    gamma = rescale_mirror(t,p,alpha,f_write=f_write,title='before optimization',n_full=n_full)
    t = gamma*t
    b,w = b+np.log(gamma),w+np.log(gamma)
    # Make a copy of adafdr-fast result
    res_fast = (np.sum(p < t), t.copy(), [a,b,w,mu,sigma,gamma])
    # Return the result without optimization.
    if fast_mode or (np.sum(p<t)<100) or (np.sum(p>1-t)<20):
        if f_write is not None:
            temp_str = 'Too few discoveries for optimization: ' +\
                       'D=%d, FD_hat=%d, '%(np.sum(p<t), np.sum(p>1-t)) +\
                       'Exit with fast mode result.\n'
            f_write.write(temp_str)
        return res_fast
    # Setting parameters for the optimization.
    # lambda0: adaptively set based on the approximation accuracy of the sigmoid function.
    lambda0,n_rej,n_fd = 1/t.mean(),max(np.sum(p<t),1),np.sum(p>1-t)
    if f_write is not None:
        f_write.write('\n## choosing lambda0\n')
        f_write.write('## lambda0=%0.1f, D=%d, FD_hat=%d, alpha_hat=%0.3f\n'%(lambda0,n_rej,n_fd,n_fd/n_rej))        
    while np.absolute(np.sum(sigmoid((lambda0*(t-p))))-n_rej)>0.02*n_rej \
        or np.absolute(np.sum(sigmoid((lambda0*(t+p-1))))-n_fd)>0.02*n_fd:
        lambda0 = lambda0+0.5/t.mean()
        if f_write is not None:
            f_write.write('# lambda0=%0.1f, D_apr=%0.1f (r err=%0.3f), FD_hat_apr=%0.3f (r err=%0.3f)\n'\
                          %(lambda0,np.sum(sigmoid((lambda0*(t-p)))),np.absolute(1-np.sum(sigmoid((lambda0*(t-p))))/n_rej),\
                            np.sum(sigmoid((lambda0*(t+p-1)))),np.absolute(1-np.sum(sigmoid((lambda0*(t+p-1))))/n_fd)))   
    # Other parameters. 
    lambda1  = 10/alpha
    lambda0, lambda1 = int(lambda0), int(lambda1)
    loss_rec = np.zeros([n_itr],dtype=float)
    n_samp   = x.shape[0]
    if verbose:
        if f_write is not None:
            f_write.write('\n## optimization paramter:\n')
            f_write.write('# n_itr=%d, n_samp=%d, lambda0=%0.4f, lambda1=%0.4f\n'%(n_itr,n_samp,lambda0,lambda1))           
    # Initialization using pytorch.
    lambda0 = Variable(torch.Tensor([lambda0]),requires_grad=False)
    lambda1 = Variable(torch.Tensor([lambda1]),requires_grad=False)
    p       = Variable(torch.from_numpy(p).float(),requires_grad=False)
    x       = Variable(torch.from_numpy(x).float(),requires_grad=False)
    a       = Variable(torch.Tensor(a),requires_grad=True)
    b       = Variable(torch.Tensor([b]),requires_grad=True)
    w       = Variable(torch.Tensor(w),requires_grad=True)
    mu      = Variable(torch.Tensor(mu),requires_grad=True)
    # Reparametrize by sigma_mean.
    sigma_mean = np.mean(sigma,axis=0)
    sigma      = Variable(torch.Tensor(sigma/sigma_mean),requires_grad=True)
    sigma_mean = Variable(torch.from_numpy(sigma_mean).float(),requires_grad=False)    
    if verbose:
        if f_write is not None:
            f_write.write('\n## method_single_fold: initialization\n')
            f_write.write('# Slope: a=%s\n'%(a.data.numpy()))
            f_write.write('         b=%0.4f\n'%(b.data.numpy()))
            for k in range(K):
                f_write.write('# Bump %d: w=%0.4f\n'%(k,w.data.numpy()[k]))
                f_write.write('         mu=%s\n'%(mu.data.numpy()[k])) 
                f_write.write('      sigma=%s\n'%(sigma.data.numpy()[k])) 
            f_write.write('\n')
    # Specifying the optimizer.
    optimizer = torch.optim.Adam([a,b,w,mu,sigma],lr=0.02)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,300,600],gamma=0.5)     
    optimizer.zero_grad()
    # Optimization.
    for l in range(n_itr):        
        scheduler.step()
        # Calculate the model.
        optimizer.zero_grad()
        sigma = sigma.clamp(min=1e-4)
        t = torch.exp(torch.matmul(x,a)+b)
        for i in range(K):
            t = t+torch.exp(w[i]-torch.matmul((x-mu[i,:])**2,sigma[i,:] * sigma_mean))
        loss1 = -torch.mean(torch.sigmoid(lambda0*(t-p)))
        loss2 = lambda1*tf.relu(torch.mean(torch.sigmoid(lambda0*(t+p-1)))\
                                - alpha*torch.mean(torch.sigmoid(lambda0*(t-p))))        
        loss  = loss1+loss2
        # Back propogation
        loss.backward()
        optimizer.step()               
        # Record important quantities.
        loss_rec[l] = loss.data.numpy()        
        if verbose:
            if l%(int(n_itr)/5)==0:
                if f_write is not None:
                    f_write.write('\n## iteration %d: \n'%(l))    
                    f_write.write('# n_rej=%d, n_rej_apr=%0.1f, FD_hat=%d, FD_hat_apr=%0.1f\n'%\
                                  (np.sum(t.data.numpy()>p.data.numpy()),\
                                   np.sum(sigmoid(lambda0.data.numpy()*(t.data.numpy()-p.data.numpy()))),\
                                   np.sum(p.data.numpy()>1-t.data.numpy()),\
                                   np.sum(sigmoid(lambda0.data.numpy()*(p.data.numpy()-1+t.data.numpy())))))
                    f_write.write('# loss1=%0.4f, loss2=%0.4f, FDP=%0.4f, FDP_hat_apr=%0.4f\n'\
                                  %(loss1.data.numpy(),loss2.data.numpy(),\
                                  np.sum((h==0)*(p.data.numpy()<t.data.numpy()))/np.sum(p.data.numpy()<t.data.numpy()),\
                                  (torch.mean(torch.sigmoid(lambda0*(t+p-1)))\
                                  /torch.mean(torch.sigmoid(lambda0*(t-p)))).data.numpy()))
                    f_write.write('# Slope: a=%s\n'%(a.data.numpy()))
                    f_write.write('         b=%0.4f\n'%(b.data.numpy()))
                    for k in range(K):
                        f_write.write('# Bump %d: w=%0.4f\n'%(k,w.data.numpy()[k]))
                        f_write.write('         mu=%s\n'%(mu.data.numpy()[k])) 
                        f_write.write('      sigma=%s\n'%(sigma.data.numpy()[k]))
                if d==1:
                    if output_folder is not None:
                        plt.figure(figsize=[8,5])
                        plot_t(t.data.numpy(),p.data.numpy(),x.data.numpy(),h)
                        plt.savefig(output_folder+'/threshold_itr_%d_fold_%d.png'%(l,fold_number))
                        plt.close()
    if verbose:
        if output_folder is not None:
            plt.figure(figsize=[6,5])
            plt.plot(np.log(loss_rec-loss_rec.min()+1e-3))
            plt.savefig(output_folder+'/loss_fold_%d.png'%fold_number)
            plt.close()      
    # Convert the results.
    p     = p.data.numpy()
    x     = x.data.numpy()
    a     = a.data.numpy()
    b     = b.data.numpy()
    w     = w.data.numpy()
    mu    = mu.data.numpy()
    sigma = (sigma * sigma_mean).data.numpy()
    # Testing.
    t = t_cal(x,a,b,w,mu,sigma)
    gamma = rescale_mirror(t,p,alpha,f_write=f_write,title='after method_single_fold',
                           n_full=n_full)   
    t *= gamma
    n_rej=np.sum(p<t)     
    if verbose: 
        if f_write is not None:
            f_write.write('\n## Test result with method_single_fold\n')
        result_summary(p<t, h=h, f_write=f_write, title='method_single_fold_%d'%fold_number)
    theta = [a,b,w,mu,sigma,gamma]
    if n_rej < 0.9*res_fast[0]:
        if f_write is not None:
            f_write.write('# Optimization not yielding meaningful result. Exit with fast mode')
        return res_fast
    return n_rej, t, theta

def reparametrize(a_init, mu_init, sigma_init, w_init, d):
    """Learn the decision threshold via optimization.

    Args:
        a_init ((d,) ndarray): slope parameter.
        mu_init,sigma_init ((K,d) ndarray): bump parameter.
        w_init ((n,) ndarray): proportion of each component.
        d (int): The dimensionality.     
        
    Returns:
        Parameters used in optimizataion and t_cal.
        a ((d,) ndarray): slope parameter.
        mu,sigma ((K,d) ndarray): bump parameter.
        w ((n,) ndarray): proportion of each component.
    """
    
    a  = a_init
    b  = np.log(w_init[0]).sum()\
        + np.log(a_init[a_init!=0]/(np.exp(a_init[a_init!=0])-1)).sum()    
    K = mu_init.shape[0]
    c = np.zeros([K, d], dtype=float)
    for k in range(K):
        for i in range(d):
            c[k, i] = sp.stats.norm.cdf(1,loc=mu_init[k,i],scale=sigma_init[k,i])\
                - sp.stats.norm.cdf(0,loc=mu_init[k,i],scale=sigma_init[k,i])
    w_init = (w_init + 1e-8)/np.sum(w_init + 1e-8)
    c = c.clip(min=0.1)
    w = np.log(w_init[1:])\
        - d/2*np.log(2*np.pi)\
        - np.log(sigma_init).sum(axis=1)\
        - np.log(c).sum(axis=1)
    mu = mu_init 
    sigma_init = sigma_init.clip(min=1e-3)
    sigma = 1/(2*sigma_init**2)   
    return a,b,w,mu,sigma

def t_cal(x,a,b,w,mu,sigma):
    """Calculate the threshold based on the learned mixture: trend+bump

    Args:
        x ((n,d) ndarray): The covaraites.
        a ((d,) ndarray): reparametrized slope parameter.
        mu,sigma ((K,d) ndarray): reparametrized bump parameter.
        w ((n,) ndarray): reparametrized proportion of each component.
        
    Returns:
        t ((n,) ndarray): threshold for each hypothesis.
    """
    if len(x.shape)==1:
        x = x.reshape([-1, 1])
    t = np.exp(x.dot(a)+b)
    for i in range(mu.shape[0]):
        t += np.exp(w[i])*np.exp(-np.sum((x-mu[i])**2*sigma[i],axis=1))
    return t

def rescale_mirror(t,p,alpha,f_write=None,title='',n_full=None):
    """Rescale to have the mirror estimate below level alpha

    Args:
        t ((n,) ndarray): The threshold for each hypothesis.
        p ((n,) ndarray): The p-value for each hypothesis.
        alpha (float): The nominal FDR level.
        f_write (file handle): The file to write on.
        title (String): A description of the task. 
        
    Returns:
        (float): the rescale factor.
    """
    def get_FD_hat(p_, t_, n_full=None):
        """ mirror estimate: when the estimated values are two small,
            use the estimates from BH
        """
        if n_full is None:
            n_full = p_.shape[0]
        FD_hat = np.sum(p_ > 1-t_)
        FD_hat_bh = np.mean(t_)*n_full
        # print('FD_hat=%d, FD_hat_bh=%0.3f, ratio=%0.3f'%
        #       (FD_hat, FD_hat_bh, FD_hat/FD_hat_bh))
        if FD_hat >= 5:
            return min(FD_hat, FD_hat_bh)
            # return FD_hat
        else:
            return min(5, FD_hat_bh)
        
    if f_write is not None:
        f_write.write('\n## rescale_mirror: %s\n'%title)    
    # Rescale t to a sensible region.
    t_999 = np.percentile(t,99.9)
    if t.clip(max=t_999).mean()>0.2:
        gamma_pre = 0.2/t.clip(max=t_999).mean()
    else: 
        gamma_pre = 1
    t = t*gamma_pre
    if f_write is not None:
        f_write.write('# quantile of t (1,25,75,99): %s\n'%(np.percentile(t,[1,25,75,99])))
        f_write.write('# gamma_pre=%0.4f\n'%gamma_pre)
    # Grid search.
    gamma_grid = np.linspace(0, 0.2/np.mean(t), 50)[1:]
    alpha_hat = np.zeros([gamma_grid.shape[0]], dtype=float)
    for i in range(gamma_grid.shape[0]):
        # alpha_hat[i] = max(1, np.sum(p>1-t*gamma_grid[i]))/max(1, np.sum(p<t*gamma_grid[i]))
        alpha_hat[i] = get_FD_hat(p, t*gamma_grid[i], n_full=n_full)/\
                                  max(1, np.sum(p<t*gamma_grid[i]))
    if np.sum(alpha_hat<alpha) > 0:
        gamma_l = np.max(gamma_grid[alpha_hat<alpha])
    else:
        gamma_l = 0
    if np.sum(alpha_hat>alpha) > 0:
        gamma_u = np.min(gamma_grid[alpha_hat>alpha])
    else:
        gamma_u = gamma_grid[-1]
    gamma_l = min(gamma_l, gamma_u-gamma_grid[1]+gamma_grid[0])
    # Binary search.
    gamma_m = (gamma_u+gamma_l)/2    
    # while (gamma_u-gamma_l>1e-2) or (max(1, np.sum(p>1-t*gamma_m))/max(1, np.sum(p<t*gamma_m)) > alpha):
    while (gamma_u-gamma_l>1e-2) or (get_FD_hat(p, t*gamma_m, n_full=n_full)/\
                                     max(1, np.sum(p<t*gamma_m)) > alpha):
        gamma_m = (gamma_l+gamma_u)/2
        D_hat = max(1, np.sum(p<t*gamma_m))
        # FD_hat = max(1, np.sum(p>1-t*gamma_m))
        FD_hat = get_FD_hat(p, t*gamma_m, n_full=n_full)
        alpha_hat = FD_hat/D_hat
        if f_write is not None:
            f_str = '# gamma_l=%0.4f, gamma_u=%0.4f, D_hat=%d, FD_hat=%0.2f, alpha_hat=%0.4f\n'%\
                    (gamma_l,gamma_u,D_hat,FD_hat,alpha_hat)
            f_write.write(f_str)
        if alpha_hat < alpha:
            gamma_l = gamma_m
        else: 
            gamma_u = gamma_m
        if (D_hat==1) and (FD_hat==1):
            gamma_u = 0
            gamma_l = 0
            break
    if f_write is not None:
        f_write.write('# final output: gamma=%0.4f\n'%((gamma_u+gamma_l)/2*gamma_pre))
    return max((gamma_u+gamma_l)/2*gamma_pre, 1e-20) 

def method_init(p_input, x_input, K, alpha=0.1, n_full=None, h=None, verbose=False,
                output_folder=None, random_state=0, fold_number=0, f_write=None):
    """Initialization for method_single_fold that fits a mixture model with bump+slope.

    Args:
        p_input ((n,) ndarray): The p-values
        x_input ((n,d) ndarray): The covaraites.
        K (int): The number of bump components.
        alpha (float): The nominal FDR level.
        n_full (int): Total number of hypotheses before filtering.
        h ((n,) ndarray): The ground truth (None if not available).
        verbose (bool): Indicate if output the computation details.
        output_folder (string): The output directory.
        random_state (int): The random seed.
        fold_number (int(0,1)): The fold number.
        f_write (file handler (write mode)): The output file.        
        
    Returns:
        a ((d,) ndarray): slope parameter.
        mu,sigma ((k,d) ndarray): bump parameter.
        w ((n,) ndarray): proportion of each component.
    """
    np.random.seed(random_state)
    p = np.copy(p_input)
    x = np.copy(x_input)
    if f_write is not None:
        f_write.write('## method_init starts\n')        
    if len(x.shape)==1: 
        x = x.reshape([-1,1])
    n_samp,d = x.shape
    if n_full is None:
        n_full = n_samp            
    # Extract the null and the alternative proportion.
    _,t_BH = bh_test(p,n_full=n_full,alpha=0.1)
    if np.sum(p<t_BH) > 100:
        x_null,x_alt = x[p>0.75],x[p<t_BH]
    else:
        x_null,x_alt = x[p>0.75],x[p<0.05]
    if f_write is not None:
        f_write.write('# t_BH=%0.6f, n_null=%d, n_alt=%d\n'%(t_BH, x_null.shape[0],x_alt.shape[0]))      
    # Fit the null distribution.
    if f_write is not None:
        f_write.write('## Learning null distribution\n')            
    a_null,mu_null,sigma_null,w_null = mixture_fit(x_null,K,verbose=verbose,\
                                                   random_state=random_state,f_write=f_write,\
                                                   output_folder=output_folder,\
                                                   suffix='_null',fold_number=fold_number)   
    # Fit the alternative distribution (weighted by null)
    x_w = 1/(f_all(x_alt,a_null,mu_null,sigma_null,w_null)+1e-5)
    x_w /= np.mean(x_w)
    if f_write is not None:
        f_write.write('## Learning alternative distribution\n')  
    a,mu,sigma,w = mixture_fit(x_alt,K,x_w=x_w,verbose=verbose,\
                               random_state=random_state,f_write=f_write,\
                               output_folder=output_folder,\
                               suffix='_alt',fold_number=fold_number)    
    
    if verbose:        
        t = f_all(x,a,mu,sigma,w)
        gamma = rescale_mirror(t,p,alpha,n_full=n_full)   
        t = t*gamma
        if f_write is not None:
            f_write.write('\n## Test result with method_init\n')        
        result_summary(p<t,h,f_write=f_write, title='method_init_%d'%fold_number)  
        # Plot the fitted results
        if output_folder is not None:
            if d==1:
                plt.figure(figsize=[8,5])
            else:
                plt.figure(figsize=[8,12])
            plot_t(t,p,x,h)        
            plt.tight_layout()
            plt.savefig(output_folder+'/threshold_after_method_init_fold_%d.png'%fold_number)
        else:
            plt.show()
        if f_write is not None:
            f_write.write('## method_init finished\n')
    return a,mu,sigma,w    

def mixture_fit(x,K=3,x_w=None,n_itr=100,verbose=False,random_state=0,f_write=None,\
                output_folder=None,suffix=None,fold_number=0): 
    """Fit a slope+bump mixture using EM algorithm.

    Args:
        x ((n,d) ndarray): The covaraites.
        K (int): The number of bump components.
        x_w ((n,) ndarray): The weights for each sample.
        n_itr (int): The maximum number of iterations for the EM algorithm
        verbose (bool): Indicate if output the computation details.
        random_state (int): The random seed.
        f_write (file handler (write mode)): The output file.
        output_folder (string): The output directory.
        suffix (string): The suffix of the output file.
        fold_number (int(0,1)): The fold number.

    Returns:
        a ((d,) ndarray): slope parameter.
        mu,sigma ((k,d) ndarray): bump parameter.
        w ((n,) ndarray): proportion of each component. 
    """
 
    np.random.seed(random_state)
    if len(x.shape)==1: 
        x = x.reshape([-1,1])
    n_samp,d = x.shape
    if x_w is None: 
        x_w=np.ones([n_samp],dtype=float)        
    # Initialization
    GMM      = GaussianMixture(n_components=K,covariance_type='diag').fit(x)
    w_old    = np.zeros([K+1])
    w        = 0.5*np.ones([K+1])/K
    w[0]     = 0.5
    a        = ML_slope(x,x_w)   
    mu,sigma = GMM.means_,GMM.covariances_**0.5
    w_samp   = np.zeros([K+1,n_samp],dtype=float)
    i        = 0
    # Print the initialization information
    if verbose:
        if f_write is not None:
            f_write.write('## mixture_fit: initialization parameters\n')
            f_write.write('# Slope: w=%0.4f, a=%s\n'%(w[0],a))
            for k in range(K):
                f_write.write('# Bump %d: w=%0.4f\n'%(k,w[k+1]))
                f_write.write('         mu=%s\n'%(mu[k])) 
                f_write.write('      sigma=%s\n'%(sigma[k])) 
            f_write.write('\n')
    # EM algorithm        
    while np.linalg.norm(w-w_old,1)>5e-3 and i<n_itr:              
        # E step       
        w_old = w
        w_samp[0,:] = w[0]*f_slope(x,a)
        for k in range(K):
            w_samp[k+1,:] = w[k+1]*f_bump(x,mu[k],sigma[k])
        w_samp = w_samp/np.sum(w_samp,axis=0)*x_w             
        # M step
        w = np.mean(w_samp,axis=1) 
        a = ML_slope(x,w_samp[0,:])
        for k in range(K):
            if w[k+1]>1e-4: mu[k],sigma[k]=ML_bump(x,w_samp[k+1,:])               
        sigma = sigma.clip(min=1e-4)
        w[w<1e-3] = 0
        w /= w.sum()
        i += 1        
    # Convergence warning       
    if i >= n_itr and verbose: 
        print('Warning: the model does not converge, w_dif=%0.4f'%np.linalg.norm(w-w_old,1))        
        if f_write is not None:
            f_write.write('Warning: the model does not converge, w_dif=%0.4f\n'%np.linalg.norm(w-w_old,1))
    # Output 
    if verbose and f_write is not None:
        f_write.write('## mixture_fit: learned parameters\n')
        f_write.write('# Slope: w=%0.4f, a=%s\n'%(w[0],a))
        for k in range(K):
            f_write.write('# Bump %d: w=%0.4f\n'%(k,w[k+1]))
            f_write.write('         mu=%s\n'%(mu[k])) 
            f_write.write('      sigma=%s\n'%(sigma[k])) 
        f_write.write('\n')        
    if output_folder is not None:
        bins_ = np.linspace(0,1,101)
        x_grid = bins_.reshape([-1,1])       
        if d==1:
            plt.figure(figsize=[8,5])
            plt.hist(x,bins=bins_,weights=x_w/np.sum(x_w)*100) 
            temp_p = f_all(x_grid,a,mu,sigma,w)      
            plt.plot(bins_,temp_p)
            plt.savefig(output_folder+'/projection%s_fold_%d.png'%(suffix,fold_number))        
        else:
            plt.figure(figsize=[8,12])
            n_figure = min(d, 5)
            for i_dim in range(n_figure):        
                plt.subplot(str(n_figure)+'1'+str(i_dim+1))
                plt.hist(x[:,i_dim],bins=bins_,weights=x_w/np.sum(x_w)*100)  
                temp_p = f_all(x_grid,a[[i_dim]],mu[:,[i_dim]],sigma[:,[i_dim]],w)  
                plt.plot(bins_,temp_p)
                plt.title('Dimension %d'%(i_dim+1))
            plt.savefig(output_folder+'/projection%s_fold_%d.png'%(suffix,fold_number))
        plt.close('all')    
    return a,mu,sigma,w

"""
    sub-routines for mixture_fit
    input:  x: (n,d) ndarray (not (n,))
            w: (n,) ndarray
            a: (d,) ndarray
            mu,sigma: (k,d) ndarray
"""
def ML_slope(x,v=None,c=0.005):  
    """
    ML fit of the slope a/(e^a-1) e^(ax), defined over [0,1]

    Parameters
    ----------
    x : (n,d) ndarray
        covaraites
    v : (n,) ndarray 
        weight for each sample
    c : float
        regularization factor 

    Returns
    -------
    (d,) ndarray
        the estiamted slope parameter
    """    
    n,d = x.shape
    a = np.zeros(d,dtype=float)
    if v is None:
        t = np.mean(x,axis=0) 
    else:
        t = np.sum((x.T*v).T,axis=0)/np.sum(v) # weighted sum along each dimension
    
    a_u=10*np.ones([d],dtype=float)
    a_l=-10*np.ones([d],dtype=float)
    # binary search 
    while np.linalg.norm(a_u-a_l)>0.01:      
        a_m  = (a_u+a_l)/2
        a_m += 1e-2*(a_m==0)
        temp = (np.exp(a_m)/(np.exp(a_m)-1) - 1/a_m + 2*c*a_m)
        a_l[temp<t] = a_m[temp<t]
        a_u[temp>=t] = a_m[temp>=t]
    return (a_u+a_l)/2
    
def f_slope(x,a):
    """
    density of the slope function

    Parameters
    ----------
    x : (n,d) ndarray
        coML estimatearaites
    a : (d,) ndarray 
        slope parameter (for each dimension)

    Returns
    -------
    (n,) array
        the slope density for each point
    """
    f_x = np.exp(x.dot(a)) # dimension-wise probability
    norm_factor = np.prod(a[a!=0]/(np.exp(a[a!=0])-1))
    f_x = f_x * norm_factor # actual probability     

    return f_x

## Very hard to vectorize. So just leave it here
def ML_bump(x,v=None,logger=None):
    """
    ML fit of the bump function

    Parameters
    ----------
    x : (n,d) ndarray
        coML estimatearaites
    v : (n,) ndarray 
        weight for each sample

    Returns
    -------
    mu : (n,d) ndarray
        bump mean parameter (for each dimension)
    sigma : (n,d) ndarray 
        bump std parameter (for each dimension)
    """   
    def ML_bump_1d(x,v,logger=None):
        def fit_f(param,x,v):                
            mu,sigma = param
            inv_sigma = 1/sigma
            Z = sp.stats.norm.cdf(1,loc=mu,scale=sigma)-sp.stats.norm.cdf(0,loc=mu,scale=sigma)            
            inv_Z = 1/Z
            phi_alpha = 1/np.sqrt(2*np.pi)*np.exp(-mu**2/2/sigma**2)
            phi_beta = 1/np.sqrt(2*np.pi)*np.exp(-(1-mu)**2/2/sigma**2)

            # Average likelihood
            if v is None:
                t1 = np.mean(x-mu)
                t2 = np.mean((x-mu)**2)
            else:
                t1 = np.sum((x-mu)*v) / np.sum(v)
                t2 = np.sum((x-mu)**2*v) / np.sum(v)
                
            l = -np.log(Z) - np.log(sigma) - t2/2/sigma**2        
            # Gradient    
            d_c_mu = inv_sigma * (phi_alpha-phi_beta)
            d_c_sig = inv_sigma * (-mu*inv_sigma*phi_alpha - (1-mu)*inv_sigma*phi_beta)
            d_l_mu = -d_c_mu*inv_Z + t1*inv_sigma**2
            d_l_sig = -d_c_sig*inv_Z - inv_sigma + t2*inv_sigma**3
            grad = np.array([d_l_mu,d_l_sig],dtype=float)  
            return l,grad
        
        ## gradient check
        #_,grad_ = fit_f([0.2,0.1],x,v)
        #num_dmu = (fit_f([0.2+1e-8,0.1],x,v)[0]-fit_f([0.2,0.1],x,v)[0]) / 1e-8
        #num_dsigma = (fit_f([0.2,0.1+1e-8],x,v)[0]-fit_f([0.2,0.1],x,v)[0]) / 1e-8          
        #print('## Gradient check ##')
        #print('#  param value: mu=%0.6f, sigma=%0.6f'%(0.2,0.1))
        #print('#  Theoretical grad: dmu=%0.8f, dsigma=%0.8f'%(grad_[0],grad_[1]))
        #print('#  Numerical grad: dmu=%0.8f, dsigma=%0.8f\n'%(num_dmu,num_dsigma))
            
        # If the variance is small and the mean is at center, 
        # directly output the empirical mean and variance.
        if v is None:
            mu = np.mean(x)
            sigma = np.std(x)
        else:       
            mu = np.sum(x*v)/np.sum(v)
            sigma = np.sqrt(np.sum((x-mu)**2*v)/np.sum(v))
        
        if sigma<0.075 and np.min([1-mu,mu])>0.15:
            return mu,sigma
        
        param = np.array([mu,sigma])    
        lr = 0.01
        max_step = 0.025
        max_itr = 100
        i_itr = 0
        l_old = -10
        
        while i_itr<max_itr:            
            l,grad = fit_f(param,x,v)
            if np.absolute(l-l_old)<0.001:
                break
            else:
                l_old=l
            update = (grad*lr).clip(min=-max_step,max=max_step)
            param += update
            i_itr +=1                 
            if np.isnan(param).any() or np.min([param[0],1-param[0],param[1]])<0:  
                return np.mean(x),np.std(x)
                
        mu,sigma = param  
        if sigma>0.25:
            sigma=1
        return mu,sigma      
  
    mu    = np.zeros(x.shape[1],dtype=float)
    sigma = np.zeros(x.shape[1],dtype=float)
    for i in range(x.shape[1]):
        mu[i],sigma[i] = ML_bump_1d(x[:,i],v,logger=logger)
    return mu,sigma

def f_bump(x,mu,sigma):
    """
    density of the bump function

    Parameters
    ----------
    x : (n,d) ndarray
        coML estimatearaites
    mu : (d,) ndarray 
        bump mean parameter (for each dimension)
     
    sigma : (d,) ndarray 
        bump std parameter (for each dimension)

    Returns
    -------
    (n,) array
        the bump density for each point
    """    
    def f_bump_1d(x,mu,sigma):
        if sigma<1e-6: 
            return np.zeros(x.shape[0],dtype=float)
        inv_sigma=1/sigma
        pmf = sp.stats.norm.cdf(1,loc=mu,scale=sigma)-sp.stats.norm.cdf(0,loc=mu,scale=sigma)
        return inv_sigma/np.sqrt(2*np.pi)*np.exp(-inv_sigma**2*(x-mu)**2/2)/pmf
       
    f_x = np.ones([x.shape[0]],dtype=float)
    for i in range(x.shape[1]):
        f_x *=  f_bump_1d(x[:,i],mu[i],sigma[i])
    return f_x       
        
def f_all(x,a,mu,sigma,w):
    """
    density of the mixture model (slope+bump)

    Parameters
    ----------
    x : (n,d) ndarray
        coML estimatearaites
    mu : (d,) ndarray 
        bump mean parameter (for each dimension)
     
    sigma : (d,) ndarray 
        bump std parameter (for each dimension)

    Returns
    -------
    (n,) array
        the bump density for each point
    """
    
    f = w[0]*f_slope(x,a)        
    for k in range(1,w.shape[0]):
        f += w[k]*f_bump(x,mu[k-1],sigma[k-1])           
    return f

'''
    Baseline comparison methods
'''
def bh_test(p,alpha=0.1,n_full=None,verbose=False):
    if n_full is None: 
        n_full = p.shape[0]
    p_sort   = sorted(p)
    n_rej    = 0
    for i in range(p.shape[0]):
        if p_sort[i] < i*alpha/n_full:
            n_rej = i
    t_rej = p_sort[n_rej]
    if verbose:
        print("## bh testing summary ##")
        print("# n_rej = %d"%n_rej)
        print("# t_rej = %0.6f"%t_rej)
        print("\n")
    return n_rej,t_rej

def sbh_test(p,alpha=0.1,lamb=0.5,n_full=None,verbose=False):
    if n_full is None: 
        n_full = p.shape[0]
    if n_full > p.shape[0]:
        lamb = np.min(p[p>0.5])
    if verbose:
        print('lamb= %0.4f'%lamb)
    pi0_hat  = (np.sum(p>lamb)/(1-lamb)/n_full).clip(max=1)  
    alpha   /= pi0_hat
    if verbose:
        print('## pi0_hat=%0.3f'%pi0_hat) 
    p_sort = sorted(p)
    n_rej = 0
    for i in range(p.shape[0]):
        if p_sort[i] < i*alpha/n_full:
            n_rej = i
    t_rej = p_sort[n_rej]
    if verbose:
        print("## sbh summary ##")
        print("# n_rej = %d"%n_rej)
        print("# t_rej = %0.6f"%t_rej)
        print("# pi_0 estimate = %0.3f"%pi0_hat)
        print("\n")
    return n_rej,t_rej,pi0_hat

"""
    some old code
""" 


## old code for feature_explore
#def feature_explore(p,x_,alpha=0.1,qt_norm=False,vis_dim=None,cate_name={},output_folder=None,h=None):
#    def plot_feature_1d(x_margin,p,x_null,x_alt,bins,meta_info,title='',cate_name=None,\
#                        output_folder=None,h=None):
#        feature_type,cate_order = meta_info        
#        if feature_type == 'continuous':         
#            ## continuous feature: using kde to estimate 
#            n_bin = bins.shape[0]-1
#            x_grid = (bins+(bins[1]-bins[0])/2)[0:-1]
#            p_null,_ = np.histogram(x_null,bins=bins) 
#            p_alt,_= np.histogram(x_alt,bins=bins)         
#            p_null = p_null/np.sum(p_null)*n_bin
#            p_alt = p_alt/np.sum(p_alt)*n_bin
#            kde_null = stats.gaussian_kde(x_null).evaluate(x_grid)
#            kde_alt = stats.gaussian_kde(x_alt).evaluate(x_grid)
#            p_ratio = (kde_alt+1e-2)/(kde_null+1e-2)        
#                 
#        else: 
#            ## discrete feature: directly use the empirical counts 
#            unique_null,cts_null = np.unique(x_null,return_counts=True)
#            unique_alt,cts_alt = np.unique(x_alt,return_counts=True)            
#            unique_val = np.array(list(set(list(unique_null)+list(unique_alt))))
#            unique_val = np.sort(unique_val)            
#            p_null,p_alt = np.zeros([unique_val.shape[0]]),np.zeros([unique_val.shape[0]])          
#            for i,key in enumerate(unique_null): 
#                p_null[unique_val==key] = cts_null[i]                
#            for i,key in enumerate(unique_alt): 
#                p_alt[unique_val==key] = cts_alt[i]           
#            n_bin = unique_val.shape[0]           
#            p_null = (p_null+1)/np.sum(p_null+1)*n_bin
#            p_alt = (p_alt+1)/np.sum(p_alt+1)*n_bin            
#            p_ratio = (p_alt+1e-2)/(p_null+1e-2)  
#            x_grid = (np.arange(unique_val.shape[0])+1)/(unique_val.shape[0]+1)
#            
#            if cate_name is None: 
#                cate_name_ = cate_order
#            else:
#                cate_name_ = []
#                for i in cate_order:
#                    cate_name_.append(cate_name[i])
#                    
#        plt.figure(figsize=[8,8])
#        plt.subplot(311)
#        rnd_idx=np.random.permutation(p.shape[0])[0:np.min([10000,p.shape[0]])]
#        p = p[rnd_idx]
#        x_margin = x_margin[rnd_idx]
#        
#        if h is not None:
#            plt.scatter(x_margin[h==1],p[h==1],color='orange',alpha=0.3,s=4,label='alt')
#            plt.scatter(x_margin[h==0],p[h==0],color='royalblue',alpha=0.3,s=4,label='null')
#        else:
#            plt.scatter(x_margin,p,color='royalblue',alpha=0.3,s=4,label='alt')
#        plt.ylim([0,(p[p<0.5].max())])
#        plt.ylabel('p-value')
#        plt.title(title+' (%s)'%feature_type)
#        plt.subplot(312)
#        plt.bar(x_grid,p_null,width=1/n_bin,color='royalblue',alpha=0.6,label='null')
#        plt.bar(x_grid,p_alt,width=1/n_bin,color='darkorange',alpha=0.6,label='alt')
#        plt.xlim([0,1])
#        if feature_type=='discrete': 
#            plt.xticks(x_grid,cate_name_,rotation=45)
#        plt.ylabel('null/alt proportion')
#        plt.legend()
#        plt.subplot(313)
#        if feature_type == 'continuous':
#            plt.plot(x_grid,p_ratio,color='seagreen',label='ratio',linewidth=4) 
#        else:
#            plt.plot(x_grid,p_ratio,color='seagreen',marker='o',label='ratio',linewidth=4)
#            plt.xticks(x_grid,cate_name_,rotation=45)
#        plt.xlim([0,1])
#        plt.ylabel('ratio')
#        plt.xlabel('Covariate $x$')
#        plt.tight_layout()
#        if output_folder is not None:
#            plt.savefig(output_folder+'/explore_%s.png'%title)
#        plt.show()
#    
#    
#    ## preprocessing
#    x,meta_info = feature_preprocess(x_,p,qt_norm=qt_norm,continue_rank=False,require_meta_info=True)   
#    x_nq,meta_info = feature_preprocess(x_,p,qt_norm=False,continue_rank=False,require_meta_info=True)   
#    d = x.shape[1]
#    
#    ## separate the null proportion and the alternative proportion
#    _,t_BH = bh(p,alpha=alpha)
#    x_null,x_alt = x[p>0.5],x[p<t_BH]      
#    x_null_nq,x_alt_nq = x_nq[p>0.5],x_nq[p<t_BH]      
#    
#    ## generate the figure
#    bins = np.linspace(0,1,26)  
#    if vis_dim is None: 
#        vis_dim = np.arange(min(4,d))    
#    
#    for i in vis_dim:
#        x_margin = x[:,i]
#        if meta_info[i][0] == 'continuous':
#            temp_null,temp_alt = x_null[:,i],x_alt[:,i]
#        else:
#            temp_null,temp_alt = x_null_nq[:,i],x_alt_nq[:,i]
#        if i in cate_name.keys():
#            plot_feature_1d(x_margin,p,temp_null,temp_alt,bins,meta_info[i],title='feature_%s'%str(i+1),\
#                            cate_name=cate_name[i],output_folder=output_folder,h=h)
#        else:
#            plot_feature_1d(x_margin,p,temp_null,temp_alt,bins,meta_info[i],title='feature_%s'%str(i+1),\
#                            output_folder=output_folder,h=h)
#    return

    #def ML_slope_1d(x,w=None,c=0.05):
    #    if w is None:
    #        w = np.ones(x.shape[0])        
    #    t = np.sum(w*x)/np.sum(w) ## sufficient statistic
    #    a_u=100
    #    a_l=-100
    #    ## binary search 
    #    while a_u-a_l>0.1:
    #        a_m  = (a_u+a_l)/2
    #        a_m += 1e-2*(a_m==0)
    #        if (np.exp(a_m)/(np.exp(a_m)-1) - 1/a_m +c*a_m)<t:
    #            a_l = a_m
    #        else: 
    #            a_u = a_m
    #    return (a_u+a_l)/2
    #
    #a = np.zeros(x.shape[1],dtype=float)
    #for i in range(x.shape[1]):
    #    a[i] = ML_slope_1d(x[:,i],w,c=c)
    #return a
    