import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import logging
import pickle
import os

""" 
    basic functions
"""
def get_grid_1d(n_grid):    
    """
    return an equally spaced covariate covering the 1d space...

    Parameters
    ----------
    n_grid: int
        number of points 

    Returns
    -------
    (n,1) ndarray
    """
    x_grid = np.linspace(0,1,n_grid).reshape(-1,1)
    return x_grid

def get_grid_2d(n_grid):
    """
    return an equally spaced covariate covering the 2d space...

    Parameters
    ----------
    n_grid: int
        number of points 

    Returns
    -------
    (n,2) ndarray
    """
    temp = np.linspace(0,1,n_grid)
    g1,g2  = np.meshgrid(temp,temp)
    x_grid = np.concatenate([g1.reshape(-1,1),g2.reshape(-1,1)],axis=1)
    return x_grid

"""
    calculate the dimension-wise rank statistics
    # fix it: for discrete features, it may be nice to keep their values the same
    
    ----- input  -----
    x: an n*d array 
    
    ----- output -----
    ranks: an n*d array, column-wise rank of x
"""
def rank(x, continous_rank=True):
    """Calculate the dimension-wise rank statistics.
    
    Args:
        x ((n,d) ndarray): The covariates.
        continous_rank (bool): Indicate if break the same value by randomization.
    
    Returns:
        ranks ((n,d) ndarray): The column-wise rank of x
    """
    ranks = np.empty_like(x)
    n,d = x.shape
    for i in range(d):
        if continous_rank:           
            temp = x[:,i].argsort(axis=0)       
            ranks[temp,i] = np.arange(n)
        else:
            ranks[:,i] = rankdata(x[:,i])-1
    return ranks

def result_summary(pred, h=None, f_write=None, title=''):
    """ Summerize the result based on the predicted value and the true value
    
    Args:
        pred ((n,) ndarray): the testing result, 1 for alternative and 0 for null.
        h ((n,) ndarray): the true values.
        f_write (file handle)
        
    """
    if title != '':
        print('## %s'%title)
    print('# Num of discovery: %d'%np.sum(pred))
    if h is not None: 
        print("# Num of alternatives:",np.sum(h))
        print("# Num of true discovery: %d"%np.sum(pred*h))
        print("# Actual FDP: %0.3f"%(1-np.sum(pred * h) / np.sum(pred)))
    print('')    
    if f_write is not None:
        f_write.write('# Num of discovery: %d\n'%np.sum(pred))
        if h is not None:
            f_write.write("# Num of alternatives: %d\n"%np.sum(h))
            f_write.write("# Num of true discovery: %d\n"%np.sum(pred*h))
            f_write.write("# Actual FDP: %0.3f\n"%(1-np.sum(pred * h) / np.sum(pred)))
        f_write.write('\n')
    return

def print_param(a,mu,sigma,w):
    print('# w=%s'%w)
    print('# a=%s'%a)
    print('# mu=%s'%mu)
    print('# sigma=%s'%sigma)
    print('')

"""
    basic functions for visualization
""" 
def plot_x(x,vis_dim=None):
    if len(x.shape)==1:
        plt.hist(x,bins=50)
    else:
        if vis_dim is None: vis_dim = np.arange(x.shape[1])            
        for i,i_dim in enumerate(vis_dim):
            plt.subplot('1'+str(len(vis_dim))+str(i+1))
            plt.hist(x[:,i_dim],bins=50)
            plt.title('dimension %s'%str(i_dim+1))   
    
def plot_t(t,p,x,h=None,color=None,label=None):
    if color is None: color = 'darkorange'
        
    if t.shape[0]>5000:
        rand_idx=np.random.permutation(x.shape[0])[0:5000]
        t = t[rand_idx]
        p = p[rand_idx]
        x = x[rand_idx]
        if h is not None: h = h[rand_idx]
            
    if len(x.shape)==1:
        sort_idx = x.argsort()
        if h is None:
            plt.scatter(x,p,alpha=0.1,color='royalblue')
        else:
            plt.scatter(x[h==0],p[h==0],alpha=0.1,color='royalblue')
            plt.scatter(x[h==1],p[h==1],alpha=0.1,color='seagreen')
        plt.plot(x[sort_idx],t[sort_idx],color=color,label=label)
        plt.ylim([0,2*t.max()])
            
    else:
        n_plot = min(x.shape[1], 5)
        for i in range(n_plot):
            plt.subplot(str(n_plot)+'1'+str(i+1))
            sort_idx=x[:,i].argsort()
            if h is None:
                plt.scatter(x[:,i],p,alpha=0.1)
            else:
                plt.scatter(x[:,i][h==0],p[h==0],alpha=0.1,color='royalblue')
                plt.scatter(x[:,i][h==1],p[h==1],alpha=0.1,color='seagreen')
            plt.scatter(x[:,i][sort_idx],t[sort_idx],s=8,alpha=0.2,color='darkorange')
            plt.ylim([0,2*t.max()])
    #plt.scatter(x,t,alpha=0.2)
    #plt.ylim([0,1.2*t.max()])
    #plt.ylabel('t')
    #plt.xlabel('x')
    
def plot_scatter_t(t,p,x,h=None,color='orange',label=None):        
    if t.shape[0]>5000:
        rand_idx=np.random.permutation(x.shape[0])[0:5000]
        t = t[rand_idx]
        p = p[rand_idx]
        x = x[rand_idx]
        if h is not None: 
            h = h[rand_idx]            
    sort_idx = x.argsort()
    if h is None:
        plt.scatter(x,p,alpha=0.1,color='steelblue')
    else:
        plt.scatter(x[h==0],p[h==0],alpha=0.1,color='steelblue')
        plt.scatter(x[h==1],p[h==1],alpha=0.3,color='seagreen')
    plt.scatter(x[sort_idx],t[sort_idx],color=color,s=4,alpha=0.6,label=label)
    plt.ylim([0, 1.5*t.max()])
        
def plot_data_1d(p,x,h,n_pt=1000):
    rnd_idx=np.random.permutation(p.shape[0])[0:n_pt]
    p = p[rnd_idx]
    x = x[rnd_idx]
    h = h[rnd_idx]
    plt.scatter(x[h==1],p[h==1],color='r',alpha=0.2,label='alt')
    plt.scatter(x[h==0],p[h==0],color='b',alpha=0.2,label='null')
    plt.xlabel('covariate')
    plt.ylabel('p-value')
    plt.title('hypotheses') 
    
def plot_data_2d(p,x,h,n_pt=1000):
    rnd_idx=np.random.permutation(p.shape[0])[0:n_pt]
    p = p[rnd_idx]
    x = x[rnd_idx,:]
    h = h[rnd_idx]
    plt.scatter(x[h==1,0],x[h==1,1],color='r',alpha=0.2,label='alt')
    plt.scatter(x[h==0,0],x[h==0,1],color='b',alpha=0.2,label='null')
    plt.xlabel('covariate 1')
    plt.ylabel('covariate 2')
    plt.title('hypotheses') 

"""
    ancillary functions
""" 
def sigmoid(x):
    x = x.clip(min=-20,max=20)
    return 1/(1+np.exp(-x))
        

def inv_sigmoid(w):
    w = w.clip(min-1e-8,max=1-1e-8)
    return np.log(w/(1-w))

"""
    Functions for generating the simulation results
"""
def get_summary_stats(filename=None, folder_r=None):
    """Extract the statstics from the simulation results
    
    Args:
        filename (str): file path for the python results.
        folder_r (str): result for r methods.
    Return:
        summary_stats (dic): a dic containing FDP and Power.
    """
    summary_stats = {}
    # Python methods
    if filename is not None:
        fil = open(filename, 'rb')
        result_dic = pickle.load(fil)
        time_dic = pickle.load(fil)
        method_list = list(result_dic.keys())
        alpha_list = np.array([0.05, 0.1, 0.15, 0.2])
        n_data = len(result_dic[method_list[0]][alpha_list[0]])
        for method in method_list:
            summary_stats[method] = {}
            summary_stats[method]['FDP'] = np.zeros([n_data, len(alpha_list)])
            summary_stats[method]['Power'] = np.zeros([n_data, len(alpha_list)])
            for i_alpha,alpha in enumerate(alpha_list):
                for i_data,data in enumerate(result_dic[method][alpha]):
                    h, h_hat = data
                    summary_stats[method]['FDP'][i_data, i_alpha] =\
                        np.sum((h==0)*(h_hat==1)) / max(np.sum(h_hat==1), 1)
                    summary_stats[method]['Power'][i_data, i_alpha] =\
                        np.sum((h==1)*(h_hat==1)) / np.sum(h==1)
    # R methods
    if folder_r is not None:
        # file_list = os.listdir(folder_r)
        file_list = []
        for filename in os.listdir(folder_r):
            if filename[0:3] == 'res':
                file_list.append(filename)
        method_r_list = ['adapt', 'ihw']
        for method in method_r_list:
            summary_stats[method] = {}
            summary_stats[method]['FDP'] = np.zeros([n_data, len(alpha_list)])
            summary_stats[method]['Power'] = np.zeros([n_data, len(alpha_list)])
            for i_data,filename in enumerate(file_list):
                file_path = folder_r + '/' + filename
                temp_data = np.loadtxt(file_path, skiprows=1, delimiter = ',')
                h = temp_data[:, 0]
                for i_alpha,alpha in enumerate(alpha_list):                                                
                    if method == 'adapt':
                        h_hat = temp_data[:, i_alpha+1]
                    else:
                        h_hat = temp_data[:, i_alpha+5]
                    summary_stats[method]['FDP'][i_data, i_alpha] =\
                        np.sum((h==0)*(h_hat==1)) / max(np.sum(h_hat==1), 1)
                    summary_stats[method]['Power'][i_data, i_alpha] =\
                        np.sum((h==1)*(h_hat==1)) / np.sum(h==1)
    return summary_stats, time_dic

def plot_size_power(summary_stats, method_mapping_dic, data_name='', output_folder=None):
    marker_list = ['o', 'v', '^', '*', 'h', 'd']
    # color_list = ['C8', 'C5', 'C1', 'C2', 'C3', 'C0']
    color_list = ['C1', 'C2', 'C3', 'C0', 'C5', 'C8']
    alpha_list = [0.05, 0.1, 0.15, 0.2]
    axes = plt.figure(figsize = [5, 4])
    method_list = list(summary_stats.keys())
    method_list = ['nfdr (fast)', 'nfdr', 'adapt', 'ihw', 'sbh', 'bh']
    n_data = summary_stats[method_list[0]]['FDP'].shape[0]
    for i_method,method in enumerate(method_list):
        y_val = np.mean(summary_stats[method]['FDP'], axis=0)
        y_err = np.std(summary_stats[method]['FDP'], axis=0) / np.sqrt(n_data) * 1.96
        plt.errorbar(alpha_list, y_val, yerr=y_err, label=method_mapping_dic[method],\
                     capsize=4, elinewidth = 1.5, linewidth=1.5,\
                     color = color_list[i_method], marker = marker_list[i_method],\
                     markersize = 6, alpha=0.8)
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    axis_min = min(x_min, y_min)
    axis_max = max(x_max, y_max)
    plt.plot([axis_min, axis_max], [axis_min, axis_max], linestyle='--', color='k')
    plt.legend(loc=2, fontsize=12)
    plt.ylabel('FDP', fontsize=16)
    plt.xlabel('nominal FDR', fontsize=16)
    if output_folder is not None:
        plt.tight_layout()
        plt.savefig(output_folder+'fdp_%s.png'%data_name)
        plt.savefig(output_folder+'fdp_%s.pdf'%data_name)
    else:
        plt.show()
    axes = plt.figure(figsize = [5, 4])
    for i_method,method in enumerate(method_list):
        y_val = np.mean(summary_stats[method]['Power'], axis=0)
        y_err = np.std(summary_stats[method]['Power'], axis=0) / np.sqrt(n_data) * 1.96
        plt.errorbar(alpha_list, y_val, yerr=y_err, label=method_mapping_dic[method],\
                     capsize=4, elinewidth = 1.5, linewidth=1.5,\
                     color = color_list[i_method], marker = marker_list[i_method],\
                     markersize = 6, alpha=0.8)
    plt.legend(loc=2, fontsize=12)
    plt.ylabel('power', fontsize=16)
    plt.xlabel('nominal FDR', fontsize=16)
    if output_folder is not None:
        plt.tight_layout()
        plt.savefig(output_folder+'power_%s.png'%data_name)
        plt.savefig(output_folder+'power_%s.pdf'%data_name)
    else:
        plt.show()
    plt.close('all')