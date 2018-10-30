import matplotlib
matplotlib.use('Agg') 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab

from scipy.stats import beta

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from numpy import array
from scipy.cluster.vq import kmeans2
from sklearn.cluster import KMeans


def generate_data_1D(job=0, n_samples=10000,data_vis=0, num_case=4):
    if job == 0: # discrete case
        pi1=np.random.uniform(0,0.3,size=num_case)
        X=np.random.randint(0, num_case, n_samples)
        
        p = np.zeros(n_samples)
        h = np.zeros(n_samples)
        
        for i in range(n_samples):
            rnd = np.random.uniform()
            if rnd > pi1[X[i]]:
                p[i] = np.random.uniform()
                h[i] = 0
            else:
                p[i] = np.random.beta(a = np.random.uniform(0.2,0.4), b = 4)
                h[i] = 1
        return p, h, X

    
    
def generate_data_1D_cont(pi1, X, job=0):
    if job == 0: # discrete case
        n_samples = len(X)
        p = np.zeros(n_samples)
        h = np.zeros(n_samples)
        
        for i in range(n_samples):
            rnd = np.random.uniform()
            if rnd > pi1[i]:
                p[i] = np.random.uniform()
                h[i] = 0
            else:
                p[i] = np.random.beta(a = np.random.uniform(0.2,0.4), b = 4)
                h[i] = 1
        return p, h, X
    
    
#def p_value_beta_fit(p, lamb=0.8, bin_num=50, vis=0):
#    pi_0=np.divide(np.sum(p>lamb), p.shape[0] * (1-lamb))
#    temp_p=np.zeros([0])
#    step_size=np.divide(1,np.float(bin_num))
#    fil_num=np.int(np.divide(pi_0*p.shape[0],bin_num))+1
#    for i in range(bin_num):
#        p1=p[p>step_size*(i-1)]
#        p1=p1[p1 <= step_size*i]
#        choice_num= np.max(p1.shape[0] - fil_num,0)
#        if choice_num > 1:
#            choice=np.random.choice(p1.shape[0], choice_num)
#            temp_p=np.concatenate([temp_p,p1[choice]]).T
#    if vis==1:
#        plt.figure()
#        plt.hist(temp_p, bins=100, normed=True)       
#    a, b, loc, scale = beta.fit(temp_p,floc=0,fscale=1)
#    return pi_0, a, b

#def beta_mixture_pdf(x,pi_0,a,b):
#    return beta.pdf(x,a,b)*(1-pi_0)+pi_0
#
#def Storey_BH(x, alpha = 0.05, lamb=0.4, n = None):
#    pi0_hat=np.divide(np.sum(x>lamb),x.shape[0] *(1-lamb))
#    alpha /= pi0_hat
#    x_s = sorted(x)
#    if n is None:
#        n = len(x_s)
#    ic = 0
#    for i in range(len(x_s)):
#        if x_s[i] < i*alpha/float(n):
#            ic = i
#    return ic, x_s[ic], pi0_hat

#def Opt_t_cal_discrete(p, X, num_case=2,step_size=0.0001,ub=0.05,n_samples=10000,alpha=0.05):
#    # Fit the beta mixture parameters
#    fit_param=np.zeros([num_case, 3])
#    for i in range(num_case):
#        fit_param[i,:]=p_value_beta_fit(p[X==i])
#
#    # Calculating the ratios 
#    t_opt=np.zeros([num_case])
#    max_idx=np.argmin(fit_param[:,0])
#    x_grid = np.arange(0, ub, step_size)
#    t_ratio=np.zeros([num_case,x_grid.shape[0]])
#    for i in range(num_case):
#        t_ratio[i,:] = np.divide(beta_mixture_pdf(
#            x_grid,fit_param[i,0],fit_param[i,1],fit_param[i,2]), fit_param[i,0])
#
#    # Increase the threshold
#    for i in range(len(x_grid)):
#        t=np.zeros([num_case])
#        # undate the search optimal threshold
#        t[max_idx]=x_grid[i]
#        c=t_ratio[max_idx,i]
#        for j in range(num_case):
#            if j != max_idx: 
#                for k in range(len(x_grid)):
#                    if k == len(x_grid)-1:
#                        t[j]=x_grid[k]
#                        break
#                    if t_ratio[j,k+1]<c:
#                        t[j]=x_grid[k]
#                        break
#        # calculate the FDR
#        num_dis=0 
#        num_fd =0 
#        for i in range(num_case):
#            num_dis+=np.sum(p[X==i] < t[i])
#            num_fd+=np.sum(X==i)*t[i]*fit_param[i,0]
#
#        if np.divide(num_fd,np.float(np.amax([num_dis,1])))<alpha:
#            t_opt=t
#        else:
#            break
#    return t_opt

def generate_data_2D(job=0, n_samples=10000,data_vis=0):
    if job == 0: # Gaussian mixtures 
        x1 = np.random.uniform(-1,1,size = n_samples)
        x2 = np.random.uniform(-1,1,size = n_samples)
        pi1 = ((mlab.bivariate_normal(x1, x2, 0.25, 0.25, -0.5, -0.2)+
               mlab.bivariate_normal(x1, x2, 0.25, 0.25, 0.7, 0.5))/2).clip(max=1)        
        p = np.zeros(n_samples)
        h = np.zeros(n_samples)
               
        for i in range(n_samples):
            rnd = np.random.uniform()
            if rnd > pi1[i]:
                p[i] = np.random.uniform()
                h[i] = 0
            else:
                p[i] = np.random.beta(a = 0.3, b = 4)
                h[i] = 1
        X = np.concatenate([[x1],[x2]]).T
        
        if data_vis == 1:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            x_grid = np.arange(-1, 1, 1/100.0)
            y_grid = np.arange(-1, 1, 1/100.0)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            pi1_grid = ((mlab.bivariate_normal(X_grid, Y_grid, 0.25, 0.25, -0.5, -0.2)+
               mlab.bivariate_normal(X_grid, Y_grid, 0.25, 0.25, 0.7, 0.5))/2).clip(max=1)  
            ax1.pcolor(X_grid, Y_grid, pi1_grid)
            
            ax2 = fig.add_subplot(122)
            alt=ax2.scatter(x1[h==1][1:50], x2[h==1][1:50],color='r')
            nul=ax2.scatter(x1[h==0][1:50], x2[h==0][1:50],color='b')
            ax2.legend((alt,nul),('50 alternatives', '50 nulls'))
            
        return p, h, X
    if job == 1: # Linear trend
        
        x1 = np.random.uniform(-1,1,size = n_samples)
        x2 = np.random.uniform(-1,1,size = n_samples)
        pi1 = 0.1 * (x1 + 1) /2 +  0.3 *(1-x2) / 2
        
        p = np.zeros(n_samples)
        h = np.zeros(n_samples)
         
        for i in range(n_samples):
            rnd = np.random.uniform()
            if rnd > pi1[i]:
                p[i] = np.random.uniform()
                h[i] = 0
            else:
                p[i] = np.random.beta(a = 0.3, b = 4)
                h[i] = 1
        X = np.concatenate([[x1],[x2]]).T
        
        if data_vis == 1:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            x_grid = np.arange(-1, 1, 1/100.0)
            y_grid = np.arange(-1, 1, 1/100.0)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            pi1_grid =  0.1 * (X_grid + 1) /2 +  0.3 *(1-Y_grid) / 2
            
            ax1.pcolor(X_grid, Y_grid, pi1_grid)
            
            ax2 = fig.add_subplot(122)
            alt=ax2.scatter(x1[h==1][1:50], x2[h==1][1:50],color='r')
            nul=ax2.scatter(x1[h==0][1:50], x2[h==0][1:50],color='b')
            ax2.legend((alt,nul),('50 alternatives', '50 nulls'))
            
        return p, h, X
        
        
        
    if job == 2: # Gaussian mixture + linear trend
        x1 = np.random.uniform(-1,1,size = n_samples)
        x2 = np.random.uniform(-1,1,size = n_samples)
        pi1 = ((mlab.bivariate_normal(x1, x2, 0.25, 0.25, -0.5, -0.2)+
               mlab.bivariate_normal(x1, x2, 0.25, 0.25, 0.7, 0.5))/2).clip(max=1)        
        pi1 = pi1 * 0.5 + 0.5*(0.5 * (x1 + 1) /2 +  0.3 *(1-x2) / 2)
        
        p = np.zeros(n_samples)
        h = np.zeros(n_samples)
               
        for i in range(n_samples):
            rnd = np.random.uniform()
            if rnd > pi1[i]:
                p[i] = np.random.uniform()
                h[i] = 0
            else:
                p[i] = np.random.beta(a = 0.3, b = 4)
                h[i] = 1
        X = np.concatenate([[x1],[x2]]).T
        
        if data_vis == 1:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            x_grid = np.arange(-1, 1, 1/100.0)
            y_grid = np.arange(-1, 1, 1/100.0)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            pi1_grid = ((mlab.bivariate_normal(X_grid, Y_grid, 0.25, 0.25, -0.5, -0.2)+
               mlab.bivariate_normal(X_grid, Y_grid, 0.25, 0.25, 0.7, 0.5))/2).clip(max=1)  * 0.5 + (0.5 * (0.5 * (X_grid + 1) /2 +  0.3 *(1-Y_grid) / 2))
            ax1.pcolor(X_grid, Y_grid, pi1_grid)
            
            ax2 = fig.add_subplot(122)
            alt=ax2.scatter(x1[h==1][1:50], x2[h==1][1:50],color='r')
            nul=ax2.scatter(x1[h==0][1:50], x2[h==0][1:50],color='b')
            ax2.legend((alt,nul),('50 alternatives', '50 nulls'))
            
        return p, h, X
    
#def BH(x, alpha = 0.05, n = None):
#    x_s = sorted(x)
#    if n is None:
#        n = len(x_s)
#    ic = 0
#    for i in range(len(x_s)):
#        if x_s[i] < i*alpha/float(n):
#            ic = i
#    return ic, x_s[ic]



#def p_value_beta_fit(p, lamb=0.8, bin_num=50, vis=0):
#    pi_0=np.divide(np.sum(p>lamb), p.shape[0] * (1-lamb))
#    temp_p=np.zeros([0])
#    step_size=np.divide(1,np.float(bin_num))
#    fil_num=np.int(np.divide(pi_0*p.shape[0],bin_num))+1
#    for i in range(bin_num):
#        p1=p[p>step_size*(i-1)]
#        p1=p1[p1 <= step_size*i]
#        choice_num= np.max(p1.shape[0] - fil_num,0)
#        if choice_num > 1:
#            choice=np.random.choice(p1.shape[0], choice_num)
#            temp_p=np.concatenate([temp_p,p1[choice]]).T
#    if vis==1:
#        plt.figure()
#        plt.hist(temp_p, bins=100, normed=True)       
#    a, b, loc, scale = beta.fit(temp_p,floc=0,fscale=1)
#    return pi_0, a, b
#def beta_mixture_pdf(x,pi_0,a,b):
#    return beta.pdf(x,a,b)*(1-pi_0)+pi_0
#


#def Opt_t_cal_discrete(p, X, num_case=2,step_size=0.0001,ub=0.05,n_samples=10000,alpha=0.05):
#    # Fit the beta mixture parameters
#    fit_param=np.zeros([num_case, 3])
#    for i in range(num_case):
#        fit_param[i,:]=p_value_beta_fit(p[X==i])
#
#    # Calculating the ratios 
#    t_opt=np.zeros([num_case])
#    max_idx=np.argmin(fit_param[:,0])
#    x_grid = np.arange(0, ub, step_size)
#    t_ratio=np.zeros([num_case,x_grid.shape[0]])
#    for i in range(num_case):
#        t_ratio[i,:] = np.divide(beta_mixture_pdf(
#            x_grid,fit_param[i,0],fit_param[i,1],fit_param[i,2]), fit_param[i,0])
#
#    # Increase the threshold
#    for i in range(len(x_grid)):
#        t=np.zeros([num_case])
#        # undate the search optimal threshold
#        t[max_idx]=x_grid[i]
#        c=t_ratio[max_idx,i]
#        for j in range(num_case):
#            if j != max_idx: 
#                for k in range(len(x_grid)):
#                    if k == len(x_grid)-1:
#                        t[j]=x_grid[k]
#                        break
#                    if t_ratio[j,k+1]<c:
#                        t[j]=x_grid[k]
#                        break
#        # calculate the FDR
#        num_dis=0 
#        num_fd =0 
#        for i in range(num_case):
#            num_dis+=np.sum(p[X==i] < t[i])
#            num_fd+=np.sum(X==i)*t[i]*fit_param[i,0]
#
#        if np.divide(num_fd,np.float(np.amax([num_dis,1])))<alpha:
#            t_opt=t
#        else:
#            break
#    return t_opt

def result_summary(h,pred):
    print("Num of alternatives:",np.sum(h))
    print("Num of discovery:",np.sum(pred))
    print("Num of true discovery:",np.sum(pred * h))
    print("Actual FDR:", 1-np.sum(pred * h) / np.sum(pred))
    
#def softmax_prob_cal(X,Centorid, intensity=1):
#    dist=np.zeros([n_samples,num_clusters])
#    dist+=np.sum(X*X,axis=1, keepdims=True)
#    dist+=np.sum(centroid.T*centroid.T,axis=0, keepdims=True)
#    dist -= 2*X.dot(centroid.T)
#    dist=np.exp(dist*intensity)
#    dist /= np.sum(dist,axis=1, keepdims=True)
#    return dist


#def get_network(num_layers = 10, node_size = 10, dim = 1, scale = 1, cuda = False):
#    
#    
#    class Model(nn.Module):
#        def __init__(self, num_layers, node_size, dim):
#            super(Model, self).__init__()
#            l = []
#            l.append(nn.Linear(dim,node_size))
#            l.append(nn.LeakyReLU(0.1))
#            for i in range(num_layers - 2):
#                l.append(nn.Linear(node_size,node_size))
#                l.append(nn.LeakyReLU(0.1))
#
#            l.append(nn.Linear(node_size,1))
#            l.append(nn.Sigmoid())
#
#            self.layers = nn.Sequential(*l)
#
#
#
#        def forward(self, x):
#            x = self.layers(x)
#            x = 0.5 * scale * x 
#            return x
#
#   
#    
#    
#    network = Model(num_layers, node_size, dim)
#    if cuda:
#        return network.cuda()
#    else:
#        return network


#def train_network_to_target_p(network, optimizer, x, target_p, num_it = 1000, dim = 1, cuda = False):
#    target = Variable(torch.from_numpy(target_p.astype(np.float32)))
#    l1loss = nn.L1Loss()
#    batch_size = len(x)
#    n_samples = len(x)
#    loss_hist = []
#    choice = range(n_samples)
#    x_input = Variable(torch.from_numpy(x[choice].astype(np.float32).reshape(batch_size,dim)))
#
#    
#    if cuda:
#        x_input = x_input.cuda()
#        target = target.cuda()
#
#    for iteration in range(num_it):
#        if iteration % 100 == 0:
#            print iteration
#        
#        optimizer.zero_grad()
#        output = network.forward(x_input) 
#
#        loss = l1loss(output, target)
#        loss.backward()
#
#        optimizer.step()
#        loss_hist.append(loss.data[0])
#    
#    return loss_hist
#
#def train_network(network, optimizer, x, p, num_it = 3000, alpha = 0.05, dim = 1, lambda_ = 20, lambda2_ = 1e3, cuda = False, fdr_scale = 1, mirror = 1):
#    
#    batch_size = len(x)
#    n_samples = len(x)
#    print(batch_size, n_samples)
#    loss_hist = []
#    soft_compare = nn.Sigmoid()
#
#    relu = nn.ReLU()
#    choice = range(n_samples)
#    x_input = Variable(torch.from_numpy(x[choice].astype(np.float32).reshape(batch_size,dim)))
#    p_input = Variable(torch.from_numpy(p[choice].astype(np.float32).reshape(batch_size,1)))
#
#    if cuda:
#        x_input = x_input.cuda()
#        p_input = p_input.cuda()
#        
#    for iteration in range(num_it):
#        if iteration % 100 == 0:
#            print iteration
#
#
#        optimizer.zero_grad()
#        output = network.forward(x_input) 
#        s = torch.sum(soft_compare((output - p_input) * lambda2_)) / batch_size #disco rate
#        s2 = torch.sum(soft_compare((p_input - (mirror - output * fdr_scale)) * lambda2_)) / batch_size /float(fdr_scale)#false discoverate rate(over all samples)
#
#        gain = s  - lambda_ * relu((s2 - s * alpha)) 
#
#        loss = -gain
#        loss.backward()
#
#        optimizer.step()
#        loss_hist.append(loss.data[0])
#    
#    return loss_hist, s, s2
#
#
##def train_network_adapt(network, optimizer, x, p, num_it = 3000, alpha = 0.05, dim = 1, lambda_ = 2, lambda2_ = 1e3, cuda = False):
##    
##    batch_size = len(x)
##    n_samples = len(x)
##    loss_hist = []
##    soft_compare = nn.Sigmoid()
##
##    relu = nn.ReLU()
##    choice = range(n_samples)
##    x_input = Variable(torch.from_numpy(x[choice].astype(np.float32).reshape(batch_size,dim)))
##    p_input = Variable(torch.from_numpy(p[choice].astype(np.float32).reshape(batch_size,1)))
##
##    if cuda:
##        x_input = x_input.cuda()
##        p_input = p_input.cuda()
##        
##    for iteration in range(num_it):
##        if iteration % 100 == 0:
##            print iteration
##
##
##        optimizer.zero_grad()
##        output = network.forward(x_input) 
##        s = torch.sum(soft_compare((output - p_input) * lambda2_)) / batch_size #disco rate
##        s2 = torch.sum(soft_compare((p_input - (1-output)) * lambda2_)) / batch_size #false discoverate rate(over all samples)
##
##        gain = s  - lambda_ * (1+iteration/1000.0) * relu((s2 - s * alpha)) 
##
##        loss = -gain
##        loss.backward()
##
##        optimizer.step()
##        loss_hist.append(loss.data[0])
##    
##    return loss_hist, s, s2
#
#
#
#def opt_threshold(x, p, k, intensity = 1):
#    n_samples = x.shape[0]
#    
#    if len(x.shape) == 1:
#        x = np.expand_dims(x,1)
#    km = KMeans(n_clusters = k)
#    cluster = km.fit_predict(x)
#    opt = Opt_t_cal_discrete(p, cluster, num_case = k, step_size=0.00001)
#    #p_target = opt[cluster]
#    
#    x2 = x.repeat(k, axis = 1)
#    center = km.cluster_centers_.repeat(n_samples, axis = 1).T
#
#    e = np.exp (- (x2 - center) ** 2 / intensity)
#    s = np.expand_dims(np.sum(e, axis = 1),1)
#    prob = e/s
#    opt = Opt_t_cal_discrete(p, cluster, num_case = 10, step_size=0.00001)
#    p_target = prob.dot(opt)
#    
#    return p_target
#
#
#def opt_threshold_multi(x, p, k, intensity = 1, alpha = 0.05):
#    n_samples = x.shape[0]
#
#    km = KMeans(n_clusters = k)
#    cluster = km.fit_predict(x)
#    opt = Opt_t_cal_discrete(p, cluster, num_case = k, step_size=0.00001, alpha = alpha)
#    center = np.expand_dims(km.cluster_centers_, axis = -1).repeat(n_samples, axis = -1).T
#    x2 = np.expand_dims(x, axis = -1).repeat(k, axis = -1)
#
#    dist = x2 - center
#    dist = np.sum((x2 - center) ** 2, axis = 1)
#
#
#    e = np.exp (- dist / intensity)
#    s = np.expand_dims(np.sum(e, axis = 1),1)
#    prob = e/s
#    p_target = prob.dot(opt)
#    
#    return p_target
#
#
#
#def train_network_val(network, optimizer, x, p, num_it = 3000, alpha = 0.05, dim = 1, lambda_ = 20, lambda2_ = 1e3, lambda3_ = 1e2, cuda = False):
#    
#    batch_size = len(x)
#    n_samples = len(x)
#    train_loss_hist = []
#    val_loss_hist = []
#    soft_compare = nn.Sigmoid()
#
#    relu = nn.ReLU()
#    train_idx = range(n_samples/2)
#    val_idx = range(n_samples/2, n_samples)
#    num_train = len(train_idx)
#    num_val = len(val_idx)
#    
#    x_input_train = Variable(torch.from_numpy(x[train_idx].astype(np.float32).reshape(num_train,dim)))
#    p_input_train = Variable(torch.from_numpy(p[train_idx].astype(np.float32).reshape(num_train,1)))
#    x_input_val = Variable(torch.from_numpy(x[val_idx].astype(np.float32).reshape(num_val,dim)))
#    p_input_val = Variable(torch.from_numpy(p[val_idx].astype(np.float32).reshape(num_val,1)))
#    
#
#    if cuda:
#        x_input_train = x_input_train.cuda()
#        p_input_train = p_input_train.cuda()
#        x_input_val = x_input_val.cuda()
#        p_input_val = p_input_val.cuda()
#        
#    for iteration in range(num_it):
#        if iteration % 100 == 0:
#            print iteration
#
#
#        optimizer.zero_grad()
#        output = network.forward(x_input_train) 
#        s = torch.sum(soft_compare((output - p_input_train) * lambda2_)) / num_train #disco rate
#        s2 = torch.sum(soft_compare((p_input_train - (1-output)) * lambda3_)) / num_train #false discoverate rate(over all samples)
#
#        gain = s  - lambda_ * relu((s2 - s * alpha)) 
#
#        loss = -gain
#        loss.backward()
#
#        optimizer.step()
#        train_loss_hist.append(loss.data[0])
#        
#        output = network.forward(x_input_val) 
#        s = torch.sum(soft_compare((output - p_input_val) * lambda2_)) / num_val #disco rate
#        s2 = torch.sum(soft_compare((p_input_val - (1-output)) * lambda3_)) / num_val #false discoverate rate(over all samples)
#
#        gain = s  - lambda_ * relu((s2 - s * alpha)) 
#
#        loss = -gain
#        val_loss_hist.append(loss.data[0])
#        
#    
#    return train_loss_hist, val_loss_hist, s, s2
#
#
#def get_scale(network, x, p, dim = 1, cuda = False, lambda_ = 20, lambda2_ = 1e3, alpha = 0.05, fit = False, scale = 1, fdr_scale = 1, mirror  = 1):
#    batch_size = len(x)
#    n_samples = len(x)
#    loss_hist = []
#    soft_compare = nn.Sigmoid()
#    
#    relu = nn.ReLU()
#    choice = range(n_samples)
#    x_input = Variable(torch.from_numpy(x[choice].astype(np.float32).reshape(batch_size,dim)))
#    p_input = Variable(torch.from_numpy(p[choice].astype(np.float32).reshape(batch_size,1)))
#
#    if cuda:
#        x_input = x_input.cuda()
#        p_input = p_input.cuda()
#    
#    hi = 10.0
#    low = 0.1
#    current = scale
#    output = network.forward(x_input) * current
#
#    s = torch.sum(soft_compare((output - p_input) * lambda2_)) / batch_size #disco rate
#    s2 = torch.sum(soft_compare((p_input - (mirror-output)) * lambda2_)) / batch_size #false discoverate rate(over all samples)
#
#
#            
#    if fit:
#        for i in range(300):
#
#            output = network.forward(x_input) * current
#
#            s = torch.sum(soft_compare((output - p_input) * lambda2_)) / batch_size #disco rate
#            s2 = torch.sum(soft_compare((p_input - (mirror -output * fdr_scale)) * lambda2_)) / batch_size / float(fdr_scale) #false discoverate rate(over all samples)
#
#            if (s2/s).cpu().data[0] > alpha:
#                hi = current
#                current = (low + current)/2
#            else:
#                low = current
#                current = (hi + current)/2
#
#    print current, (s2/s).cpu().data[0]
#        
#    return current, (s2/s).cpu().data[0]
#
#import os
#import markdown2
#def generate_report(x = None, p = None, h = None, out_dir = '', url_prefix = '', info = {}, loss1 = None, loss2 = None, efdr = None, scales = None, x_prob = None, outputs = None, grids = None):
#    out_filename =  os.path.join(out_dir, 'report.html')
#    if not os.path.exists(out_dir):
#        os.makedirs(out_dir)
#    
#    if grids:
#        X_grid, Y_grid = grids
#    
#    f = open(out_filename, 'w')
#    s = ''
#    s += '# Report\n'
#    
#    s += '## Basic information\n'
#    for k in sorted(info.keys()):
#        v = info[k]
#        s += (str(k) + ':' + str(v) + '\n\n')
#    
#    dim = 1
#    if not x is None and not p is None :
#        n_samples = len(x)
#        s += '## Data description\n'
#        s += '- number of samples: {}\n'.format((n_samples))
#        s += '- dimension of data: {}\n'.format((x.shape[1]))
#        
#        dim = x.shape[1]
#        
#        plt.figure()
#        plt.scatter(x[:,0],p, 1, alpha = 0.3)
#        plt.xlabel('x')
#        plt.ylabel('p-value')
#        plt.savefig(out_dir + '/data.png')
#
#        s +=  '\n'
#        s += '![]({})\n'.format('data.png')
#        
#        if not h is None:
#            if not np.isnan(h[0]):
#                plt.figure()
#                plt.scatter(x[h==0,0], p[h==0], 1, alpha = 0.3, label = 'Null')
#                plt.scatter(x[h==1,0], p[h==1], 1, alpha = 0.3, label = 'Alternative')
#                plt.legend()
#                plt.xlabel('x')
#                plt.ylabel('p-value')
#                plt.savefig(out_dir + '/data_gt.png')
#        
#                s +=  '\n'
#                s += '![]({})\n'.format('data_gt.png')
#    
#    if not p is None:
#        plt.figure()
#        plt.hist(p, 50)
#        plt.xlabel('pvalues')
#        plt.savefig(out_dir + '/p_hist.png')
#        s +=  '\n'
#        s += '![]({})\n'.format('p_hist.png')
#    
#    if not loss1 is None and not loss2 is None:
#        s += '## Loss history\n'
#        plt.figure()
#        for i in range(len(loss1)):
#            plt.plot(np.log(loss1[i]), label = 'Loss history {}'.format(i))
#        plt.legend()
#        plt.savefig(out_dir + '/loss1.png')
#        s +=  '\n'
#        s += '![]({})\n'.format('loss1.png')
#        
#        plt.figure()
#        for i in range(len(loss2)):
#            plt.plot(loss2[i], label = 'Loss history {}'.format(i))
#        plt.legend()
#        plt.savefig(out_dir + '/loss2.png')
#        s +=  '\n'
#        s += '![]({})\n'.format('loss2.png')
#    
#    
#    if not scales is None and not efdr is None:
#        s += '## Scaling factors and expected FDR during training\n\n'
#        
#        s += str(efdr) + '\n\n'
#        
#        s += str(scales) + '\n\n'
#    
#    
#    if not len(outputs) == 0:
#        s += '## Threshold \n\n'
#        if dim == 1:
#            plt.figure()
#            for i in range(len(outputs)):
#                plt.plot(x_prob, outputs[i], label = 'threshold {}'.format(i))
#            
#            plt.xlabel('x')
#            plt.ylabel('t')
#            plt.legend()
#            plt.savefig(out_dir + '/threshold.png')
#
#            
#            s += '![]({})\n'.format('threshold.png')
#        elif dim == 2:
#            for i in range(len(outputs)):
#                plt.figure()
#                z = outputs[i].reshape(X_grid.shape)
#                plt.pcolor(X_grid, Y_grid, z)
#                plt.colorbar()
#                plt.savefig(out_dir + '/threshold_{}.png'.format(i))
#                s += '![]({})\n\n'.format('threshold_{}.png'.format(i))
#            
#     
#    html = markdown2.markdown(s)
#    
#    f.write(html)
#    f.close()
#    url = url_prefix + '/' + out_dir + '/report.html'
#    return url


