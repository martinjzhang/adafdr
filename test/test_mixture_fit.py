import numpy as np
import scipy as sp

from adafdr.util import *
import adafdr.method as md
import adafdr.data_loader as dl

def test_f_slope():
    ''' a 1d example '''
    n_grid = 101
    x = get_grid_1d(n_grid)
    a = np.array([2])
    fx = dl.f_slope(x,a)
    p_true = np.exp(x*a).flatten()
    p_true = p_true / np.sum(p_true)    
    print('1d output dimension check:',fx.shape)
    assert fx.shape == (n_grid,)
    error = np.linalg.norm(fx/n_grid-p_true,1)
    print('1d n_grid normalize l1 error = %0.8f'%error)
    assert error<0.005    
    error = np.linalg.norm(fx/np.sum(fx)-p_true,1)
    print('1d self normalize l1 error = %0.8f'%error)
    assert error<1e-8
    
    ''' a 2d example '''
    n_grid = 101
    x = get_grid_2d(n_grid)
    a = np.array([2,0])
    fx = dl.f_slope(x,a)
    p_true = np.prod(np.exp(x*a),axis=1)
    p_true = p_true / np.sum(p_true)
    error = np.linalg.norm(fx/n_grid**2-p_true,1)
    print('2d output dimension check:',fx.shape)
    assert fx.shape == (n_grid**2,)
    print('2d l1 error = %0.8f'%error)
    assert error<0.005
    error = np.linalg.norm(fx/np.sum(fx)-p_true,1)
    print('2d self normalize l1 error = %0.8f'%error)
    assert error<1e-8
    
def test_ML_slope():
    np.random.seed(0)
    def get_l(a,x,c,v=None):
        if v is None:
            l = np.sum(np.log(a[a!=0]/(np.exp(a[a!=0])-1))) + \
                a.dot(np.mean(x,axis=0)) - c*np.sum(a**2)
        else:
            l = np.sum(np.log(a[a!=0]/(np.exp(a[a!=0])-1))) + \
                a.dot(np.sum((x.T*v).T,axis=0))/np.sum(v) - c*np.sum(a**2)
        return l
    
    x,param = dl.load_x_mixture(opt=0)
    a_true = param 
    c = 0
    
    ''' without weights '''
    a_hat = md.ML_slope(x,c=c)    
    error = np.linalg.norm(a_hat-a_true,1)
    print('without weights: a_true = %s'%a_true)
    print('without weights: a_hat = %s'%a_hat)
    print('without weights: a_error = %s\n'%error)
    assert error < 0.2
    
    l_hat = get_l(a_hat,x,c)
    l_true = get_l(a_true,x,c)
    error = np.absolute(l_hat-l_true)
    print('without weights: l_true = %s'%l_true)
    print('without weights: l_hat = %s'%l_hat)
    print('without weights: l_error = %s\n'%error)    
    assert error < 0.001
    
    ''' with weights '''
    v = np.random.randn(x.shape[0]).clip(min=0)  
    a_hat = md.ML_slope(x,v=v,c=c)
    error = np.linalg.norm(a_hat-a_true,1)
    print('with weights: a_true = %s'%a_true)
    print('with weights: a_hat = %s'%a_hat)
    print('with weights: a_error = %s\n'%error)
    assert error < 0.2
    
    l_hat = get_l(a_hat,x,c,v)
    l_true = get_l(a_true,x,c,v)
    error = np.absolute(l_hat-l_true)
    print('with weights: l_true = %s'%l_true)
    print('with weights: l_hat = %s'%l_hat)
    print('with weights: l_error = %s\n'%error)    
    assert error < 0.001
        
def test_f_bump():
    ''' a 1d example '''
    n_grid = 101
    x = get_grid_1d(n_grid)
    mu = np.array([0.5],dtype=float)
    sigma = np.array([0.1],dtype=float)    
    fx = dl.f_bump(x,mu,sigma)    
    p_true = np.exp(-(x-mu)**2/sigma**2/2).reshape(-1)
    p_true = p_true / np.sum(p_true)    
    print('1d output dimension check:',fx.shape)
    assert fx.shape == (n_grid,)
    error = np.linalg.norm(fx/n_grid-p_true,1)
    print('1d n_grid normalize l1 error = %0.8f'%error)
    assert error<0.01  
    error = np.linalg.norm(fx/np.sum(fx)-p_true,1)
    print('1d self normalize l1 error = %0.8f'%error)
    assert error<1e-8
    
    ''' a 2d example '''
    n_grid = 101
    x = get_grid_2d(n_grid)
    mu = np.array([0.5,0.2],dtype=float)
    sigma = np.array([0.1,0.1],dtype=float)    
    fx = dl.f_bump(x,mu,sigma)    
    p_true = np.exp(-np.sum((x-mu)**2/sigma**2/2,axis=1)).reshape(-1)
    p_true = p_true / np.sum(p_true)    
    print('2d output dimension check:',fx.shape)
    assert fx.shape == (n_grid**2,)
    print('2d l1 error = %0.8f'%error)
    assert error<0.01
    error = np.linalg.norm(fx/np.sum(fx)-p_true,1)
    print('2d self normalize l1 error = %0.8f'%error)
    assert error<1e-8
    
def test_ML_slope():
    np.random.seed(0)
    def get_l(param,x,v=None): # omitting the constant term 
        l = 0
        mu,sigma = param
        for j in range(x.shape[1]):        
            Z = sp.stats.norm.cdf(1,loc=mu[j],scale=sigma[j])-sp.stats.norm.cdf(0,loc=mu[j],scale=sigma[j])
            print(Z)

            if v is None:
                l += -np.log(Z) - np.log(sigma[j]) - 1/2/sigma[j]**2*np.mean((x[:,j]-mu[j])**2)
            else:
                t = np.sum((x[:,j]-mu[j])**2*v) / np.sum(v)
                l += -np.log(Z) - np.log(sigma[j]) - 1/2/sigma[j]**2*t - 1/2*np.log(2*np.pi)
        return l

    x,param = dl.load_x_mixture(opt=1)
    mu_true,sigma_true = param
    
    ''' without weights '''
    mu_hat,sigma_hat = md.ML_bump(x)
    error = np.linalg.norm(mu_true-mu_hat,1) + np.linalg.norm(sigma_true-sigma_hat,1)
    print('without weights: mu_true = %s, sigma_true = %s'%(mu_true,sigma_true))
    print('without weights: mu_hat = %s, sigma_hat = %s'%(mu_hat,sigma_hat))
    print('without weights: param_error = %s\n'%error)
    assert error < 0.2

    l_hat = get_l((mu_hat,sigma_hat),x)
    l_true = get_l((mu_true,sigma_true),x)
    error = np.absolute(l_hat-l_true)
    print('without weights: l_true = %s'%l_true)
    print('without weights: l_hat = %s'%l_hat)
    print('without weights: l_error = %s\n'%error)    
    assert error < 0.01

    ''' with weights '''
    v = np.random.randn(x.shape[0]).clip(min=0)  
    mu_hat,sigma_hat = md.ML_bump(x,v=v)    
    error = np.linalg.norm(mu_true-mu_hat,1) + np.linalg.norm(sigma_true-sigma_hat,1)
    print('with weights: mu_true = %s, sigma_true = %s'%(mu_true,sigma_true))
    print('with weights: mu_hat = %s, sigma_hat = %s'%(mu_hat,sigma_hat))
    print('with weights: param_error = %s\n'%error)
    assert error < 0.2

    l_hat = get_l((mu_hat,sigma_hat),x,v=v)
    l_true = get_l((mu_true,sigma_true),x,v=v)
    error = np.absolute(l_hat-l_true)
    print('with weights: l_true = %s'%l_true)
    print('with weights: l_hat = %s'%l_hat)
    print('with weights: l_error = %s\n'%error)    
    assert error < 0.01  
    # assert False
    
def test_f_all():
    n_grid = 101
    x = get_grid_2d(n_grid)
    w = np.array([0.4,0.3,0.3],dtype=float)
    a = np.array([2,0],dtype=float)
    mu = np.array([[0.2,0.2],[0.7,0.7]],dtype=float)
    sigma = np.array([[0.1,0.2],[0.1,0.1]],dtype=float) 
    
    fx = dl.f_all(x,a,mu,sigma,w)    
    fx0 = md.f_slope(x,a)
    fx1 = md.f_bump(x,mu[0],sigma[0])
    fx2 = md.f_bump(x,mu[1],sigma[1])
    error = np.linalg.norm(fx - (w[0]*fx0 + w[1]*fx1 + w[2]*fx2),1)
    print('theoretical error = %0.8f'%error)
    
    
    p0 = np.prod(np.exp(x*a),axis=1)
    p0 /= p0.sum()
    p1 = np.exp(-np.sum((x-mu[0])**2/sigma[0]**2/2,axis=1)).reshape(-1)
    p1 /= p1.sum()
    p2 = np.exp(-np.sum((x-mu[1])**2/sigma[1]**2/2,axis=1)).reshape(-1)
    p2 /= p2.sum() 
    p_true = w[0]*p0 + w[1]*p1 + w[2]*p2
    error = np.linalg.norm(fx/n_grid**2-p_true,1)
    print('output dimension check:',fx.shape)
    assert fx.shape == (n_grid**2,)
    print('all l1 error = %0.8f'%error)
    assert error<0.01
    
def test_mixture_fit():
    def get_l(x,a,mu,sigma,w):
        fx = md.f_all(x,a,mu,sigma,w)
        return np.mean(np.log(fx))
    # 2d slope+bump
    x,param = dl.load_x_mixture(opt=2)
    a_true,mu_true,sigma_true,w_true = param
    l_true = get_l(x,a_true,mu_true,sigma_true,w_true)
    a_hat,mu_hat,sigma_hat,w_hat = md.mixture_fit(x)
    l_hat = get_l(x,a_hat,mu_hat,sigma_hat,w_hat)
    print('# 2d l_true=%s'%l_true)
    print('# 2d l_hat=%s'%l_hat)   
    error = np.absolute(l_hat-l_true)
    assert error<0.01
    
    # 10d slope+bump
    x,param = dl.load_x_mixture(opt=3)
    a_true,mu_true,sigma_true,w_true = param
    l_true = get_l(x,a_true,mu_true,sigma_true,w_true)
    a_hat,mu_hat,sigma_hat,w_hat = md.mixture_fit(x)
    l_hat = get_l(x,a_hat,mu_hat,sigma_hat,w_hat)
    print('# 10d l_true=%s'%l_true)
    print('# 10d l_hat=%s'%l_hat)   
    error = np.absolute(l_hat-l_true)
    assert error<0.01
