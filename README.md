## AdaFDR
A fast and covariate-adaptive method for multiple hypothesis testing. 

Software accompanying the paper "AdaFDR: a Fast, Powerful and Covariate-Adaptive Approach to Multiple Hypothesis Testing", 2018.

## Installation
```
pip install adafdr
```

## Usage
`adafdr` mainly offers two methods: `adafdr_explore` for covariate visualization and 
`adafdr_test` for multiple hypothesis testing. The input data `p,x` has the following 
format:

* `p`: (N,) numpy.ndarray, p-values for N hypotheses.
* `x`: (N,d) numpy.ndarray, d-dimensional covariate for each hypothesis. When d=1, 
`x` is allowed to be (N,) numpy.ndarray or (N,1) numpy.ndarray.

The covariate visualization method `adafdr_explore` can be used as 
* `adafdr_explore(p, x, output_folder=None)`

where if the output_folder in not `None`, the covariate visualization figures will be 
saved into `output_folder`. Otherwise, they will show up on the console.

The multiple hypotehsis testing method `adafdr_test` can be used as 
* fast version: `n_rej,t_rej,theta = md.adafdr_test(p, x, alpha=0.1)`
* regular version: `n_rej,t_rej,theta = md.adafdr_test(p, x, alpha=0.1, fast_mode=False)`
* regular version with multi-core: `n_rej,t_rej,theta = md.adafdr_test(p, x, alpha=0.1, fast_mode=False, single_core=false)`

where `n_rej` is the number of rejections; `t_rej` is a (N,) numpy.ndarray for decision threshold for each hypothesis;
`theta` is a list of learned parameters. If `output_folder` is a folder path, log files will be saved in the folder. 

## Example on airway data
The following is an example on the airway data
used in the paper.
### Import package and load data
`adafdr.method` contains the algorithm implementation while `adafdr.data_loader` can be 
used to load the data used in the paper. Here we load the *airway* data used in the paper.
See vignette for other data accompanied with the package. 
```python
import adafdr.method as md
import adafdr.data_loader as dl
p,x = dl.data_airway()
```

### Covariate visualization using `adafdr_explore`
```python
md.adafdr_explore(p, x, output_folder=None)
```

![p_scatter](https://raw.githubusercontent.com/martinjzhang/adafdr/master/images/explore_p_feature_1.png ) 
![ratio](https://raw.githubusercontent.com/martinjzhang/adafdr/master/images/explore_ratio_feature_1.png )

Here, the left is a scatter plot of each hypothesis with p-values (y-axis) against the covariate (x-axis). 
The right are the estimated null hypothesis distribution (blue) and the estimated alternative hypothesis 
distribution (orange) with respect to the covariate. Here we can conclude that a hypothesis is more likely
to be significant if the covariate (gene expression) value is larger.

### Multiple hypothesis testing using `adafdr_test`
```python
n_rej,t_rej,theta = md.adafdr_test(p, x, fast_mode=True, output_folder=None)
```

Here, the learned threshold looks as follows. Note that the two lines correspond to the data from two folds via
hypothesis splitting.

![p_scatter](https://raw.githubusercontent.com/martinjzhang/adafdr/master/images/threshold.png)

## Quick Test
### Basic test
Here is a quick test. First check if the package can be succesfully imported:
```python
import adafdr.method as md
import adafdr.data_loader as dl
```
Next, run a small example which should take a few seconds:
```python
import numpy as np
p,x,h,_,_ = dl.load_1d_bump_slope()
n_rej,t_rej,theta = md.adafdr_test(p, x, alpha=0.1, fast_mode=True)
D = np.sum(p<=t_rej)
FD = np.sum((p<=t_rej)&(~h))
print('# AdaFDR successfully finished! ')
print('# D=%d, FD=%d, FDP=%0.3f'%(D, FD, FD/D))
```
It runs *AdaFDR-fast* on a 1d simulated data. If the package is successfully imported, 
the result should look like:
```
# AdaFDR successfully finished! 
# D=840, FD=80, FDP=0.095
```

### Compatibility testing for multi-core processing
*AdaFDR* also offers a multi-core version where the hypotheses from the two folds 
are processed in parallel. Due to some compatibility issues (of `PyTorch` and `multiprocessing`),
in some rare cases the machine will get stuck when running the regular version of *adafdr_test* 
with multi-core processing (the fast version is always fine). To check it, run the following 

```python
import adafdr.method as md
import adafdr.data_loader as dl
import numpy as np
p,x,h,_,_ = dl.load_1d_bump_slope()
n_rej,t_rej,theta = md.adafdr_test(p, x, alpha=0.1, fast_mode=False, single_core=False)
D = np.sum(p<=t_rej)
FD = np.sum((p<=t_rej)&(~h))
print('# AdaFDR successfully finished! ')
print('# D=%d, FD=%d, FDP=%0.3f'%(D, FD, FD/D))
```

If the machine is compatible with the multi-core processing, the following output will show up within a minute or two:
```
# AdaFDR successfully finished! 
# D=823, FD=83, FDP=0.101
```
If nothing shows up in more than 3 minutes, then the machine is not compatible with 
multi-core processing. Then it is recommended to use `md.adafdr_test(p, x, alpha=0.1)` 
for the fast version and `md.adafdr_test(p, x, alpha=0.1, fast_mode=False)` for the regular 
version with single-core processing.

## Citation information
Coming soon.
