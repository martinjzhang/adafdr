## AdaFDR
[![GitHub version](https://badge.fury.io/gh/martinjzhang%2Fadafdr.svg)](https://badge.fury.io/gh/martinjzhang%2Fadafdr) [![PyPI version](https://badge.fury.io/py/adafdr.svg)](https://badge.fury.io/py/adafdr)

A fast and covariate-adaptive method for multiple hypothesis testing. 

Software accompanying the paper "AdaFDR: a Fast, Powerful and Covariate-Adaptive Approach to Multiple Hypothesis Testing", 2018.

## Requirement
* AdaFDR runs on python 3.

## Installation
```
pip install adafdr
```

## Usage
### Import package
`adafdr.method` contains all methods while `adafdr.data_loader` contains the data.
They can be imported as 
```python
import adafdr.method as md
import adafdr.data_loader as dl
```
Other ways of importing are usually compatible. For example, one can import the package with `import adafdr`
and call method `xxx` in the method module via `adafdr.method.xxx()`

### Input format
For a set of N hypotheses, the input data includes the p-values `p` and the d-dimensional covariate `x`, 
with the following format:

* `p`: (N,) numpy.ndarray.
* `x`: (N,d) numpy.ndarray. 

When d=1, `x` is allowed to be either (N,) numpy.ndarray 
or (N,1) numpy.ndarray.

### Covariate visualization
The covariate visualization method `adafdr_explore` can be used as 
```python
adafdr.method.adafdr_explore(p, x, output_folder=None)
```
If the `output_folder` is not `None`, the covariate visualization figures will be 
saved in `output_folder`. Otherwise, they will show up in the console.

### Multiple testing
The multiple hypothesis testing method `adafdr_test` can be used as 
* fast version (default): `res = adafdr.method.adafdr_test(p, x, alpha=0.1)`
* regular version: `res = adafdr.method.adafdr_test(p, x, alpha=0.1, fast_mode=False)`
* regular version with multi-core: `res = adafdr.method.adafdr_test(p, x, alpha=0.1, fast_mode=False, single_core=False)`

`res` is a dictionary containing the results, including:
* `res['decision']`: a (N,) boolean vector, decision for each hypothesis with value 1 meaning rejection.
<!-- * `res['n_rej']`: the number of rejections (on each fold). -->
* `res['threshold']`: a (N,) float vector, threshold for each hypothesis.
<!--* `res['theta']`: a list of learned parameters. -->
If `output_folder` is a folder path, log files will be saved in the folder. 

## Example on airway RNA-seq data
The following is an example on the airway RNA-seq data
used in the paper.
### Import package and load data
Here we load the *airway* data used in the paper.
See [vignettes](./vignettes) for other data accompanied with the package. 
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

Here, the left is a scatter plot of each hypothesis with p-values (y-axis) plotted against the covariate (x-axis). 
The right panel shows the estimated null hypothesis distribution (blue) and the estimated alternative hypothesis 
distribution (orange) with respect to the covariate. Here we can conclude that a hypothesis is more likely
to be significant if the covariate (gene expression) value is higher.

### Multiple hypothesis testing using `adafdr_test`
```python
res = md.adafdr_test(p, x, fast_mode=True, output_folder=None)
```

Here, the learned threshold `res['threshold']` looks as follows.

![p_scatter](https://raw.githubusercontent.com/martinjzhang/adafdr/master/images/threshold.png)

Each orange dot corresponds to the threhsold to one hypothesis. The discrepancy at the right is due to the difference between the thresholds learned by the two folds.

## Quick Test
<!-- ### Basic test -->
Here is a quick test. First check if the package can be successfully imported:
```python
import adafdr.method as md
import adafdr.data_loader as dl
```
Next, run a small example which should take a few seconds:
```python
import numpy as np
p,x,h,_,_ = dl.load_1d_bump_slope()
res = md.adafdr_test(p, x, alpha=0.1)
t = res['threshold']
D = np.sum(p<=t)
FD = np.sum((p<=t)&(~h))
print('# AdaFDR successfully finished!')
print('# D=%d, FD=%d, FDP=%0.3f'%(D, FD, FD/D))
```
It runs *AdaFDR-fast* on a 1d simulated data. If the package is successfully imported, 
the result should look like:
```
# AdaFDR successfully finished! 
# D=840, FD=80, FDP=0.095
```

<!-- 
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
print('# AdaFDR successfully finished!')
print('# D=%d, FD=%d, FDP=%0.3f'%(D, FD, FD/D))
```

If the machine is compatible with the multi-core processing, the following output will show up within a minute or two:
```
# AdaFDR successfully finished! 
# D=819, FD=75, FDP=0.092
```
If nothing shows up in more than 3 minutes, then the machine is not compatible with 
multi-core processing. Then it is recommended to use `md.adafdr_test(p, x, alpha=0.1)` 
for the fast version and `md.adafdr_test(p, x, alpha=0.1, fast_mode=False)` for the regular 
version with single-core processing.
-->

## Citation information
Coming soon.
