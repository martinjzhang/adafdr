## AdaFDR
A fast and covariate-adaptive method for multiple hypothesis testing.

## Installation
```
pip install adafdr
```

## Usage
`adafdr` mainly offers two methods: `adafdr_explore` for covariate visualization and 
`adafdr_test` for multiple hypothesis testing. 

### Import package and load data
`adafdr.method` contains the algorithm implementation while `adafdr.data_loader` can be 
used to load the data used in the paper. 
```python
import adafdr.method as md
import adafdr.data_loader as dl
p,x = dl.data_airway()
```
The data `p,x` has the following format:
* `p`: (n,) numpy.ndarray, p-values for each hypothesis.
* `x`: (n,d) numpy.ndarray, d-dimensional covariate for each hypothesis. When d=1, 
`x` is allowed to be (n,) numpy.ndarray or (n,1) numpy.ndarray.

### Covariate visualization using adafdr_explore
```python
md.adafdr_explore(p, x, output_folder=None)
```


### Multiple hypothesis testing using adafdr_test
```python
n_rej,t_rej,theta = md.adafdr_test(p, x)
```

## Quick Test
