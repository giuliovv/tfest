# Tfest
[![PyPI version](https://badge.fury.io/py/tfest.svg)](https://badge.fury.io/py/tfest)
[![Downloads](https://static.pepy.tech/personalized-badge/tfest?period=total&units=international_system&left_color=black&right_color=red&left_text=Downloads)](https://pepy.tech/project/tfest)

Transfer function estimation with frequency response. 
Python equivalent of MATLAB tfest (but the algorithm is not exactly the same)

Only works with SISO systems for now.


### Installation:
```shell
pip install tfest
```

### Example:
To get a scipy transfer function:
```python
import tfest
# u: input
# y: output
te = tfest.tfest(u, y)
# n_zeros, n_poles
te.estimate(3, 4, time=1)
te.get_transfer_function()
```
"time" is simulation length in seconds.
To plot its bode diagram:
```python
te.plot_bode()
```
Default method to calculate the frequency response Y/U is H1 estimator, if you want to use H2 or frequency/frequency just set the method to "h2" or "fft" and specify the time length of the simulation (in seconds, default is 1):
```python
te.estimate(3, 4, method="h2", time=1)
te.estimate(3, 4, method="fft", time=1)
```
To use L2 regularization set the value of lambda l1 (default l1=0):
```python
te.estimate(3, 4, time=1, l1=0.1)
```
