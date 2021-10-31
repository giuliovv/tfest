# Tfest
Transfer function estimation with frequency response. 
Python equivalent of MATLAB tfest (but the algorithm is not exactly the same)

Only works wuth SISO systems for now.


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
To use L1 normalization set the value of lambda l1 (default l1=0):
```python
te.estimate(3, 4, time=1, l1=0.1)
```
