# Tfest
Transfer function estimation with frequency response. Only works for SISO systems for now.
Python equivalent of MATLAB tfest

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
te.estimate(3, 4)
te.get_transfer_function()
```
To plot its bode diagram:
```python
te.plot_bode()
```
Default method to calculate the frequency response Y/U is cross density/ power density, if you want to use frequency/frequency just set the method to "fft" and specify the time length of the simulation (in seconds, default is 1):
```python
te.estimate(3, 4, method="fft", time=1)
```