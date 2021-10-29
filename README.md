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
te.estimate()
te.get_transfer_function()
```
To plot its bode diagram:
```python
te.plot_bode()
```
