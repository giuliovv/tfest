import matplotlib.pyplot as plt
import numpy as np

from matplotlib.mlab import psd, csd
from scipy.optimize import minimize

class TfEst:
    def __init__(self, u, y):
        self.u = u
        self.y = y

    def loss(x, npoles, freq, H):
        poles = x[:npoles]
        zeros = x[npoles:]
        risp = np.array([
            (1+sum([a*1j*s**i for i, a in enumerate(zeros)]))/(1+sum([b*1j*s**i for i, b in enumerate(poles)]))
            for s in freq])
        return np.linalg.norm(H-risp).sum()
        #  + np.abs(x.sum())

    def frequency_response(self):
        cross_sd, frequency = csd(self.y, self.u)
        power_sd, _ = psd(self.u)
        return cross_sd/power_sd, frequency

    def estimate(self, npoles, nzeros, init_value=1):
        x0 = [init_value]*(npoles+nzeros)
        H, frequency = self.frequency_response()
        pass_to_loss = lambda x: self.loss(x, npoles, frequency, H)
        res = minimize(pass_to_loss, x0, method='nelder-mead', options={'xatol': 1e-4, 'disp': True})
        return res