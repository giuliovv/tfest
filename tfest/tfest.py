import matplotlib.pyplot as plt
import numpy as np

from matplotlib.mlab import psd, csd
from scipy.optimize import minimize
from scipy import signal

class TfEst:
    def __init__(self, u, y):
        self.u = u
        self.y = y
        self.res = None
        self.frequency = None
        self.H = None
        self.npoles = 0
        self.nzeros = 0
        self.init_value = 1

    def loss(x, npoles, freq, H):
        """
        x: array of zeros and poles
        npoles: number of poles
        freq: frequency
        H: frequency response

        return: loss
        """
        poles = x[:npoles]
        zeros = x[npoles:]
        risp = np.array([
            (1+sum([a*1j*s**i for i, a in enumerate(zeros)]))/(1+sum([b*1j*s**i for i, b in enumerate(poles)]))
            for s in freq])
        return np.linalg.norm(H-risp).sum() # + np.abs(x.sum())

    def frequency_response(self):
        """
        return: frequency response and frequency
        """
        cross_sd, frequency = csd(self.y, self.u)
        power_sd, _ = psd(self.u)
        self.frequency = frequency
        self.H = cross_sd/power_sd
        return self.H, frequency

    def estimate(self, npoles, nzeros, init_value=1, options={'xatol': 1e-4, 'disp': True}):
        """
        npoles: number of poles
        nzeros: number of zeros
        init_value: initial value for optimization
        options: options for scipy.optimize.minimize

        return: scipy.optimize.minimize.OptimizeResult
        """
        self.npoles = npoles
        self.nzeros = nzeros
        self.init_value = init_value

        x0 = [init_value]*(npoles+nzeros)
        H, frequency = self.frequency_response()
        pass_to_loss = lambda x: self.loss(x, npoles, frequency, H)
        self.res = minimize(pass_to_loss, x0, method='nelder-mead', options=options)
        return self.res

    def get_transfer_function(self):
        """
        return: transfer function
        """
        if self.res == None:
            raise Exception("Please run .estimate(npoles, nzeros) before plotting.")
        zeros = self.res.x[:self.npoles]
        poles = self.res.x[self.npoles:]
        return signal.TransferFunction(zeros, poles)
    
    def plot(self):
        """
        Plot the frequency response and the transfer function
        """
        if self.res == None:
            raise Exception("Please run .estimate(npoles, nzeros) before plotting.")
        zeros = self.res.x[:self.npoles]
        poles = self.res.x[self.npoles:]
        risp = np.array([
            (1+sum([a*1j*s**i for i, a in enumerate(zeros)]))/(1+sum([b*1j*s**i for i, b in enumerate(poles)]))
            for s in self.frequency])
        plt.plot(-np.log(risp), label="estimation")
        plt.plot(np.log(self.H), label="train data")
        plt.legend(loc="upper right")
        plt.show()