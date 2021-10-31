import matplotlib.pyplot as plt
import numpy as np
import warnings

from matplotlib.mlab import psd, csd
from scipy.optimize import minimize
from scipy import signal

from numpy.fft import fft, fftfreq

class tfest:
    def __init__(self, u, y):
        self.u = u
        self.y = y
        self.res = None
        self.frequency = None
        self.H = None
        self.npoles = 0
        self.nzeros = 0
        self.init_value = 1

    def loss(self, x, nzeros, freq, H, l=0):
        """
        x: array of zeros and poles
        nzeros: number of zeros
        freq: frequency
        H: frequency response
        l: L1 norm

        return: loss
        """
        zeros = x[:nzeros]
        poles = x[nzeros:]
        risp = np.array([np.polyval(zeros, s) for s in 1j*freq])
        risp /= np.array([np.polyval(poles, s) for s in 1j*freq])
        if l == 0:
            L = 0
        else:
            L = l*np.sqrt((x**2).sum())
        return np.linalg.norm((risp-H).reshape(-1, 1), axis=1).sum() + L

    def frequency_response(self, method="h1", time=None):
        """
        method: "fft" or "h1" (default) or "h2"
        time: time for fft

        return: frequency response and frequency
        """
        if time == None:
            warnings.warn("Setting default time=1")
            time = 1
        dt = time/len(self.u)
        if method == "fft":
            u_f = fft(self.u)
            y_f = fft(self.y)
            u_no_zero = u_f[np.nonzero(u_f)]
            y_no_zero = y_f[np.nonzero(u_f)]
            frequency = fftfreq(u_f.size, d=dt)[np.nonzero(u_f)]
            H = y_no_zero/u_no_zero
        elif method == "h1":
            # https://dsp.stackexchange.com/questions/71811/understanding-the-h1-and-h2-estimators
            cross_sd, frequency = csd(self.u, self.y, Fs=1/dt, NFFT=len(self.u))
            power_sd, _ = psd(self.u, Fs=1/dt, NFFT=len(self.u))
            H = cross_sd/power_sd
        elif method == "h2":
            cross_sd, frequency = csd(self.y, self.u, Fs=1/dt, NFFT=len(self.u))
            power_sd, _ = psd(self.y, Fs=1/dt, NFFT=len(self.u))
            H = power_sd/cross_sd
        else:
            raise Exception("Unknown method")
        self.frequency = frequency
        self.H = H
        return self.H, frequency

    def estimate(self, nzeros, npoles, init_value=1, options={'xatol': 1e-3, 'disp': True}, method="h1", time=None, l=0):
        """
        npoles: number of poles
        nzeros: number of zeros
        init_value: initial value for optimization
        options: options for scipy.optimize.minimize
        method: "fft" or "density"
        time: time for fft
        l: L1 norm

        return: scipy.optimize.minimize.OptimizeResult
        """
        npoles += 1
        nzeros += 1
        self.npoles = npoles
        self.nzeros = nzeros
        self.init_value = init_value

        x0 = [init_value]*(npoles+nzeros)
        H, frequency = self.frequency_response(method=method, time=time)
        pass_to_loss = lambda x: self.loss(x, nzeros, frequency, H, l)
        self.res = minimize(pass_to_loss, x0, method='nelder-mead', options=options)
        return self.res

    def get_transfer_function(self):
        """
        return: transfer function
        """
        if self.res == None:
            raise Exception("Please run .estimate(npoles, nzeros) before plotting.")
        zeros = self.res.x[:self.nzeros]
        poles = self.res.x[self.nzeros:]
        return signal.lti(zeros, poles)

    def plot_bode(self):
        """
        Plot the bode diagram
        """
        if self.res == None:
            raise Exception("Please run .estimate(npoles, nzeros) before plotting.")
        tf = self.get_transfer_function()
        w, mag, phase = tf.bode()
        plt.figure()
        plt.title('Bode magnitude plot')
        plt.semilogx(w, mag)
        plt.grid()
        plt.figure()
        plt.title('Bode phase plot')
        plt.semilogx(w, phase)
        plt.grid()
        plt.show()
           
    def plot(self):
        """
        Plot the frequency response and the transfer function
        """
        if self.res == None:
            raise Exception("Please run .estimate(npoles, nzeros) before plotting.")
        zeros = self.res.x[:self.nzeros]
        poles = self.res.x[self.nzeros:]
        omega = np.array([(1j*s) for s in self.frequency])
        risp = np.array([np.polyval(zeros, s) for s in omega])
        risp /= np.array([np.polyval(poles, s) for s in omega])
        plt.plot(np.log(risp), label="estimation")
        plt.plot(np.log(self.H), label="train data")
        plt.legend(loc="upper right")
        plt.grid()
        plt.show()