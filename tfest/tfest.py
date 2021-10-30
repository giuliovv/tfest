import matplotlib.pyplot as plt
import numpy as np

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

    def loss(self, x, nzeros, freq, H):
        """
        x: array of zeros and poles
        nzeros: number of zeros
        freq: frequency
        H: frequency response

        return: loss
        """
        zeros = x[:nzeros]
        poles = x[nzeros:]
        risp = np.array([
            (sum([a*(1j*s)**i for i, a in enumerate(zeros)]))/(sum([b*(1j*s)**i for i, b in enumerate(poles)]))
            for s in freq])
        return np.square(np.linalg.norm((risp-H).reshape(-1, 1), axis=1)).sum() #+ np.abs(x).sum()

    def frequency_response(self, method="density", time=1):
        """
        method: "fft" or "density"
        time: time for fft

        return: frequency response and frequency
        """
        if method == "fft":
            u_f = fft(self.u)
            y_f = fft(self.y)
            u_no_zero = u_f[np.nonzero(u_f)]
            y_no_zero = y_f[np.nonzero(u_f)]
            frequency = fftfreq(len(u_no_zero), time/len(u_no_zero))
            H = y_no_zero/u_no_zero
        elif method == "density":
            cross_sd, frequency = csd(self.y, self.u)
            power_sd, _ = psd(self.u)
            H = cross_sd/power_sd
        self.frequency = frequency
        self.H = H
        return self.H, frequency

    def estimate(self, nzeros, npoles, init_value=1, options={'xatol': 1e-2, 'disp': True}, method="density", time=1):
        """
        npoles: number of poles
        nzeros: number of zeros
        init_value: initial value for optimization
        options: options for scipy.optimize.minimize
        method: "fft" or "density"
        time: time for fft

        return: scipy.optimize.minimize.OptimizeResult
        """
        npoles += 1
        nzeros += 1
        self.npoles = npoles
        self.nzeros = nzeros
        self.init_value = init_value

        x0 = [init_value]*(npoles+nzeros)
        H, frequency = self.frequency_response(method=method, time=time)
        pass_to_loss = lambda x: self.loss(x, nzeros, frequency, H)
        self.res = minimize(pass_to_loss, x0, method='nelder-mead', options=options)
        return self.res

    def get_transfer_function(self):
        """
        return: transfer function
        """
        if self.res == None:
            raise Exception("Please run .estimate(npoles, nzeros) before plotting.")
        zeros = list(reversed(self.res.x[:self.nzeros]))
        poles = list(reversed(self.res.x[self.nzeros:]))
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
        risp = np.array([
            (sum([a*(1j*s)**i for i, a in enumerate(zeros)]))/(sum([b*(1j*s)**i for i, b in enumerate(poles)]))
            for s in self.frequency])
        plt.plot(np.log(risp), label="estimation")
        plt.plot(np.log(self.H), label="train data")
        plt.legend(loc="upper right")
        plt.grid()
        plt.show()