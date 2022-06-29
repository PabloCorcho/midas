import numpy as np
from numba import njit

class CubicSplineKernel(object):

    def __init__(self, dim=1, h=2):
        """
        :type dim: int
            Number of dimensions
        :type h: float
            Characteristic smoothing scale
        """
        print(
            ' [KERNEL] Initialising Cubic Spline Kernel\n  · dim={}\n  · h={}'
            .format(dim, h))
        self.name = 'CubicSplineKernel'
        self.dim = dim
        self.h = h
        self.kernel_params = {'h':h, 'dim':dim}
        self.get_norm()

    def get_norm(self):
        if self.dim == 1:
            self.norm = 2 / 3 / self.h
        elif self.dim == 2:
            self.norm = 10 / 7 / np.pi / self.h ** 2
        elif self.dim == 3:
            self.norm = 1 / np.pi / self.h ** 3


    def kernel(self, r):
        q = r / self.h
        W = np.empty_like(q)
        mask = q <= 1
        W[mask] = 1 - 1.5 * q[mask]**2 + .75 * q[mask]**3
        mask = (q > 1) & (q <= 2)
        W[mask] = 0.25 * (2 - q[mask]) ** 3
        W[q > 2] = 0
        return W * self.norm


class GaussianKernel(object):

    def __init__(self, mean=0, sigma=1, sigma_trunc=3):
        """
        :type dim: int
            Number of dimensions
        :type h: float
            Characteristic smoothing scale
        """
        print(
            ' [KERNEL] Initialising Gaussian Kernel\n  · mean={}\n  · sigma={}\n  · sigma_trunc={}'
            .format(mean, sigma, sigma_trunc))
        self.name = '2DGaussianKernel'
        self.mean = mean
        self.sigma = sigma
        self.sigma_trunc = sigma_trunc
        self.kernel_params = {'sigma': sigma}

    @staticmethod
    @njit
    def kernel(r, sigma, mean=0, sigma_trunc=3):
        w = np.zeros(r.shape, dtype=np.float32)
        e = (r-mean)**2 / sigma**2
        in_trunc = np.where(e < sigma_trunc**2)[0]
        w[in_trunc] = (np.exp(- 0.5 * e[in_trunc]) / np.sqrt(2*np.pi) / sigma)
        w /= (np.sum(w[in_trunc]) + 1e-10)
        return w


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from time import time
    r = np.linspace(-10, 10, 100000)
    # kernel = CubicSplineKernel(dim=2, h=.2)
    kernel = GaussianKernel(mean=1, sigma=.5)
    s = time()
    W = kernel.kernel(r, kernel.mean, kernel.sigma)
    e = time()
    print('Time Initialization: ', e-s)
    s = time()
    W = kernel.kernel(r, kernel.mean, kernel.sigma)
    e = time()
    print('Time after compilation: ', e-s)
    print('Volume integration: ', np.trapz(W, r))

# Mr Krtxo \(ﾟ▽ﾟ)/