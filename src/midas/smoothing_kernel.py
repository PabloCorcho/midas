import numpy as np


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

    def __init__(self, mean=0, sigma=1):
        """
        :type dim: int
            Number of dimensions
        :type h: float
            Characteristic smoothing scale
        """
        print(
            ' [KERNEL] Initialising Gaussian Kernel\n  · mean={}\n  · sigma={}'
            .format(mean, sigma))
        self.name = '2DGaussianKernel'
        self.mean = mean
        self.sigma = sigma
        self.kernel_params = {'sigma': sigma}

    def kernel(self, r):
        W = (np.exp(- 0.5 * (r-self.mean)**2 / self.sigma**2)
             / np.sqrt(2*np.pi) / self.sigma)
        return W


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    r = np.linspace(-10, 10, 100)
    # kernel = CubicSplineKernel(dim=2, h=.2)
    kernel = GaussianKernel(mean=1, sigma=.5)
    W = kernel.kernel(r)
    print(W)
    plt.plot(r, W)
    plt.show()
    plt.close()
    print('Volume integration: ', np.trapz(W, r))