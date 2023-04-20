#from numba import njit
import numpy as np

#@njit
def fast_vector_norm(x, y):
    return np.sqrt(x*x + y*y)

#@njit
def fast_interpolation(f, dx, x, xnew, dxnew):
    cumulative = np.cumsum(f * dx)
    interpolation = np.interp(xnew, x, cumulative)
    fnew = np.diff(interpolation) / dxnew
    return fnew

def inter_2d(array, x_range, y_range, x_inter, y_inter, extrapolate=0.0, log=False):
    """"This function performs 2D linear interpolation on a given 2D array."""
    if (x_inter > max(x_range) or x_inter < min(x_range)
        or y_inter > max(y_range) or y_inter < min(y_range)):
        print("Value out of range")
        return extrapolate
    else:
        indx1 = np.searchsorted(x_range, x_inter)
        indx2 = np.searchsorted(y_range, y_inter)
        # Linear or logarithmic interpolation
        if not log:
            w1 = (x_inter - x_range[indx1 - 1]) / (x_range[indx1] - x_range[indx1 - 1])
            w2 = (y_inter - y_range[indx2 - 1]) / (y_range[indx2] - y_range[indx2 - 1])
        else:
            w1 = np.log(x_inter / x_range[indx1 - 1]) / np.log(x_range[indx1] / x_range[indx1 - 1])
            w2 = np.log(y_inter / y_range[indx2 - 1]) / np.log(y_range[indx2] / y_range[indx2 - 1])
        
        interp_array =  (w1  * w2 * array[:, indx1, indx2]
                + (1 - w1) * w2 * array[:, indx1 - 1, indx2]
                + w1 * (1 - w2) * array[:, indx1, indx2 - 1]
                + (1 - w1) * (1 - w2) * array[:, indx1 - 1, indx2 - 1])
        return interp_array

def gaussian1d_conv(f, sigma, deltax):
    """Apply a gaussian convolution to a 1D array f(x).

    params
    ------
    - f: (array) 1D array containing the data to be convolved with.
    - sigma (array) 1D array containing the values of sigma at each value of x
    - deltax: (float) Step size of x in "physical" units.
    """
    sigma_pixels = sigma / deltax
    pix_range = np.arange(0, f.size, 1)
    if len(pix_range) < 2e4:
        XX = pix_range[:, np.newaxis] - pix_range[np.newaxis, :]
        g = np.exp(- (XX)**2 / 2 / sigma_pixels[np.newaxis, :]**2) / (
                   sigma_pixels[np.newaxis, :] * np.sqrt(2 * np.pi))
        g /= g.sum(axis=1)[:, np.newaxis]
        f_convolved = np.sum(f[np.newaxis, :] * g, axis=1)
    else:
        print(' WARNING: TOO LARGE ARRAY --- APPLYING SLOW CONVOLUTION METHOD ---')
        f_convolved = np.zeros_like(f)
        for pixel in pix_range:
            XX = pixel - pix_range
            g = np.exp(- (XX)**2 / 2 / sigma_pixels**2) / (
                       sigma_pixels * np.sqrt(2 * np.pi))
            g /= g.sum()
            f_convolved[pixel] = np.sum(f * g)
    return f_convolved

def gaussian(x, a, m, s):
    """Gaussian profile."""
    i = a / np.sqrt(2 * np.pi) / s**2 * np.exp(- (x - m)**2 / s**2 / 2)
    return i

def gaussian1d_conv(f, sigma, deltax):
    """Apply a gaussian convolution to a 1D array f(x).

    params
    ------
    - f: (array) 1D array containing the data to be convolved with.
    - sigma (array) 1D array containing the values of sigma at each value of x
    - deltax: (float) Step size of x in "physical" units.
    """
    sigma_pixels = sigma / deltax
    pix_range = np.arange(0, f.size, 1)
    if len(pix_range) < 2e4:
        XX = pix_range[:, np.newaxis] - pix_range[np.newaxis, :]
        g = np.exp(- (XX)**2 / 2 / sigma_pixels[np.newaxis, :]**2) / (
                   sigma_pixels[np.newaxis, :] * np.sqrt(2 * np.pi))
        g /= g.sum(axis=1)[:, np.newaxis]
        f_convolved = np.sum(f[np.newaxis, :] * g, axis=1)
    else:
        print(' WARNING: TOO LARGE ARRAY --- APPLYING SLOW CONVOLUTION METHOD ---')
        f_convolved = np.zeros_like(f)
        for pixel in pix_range:
            XX = pixel - pix_range
            g = np.exp(- (XX)**2 / 2 / sigma_pixels**2) / (
                       sigma_pixels * np.sqrt(2 * np.pi))
            g /= g.sum()
            f_convolved[pixel] = np.sum(f * g)
    return f_convolved

if __name__ == '__main__':
    from time import time
    import numpy as np
    a = np.ones((200, 100))
    b = np.ones((200, 100))
    s = time()
    c = fast_vector_norm(a, b)
    e = time()
    print('Initial: ', e - s)

    s = time()
    c = fast_vector_norm(a, b)
    e = time()
    print('After compilation: ', e - s)
    ###
    print('INTERPOLATION')
    x = np.linspace(10, 100, 100000)
    newx = np.linspace(30, 40, 100000)
    f = np.sin(x)
    s = time()
    c = fast_interpolation(f, np.diff(x)[0], x, newx, np.diff(newx)[0])
    e = time()
    print('Initial: ', e - s)
    s = time()
    c = fast_interpolation(f, np.diff(x)[0], x, newx, np.diff(newx)[0])
    e = time()
    print('After: ', e - s)

# Mr Krtxo \(ﾟ▽ﾟ)/