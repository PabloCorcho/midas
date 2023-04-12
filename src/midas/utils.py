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

def inter_2d(array, x_range, y_range, x_inter, y_inter, extrapolate=0.0):
    """"This function performs 2D linear interpolation on a given 2D array."""
    if (x_inter > max(x_range) or x_inter < min(x_range)
        or y_inter > max(y_range) or y_inter < min(y_range)):
        print("Value out of range")
        return extrapolate
    else:
        indx1 = np.searchsorted(x_range, x_inter)
        indx2 = np.searchsorted(y_range, y_inter)
        
        # weights
        w1 = (x_inter - x_range[indx1 - 1]) / (x_range[indx1] - x_range[indx1 - 1])
        w2 = (y_inter - y_range[indx2 - 1]) / (y_range[indx2] - y_range[indx2 - 1])
    
        interp_array =  (w1  * w2 * array[indx1, indx2,:]
                + (1 - w1) * w2 * array[indx1 - 1, indx2, :]
                + w1 * (1 - w2) * array[indx1, indx2 - 1, :]
                + (1 - w1) * (1 - w2) * array[indx1 - 1, indx2 - 1, :])
        return interp_array


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