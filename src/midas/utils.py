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
