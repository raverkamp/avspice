"""unit tests for util"""

import unittest
from math import sin, cos
import numpy as np
from avspice.util import drange, ndiffn, ndiff


class Test_ndiff(unittest.TestCase):
    """test for transistor"""

    def test_ndiff(self):
        def f(x):
            return sin(x)

        for x in drange(0.1, 2, 0.1):
            df = ndiff(f, x)
            # 1e-8 as limit just works
            self.assertTrue(abs(cos(x) - df) < 1e-8)

    def test_ndiffn(self):
        def f(x):
            res = np.zeros(3)
            res[0] = sin(x[0] + 2 * x[1] + 3 * x[2])
            res[1] = 3 * x[0] ** 2 - 7 / x[1] + x[2]
            res[2] = x[0] * x[1] ** 2 / x[2]
            return res

        def df(x):
            res = np.zeros((3, 3))
            res[0][0] = cos(x[0] + 2 * x[1] + 3 * x[2])
            res[0][1] = 2 * cos(x[0] + 2 * x[1] + 3 * x[2])
            res[0][2] = 3 * cos(x[0] + 2 * x[1] + 3 * x[2])

            res[1][0] = 6 * x[0]
            res[1][1] = 7 / x[1] ** 2
            res[1][2] = 1

            res[2][0] = x[1] ** 2
            res[2][1] = x[0] * 2 * x[1] / x[2]
            res[2][2] = -x[0] * x[1] ** 2 / x[2] ** 2
            return res

        for x0 in drange(0.1, 2, 0.1):
            for x1 in drange(0.1, 2, 0.1):
                for x2 in drange(0.1, 2, 0.1):
                    x = np.zeros(3)
                    x[0] = x0
                    x[1] = x1
                    x[2] = x2
                    ndf = ndiffn(f, x)
                    cdf = df(x)
                    dd = ndf - cdf
                    a = True
                    for i0 in range(2):
                        for i1 in range(2):
                            # 1e-5 just works
                            a = a and abs(dd[i0][i1]) < 1e-5
                    if not a:
                        print(x)
                        print(ndf)
                        print(cdf)
                    self.assertTrue(a)


if __name__ == "__main__":
    unittest.main()
