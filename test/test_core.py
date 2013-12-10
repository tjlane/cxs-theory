
"""
Tests for cxsinfer.core
"""


import numpy as np
from numpy.testing import *
import numdifftools as nd

from cxsinfer import core


class TestAutocorrRegressor(object):
    
    def setup(self):
        
        n_coefficients = 3 # this is lambdas
        self.legendre_coefficients = np.abs(np.random.randn(n_coefficients))
        self.ar = core.AutocorrRegressor(self.legendre_coefficients)
        
        return
        
    def test_analytical_gradient(self):

        trials = 5

        numerical_grad = nd.Gradient(self.ar._objective)
        analytic_grad  = self.ar._objective_grad

        x = self.ar.lbd_max**2 + 2*self.ar.lbd_max + 1
        for i in range(trials):
            g = np.random.randn(x*2)
            assert_allclose(analytic_grad(g), numerical_grad(g))

