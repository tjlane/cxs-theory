
"""
TJL Nov 2013
"""

import numpy as np
from scipy import sparse
from scipy import optimize

from odin import xray
from odin.math2 import Wigner3j


class GkCoefficients(object):
    """
    
    """
    
    def __init__(self, ell_max):
        self._ell_max = int(ell_max)
        self._Gk = np.zeros(self._ell_max * (self._ell_max + 1), dtype=np.complex128)
        return
        
    @property
    def ell_max(self):
        return self._ell_max
    
    @property
    def n_param(self):
        return self._ell_max * (self._ell_max + 1) + 1
        
    @property
    def n_param_compact(self):
        return (self._ell_max ** 2 + 3 * self._ell_max) / 2 + 1
        
    @property
    def linear_array(self):
        return self._Gk
        
    @property
    def real(self):
        return np.real(self._Gk)
        
    @property
    def imag(self):
        return np.imag(self._Gk)
    
    @property
    def conj(self):
        return np.conjugate(self._Gk)
    
    @property
    def compact_linear_array(self):
        
        gk_compact = []
        for k in range(len(self._Gk)):
            l,m = self._backward_index(k)
            if m >= 0:
                gk_compact.append(self._Gk[k])
                
        assert len(gk_compact) == self.n_param_compact
        
        return np.array(gk_compact)

    @property
    def lm_array(self):
        raise NotImplementedError()

    @property
    def _param_array(self):
        raise NotImplementedError()

    @classmethod
    def from_linear_array(cls, new_gk):
        """
        Create an instance from a linear non-compact g_k array.
        """
        
        if not len(new_gk.shape) == 1:
            raise ValueError('`new_gk` argument must be a one-D '
                             'array of g_k coefficients in the standard '
                             '(non-compact) k-indexing scheme')
        
        ell_max, m_max = cls._backward_index(len(new_gk) - 1)
        
        if not m_max == ell_max:
            raise ValueError('`new_gk` array does not have an appropriate'
                             'length to specify a complete set of sph hrm'
                             'coeffients. Should be len mod l*(l+2)+1, l int. '
                             'Got shape: %s, looking for: %d' % (str(new_gk.shape), ell_max*(ell_max+2)+1))
        
        instance = cls(ell_max)
        instance._Gk = new_gk
        
        return instance
    
        
    @classmethod
    def from_compact_linear_array(cls, new_gk):
        """
        Create an instance from a linear compact g_k array.
        """
        gk = np.zeros(self.n_param, dtype=np.complex128)
        
        for k in range(new_gk.shape[0]):
            
            l,m = cls._backward_index_compact(k)
            assert m >= 0
            
            kp1 = cls.forward_index(l,m)
            gk[kp1] = new_gk[k]
            
            kp2 = cls.forward_index(l,-m)
            sign = -1 ** int(np.abs(m))
            gk[kp1] = sign * np.conjugate(new_gk[k])
            
        return cls.from_linear_array(gk)
        
        
    @classmethod
    def from_lm_array(cls, new_gk):
        raise NotImplementedError()
        
        
    @classmethod
    def _from_param_array(cls, param_gk):
        """
        """

        param_gk = param_gk.reshape(len(param_gk)/2, 2) # split real and imag
        param_gk = param_gk.astype(np.complex128)

        l_max, _ = cls._backward_index_compact(len(param_gk)-1)
        nm = l_max ** 2 + 2 * l_max + 1
        Gk = np.zeros(nm, dtype=np.complex128)

        for k in range(nm):

            l,m = cls._backward_index(k)

            if m >= 0:
                k_comp = cls._forward_index_compact(l,m)
                Gk[k] = np.copy(param_gk[k_comp,0] + 1j*param_gk[k_comp,1]) # coefficient

            else: # note switch in complex conj
                k_comp = cls._forward_index_compact(l, np.abs(m))
                sign = -1 ** int(np.abs(m)) 
                Gk[k] = sign * np.copy(param_gk[k_comp,0] - 1j*param_gk[k_comp,1]) # complex conjugate
                
        return cls.from_linear_array(Gk)
    

    @staticmethod
    def _forward_index(ell, m):
        """
        Convert spherical harmonic indices (l,m) into a linear indexing scheme.
        
            (ell, m) --> k = 0,1,2,...
            
        Parameters
        ----------
        ell : int
            The angular momentum quantum number
        m : int
            The magnetic quantum number
            
        Returns
        -------
        k : int
            The new linear index corresponding to (ell,m)
            
        See Also
        --------
        backward_index : go the other way
        forward_index_compact : non-redundant linear index
        """
        return int(ell * (ell+1) + m)
    

    @staticmethod
    def _backward_index(k):
        """
        Turn a linear spherical harmonic index back into (ell,m)
        
            k --> (ell, m)
        
        Parameters
        ----------
        k : int
            The new linear index corresponding to (ell,m)

        Returns
        -------
        ell : int
            The angular momentum quantum number
        m : int
            The magnetic quantum number
            
        See Also
        --------
        forward_index_compact : go the other way
        """
        ell = int(np.floor(np.sqrt(k)))
        m = int(k - ell*(ell+1))
        return ell, m
    

    @staticmethod
    def _forward_index_compact(ell, m):
        """
        Convert spherical harmonic indices (l,m) into a linear indexing scheme
        that is non-redundant in the sense that only positive m values are
        considered. The opposite m-values (-m) can be infered from their
        positive cousins.
        
            (ell, m) --> k = 0,1,2,...
            
        Parameters
        ----------
        ell : int
            The angular momentum quantum number
        m : int
            The magnetic quantum number
            
        Returns
        -------
        k : int
            The new linear index corresponding to (ell,m).
            
        See Also
        --------
        backward_index_compact : go the other way
        forward_index : redundant linear index
        """
        return int((ell * (ell+1))/2 + m)
    

    @staticmethod
    def _backward_index_compact(k):
        """
        Turn a 'compact' linear spherical harmonic index back into (ell,m)
        
            k --> (ell, m)
        
        Parameters
        ----------
        k : int
            The new linear index corresponding to (ell,m)

        Returns
        -------
        ell : int
            The angular momentum quantum number
        m : int
            The magnetic quantum number
            
        See Also
        --------
        forward_index_compact : go the other way
        """
        ell = int(np.floor( (np.sqrt(8*k+1) - 1)/2 ))
        m = int( k - (ell*(ell+1))/2 )
        return ell, m
    
        
    def lsqd_coef(self):
        """
        Take the output from the optimizer (which has the real & complex components separated)
        and compute the ell-squared coefficient,

            ell-squared = sum{ G_{ell m} ^ 2 }
                           m
                           
        Returns
        -------
        lsqd : np.ndarray
            The ell-squared coefficient, in order for ell = 0, 1, 2, ...
        """

        Gk_mag = np.square( np.abs(self._Gk) ) # k-indexing now correct
        k_max = len(Gk_mag)

        l_max = backward_index(k_max)[0]
        lsqd = np.zeros(l_max+1)

        for k in range(k_max):
            l,m = backward_index(k)
            lsqd[l] += Gk_mag[k]

        return lsqd
    

class HTensor(object):
    """
    Note: all the matrices contained in this class are in full (non-compact)
    linear index space.
    """
    
    def __init__(self, lbd_max):
        self._lbd_max = lbd_max
        
        self._matrix_list = []
        for l in range(self._lbd_max+1):
            self._matrix_list.append(self._compute_H(l))
            
        return
        
        
    def __call__(self):
        return self._matrix_list
    
        
    def _compute_h(self, l, L, m, M, lbd):
        N = np.sqrt( (2.0*l + 1.0)*(2.0*L + 1.0)*(2.0*lbd + 1.0) / (4.0*np.pi) ) *\
            Wigner3j(l, L, lbd, 0.0, 0.0, 0.0)
        return (-1.0)**M * N * Wigner3j(l, L, lbd, m, -M, M-m)
    
        
    def _compute_H(self, lbd):
        """
        Return the matrix H_(kK)
        """
        n = (lbd ** 2) + 2 * lbd + 1 # same as l * (l+1) + m_max
        H = sparse.lil_matrix((n, n))
        
        for k in range(n):
            for K in range(k, n):

                l, m = GkCoefficients._backward_index(k)
                L, M = GkCoefficients._backward_index(k)

                h = self._compute_h(l, L, m, M, lbd)
                if h != 0.0:
                    # TJL : we could get 2x memory savings here
                    H[k,K] = h
                    H[K,k] = h
                
        return H
    
        
    @property
    def lbd_max(self):
        return self._lbd_max
        
    
    def lbd(self, lbd):
        if lbd > self._lbd_max:
            raise ValueError('`lbd` > lambda_max (%d/%d respectively)' % \
                              (lbd, self._lbd_max))
        return self._matrix_list[lbd]
        
        
    def dirac_product(self, bra, ket, lbd):
        """
        Compute the Dirac product
        
            < bra | H^{lbd} | ket >
            
        on the dense part of H^{lbd}. If the Bra/Ket lengths exceed the size of
        H^{lbd}, we employ the fact that H^{lbd} is sparse to infinity and
        dicard the excess subspace.
        
        Parameters
        ----------
        bra/ket : np.ndarray, complex
            Two one-dimensional arrays of the same length.
            
        lbd : int
            The index of the H^{lbd} matrix to use in the product
        
        Returns
        -------
        product : float
            The final Dirac product.
        """
        
        if (not bra.shape == ket.shape) or (not len(bra.shape) == 1):
            raise ValueError('`bra` and `ket` must be the same one-dimensional '
                             'shape, got %s and %s respectively.' % (str(bra.shape), str(ket.shape)))
        
        H = self.lbd(lbd)
        n = H.shape[0]
        
        if n > bra.shape[0]:
            raise ValueError('bra/ket length too small -- does not cover the '
                             'dense space of H')
        
        product = np.dot(bra[:n], H.dot(ket[:n]))
        
        return product
        


class AutocorrRegressor(object):
    
    def __init__(self, legendre_coefficients, alpha=0.1):
        
        self._legendre_coefficients = legendre_coefficients
        self._M_lbd = self._C_lbd_to_M(self._legendre_coefficients)
        self.alpha = alpha
        
        self._lbd_max = len(self._legendre_coefficients) - 1
        
        self._n_param = (self._lbd_max ** 2 + 3 * self._lbd_max) / 2 + 1
        self._HT = HTensor(self._lbd_max)
        
        return
    
    @property
    def lbd_max(self):
        return self._lbd_max
    
        
    @property
    def n_param(self):
        return self._n_param
    
    
    @staticmethod
    def _C_lbd_to_M(c_vector):
        d = np.array([2.0 * l + 1.0 for l in range(len(c_vector))])
        M = np.sqrt( np.abs(c_vector) / d)
        return M / M[1]
    
        
    def _sigma_gk(self, Gk_v):
        
        sigma = np.zeros((len(Gk_v), 2))
        
        sigma[np.real(Gk_v) > 0.0, 0]  =  1.0
        sigma[np.real(Gk_v) < 0.0, 0]  = -1.0
        
        sigma[np.imag(Gk_v) > 0.0, 1]  =  1.0
        sigma[np.imag(Gk_v) < 0.0, 1]  = -1.0
        
        # sigma[ Gk_v == 0.0 ] =  0.0 # already done implicitly
        
        return sigma
    
        
    # def _Gparam_to_Gk(self, G_param):
    #     
    #     gk = GkCoefficients._from_param_array(G_param)
    #     Gk_v = gk.linear_array
    #     Gk_s = np.conjugate(Gk_v)
    #     
    #     return Gk_v, Gk_s
    

    def _objective(self, G_param):
        
        s = G_param.shape[0]/2
        Gk_v = G_param[:s] + 1j * G_param[s:]
        Gk_s = np.conjugate(Gk_v)

        obj = 0.0
        for lbd in range(self.lbd_max+1):
            obj += np.power( np.abs(self._HT.dirac_product(Gk_s, Gk_v, lbd) - self._M_lbd[lbd]), 2)

        obj += np.sum(np.abs(G_param)) * self.alpha

        return obj
    

    def _objective_grad(self, G_param):
        
        s = G_param.shape[0]/2
        grad = np.zeros(s, dtype=np.complex)
        Gk_v = G_param[:s] + 1j * G_param[s:]
        Gk_s = np.conjugate(Gk_v)

        # derivative = 2f * f' + alpha * sigma
        for lbd in range(self.lbd_max+1):
            n = lbd * (lbd+2) + 1
            
            # f
            g = np.zeros(n, dtype=np.complex128) # tmp storage
            g += 4.0 * (self._HT.dirac_product(Gk_s, Gk_v, lbd) - self._M_lbd[lbd])
            
            # f' -- term from the chain rule
            g[:] *= self._HT.lbd(lbd).dot(Gk_v[:n]) # (Hg)_k
            
            # accumulate
            grad[:n] += g
        
        # reformat the answer
        # and add (finally) the term due to the L1 regularization
        ret = np.zeros(s*2)
        ret[:s] = np.real(grad) + self.alpha * self._sigma_gk(Gk_v)[:,0]
        ret[s:] = np.imag(grad) + self.alpha * self._sigma_gk(Gk_v)[:,1]
        
        return ret
    
        
    def optimize_coefficients(self):
    
        x = self.lbd_max**2 + 2*self.lbd_max + 1
        Gk0 = np.random.randn(x*2) / 10. # x2 for complex
        
        gk_opt = optimize.fmin_bfgs(self._objective, Gk0, 
                                    fprime=self._objective_grad,
                                    #fprime=numerical_grad,
                                    #fprime=None,
                                    maxiter=int(1e6), disp=1)

        # reshape final array into a linear complex one
        s = gk_opt.shape[0]/2
        gk = gk_opt[:s] + 1j * gk_opt[s:]
    
        return GkCoefficients.from_linear_array(gk)
    

    