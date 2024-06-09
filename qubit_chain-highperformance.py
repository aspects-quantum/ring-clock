import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

from scipy.integrate import odeint, complex_ode


from scipy.optimize import minimize
from snr_toolbox import SNR

from qubit_chain_params import expWrapper, getExpParam, expWrapperPlus

from multiprocessing import Pool

w = 2.0
bH = 1e-1
bC = np.inf
gH = 1e-1
gC = 1e-1

Gamma = np.array([1e0])


def mySNR(g):
    n_qubits = len(g)+1
    dim = n_qubits

    # Set Hamiltonian
    # H0
    H0 = np.eye(dim)*w

    # HI
    H0 += np.diag(g,-1) + np.diag(g,1)
    
    # # Inverse beta thermalization
    # JH = np.zeros((dim,dim))
    # JH[1,0] = np.sqrt(gH)

    # Set Tick Operator
    JTick = []
    for k,GammaK in enumerate(Gamma):
        # Generate jump matrix
        _JTick = np.zeros((dim,dim))
        _JTick[0,-k-1] = np.sqrt(GammaK)

        JTick.append(_JTick)

    # Set initial state
    rho0 = np.zeros((dim,dim))
    rho0[0,0] = 1.0

    return SNR(H0,[],JTick,rho0,method="diag")[1]

def func_vec(y):
    return -np.apply_along_axis(mySNR,1,np.atleast_2d(y)**2)

def optimizeCouplings(g):
    params_init = np.sqrt(g)

    res = minimize(func_vec,params_init)

    print(res.message)

    params_opt = res.x**2

    return params_opt

def optimizeExpCouplings(g,_d,mode="standard"):
    """
    Returns optimized coupling parameters g for the exponential coupling model
    Input:
        g : initial values, numpy array of length 4: (g1,g2,g3,lambda1,lambda2)
        d : length of the chain, should be >5.
     mode : "standard" or "plus"
    Returns:
        g_opt : optimized couplings
    """
    # Ensure array-type and sorted array
    _d = np.sort(np.atleast_1d(_d))

    for k,d in enumerate(_d):
        # Helper Functions
        # Returns SNR for given exp coupling parameters
        if mode == "standard":
            wrapper = expWrapper
            def expSNRWrapper(_g):
                return -np.apply_along_axis(mySNR,1,np.atleast_2d(expWrapper(_g,d)))
        elif mode == "plus":
            wrapper = expWrapperPlus
            def expSNRWrapper(_g):
                return -np.apply_along_axis(mySNR,1,np.atleast_2d(expWrapperPlus(_g,d)))

        # Initialize Parameters
        if k == 0:
            params_init = g.copy()
        else:
            if mode == "standard":
                params_init = np.load("params_exp/chain_exp_nq_"+str(_d[k-1])+".npy")
            elif mode == "plus":
                params_init = np.load("params_expPlus/chain_exp_nq_"+str(_d[k-1])+".npy")

        bound_param = 1e-1
        bounds_upper = params_init * (1+bound_param)
        bounds_lower = params_init * (1-bound_param)

        bounds = []
        for bl, bu in zip(bounds_lower,bounds_upper):
            bounds.append((bl,bu))

        res = minimize(expSNRWrapper,params_init, method="Nelder-Mead",bounds=bounds,options={"adaptive":True}) # method="L-BFGS-B",bounds=bounds,options={"eps": 1e-3})

        # print(res.message)

        params_opt = res.x

        print("\nResult for d="+str(d))
        print("init. val SNR: ",mySNR(wrapper(params_init,d)))
        print("optimized SNR: ",mySNR(wrapper(params_opt,d)))
        # print(params_opt)

        if mode == "standard":
            np.save("params_exp/chain_exp_nq_"+str(d)+".npy",params_opt)
        elif mode == "plus":
            np.save("params_expPlus/chain_exp_nq_"+str(d)+".npy",params_opt)

    pass

def generate_init(nq,k=1):
    nq = int(nq)
    temp = np.load("params/chain_nq_"+str(nq-k)+".npy")
    if (nq-k+1)%2==0:
        nq = nq - k + 1
        mid = np.ones((k))*temp[int(nq/2-1)]
        return np.concatenate((temp[:int(nq/2-1)],mid,temp[int(nq/2-1):]))
    else:
        nq = nq - k + 1
        mid = np.ones((k))*temp[int((nq-1)/2-1)]
        return np.concatenate((temp[:int((nq-1)/2)],mid,temp[int((nq-1)/2):]))

def optimizeCouplingRange(nq_min,nq_max):
    """
    Optimizes coupling rates for given range of qubits,
        nq = nq_min, nq_min+1, ..., nq_max-1.
    Results are saved in params/chain_nq_??.npy.
    """
    nq_range = np.arange(nq_min,nq_max)

    start = time.time()

    res_params = []
    for n in nq_range:
        start_temp = time.time()
        print("Start optimization for nq =",n,"chain clock")
        
        temp = optimizeCouplings(generate_init(n))

        np.save("params/chain_nq_"+str(n)+".npy",temp)
        
        end_temp = time.time()

        res_params.append(temp)
        print("old SNR:",mySNR(generate_init(n)))
        print("new SNR:",mySNR(res_params[-1]))
        print("duration:",np.round(end_temp-start_temp,3),"s\n")

    end = time.time()

    print("\neval time:",np.round(end-start,3),"s")

    pass

if __name__ == "__main__":
    optimizeExpCouplings(np.load("params_expPlus/chain_exp_nq_10.npy"),np.arange(10,201,1),mode="plus")
    optimizeExpCouplings(np.load("params_expPlus/chain_exp_nq_200.npy"),np.arange(200,501,10),mode="plus")
    optimizeExpCouplings(np.load("params_expPlus/chain_exp_nq_500.npy"),np.arange(520,1001,20),mode="plus")
    optimizeExpCouplings(np.load("params_expPlus/chain_exp_nq_1000.npy"),np.arange(1050,2001,50),mode="plus")
