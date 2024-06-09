import numpy as np

def getOptParam(nq):
    nq = np.atleast_1d(nq)

    list = []
    for n in nq:
        list.append(np.load("params/chain_nq_"+str(n)+".npy"))
    
    if len(nq)==1:
        return list[0]
    else:
        return list

def expWrapperPlus(_g,d):
    """
    Helper Function to turn coupling parameters g for the exponential coupling model
    into couplings from the Hamiltonian. Here, the d^1/3 scaling of the exponential ramps is hard-coded into
    the function.
    Input:
        g : numpy array of length 4: (g1,g2,g3,lambda1,lambda2)
        d : length of the chain
    Returns:
        g_full : Hamiltonian couplings
    """
    # Ensure _g is 2d
    _g = np.atleast_2d(_g)

    # Functional form for the couplings
    g_of_n = lambda n : -_g[:,0]*np.exp(-n/(_g[:,3]*(d/10)**(1/3))) + _g[:,1] + (_g[:,2])*np.exp((n-(d-2))/(_g[:,4]*(d/10)**(1/3)))

    # Returns SQRT of the correct coupling
    return g_of_n(np.linspace(0,d-2,d-1))

def expWrapperPlusSingleRamp(_g,d):
    """
    Note: Only includes the LEFT ramp
    Helper Function to turn coupling parameters g for the exponential coupling model
    into couplings from the Hamiltonian. Here, the d^1/3 scaling of the exponential ramps is hard-coded into
    the function.
    Input:
        g : numpy array of length 4: (g1,g2,g3,lambda1,lambda2)
        d : length of the chain
    Returns:
        g_full : Hamiltonian couplings
    """
    # Ensure _g is 2d
    _g = np.atleast_2d(_g)

    # Functional form for the couplings
    g_of_n = lambda n : -_g[:,0]*np.exp(-n/(_g[:,3]*(d/10)**(1/3))) + _g[:,1]

    # Returns SQRT of the correct coupling
    return g_of_n(np.linspace(0,d-2,d-1))

def expWrapper(_g,d):
    """
    Helper Function to turn coupling parameters g for the exponential coupling model
    into couplings from the Hamiltonian
    Input:
        g : numpy array of length 4: (g1,g2,g3,lambda1,lambda2)
        d : length of the chain
    Returns:
        g_full : Hamiltonian couplings
    """
    # Ensure _g is 2d
    _g = np.atleast_2d(_g)

    # Functional form for the couplings
    g_of_n = lambda n : -_g[:,0]*np.exp(-n/_g[:,3]) + _g[:,1] + (_g[:,2])*np.exp((n-(d-2))/_g[:,4])

    # Returns SQRT of the correct coupling
    return g_of_n(np.linspace(0,d-2,d-1))

def expWrapperSingleRamp(_g,d):
    """
    Note: Only includes the LEFT ramp
    Helper Function to turn coupling parameters g for the exponential coupling model
    into couplings from the Hamiltonian
    Input:
        g : numpy array of length 4: (g1,g2,g3,lambda1,lambda2)
        d : length of the chain
    Returns:
        g_full : Hamiltonian couplings
    """
    # Ensure _g is 2d
    _g = np.atleast_2d(_g)

    # Functional form for the couplings
    g_of_n = lambda n : -_g[:,0]*np.exp(-n/_g[:,3]) + _g[:,1]

    # Returns SQRT of the correct coupling
    return g_of_n(np.linspace(0,d-2,d-1))
    
def getExpParam(nq,mode="standard",oneramp=False):
    """
    Obtain exponential parameters from data
        nq  :   number of sites
      mode  :   "standard" or "plus" (plus is where d^1/3 is separated out)
   oneramp  :   False for standard potential, True for single-sided
    """
    nq = np.atleast_1d(nq)

    list = []
    for n in nq:
        if mode == "standard":
            tmp = np.load("params_exp/chain_exp_nq_"+str(n)+".npy")
            if oneramp:
                list.append(expWrapperSingleRamp(tmp,n))
            else:
                list.append(expWrapper(tmp,n))
        elif mode == "plus":
            tmp = np.load("params_expPlus/chain_exp_nq_"+str(n)+".npy")
            if oneramp:
                list.append(expWrapperPlusSingleRamp(tmp,n))
            else:
                list.append(expWrapperPlus(tmp,n))
        else:
            raise Exception("Error: mode "+mode+" is unknown in call to getExpParam().\nPlease use either mode=\"standard\" or mode=\"plus\"")

    if len(nq)==1:
        return list[0]
    else:
        return list