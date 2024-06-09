import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.linalg import null_space
from scipy.integrate import solve_ivp

from scipy.sparse import csr_matrix, csr_array
from scipy.sparse.linalg import eigs, spsolve

from snr_toolbox import superL, SNR, notick_evolution, tiltedSuperL, generalizedSNR, tiltedEvals
from qubit_chain_params import getOptParam, getExpParam, expWrapper

from qutip_extension import diss

class RingClock:
    # Class to instantiate qubit chain clock
    # Here: simplified description with only d dimensions
    def __init__(self,g,Gamma,s):
        """
        Input:
            g      : coherent coupling
            Gamma  : clock decay-tick rate
            s      : entropy per tick
        """
        self.n_qubits = len(g) + 1
        self.updateClock(g,Gamma,s)
        pass

    def updateClock(self,g,Gamma,s):
        """
        Set desired clock parameters
        """
        # Set Parameters
        self.s = s
        self.Gamma = Gamma
        self.g = g

        # Initialize clock with given params
        self.generateOperators()
        pass

    def set_H(self):
        """
        Generates and returns interaction Hamiltonian between the qubit sites,
        """
        # Full dimensionality
        dim = self.n_qubits

        _g = np.atleast_1d(self.g)

        # Generic case
        H = np.diag(_g,-1) + np.diag(_g.conj(),1)

        # Set
        self.H = H
        pass
    
    def set_JTick(self):
        """
        Generates and returns dissipator for the ticking and unticking
        """
        # Full dimensionality
        dim = self.n_qubits
    
        # Generate jump matrix
        _JTick = np.zeros((dim,dim))
        _JTick[0,-1] = np.sqrt(self.Gamma)

        self.JTick_fw = [_JTick]
        self.JTick_bw = [_JTick.conj().T * np.exp(-self.s / 2.0)]
            
        pass

    def generateOperators(self):
        """
        Generates all the necessary clock operators with latest
        instantiated parameters
        """
        dim = self.n_qubits

        # Set Hamiltonian
        self.set_H()

        # Set dissipators
        self.set_JTick()

        # Set Lindblad Superoperator
        # self.superL = superL(self.H,[],self.JTick_fw)

        # Set initial state
        _rho0 = np.zeros((dim,dim))
        _rho0[0,0] = 1.0

        self.rho0 = _rho0
        pass

    def setSteadyState(self):
        """
        Obtain Clock steady-state
        """
        revSuperL = tiltedSuperL(self.H,[],self.JTick_fw,self.JTick_bw,0.0,sparse=True)

        # Calculate steady-state and normalize
        rhoSS = eigs(revSuperL,k=1,sigma=0.0,which="LM")[1].reshape(self.rho0.shape)

        self.rhoSS = rhoSS / rhoSS.trace()

        # rhoSS = np.reshape(null_space(revSuperL)[:,0],self.rho0.shape)
        # self.rhoSS = rhoSS / np.trace(rhoSS)

        pass

    def getEntropyProductionRate(self):
        """
        Calculate Clock entropy production
        """
        # Generate Clock steady-state
        self.setSteadyState()

        _shape = self.rhoSS.shape

        # Create vector-shaped steady-state
        rhoSS_vector = self.rhoSS.ravel()

        # Net dissiüator for the tick
        JNet = self.JTick_fw[0].conj().T.dot(self.JTick_fw[0]) - self.JTick_bw[0].conj().T.dot(self.JTick_bw[0])

        return np.real_if_close(self.s * (JNet.dot(self.rhoSS)).trace())

    def getGeneralizedSNR(self,sparse=False):
        """
        Obtain Clock SNR and avgN for current parameters

        Output:
            SNR, avgN
        """
        self.setSteadyState()

        return generalizedSNR(self.H,[],self.JTick_fw,self.JTick_bw,rhoSS=self.rhoSS,sparse=sparse)

    def getTiltedEvals(self,sparse=False):
        """
        Obtain Clock SNR and avgN for current parameters

        Output:
            SNR, avgN
        """
        self.setSteadyState()

        return tiltedEvals(self.H,[],self.JTick_fw,self.JTick_bw,rhoSS=self.rhoSS,sparse=sparse)

    def getSNR(self,method="lyapunov"):
        """
        Obtain Clock SNR for current parameters
        Input:
            method : "lyapunov" or "lindbladian"
        Output:
            nu, N
        """
        return SNR(self.H,[],self.JTick_fw,self.rho0,method=method)
    
    def getTimeEvol(self,times,psi0=None):
        """
        Obtain time evolved state

        Input:
            times: (1,) shaped np.array of evaluation times
        
        Output:
            t, psi(t)  (shapes (len,) and (nq,len))
        """
        # Generate non-Hermitian Hamiltonian
        # H - 0.5j J^dag J
        H_nonHermitian = self.H - 0.5j*(self.JTick_fw[0].conj().T).dot(self.JTick_fw[0])

        # Diagonalize
        w, V = np.linalg.eig(H_nonHermitian)

        # Timespan
        times = np.atleast_1d(times)
        t_span = (times[0],times[-1])

        # Initial state
        if psi0 is None:
            psi0 = np.zeros(self.n_qubits,dtype=complex)
            psi0[0] = 1.0

        Vt = np.exp(-1j*np.outer(times,w))

        psit = np.dot(V,(Vt*np.linalg.solve(V,psi0)).T)

        return times, psit
    
    def getBulkMomentum(self,time,psi0=None):
        """
        Obtain momentum space representation of time-evolved state

        Input:
            times: evaluation time
        
        Output:
            psi_k shape (nq,len)
        """
        # Solve EOM
        y = self.getTimeEvol(np.array([0.0,time]))[1]

        # Obtain final state in k-space
        k_range = np.arange(self.n_qubits) * 2 * np.pi / self.n_qubits

        return np.exp(1j*np.outer(k_range,np.arange(self.n_qubits))).dot(y[:,-1]) / np.sqrt(self.n_qubits)