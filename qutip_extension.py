import numpy as np
import qutip as qt

from scipy.sparse import csr_matrix, eye, kron

def adjoint_superoperator(H,sparse=False):
    '''
    Returns the Commutator with H in the form of a superoperator
    '''
    dim = H.shape[0]
    if sparse:
        if not isinstance(H,csr_matrix):
            H = csr_matrix(H)
        return kron(H,eye(dim))-kron(eye(dim),H.transpose())

    return np.kron(H,np.eye(dim))-np.kron(np.eye(dim),np.transpose(H))

def anticommutator_superoperator(Pi,sparse=False):
    '''
    Returns the Anticommutator with Pi in superoperator form
    '''
    dim = Pi.shape[0]

    # Sparsity routine
    if sparse:
        if not isinstance(Pi,csr_matrix):
            Pi = csr_matrix(Pi)
        return kron(Pi,eye(dim))+kron(eye(dim),Pi.transpose())

    return np.kron(Pi,np.eye(dim))+np.kron(np.eye(dim),Pi.T)

def diss(J,sparse=False):
    '''
    Returns the superoperator of the dissipator with J as the collapse operator.
    '''
    dim = J.shape[0]

    # Sparsity routine
    if sparse:
        if not isinstance(J,csr_matrix):
            J = csr_matrix(J)
        return kron(J,J.conjugate())-0.5*(kron((J.transpose().conjugate()).dot(J),eye(dim))+kron(eye(dim),(J.conjugate().transpose()).dot(J)))
    
    return np.kron(J,np.conj(J))-0.5*(np.kron(np.conj(J.T).dot(J),np.eye(dim))+np.kron(np.eye(dim),np.conj(J.T).dot(J)))

def tilted_diss(J,chi,sparse=False):
    '''
    Returns the tilted superoperator of the dissipator with J as the collapse operator.
    With counting field chi
    '''
    dim = J.shape[0]

    # Sparsity routine
    if sparse:
        if not isinstance(J,csr_matrix):
            J = csr_matrix(J)
        return np.exp(1j*chi)*kron(J,J.conjugate())-0.5*(kron((J.conjugate().transpose()).dot(J),eye(dim))+kron(eye(dim),(J.transpose().conjugate()).dot(J)))

    return np.exp(1j*chi)*np.kron(J,np.conj(J))-0.5*(np.kron(np.conj(J.T).dot(J),np.eye(dim))+np.kron(np.eye(dim),np.conj(J.T).dot(J)))