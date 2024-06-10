import numpy as np
import qutip as qt

import time as time

from scipy.integrate import solve_ivp

from scipy.linalg import solve_continuous_lyapunov, eig

from scipy.sparse.linalg import spsolve, eigs
from scipy.sparse import csr_matrix

from qutip_extension import adjoint_superoperator, anticommutator_superoperator, diss, tilted_diss

def SNR(_H,_c_ops,_JTick,_rho0,data_type="numpy",method="lyapunov"):
    """
    Computes the SNR for a clock with
    Input:
              H :   Hamiltonian of the clock
          c_ops :   List [c1,...] of Lindblad jump operators (standard continuous dissipation)
          JTick :   Jump operator generating the clock's tick
           rho0 :   Initial state of the clock
      data_type :   "numpy" if L, Jtick and rho0 are of numpy type; "qutip" if they are Qobj.
         method :   "lyapunov" or "lindbladian" or "diag"
    
    Return:
        nu      :   frequency 1/mu
        SNR     :   SNR mu^2 / sigma^2
    """
    # Convert datatypes if necessary
    if data_type=="qutip":
        rho0 = _rho0.data.toarray()
    elif data_type=="numpy":
        rho0 = _rho0
    else:
        # Raise exception if none of the predefined datatypes is selected
        raise Exception("Error: data_type",data_type,"is unknown in call to SNR().\nPlease use either data_type=\"numpy\" or data_type=\"qutip\"")
    
    # Dimensions of density matrix and transformation into a vector
    state_shape = rho0.shape
    rho0 = np.ravel(rho0)

    # Moments of the tick distribution
    if method == "lyapunov":
        # Lyapunov method
        A = -1j*_H 
        for JT in _JTick:
            A -= 0.5*(JT.T.conj().dot(JT))

        X = solve_continuous_lyapunov(A,_rho0)
        Y = 2*solve_continuous_lyapunov(A,X)

    elif method == "lindbladian":
        L = superL(_H,_c_ops,_JTick,data_type=data_type)

        X = np.linalg.solve(L,rho0)
        Y = 2*np.linalg.solve(L.dot(L),rho0)

    elif method == "diag":
        # Directly solve Lyapunov with diagonal matrix
        # Get effective Hamiltonian
        A = -1j*_H 
        for JT in _JTick:
            A -= 0.5*(JT.T.conj().dot(JT))

        # Diagonalize
        # start = time.time()
        w, V = eig(A,right=True)
        # end = time.time()
        # print("diagonalization : "+str(np.round(end-start,3))+" s")

        # start = time.time()
        Vinv = np.linalg.solve(V,np.eye(len(w)))
        # end = time.time()
        # print(" matrix inverse : "+str(np.round(end-start,3))+" s")

        # Obtain diag-tensor
        # start = time.time()
        I = np.ones_like(w)
        Lsup = (np.kron(w,I)+np.kron(I,w).conj())

        # Solve in diagonal basis
        B = np.ravel(np.dot(Vinv,np.dot(_rho0,Vinv.T.conj())))

        X = np.reshape((1/Lsup)*B,state_shape)
        Y = 2*np.reshape((1/Lsup**2)*B,state_shape)

        X = np.dot(V,np.dot(X,V.T.conj()))
        Y = np.dot(V,np.dot(Y,V.T.conj()))
        # end = time.time()

        # print(" postprocessing : "+str(np.round(end-start,3))+" s")

    else:
        # Raise exception if none of the predefined methods is selected
        raise Exception("Error: method",data_type,"is unknown in call to SNR().\nPlease use either data_type=\"lyapunov\" or data_type=\"lindbladian\"")

    mu = -np.real(np.trace(np.reshape(X,state_shape)))
    sigma2 = np.real(np.trace(np.reshape(Y,state_shape)))-mu**2

    # print(mu**2/sigma2)

    # Return statement
    return 1/mu,mu**2/sigma2

def tiltedSuperL(_H,_c_ops,_JTick_fw,_JTick_bw,chi,data_type="numpy",sparse=True):
    """
    Returns tilted Superoperator in matrix form for full counting statistics
    Input:
              H :   Hamiltonian of the clock
          c_ops :   List [c1,...] of Lindblad jump operators (standard continuous dissipation)
       JTick_fw :   Jump operator generating the clock's ticks forward
       JTick_bw :   Jump operator generating the clock's ticks backward
      data_type :   "numpy" if L, Jtick and rho0 are of numpy type; "qutip" if they are Qobj.
                    WARNING: qutip not yet implemented
         sparse :   True if input data is sparse, else False
    Return:
              L :   matrix superoperator
    """
    L = superL(_H,_c_ops,[],data_type,sparse=sparse)

    for fw_temp in _JTick_fw:
        L += tilted_diss(fw_temp,chi,sparse=sparse)
    for bw_temp in _JTick_bw:
        L += tilted_diss(bw_temp,-chi,sparse=sparse)

    return L

def ddx(f,x,diff = 1e-3):
    # Numerical first order derivative
    return (f(x+diff/2)-f(x-diff/2))/diff

def ddx2(f,x,diff = 1e-3):
    # Numerical second order derivative
    return (f(x+diff)-2*f(x)+f(x-diff))/diff**2

def sparsify(_H,_c_ops,_JTick_fw,_JTick_bw):
    """
    Local function to sparsify Hamiltonian and jump operators
    """
    # Convert into list if is not one already
    if not isinstance(_JTick_fw,list):
        _JTick_fw = [_JTick_fw]
    if not isinstance(_JTick_bw,list):
        _JTick_bw = [_JTick_bw]
    if not isinstance(_c_ops,list):
        _c_ops = [_c_ops]

    # Sparsify
    if not isinstance(_H,csr_matrix):
            _H = csr_matrix(_H)
    for k, c_op in enumerate(_c_ops):
        if not isinstance(c_op,csr_matrix):
            _c_ops[k] = csr_matrix(c_op)
    for k, J_fw in enumerate(_JTick_fw):
        if not isinstance(J_fw,csr_matrix):
            _JTick_fw[k] = csr_matrix(J_fw)
    for k, J_bw in enumerate(_JTick_bw):
        if not isinstance(J_bw,csr_matrix):
            _JTick_bw[k] = csr_matrix(J_bw)

    return _H,_c_ops,_JTick_fw,_JTick_bw


def generalizedSNR(_H,_c_ops,_JTick_fw,_JTick_bw,rhoSS=None,data_type="numpy",sparse=False):
    """
    Returns tilted Superoperator in matrix form for full counting statistics
    Input:
              H :   Hamiltonian of the clock
          c_ops :   List [c1,...] of Lindblad jump operators (standard continuous dissipation)
       JTick_fw :   Jump operator generating the clock's ticks forward
       JTick_bw :   Jump operator generating the clock's ticks backward
         rho_SS :   Steady-state as starting vector for eigenvalue search
      data_type :   "numpy" if L, Jtick and rho0 are of numpy type; "qutip" if they are Qobj.
                    WARNING: qutip not yet implemented
         sparse :   True if input data is sparse, else False
    Return:
         genSNR :   Generalized SNR in long-time limit
    """
    # Check whether input matrix is sparse
    if sparse:
        _H,_c_ops,_JTick_fw,_JTick_bw = sparsify(_H,_c_ops,_JTick_fw,_JTick_bw)

    def maxEval(chi):
        # Determine and return eigenvalue of Lindbladian with greatest real part
        _M = tiltedSuperL(_H,_c_ops,_JTick_fw,_JTick_bw,chi,data_type=data_type,sparse=sparse)

        if sparse:
            if rhoSS is None:
                eigenvalues = eigs(_M,k=1,sigma=0.0,which="SR")[0]
            else:
                eigenvalues = eigs(_M,k=1,sigma=0.0,v0=rhoSS.ravel(),which="LM")[0]
        else:
            eigenvalues = np.linalg.eigvals(_M)
        return eigenvalues[np.argmax(np.real(eigenvalues))]

    # Determine average number
    avgN = -1j*ddx(maxEval,0)

    # Determine number variance
    varN = -ddx2(maxEval,0)

    _SNR = np.abs(np.real(avgN/varN))

    return np.real(avgN), _SNR

def tiltedEvals(_H,_c_ops,_JTick_fw,_JTick_bw,rhoSS=None,sparse=False):
    """
    Returns derivatives of dominant eigenvalue of the
    tilted Superoperator in matrix form for full counting statistics, with
    perturbations of backwards tick operators.
    Input:
              H :   Hamiltonian of the clock
          c_ops :   List [c1,...] of Lindblad jump operators (standard continuous dissipation)
       JTick_fw :   Jump operator generating the clock's ticks forward
       JTick_bw :   Jump operator generating the clock's ticks backward
         rho_SS :   Steady-state as starting vector for eigenvalue search
         sparse :   True if input data is sparse, else False
    Return:
          coeff :   array [01, 10, 11, 20, 21]
    """
    # Check whether input matrix is sparse
    if sparse:
        _H,_c_ops,_JTick_fw,_JTick_bw = sparsify(_H,_c_ops,_JTick_fw,_JTick_bw)

    def maxEval(chi,delta):
        # Determine and return eigenvalue of Lindbladian with greatest real part
        # dependent on the counting field chi and perturbation delta of tick ops
        local_JT_bw = []

        for k, JTb in enumerate(_JTick_bw):
            delta = 0.0 if delta < 0 else delta
            local_JT_bw.append(np.sqrt(2*delta)*JTb)

        # Obtain Superoperator
        _M = tiltedSuperL(_H,_c_ops,_JTick_fw,local_JT_bw,chi,data_type="numpy",sparse=sparse)

        if sparse:
            if rhoSS is None:
                eigenvalues = eigs(_M,k=1,sigma=0.0,which="SR")[0]
            else:
                eigenvalues = eigs(_M,k=1,sigma=0.0,v0=rhoSS.ravel(),which="LM")[0]
        else:
            eigenvalues = np.linalg.eigvals(_M)
        return eigenvalues[np.argmax(np.real(eigenvalues))]

    # Result placeholder
    res = []

    # Chi^0 Delta^1
    res.append((ddx(lambda x : maxEval(0,x),0,diff=1e-8)))

    # Chi^1 Delta^0
    res.append((ddx(lambda x : maxEval(x,0),0)))

    # Chi^1 Delta^1
    res.append((ddx(lambda y : ddx(lambda x : maxEval(x,y),0),0,diff=1e-8)))

    # Chi^2 Delta^0
    res.append((ddx2(lambda x : maxEval(x,0),0)))

    # Chi^2 Delta^1
    res.append((ddx2(lambda y : ddx(lambda x : maxEval(y,x),0,diff=1e-8),0)))

    # Add first non-zero eVal
    _M = tiltedSuperL(_H,_c_ops,_JTick_fw,_JTick_bw,0.0,data_type="numpy",sparse=sparse)
    if sparse:
        if rhoSS is None:
            eigenvalues = eigs(_M,k=2,sigma=0.0,which="SR")[0]
        else:
            eigenvalues = eigs(_M,k=2,sigma=0.0,v0=rhoSS.ravel(),which="LM")[0]
    else:
        eigenvalues = np.linalg.eigvals(_M)
    res.append(eigenvalues[np.argmax(np.real(eigenvalues))-1])

    return res

def superL(_H,_c_ops,_JTick,data_type="numpy",sparse=False):
    """
    Returns Superoperator in matrix form
    Input:
              H :   Hamiltonian of the clock
          c_ops :   List [c1,...] of Lindblad jump operators (standard continuous dissipation)
          JTick :   Jump operator generating the clock's tick
      data_type :   "numpy" if L, Jtick and rho0 are of numpy type; "qutip" if they are Qobj.
         sparse :   True if input data is sparse, else False

    Return:
              L :   matrix superoperator
    """
    # Empty list for collapse operators and ticking operator
    c_ops = []
    JTick = []

    # Convert JTick into list if is not one already
    if isinstance(_JTick,list):
        pass
    else:
        _JTick = [_JTick]

    # Convert datatypes if necessary
    if data_type=="qutip":
        for c_op in _c_ops:
            c_ops.append(c_op.data.toarray())
        H = _H.data.toarray()
        for _JT in _JTick:
            JTick.append(_JT.data.toarray())
    elif data_type=="numpy":
        for c_op in _c_ops:
            c_ops.append(c_op)
        H = _H
        for _JT in _JTick:
            JTick.append(_JT)
    else:
        # Raise exception if none of the predefined datatypes is selected
        raise Exception("Error: data_type",data_type,"is unknown in call to SNR().\nPlease use either data_type=\"numpy\" or data_type=\"qutip\"")
    
    if sparse:
        _H, c_ops, _JTick = sparsify(_H,c_ops,_JTick,[])[:3]

    # Generate Super-Operator
    L = -1j*adjoint_superoperator(H,sparse=sparse)
    for c_op in c_ops:
        L+= diss(c_op,sparse=sparse)
    for JT in JTick:
        L-= 0.5*anticommutator_superoperator((JT.conjugate().transpose()).dot(JT),sparse=sparse)

    # Return Superoperator 
    return L