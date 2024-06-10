import numpy as np

import time as time

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import expm_multiply

from qubit_chain_params import getOptParam, getExpParam
from qubit_chain import RingClock

from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8

})

col_scheme_BR = ["navy","mediumblue","slateblue","darkslateblue","rebeccapurple","indigo","darkviolet","darkmagenta","mediumvioletred","deeppink","crimson","orangered","sienna","peru"]


def plotTransmission(d):
    """
    Plot transmission functions
    (a) flat spectrum, wiggly lines,  (b) apodized with energy distribution
    """
    # Obtain parameters
    g = getExpParam(d,mode="plus").astype(complex)

    # Bare flat spectrum
    g_flat = g.copy()
    g_flat[:] = g[int(d/2)]
    
    # Symmetrized apodized spectrum
    g_symm = g.copy()
    g_symm[:int(d/2)] = (g[-int(d/2):])[::-1]

    # Build Hamiltonians
    # Flat
    H_flat = np.diag(g_flat,-1)+np.diag(g_flat,+1)
    H_flat[-1,-1] = -0.5j
    H_flat[0,0] = -0.5j

    # Symmetrized
    H_symm = np.diag(g_symm,-1)+np.diag(g_symm,+1)
    H_symm[-1,-1] = -0.5j
    H_symm[0,0] = -0.5j

    # Shifted Hamiltonian
    A = lambda w, H : np.diag(np.ones((d))*w) - H

    # Transmission function calculator
    def T(w,H):
        res = []
        for w in omegas:
            res.append(np.dot(vD,np.linalg.solve(A(w,H),v1)))
        return np.abs(np.array(res))**2
    
    # Helper vectors
    vD = np.zeros((d))  # final site
    vD[-1] = 1.0

    v1 = np.zeros((d))  # first site
    v1[0] = 1.0
    
    # Frequency range
    omegas = np.linspace(-0.5,0.5,2000)

    # Plotting
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(7.08333,2),sharey=True)

    ax1.plot(omegas,T(omegas,H_flat),color=col_scheme_BR[6],label=r"$T(\omega)$")
    ax1.vlines(x=np.real(np.array([-2*g[int(d/2)],2*g[int(d/2)]])),ymin=-0.1,ymax=1.1,color="black",linestyle="--",label=r"$\pm 2g$")
    ax1.set_ylim(-0.1,1.1)
    ax1.set_xlim(-0.5,0.5)
    ax1.set_ylabel(r"transmittance $=[1]$")
    ax1.set_xlabel(r"$\omega = [\Gamma]$")
    ax1.legend(loc="lower right")

    # Obtain energies
    myClock = RingClock(g,1.0,5.0)
    psi_k = myClock.getBulkMomentum(d*2.2)
    ax2.plot(2*np.real(g[int(d/2)]) * np.cos(2*np.pi * np.arange(int(d/2)) / d),0.6*(np.abs(psi_k)**2)[:int(d/2)]*d**(1/3),color=col_scheme_BR[0],linestyle="--",marker=".",label=r"$\propto p(E)$")
    # ax1.plot(2*np.real(g[int(d/2)]) * np.cos(2*np.pi * np.arange(int(d/2)) / d),0.6*(np.abs(psi_k)**2)[:int(d/2)]*d**(1/3),color=col_scheme_BR[0],linestyle="--",marker=".",label=r"$\propto p(E)$")
    ax2.plot(omegas,T(omegas,H_symm),color=col_scheme_BR[6],label=r"$T(\omega)$")
    ax2.vlines(x=np.real(np.array([-2*g[int(d/2)],2*g[int(d/2)]])),ymin=-0.1,ymax=1.1,color="black",linestyle="--",label=r"$\pm 2g$")
    ax2.set_ylim(-0.1,1.1)
    ax2.set_xlim(-0.5,0.5)
    ax2.set_xlabel(r"$\omega = [\Gamma]$")
    ax2.legend(loc="lower right")

    ax1.text(-0.158, 0.94, "(a)",transform=ax1.transAxes)
    ax2.text(-0.08, 0.94, "(b)",transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig("figures/transmittance.jpg",dpi=600)
    # plt.show()
    plt.close()
    pass

def widthVarPlausibilization(nq_range,paramGetter,mu=[0.5,0.9],lambda_min=10,lambda_max=1000):
    """
    Generate a plot as follows
    (a) plotting of the width versus ramp length        (b) plotting of variance/width versus the length
        scale lambda,                                       of the ring

    Input
        nq_range    :   range of ring lengths for the plot b
        lambda_min/max: min and max length of ramp
    """
    ############################
    # Obtain data for plot (a) #
    ############################
    lambdas = np.exp(np.linspace(np.log(lambda_min),np.log(lambda_max),30))
    widths_a  = np.zeros_like(lambdas)
    widths_b  = np.zeros_like(lambdas)

    for k, lambda_k in enumerate(lambdas):
        # Coupling values
        gk_a = 1.0 - mu[0]*np.exp(-np.arange(10*lambda_k)/lambda_k)
        gk_b = 1.0 - mu[1]*np.exp(-np.arange(10*lambda_k)/lambda_k)

        # Hamiltonian
        H_a = diags(gk_a,-1) + diags(gk_a,+1)
        H_b = diags(gk_b,-1) + diags(gk_b,+1)

        # Initial state
        psi0 = np.zeros((len(gk_a)+1))
        psi0[0] = 1.0

        tf = lambda_k*4
        psit_a = expm_multiply(-1j*H_a,psi0,start=0.0,stop=tf,num=2,endpoint=True)
        psit_b = expm_multiply(-1j*H_b,psi0,start=0.0,stop=tf,num=2,endpoint=True)

        p_a = np.abs(psit_a[1,:])**2
        p_b = np.abs(psit_b[1,:])**2

        x = np.arange(len(p_a))

        def dev(p):
            mean = np.sum(x*p)
            return np.sqrt(np.sum(p*(x - mean)**2))

        widths_a[k] = dev(p_a)
        widths_b[k] = dev(p_b)

        # plt.figure()
        # plt.plot(p,marker=".",linestyle="")
        # plt.show()

    ############################
    # Obtain data for plot (b) #
    ############################
    devTrue = np.zeros((len(nq_range)))
    devFake = np.zeros((len(nq_range)))

    avgTrue = np.zeros((len(nq_range)))
    avgFake = np.zeros((len(nq_range)))

    for k, nq in enumerate(nq_range):
        times = np.arange(8*nq)

        def dev(f,times):
            # Pedestrian method
            f_norm = np.sum(f)
            avg = np.sum(f*times)/f_norm
            var = np.sum(f*(times - avg)**2)/f_norm
            return avg,np.sqrt(var)

        myClock = RingClock(paramGetter(nq),1.0,1.0)

        # Solve EOM
        y = myClock.getTimeEvol(times)[1]

        # Determine tick probability
        tickPDF = np.abs(y[-1,:])**2

        avgTrue[k],devTrue[k] = dev(tickPDF,times)

        # Determine free evolution
        # Hamiltonian
        gl = paramGetter(nq,oneramp=True)
        gr = np.ones_like(gl)*gl[-1]
        g = np.concatenate((gl,gr))
        H = np.diag(g,-1) + np.diag(g,1)

        # Diagonalize
        w, V = np.linalg.eig(H)

        # Initial state
        psi0 = np.zeros(2*nq-1,dtype=complex)
        psi0[0] = 1.0

        Vt = np.exp(-1j*np.outer(times,w))

        psit = np.dot(V,(Vt*np.dot(V.T.conj(),psi0)).T)

        proj_density = np.abs(psit[nq,:])**2

        avgFake[k],devFake[k] = dev(proj_density,times)

        print(k,len(nq_range))

        # plt.figure()
        # plt.plot(tickPDF)
        # plt.plot(proj_density)
        # plt.show()

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(7.08333,2))

    ax1.loglog(lambdas,widths_a,marker=".",linestyle="--",color=col_scheme_BR[0],label=r"discrete, $\mu_\ell = "+str(mu[0])+r" g$")
    ax1.loglog(lambdas,widths_b,marker=".",linestyle="--",color=col_scheme_BR[2],label=r"discrete, $\mu_\ell = "+str(mu[1])+r" g$")
    ax1.loglog(lambdas[14:],lambdas[14:]/10,color=col_scheme_BR[6],linestyle="-.",label=r"width $\propto \lambda_\ell$")

    ax1.set_xlabel(r"ramp width $\lambda_\ell = [1]$")
    ax1.set_ylabel(r"width $=[1]$")

    ax1.legend(loc="best")
    ax1.grid(linestyle="--",color="black",alpha=0.2)

    ax2.loglog(nq_range,np.abs(devTrue-devFake),color=col_scheme_BR[0],marker=".",linestyle="--",label=r"$|\mathrm{Var}[T]^{1/2}-\mathrm{Var}[T]_\text{free}^{1/2}|$")
    ax2.loglog(nq_range,np.abs(avgTrue-avgFake),color=col_scheme_BR[2],marker=".",linestyle="--",label=r"$|\mathrm{E}[T]-\mathrm{E}[T]_\text{free}|$")
    # ax2.loglog(nq_range,avgTrue,label=r"$\mathrm{E}[T]$")
    # ax2.loglog(nq_range,avgFake,label=r"$\mathrm{E}[T]_\text{no loss}$")

    ax2.set_xlabel(r"number of sites $n$")
    ax2.set_ylabel(r"$T=[\Gamma^{-1}]$")

    ax2.legend(loc="lower left")
    ax2.grid(linestyle="--",color="black",alpha=0.2)

    ax1.text(-0.16, 0.94, "(a)",transform=ax1.transAxes)
    ax2.text(-0.19, 0.94, "(b)",transform=ax2.transAxes)


    plt.tight_layout()
    plt.savefig("figures/width_scaling.jpg",dpi=600)
    # plt.show()
    plt.close()
    pass

def comparisonExpTrue(file_full,file_exp):
    """
    Generate plot showing
    (a) difference couplings, (b) difference precision
    """
    # Range of dimensions
    d = 50
    d_range = np.arange(10,51)

    # Load results
    res_full = np.load(file_full)
    res_exp = np.load(file_exp)

    # Find indices for nq = 10,11,...,50
    idx_full = np.where((res_full[0,:]<= max(d_range))*(res_full[0,:]>= min(d_range)))[0]
    idx_exp = np.where((res_exp[0,:]<= max(d_range))*(res_exp[0,:]>= min(d_range)))[0]

    fig, (ax1,ax3) = plt.subplots(1,2,figsize=(7.08333,2))

    # ax2 = ax1.twinx()

    # Plot couplings
    g_full = getOptParam(d)
    g_exp  = getExpParam(d,mode="plus")

    ax1.plot(np.arange(d-1),g_full,marker=".",linestyle="--",color=col_scheme_BR[6],label=r"$g_j^{\rm opt}$")
    ax1.plot(np.arange(d-1),np.abs(g_full-g_exp),marker=".",linestyle="--",color=col_scheme_BR[8],label=r"$|g_j^{\rm exp}-g_j^{\rm opt}|$")
    # ax2.semilogy(np.arange(d-1)+0.5,np.abs(g_full-g_exp),linestyle="-",color=col_scheme_BR[0],label="difference")

    idx = np.concatenate((np.arange(0,24),np.arange(26,28),np.arange(30,41)))

    ax3.plot(d_range[idx],(np.abs(res_full[1,idx_full]/res_exp[1,idx_exp]-1))[idx],linestyle="--",marker=".",color=col_scheme_BR[0],label="rel. precision difference")

    ax1.set_xlabel(r"site index $j$")
    ax3.set_xlabel(r"number of sites $n$")

    ax1.set_ylabel(r"couplings $g=[\Gamma]$")
    # ax2.set_ylabel(r"coupling diff. $|g_n^{\rm opt}/g_n^{\rm exp}-1|$")
    ax3.set_ylabel(r"$\mathcal N_\infty^{\rm opt} / \mathcal N_\infty^{\rm exp} - 1$")

    ax1.legend(loc="upper left")
    # ax2.legend(loc="lower right")
    # ax3.legend(loc="upper right")

    ax1.grid(linestyle="--",color="black",alpha=0.2)
    ax3.grid(linestyle="--",color="black",alpha=0.2)

    ax1.text(-0.15, 0.94, "(a)",transform=ax1.transAxes)
    ax3.text(-0.20, 0.94, "(b)",transform=ax3.transAxes)

    # ax3.yaxis.tick_right()
    # ax3.yaxis.set_label_position("right")

    plt.tight_layout()
    plt.savefig("figures/comparison_true_exp.png",dpi=600)
    # plt.show()
    plt.close()
    pass

def tickNumberFake():
    """
    Generate plot for the main text illustrating a toy example of ticking statistics
    for N(t)
    """
    # Trajectories
    tgood = 2*np.array([0.0,1.0,1.9,2.8,3.5,4.8,5.3])
    Ngood = np.array([0,1,2,3,4,5])

    tbad = 2*np.array([0.0,0.9,1.8,2.1,2.30,3.35,4.30,5.3])
    Nbad = np.array([0.0,1,2,1,2,3,4])

    # Avg
    times = np.linspace(0,6,200)

    plt.figure(figsize=(3.417,1.4))
    # plt.stairs(Ngood,tgood,baseline=None,color=col_scheme_BR[0],label="traj. 1",linewidth=1.2)
    plt.stairs(Nbad,tbad,baseline=None,color=col_scheme_BR[0],label="trajectory",linewidth=1.2)
    plt.plot(2*times,np.sqrt(0.5 + times**2)-np.sqrt(0.5),color="black",linestyle="--",label=r"$\mathrm{E}[N(t)]$",linewidth=1.2)
    plt.fill_between(2*times,1.3*times,(0.5+(0.7*times)**2)**(1/2) -np.sqrt(0.5),alpha=0.4,facecolor="gray",edgecolor="gray",linestyle="-",linewidth=0.5,label="uncertainty band")
    plt.legend(loc="best")
    plt.ylabel(r"tick number $N(t)$")
    plt.xlabel(r"time $t=[{\rm a.u.}]$")
    plt.xlim(-1.0,10.6)
    plt.tight_layout()
    plt.savefig("figures/Ntrajectory.jpg",dpi=600)
    # plt.show()
    plt.close()
    pass

def accuracyComparison(data_files,alpha,relative=False):
    """
    Generate a plot of the perturbations of the clock accuracy
    (a) plot of difference |N-N_T|      (b) plot of N for several values of alpha

    Input:
        data_files  :   number of sites
    """
    data_files = [data_files] if not isinstance(data_files,list) else data_files


    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(7.08333,2))#,height_ratios=(1.5,1))

    # Plot for each data file separately
    for k, file in enumerate(data_files):

        # Obtain data file
        data = np.load("data/"+file)
        
        # Obtain relevant data points
        nq_range= data[0,:]

        ET = 1/data[2,:]
        VarT = ET**2/data[1,:]

        accT = data[1,:]

        dEN = data[4,:]
        dVarN = dEN/data[3,:]

        accN = data[3,:]

        if k <= 2:
            if k<= 1:
                x=-10
            else:
                x=-1
            # x = 5 if k>= 1 else 0
            ax1.loglog(nq_range[:x],(np.abs(accT-accN))[:x],marker=".",color=col_scheme_BR[2*k],label=r"$\beta="+str(alpha[k])+"$")
        if k==0:
            ax2.loglog(nq_range,accT,marker=".",color="black",label=r"$\mathcal N_\infty$")
        if k>=2:
            ax2.loglog(nq_range,accN,marker=".",color=col_scheme_BR[2*k],label=r"$\beta="+str(alpha[k])+"$")

        # z = np.polyfit(np.log(nq_range),np.log(accT - accN),deg=1)
        # ax2.loglog(nq_range,np.exp(z[1])*nq_range**z[0],color="black",linestyle="--")#,label=r"fit $\sim d^{"+str(np.round(z[0],3))+r"}$")
        # ax2.loglog(nq_range,accT - accN,marker=".",linestyle="--",color=col_scheme_BR[2*k],label=r"$\alpha="+str(alpha[k])+r"$, fit $\sim d^{"+str(np.round(z[0],3))+r"}$")

    ax1.set_ylabel(r"$|\mathcal N_\infty-\mathcal N_\Sigma|$")
    ax2.set_ylabel(r"precision $\mathcal N$")

    ax1.set_xlabel(r"number of sites $n$")
    ax2.set_xlabel(r"number of sites $n$")

    ax1.legend(loc="best")
    ax2.legend(loc="best")

    ax1.grid(linestyle="--",color="black",alpha=0.2)
    ax2.grid(linestyle="--",color="black",alpha=0.2)

    ax1.text(-0.18, 0.94, "(a)",transform=ax1.transAxes)
    ax2.text(-0.16, 0.94, "(b)",transform=ax2.transAxes)


    plt.tight_layout()
    plt.savefig("figures/accuracy_comparison.jpg",dpi=600)
    # plt.show()
    plt.close()
    pass

def tiltedEvalPlotter(nq_range,paramGetter):
    """
    Generate a plot of the perturbations of the zero eigenvalue
    for finite entropies

    Input:
        nq_range    :   number of sites
     paramGetter    :   function to obtain parameters
    """

    # Generate the data points
    l11 = np.zeros((len(nq_range)),dtype=complex)
    l21 = np.zeros((len(nq_range)),dtype=complex)
    l20 = np.zeros((len(nq_range)),dtype=complex)
    gap = np.zeros((len(nq_range)),dtype=complex)

    print("Tilted eval plotter started.\n")

    for k, nq in enumerate(nq_range):
        # Obtain params and generate clock
        g = paramGetter(nq)
        myClock = RingClock(g,1.0,0.0)

        # Obtain tilted evals
        res = myClock.getTiltedEvals(sparse=True)
        l11[k] = res[2]
        l21[k] = res[4]
        l20[k] = res[3]
        gap[k] = res[-1]

        print("> round "+str(k+1)+"/"+str(len(nq_range))+", nq = "+str(nq)+" completed.")

    fig, axs = plt.subplots(1,2,figsize=(7.08333,2))

    z,V = np.polyfit(np.log(nq_range),np.log(np.abs(gap)),deg=1,cov=True)
    print(V)
    alpha_gap = z[0]
    axs[0].loglog(nq_range,np.abs(gap),label=r"$|\lambda_\mathrm{gap}|\sim n^{"+str(np.round(z[0],2))+r"}$",marker=".",color=col_scheme_BR[6])
    # axs[0].loglog(nq_range,np.exp(z[1])*nq_range**z[0],color="black",linestyle="--",label=r"fit $|\lambda_\mathrm{gap}| = O(d^{"+str(np.round(z[0],2))+r"})$")
    axs[0].legend(loc="upper right")
    axs[0].set_xlabel(r"number of sites $n$")
    axs[0].set_ylabel(r"spectral gap $[\lambda]=\Gamma$")
    axs[0].grid(linestyle="--",color="black",alpha=0.2)

    z = np.polyfit(np.log(nq_range),np.log(np.abs(l11)),deg=1)
    axs[1].loglog(nq_range,np.exp(z[1])*nq_range**z[0],color="black",linestyle="--")
    axs[1].loglog(nq_range,np.abs(l11),label=r"$|\lambda_{11}|\sim n^{"+str(np.round(z[0],2))+r"}$",linestyle="--",marker=".",color=col_scheme_BR[0])
    axs[1].loglog(nq_range,np.abs(l11)[0]*(nq_range/10)**(-alpha_gap),color=col_scheme_BR[0],linestyle="-",label=r"$\lambda^\mathrm{bound}_{11}\sim n^{"+str(np.round(-alpha_gap,2))+r"}$")

    z = np.polyfit(np.log(nq_range),np.log(np.abs(l21)),deg=1)
    axs[1].loglog(nq_range,np.exp(z[1])*nq_range**z[0],color="black",linestyle="--")
    axs[1].loglog(nq_range,np.abs(l21),label=r"$|\lambda_{21}|\sim n^{"+str(np.round(z[0],2))+r"}$",linestyle="--",marker=".",color=col_scheme_BR[2])
    axs[1].loglog(nq_range,np.abs(l21)[0]*(nq_range/10)**(-2*alpha_gap),color=col_scheme_BR[2],linestyle="-",label=r"$\lambda^\mathrm{bound}_{21}\sim n^{"+str(np.round(-2*alpha_gap,2))+r"}$")

    # z = np.polyfit(np.log(nq_range),np.log(np.abs(l20)),deg=1)
    # axs[0].loglog(nq_range,np.exp(z[1])*nq_range**z[0],color="black",linestyle="--")
    # axs[0].loglog(nq_range,np.abs(l20),label=r"$|\lambda_{20}|$, fit $\sim d^{"+str(np.round(z[0],3))+r"}$",linestyle="--",marker=".",color=col_scheme_BR[4])

    axs[1].legend(loc="upper right")
    axs[1].set_xlabel(r"number of sites $n$")
    axs[1].set_ylabel(r"coefficients $[\lambda_{ij}]=\Gamma$")
    axs[1].grid(linestyle="--",color="black",alpha=0.2)

    axs[1].set_ylim(0.5e-2,5e1)

    axs[0].text(-0.185, 0.94, "(a)",transform=axs[0].transAxes)
    axs[1].text(-0.193, 0.94, "(b)",transform=axs[1].transAxes)

    plt.tight_layout()
    plt.savefig("figures/spectral_gap.jpg",dpi=800)
    # plt.show()
    plt.close()
    pass


def kspacePlotter(nq_range,paramGetter):
    """
    Generate a plot of the k-space distribution for the wave function

    Input:
        nq_range    :   number of sites
     paramGetter    :   function to obtain parameters
    """
    # Preliminaries
    nq_range = np.atleast_1d(nq_range)

    fig, ax1 = plt.subplots(figsize=(3.417,1.6))

    for k, nq in enumerate(nq_range):
        times = np.linspace(0,nq*2.2,5)

        myClock = RingClock(paramGetter(nq),1.0,1.0)

        # Solve EOM
        y = myClock.getTimeEvol(times)[1]

        # Obtain final state in k-space
        k_range = np.arange(nq) * 2 * np.pi / nq

        yk = np.exp(1j*np.outer(k_range,np.arange(nq))).dot(y[:,-1]) / np.sqrt(nq)

        print(np.linalg.norm(yk))

        ax1.semilogy(k_range,np.abs(yk)**2,label=r"$|\psi_k|^2,$ $n="+str(nq)+"$",marker=".",color=col_scheme_BR[k*2],zorder=2)

        # Polyfit close to peak
        idxs = np.arange(int(3*nq/16),int(5*nq/16))
        z = np.polyfit(k_range[idxs],(np.log(np.abs(yk)**2))[idxs],deg=2)
        p = np.poly1d(z)

        idxs = np.arange(0,int(nq/2))
        # ax1.semilogy(k_range[idxs],np.exp(p(k_range[idxs])),label=r"quadratic polyfit for $n="+str(nq)+"$ sites",color=col_scheme_BR[k*2+1])

        # ax1.semilogy(k_range,np.abs(y[:,-1])**2,label=r"$|\psi_k|^2,$ for $n="+str(nq)+"$ sites",color=col_scheme_BR[k*2])

    k_range_smooth = np.linspace(0,2*np.pi,1000)

    ax2 = ax1.twinx()


    ax2.plot(k_range_smooth,np.cos(k_range_smooth),label=r"$E(k)$",color="black",zorder=0)

    # ax2.vlines(np.pi/2,-1.2,1.2,color="black",linestyle="--",label=r"$\frac{\pi}{2}$-point")

    ax2.set_ylim(-1.1,1.1)

    ax2.set_ylabel(r"dispersion $E=[2g]$")

    ax1.set_ylabel(r"population $p=[1]$")
    ax1.set_ylim(1e-4,5e-1)
    ax1.set_xticks(np.array([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]),["0",r"$\frac{\pi}{2}$",r"$\pi$",r"$\frac{3\pi}{2}$",r"$2\pi$"])
    ax1.grid(linestyle="--",color="black",alpha=0.2)
    ax1.set_xlabel(r"momentum $k$")


    ax1.legend(loc="lower right").set_zorder(100)
    ax2.legend(loc="upper right").set_zorder(100)

    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.set_frame_on(False)
    plt.tight_layout()
    plt.savefig("figures/kspace.jpg",dpi=600)    
    # plt.show()
    plt.close()
    pass

def expParamPlotter(nq_range,mode=""):
    """
    Generates a plot of all the parameters how the change with nq, number of lattice sites.
    Loglog format
    
    Input
           nq_range :   number of sites
     expParamGetter :   function to obtain the parameters
               type :   "" nothing or "Plus"
    """
    fig, axs = plt.subplots(1,2,figsize=(7.08333,2.0))

    # Obtain the parameters
    params = np.zeros((len(nq_range),5))
    for k, nq in enumerate(nq_range):
        params[k,:] = np.load("params_exp"+mode+"/chain_exp_nq_"+str(nq)+".npy")

    # Rescaling by d^1/3
    if mode == "Plus":
        params[:,3]*=(nq_range/10)**(1/3)
        params[:,4]*=(nq_range/10)**(1/3)

    print(params[-1,4])

    z, V = np.polyfit(np.log(nq_range),np.log(np.abs(params[:,3])),deg=1,cov=True)
    print(V)
    # axs[0].loglog(nq_range,np.exp(z[1])*nq_range**z[0],color="black",linestyle="--")
    axs[0].loglog(nq_range,params[:,3],linestyle="--",marker=".",label=r"$\lambda_{\ell}\sim n^{"+str(np.round(z[0],2))+"}$",color=col_scheme_BR[0])
    axs[0].loglog(nq_range,params[:,4],linestyle="--",marker=".",label=r"$\lambda_{r}$",color=col_scheme_BR[5])
    # axs[0].loglog(nq_range,nq_range**(1/3),linestyle="--",label=r"$n^{1/3}$",color="black")

    axs[1].plot(nq_range,params[:,0],linestyle="--",marker=".",label=r"$\mu_{\ell}$",color=col_scheme_BR[0])
    axs[1].plot(nq_range,params[:,1],linestyle="--",marker=".",label=r"$g$",color=col_scheme_BR[3])
    axs[1].plot(nq_range,params[:,2],linestyle="--",marker=".",label=r"$\mu_{r}$",color=col_scheme_BR[6])
    # axs[1].loglog(nq_range,(nq_range/nq_range[-1])**(1/9)*params[-1,0],linestyle="--",label=r"$n^{1/12}$",color="black")

    for ax in axs:    
        ax.grid(True,linestyle="--",color="gray")
        ax.set_xlabel(r"number of sites $n$")
    axs[0].legend(loc="upper left")
    axs[1].legend(loc="lower right")

    axs[0].set_ylabel(r"ramp scale $\lambda = [1]$")
    axs[1].set_ylabel(r"energy scale $g = [\Gamma]$")

    axs[0].text(-0.16, 0.94, "(a)",transform=axs[0].transAxes)
    axs[1].text(-0.17, 0.94, "(b)",transform=axs[1].transAxes)

    plt.tight_layout()
    plt.savefig("figures/params_exp_scaling.jpg",dpi=600)
    # plt.show()
    plt.close()
    pass

def evolutionPlotPaper(nq,paramGetter):
    """
    Generate plot for 3d evolution figure
    Input
        nq          : number of qubit sites
        s           : entropy
        paramGetter : function to obtain parameters
    """
    g = paramGetter(nq)
    myClock = RingClock(g,1.0,1.0)

    time_eval = np.arange(0,7*nq,20)

    t, y = myClock.getTimeEvol(time_eval)

    fig = plt.figure(figsize=(1.8*3.417,1.5*3.417))
    ax = fig.add_subplot(projection='3d')

    colors = ["navy","mediumblue","slateblue","darkslateblue","rebeccapurple","indigo","darkviolet","darkmagenta","mediumvioletred","deeppink","crimson"]
    yticks = t

    colors = colors[:len(yticks)]

    ptick = []

    for j, (c, k) in enumerate(zip(colors[::-1], yticks[::-1])):
        # Generate the random data for the y=k 'layer'.
        xs = np.arange(nq)
        ys = np.abs(y[:,-j-1])**2

        ptick.append(1-np.sum(ys))

        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        if j==len(yticks)-1:
            ax.plot(xs, ys, zs=k, zdir='y', linestyle="--",marker=".", color=c, alpha=0.8,label=r"$|\langle n|\psi(t)\rangle|^2$")
        else:
            ax.plot(xs, ys, zs=k, zdir='y', linestyle="--",marker=".", color=c, alpha=0.8)


    ax.plot(yticks, ptick[::-1], zs=nq+1, zdir='x', linestyle=":", marker="o", color="green", alpha=0.8,label=r"CDF $P[T\leq t]$")

    ax.legend(loc="best")
    ax.set_xlabel(r'lattice site $n$')
    ax.set_ylabel(r'time $t=[\Gamma]$')
    ax.set_zlabel(r'probability')

    # On the y-axis let's only label the discrete values that we have data for.
    ax.set_yticks(yticks[::2])

    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.0, 1.0, 0.7, 1]))

    plt.tight_layout()
    plt.savefig("figures/evolution.jpg",dpi=600)
    # plt.show()
    plt.close()
    pass

def triplePanelPlot(data_small_file,data_full_file):
    """
    Function to generate the three plots for the paper
        var/E vs. dim. | acc. vs. dim. | acc. vs. ent.
    
    Input:
        data_file   :   "res_nq_[min]_[max].npy"
    """
    # Read data file
    data_full = np.load("data/"+data_full_file)
    data_small = np.load("data/"+data_small_file)

    nq_range_irrev = data_full[0,::5]
    acc_range_irrev = data_full[1,::5]
    res_range_irrev = data_full[2,::5]

    avg_range_irrev = 1/res_range_irrev
    dev_range_irrev = avg_range_irrev/np.sqrt(acc_range_irrev)

    nq_range_rev = data_small[0,:]
    acc_range = data_small[3,:]
    res_range = data_small[4,:]
    ent_range = data_small[5,:]

    fig, axs = plt.subplots(1,3,figsize=(7.08333,2.5))#,gridspec_kw={'wspace':0.14})

    # AVG VAR Plot
    # Polyfit
    z, V = np.polyfit(np.log(nq_range_irrev),np.log(dev_range_irrev),deg=1,cov=True)
    # axs[0].loglog(nq_range_irrev,np.exp(z[1])*nq_range_irrev**z[0],color=col_scheme_BR[0],linestyle="--")#,label=r"fit $\sim d^{"+str(np.round(z[0],2))+r"}$")
    axs[0].loglog(nq_range_irrev,dev_range_irrev,color=col_scheme_BR[0],marker=".",label=r"$\mathrm{Var}[T]^{1/2}\sim n^{"+str(np.round(z[0],2))+r"}$")
    z = np.polyfit(np.log(nq_range_irrev),np.log(avg_range_irrev),deg=1)
    # axs[0].loglog(nq_range_irrev,np.exp(z[1])*nq_range_irrev**z[0],color=col_scheme_BR[2],linestyle="--")#,label=r"fit $\sim d^{"+str(np.round(z[0],2))+r"}$")
    axs[0].loglog(nq_range_irrev,avg_range_irrev,color=col_scheme_BR[2],marker=".",label=r"$\mathrm{E}[T]\sim n^{"+str(np.round(z[0],2))+r"}$")
    # axs[2].loglog(nq_range,acc_range_irrev,color="violet",linestyle="",marker="+",label="irrev. simulation")
    axs[0].set_xlabel(r"number of sites $n = [1]$")
    axs[0].set_ylabel(r"$T = [\Gamma^{-1}]$")
    axs[0].legend(loc="upper left")

    axs[0].text(-0.26, 0.94, "(a)",transform=axs[0].transAxes)

    # ACCURACY DIMENSION
    axs[1].loglog(nq_range_irrev,nq_range_irrev,color=col_scheme_BR[2],linestyle="--",label=r"$\mathcal N_\infty= n$")
    axs[1].loglog(nq_range_irrev,nq_range_irrev**2,color=col_scheme_BR[4],label=r"$\mathcal N_\infty = n^2$")    
    # Polyfit
    z, V = np.polyfit(np.log(nq_range_irrev),np.log(acc_range_irrev),deg=1,cov=True)
    print(V)
    # axs[1].loglog(nq_range_irrev,np.exp(z[1])*nq_range_irrev**z[0],color=col_scheme_BR[0],linestyle="--")#,label=r"fit $\sim d^{"+str(np.round(z[0],2))+r"}$")
    axs[1].loglog(nq_range_irrev,acc_range_irrev,color=col_scheme_BR[0],marker=".",label=r"$\mathcal N_\infty\sim n^{"+str(np.round(z[0],2))+r"}$")
    axs[1].set_xlabel(r"number of sites $n = [1]$")
    axs[1].set_ylabel(r"precision $\mathcal N = [1]$")
    axs[1].legend(loc="upper left")

    axs[1].text(-0.27, 0.94, "(b)",transform=axs[1].transAxes)

    # ACCURACY ENTROPY
    ent_temp = np.linspace(1,max(ent_range),500)
    axs[2].plot(ent_temp,ent_temp/2,color=col_scheme_BR[2],linestyle="--",label=r"$\mathcal N_\Sigma = \Sigma/2$ (C)")
    axs[2].plot(ent_temp,ent_temp**2,color=col_scheme_BR[4],label=r"$\mathcal N_\Sigma = \Sigma^2$")
    axs[2].plot(ent_range,acc_range,color=col_scheme_BR[0],marker=".",label=r"$\mathcal N_\Sigma$ (numerics)")
    axs[2].set_xlabel(r"entropy per tick $\Sigma_\mathrm{tick}=[k_B]$")
    axs[2].set_ylabel(r"precision $\mathcal N = [1]$")
    axs[2].legend(loc="upper left")

    axs[2].text(-0.3, 0.94, "(c)",transform=axs[2].transAxes)

    # # ACCURACY RESOLUTION
    # axs[1].plot(res_range,1/res_range,col_scheme_BR[0],label=r"classical $\mathcal N \leq \frac{\Gamma}{\nu}$")
    # axs[1].plot(res_range,1/res_range**2,color=col_scheme_BR[2],label=r"general $\mathcal N \leq \frac{\Gamma^2}{\nu^2}$")
    # # Polyfit
    # z = np.polyfit(np.log(res_range),np.log(acc_range),deg=1)
    # axs[1].loglog(res_range,np.exp(z[1])*res_range**z[0],color="black",linestyle="--",label=r"fit $\mathcal N = O(d^{"+str(np.round(z[0],3))+r"})$")
    # axs[1].loglog(res_range,acc_range,color=col_scheme_BR[-3],linestyle="",marker=".",label="simulation")
    # axs[1].set(xlabel=r"resolution $\nu = [\Gamma]$")
    # axs[1].legend(loc="upper right")

    for ax in axs:
        ax.grid(linewidth="0.6",linestyle="--",alpha=0.5,color="gray")

    plt.tight_layout()
    plt.savefig("figures/accuracy_panel.jpg",dpi=600)
    # plt.show()
    plt.close()
    pass


def plotCouplingsPaper(_d,paramGetter=getOptParam):
    """
    Plot coupling parameters for paper.
    Input:
        d   :   dimension(s) for the comparison
    """
    # Ensure iterable dimension array.
    _d = np.atleast_1d(_d)

    # Labels
    label2=r"$g_n$"

    colx = ["navy","blue","darkslateblue","rebeccapurple","indigo","darkviolet","darkmagenta","mediumvioletred","crimson"]

    plt.figure(figsize=(3.417,1.4))
    for k,d in enumerate(_d):
        if k>=1:
            label2 = None
        plt.plot(np.arange(d-1),paramGetter(d),linestyle="--",marker=".",color=colx[k*2],label=label2)
    plt.xlabel(r"coupling site index $n$")
    plt.ylabel(r"couplings $g=[\Gamma]$")
    # plt.grid(True,linewidth="0.6",linestyle="--",alpha=0.5,color="gray")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("figures/rates_comparison.jpg")
    # plt.show()
    plt.close()
    pass