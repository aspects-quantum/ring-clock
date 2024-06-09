import numpy as np
import matplotlib.pyplot as plt

import time as time

from scipy.optimize import minimize
from scipy.linalg import null_space

from snr_toolbox import superL, SNR, notick_evolution, tiltedSuperL, generalizedSNR
from qubit_chain_params import getOptParam, getExpParam, expWrapper

from qubit_chain import RingClock

from qubit_chain_plots import plotTransmission, widthVarPlausibilization, comparisonExpTrue, tickNumberFake, accuracyComparison, tiltedEvalPlotter, kspacePlotter, expParamPlotter, triplePanelPlot, plotCouplingsPaper, evolutionPlotPaper, avgVarPlot

from qutip_extension import diss

def generatePlottableData(nq_range,paramGetter,id="",mode="full",alpha=3.0):
    """
    Function to generate plottable data
        nq_range    :   chain length range
        paramGetter :   function get[...]Param(nq)
        mode        :   "full" vs "irrev"
        alpha       :   entropy prod parameter

    Output: (5,N) dimensional array
        [0,:] -> nq_min, nq_min+1, ..., nq_max-1
        [1,:] -> SNR irreversible
        [2,:] -> res irreversible
        [3,:] -> SNR reversible
        [4,:] -> res reversible
        [5,:] -> ent reversible
    """
    # Initialize
    if mode == "full":
        q = 6
    elif mode == "irrev":
        q = 3
    else:
        # Raise exception if none of the predefined methods is selected
        raise Exception("Error: mode "+mode+" is unknown in call to generatePlottableData().\nPlease use either mode=\"full\" or mode=\"irrev\"")

    res = np.zeros((q,len(nq_range)))

    res[0,:] = nq_range

    # Entropy parameters (NEED TO IMPROVE)
    s_list = alpha*np.log(nq_range + 1.0)

    print("... running generatePlottableData()")
    print("nq_min : "+str(min(nq_range))+", nq_max : "+str(max(nq_range))+", total # : "+str(len(nq_range)))

    for k,nq in enumerate(nq_range):
        myClockyClock = RingClock(paramGetter(nq),1.0,s_list[k])
        
        res[2,k], res[1,k] = myClockyClock.getSNR()
        if mode == "full":
            res[4,k], res[3,k] = myClockyClock.getGeneralizedSNR(sparse=True)
            res[5,k] = myClockyClock.getEntropyProductionRate()/res[4,k]

        print("round "+str(k+1)+"/"+str(len(nq_range))+" completed. nq = "+str(nq))

    # Save results with format
    # data/res_nq_[nqmin]_[nqmax].npy
    np.save("data/result_nq_"+str(min(nq_range))+"_"+str(max(nq_range))+"_"+mode+id+".npy",res)

    pass

if __name__ == "__main__":
    data_files = ["result_nq_10_1200_irrev.npy"]
    labels = ["forced exp."]

    d_range = np.arange(10,201,5)

    plotTransmission(32)

    widthVarPlausibilization(np.concatenate((np.arange(30,101,10),np.arange(100,1201,100))),lambda nq, oneramp=False : getExpParam(nq,mode="plus",oneramp=oneramp))

    comparisonExpTrue("data/res_nq_1_53.npy","data/result_nq_10_1200_irrev.npy")

    tiltedEvalPlotter(d_range,lambda nq : getExpParam(nq,mode="plus"))

    tickNumberFake()

    alphas = np.array([4.5,4.0,3.5,1.5,1.0])
    data_files = ["result_nq_10_200_full_stp5_alpha4.5.npy",
                  "result_nq_10_200_full_stp5_alpha4.0.npy",
                  "result_nq_10_200_full_stp5_alpha3.5.npy",
                  "result_nq_10_200_full_stp5_alpha1.5.npy",
                  "result_nq_10_200_full_stp5_alpha1.0.npy"]
    
    accuracyComparison(data_files,alphas,relative=True)
    # for alpha in [3.5,4.5]:
    #     generatePlottableData(d_range,lambda nq : getExpParam(nq,"plus"),id="_stp5_alpha"+str(alpha),mode="full",alpha=alpha)

    avgVarPlot(data_files,labels)

    kspacePlotter(np.array([100, 1000]),lambda nq : getExpParam(nq,mode="plus"))

    rng = np.concatenate((np.arange(10,201,1),np.arange(210,501,10),np.arange(520,1001,20),np.arange(1050,1201,50)))
    
    expParamPlotter(rng[::5],mode="Plus")

    evolutionPlotPaper(7,lambda nq : getOptParam(nq))

    triplePanelPlot("result_nq_10_200_full_stp5_alpha4.0.npy","result_nq_10_1200_irrev.npy")

    plotCouplingsPaper(np.array([40]),lambda nq : getExpParam(nq,mode="plus"))