import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import p_m
from scipy.stats import ks_2samp


# ============ MAIN ===============

def main():

    w1 = np.genfromtxt('./dsf_wt/weights_mathdsf512_1.csv',delimiter=",")
    w2 = np.genfromtxt('./ffn_wt/weights_512_1.csv',delimiter=",")
    neurons = 16

    w3 = p_m.floor(w2) #w3 is the non-negative component matrix of the ANN matrix
    im_size = 32

    # p_m.graphing(im_size, save_covariance = False, run_metrics = False, graph_components = False, graph_report = True,w1 = w1,w2=w2)

    ent1 = p_m.bin_total_entropy(w1)
    ent2 = p_m.bin_total_entropy(w2)
    ent3 = p_m.bin_total_entropy(w3)

    # p_m.gauss_fit(ent1)
    # p_m.gauss_fit(ent2)

    entropies = np.zeros((3,len(ent1[0])),dtype = float)
    entropies[0]=ent1
    entropies[1]=ent2
    entropies[2]=ent3
    # np.savetxt("./REUTERS/entropies_%s.csv"%str(neurons),entropies, delimiter=",")
    # #
    # #
    means = np.zeros((3,1),dtype = float)
    stdevs = np.zeros((3,1),dtype = float)
    #
    for j in range(0,3):
        means[j] = np.mean(entropies[j])
        stdevs[j] = np.std(entropies[j])
        print("w%s: mean - "%j, means[j],", stdev - ",stdevs[j] )
    #
    # entropies = np.genfromtxt('./REUTERS/entropies_32.csv',delimiter=",")
    # ent1 = entropies[0]
    # ent2 = entropies[1]
    # ent3 = entropies[2]
    print("Averages: w1:",entropies[0].mean(),", w2: ",entropies[1].mean(),", w3: ",entropies[2].mean())
    # #
    # # # #
    # print(ks_2samp(np.ndarray.flatten(ent1), np.ndarray.flatten(ent2)))
    # print(ks_2samp(np.ndarray.flatten(ent1), np.ndarray.flatten(ent3)))
    # print(ks_2samp(ent1, ent3))



if __name__ == "__main__":
    main()
