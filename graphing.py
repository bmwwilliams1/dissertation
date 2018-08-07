import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import p_m

# ============ MAIN ===============

def main():

    w1 = np.genfromtxt ('weights1.csv', delimiter=",")
    w2 = np.genfromtxt ('weights2.csv', delimiter=",")
    print('w1 dimensions:',len(w1),'x',len(w1[0]))
    print('w2 dimensions:',len(w2),'x',len(w2[0]))

    image = np.reshape(w1[:,3],(28,28))
    save_covariance = False
    graph_components = False
    graph_average = False


    #  Graphing the components, here we can graph a variable number of the graphs individually
    if (graph_components==True):
        n_points = 28
        a = np.linspace(1, 28, n_points)
        b = np.linspace(1, 28, n_points)
        a, b = np.meshgrid(a, b)

        ncols = 5
        fig,ax = plt.subplots(ncols,ncols)
        i = 0
        for sub in ax:
            for subi in sub:
                p_m.plot(subi,a,b,i,w1)
                subi.get_xaxis().set_visible(False)
                subi.get_yaxis().set_visible(False)
                i=i+1

        plt.show()

    mean, cov = p_m.metrics(w1)

    if (save_covariance==True):
        np.savetxt("cov.csv", cov, delimiter=",")

    # ================================================

    # Printing the average image
    # print(diff)
    if (graph_average==True):
        mean_image = np.reshape(mean,(28,28))
        fig,ax = plt.subplots(1,1)
        n_points = 28
        a = np.linspace(1, 28, n_points)
        b = np.linspace(1, 28, n_points)
        a, b = np.meshgrid(a, b)
        im = ax.pcolormesh(a, -b, mean_image)
        ax.axis('tight')
        plt.show()

if __name__ == "__main__":
    main()
