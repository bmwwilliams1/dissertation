import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot(ax,a,b,i,w,off):
    image = np.reshape(w[:,i+off],(28,28))
    im = ax.pcolormesh(a,-b, image)
    # fig.colorbar(im)
    ax.axis('tight')

def meanplot(ax,a,b,i,w,sizes):
    image = np.reshape(w,(28,28))
    im = ax.pcolormesh(a,-b, image)
    # fig.colorbar(im)
    ax.axis('tight')
    ax.set_title(str(sizes[i]))

def distance(a,b):

    if not (len(a)==len(b)):
        print('Error: vectors unequal length')
        return 0
    dist = 0
    for elem in range(0,len(a)-1):
        dist = dist+(a[elem]-b[elem])*(a[elem]-b[elem])
    return np.sqrt(dist)

def mean(weights):

    mean = np.zeros((len(weights[:,0]),),dtype=float)
    i= 0
    for pixel in weights:
        mean[i] = sum(pixel)/len(pixel)
        i=i+1

    return mean

def metrics(weights):

    mean = np.zeros((len(weights[:,0]),),dtype=float)
    i= 0
    for pixel in weights:
        mean[i] = sum(pixel)/len(pixel)
        i=i+1


    diff = np.zeros((len(weights),len(weights[0])),dtype=float)

    # print('start of mean vector: ',mean[0],mean[1],mean[2])
    # print('mean vector shape: ',len(mean))

    for row in range(0,len(weights)):
        for col in range(0,len(weights[0])):
            diff[row,col]=weights[row,col]-mean[row]

    # print(len(mean))
    # print('diff dimensions:',len(diff),'x',len(diff[0]))

    cov = np.zeros((len(weights[0]),len(weights[0])),dtype=float)

    for row in range(0,len(weights[0])):
        for col in range(0,len(weights[0])):
            for ex in range(0,len(weights)):
                cov[row,col] = cov[row,col]+(diff[ex,row]*diff[ex,col])

    cov = cov/(len(weights)-1)

    return mean, cov
