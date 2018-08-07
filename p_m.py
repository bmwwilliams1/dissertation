import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot(ax,a,b,i,w):
    image = np.reshape(w[:,i+102],(28,28))
    im = ax.pcolormesh(a, -b, image)
    # fig.colorbar(im)
    ax.axis('tight')

def distance(a,b):

    if not (len(a)==len(b)):
        print('Error: vectors unequal length')
        return 0
    dist = 0
    for elem in range(0,len(a)-1):
        dist = dist+(a[elem]-b[elem])*(a[elem]-b[elem])
    return np.sqrt(dist)

def metrics(weights):

    mean = np.zeros((len(weights[:,0]),),dtype=float)
    i= 0
    for pixel in weights:
        mean[i] = sum(pixel)/len(pixel)
        i=i+1

    diff = np.zeros((len(weights),len(weights[0])),dtype=float)

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

    # print some of the covariance matrix


    cov = cov/(len(weights)-1)
    print(cov[0,0],',',cov[1,0],',',cov[0,1])

    return mean, cov
