#!/usr/bin/env python

import matplotlib.pyplot as plt
import scipy as sp
from SVM.svm import SupportVectorMachine

def generate_dataset(seed=None, N=100):
    x = sp.empty((N))
    y = sp.empty((N))
    t = sp.empty((N))

    mb = sp.array([0.7, 0.6])
    sb = sp.array( [[0.1, 0.],[0., 0.15]] )
    ms = sp.array([0.05, 0.20])
    ss = sp.array( [[0.05, 0.],[0., 0.05]] )
    
    sp.random.seed(seed)
    for i in range(N):
       r = sp.random.uniform(0,1)
       if r < .5:
           x[i], y[i] = sp.random.multivariate_normal(mb, sb)
           t[i] = -1
       else:
           x[i], y[i] = sp.random.multivariate_normal(ms, ss)
           t[i] = 1
    return x,y,t

if __name__ == "__main__":
    #x,y,t = generate_dataset(seed=123,N=100)
    #x,y,t = generate_dataset(seed=75713,N=10)
    x,y,t = generate_dataset(seed=None,N=100)
    #x,y,t = generate_dataset(seed=1451468981,N=30)
    testx, testy, testt = generate_dataset(seed=None, N=50)

    pars = {
            "w": [-2.69, -1.19],
            "b": 1.53,
            }

    svm = SupportVectorMachine(debug=True)
    #svm.Setup(pars)

    svm.SetTrainingData(zip(x,y,t))
    svm.SetTestingData(zip(testx,testy,testt))
    
    #kern = lambda xn,xm,pars: sp.dot(xn,xm)
    #svm.SetKernelFunction(kern)

    svm.PrepareTraining()

    #methods = ["dual", "canonical"]
    methods = ["auto"]
    colors = ['red' if l == 1 else 'blue' for l in t]
    plt.scatter(x, y, marker='o', color=colors)
    plt.axis('equal')
    x1,x2,y1,y2 = plt.axis()
    xx = sp.linspace(sp.amin(x),sp.amax(x))
    for m in methods:
        svm.Train(m)
        Y = svm.Test()
        testcolors = ['orange' if l >= 0 else 'purple' for l in Y]
        a = svm.a
        b = svm.b
        w1 = svm.w[0]
        w2 = svm.w[1]
        print("b,w = ", b, w1, w2)
        yy = (-w1*xx - b)/w2
        style='k-.'
        plt.plot(xx,yy,style,linewidth=2.0)
        plt.scatter(testx, testy, marker='o', color=testcolors)
    plt.axis((x1,x2,y1,y2))
    plt.show()
