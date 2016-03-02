import scipy as sp
from scipy.optimize import minimize
import time

class SupportVectorMachine(object):
    def __init__(self, name="svm", debug=False):
        self.name = name
        self.training_data = None
        self.testing_data = None
        self.GramMatrix = None
        self.IsSet_GramMatrix = False
        self.debug = debug
        self.N = None
        self.D = None
        self.testN = None
        self.testD = None
        self.missClassRateS = None
        self.missClassRateB = None
        self.missClassRate = None
        self.a = None
        self.w = None
        self.b = None
        self.xi = None
        self.C = None
        self.aTol = 1e-6
        self.IsSet_C = False
        self.Kernel = self.DotProdKernel
        self.KernelParameters = None
        self.UserSet_Kernel = False
        if (debug):
            self.msg("Hi I'm " + name + ". Nice to meet you.")

    def __str__(self):
        return "--- " + self.name + " : "

    def Debug(self, message):
        if self.debug:
            self.msg(message)

    def msg(self, message):
        print(str(self) + str(message))

    def SetTrainingData(self, training_data):
        self.training_data = sp.array(training_data)
        self.N = self.training_data.shape[0]
        self.D = self.training_data.shape[1] - 1
        self.Debug("Training data set to N={} data points (of D={}).".format(self.N, self.D))

    def SetTestingData(self, testing_data):
        self.testing_data = sp.array(testing_data)
        self.testN = self.testing_data.shape[0]
        self.testD = self.testing_data.shape[1] - 1
        self.Debug("Testing data set to N={} data points (of D={}).".format(self.testN, self.testD))

    def PrepareTraining(self):
        self.Debug("Preparing training...")
        if not self.IsSet_C:
            self.SetC()
        self.GramMatrix = self.GetGramMatrix(self.Kernel)

    def SetC(self, C=1.):
        self.Debug("Setting penalty coefficient C = {}".format(C))
        self.C = C
        self.IsSet_C = True

    def Setup(self, options):
        try:
            self.w = sp.array(options["w"])
            self.b = options["b"]
        except:
            self.msg("Couldn't setup classifier with options" + str(options))

    def GetGramMatrix(self, kfunc=None, pars=None):
        if not self.IsSet_GramMatrix:
            self.Debug("GramMatrix not set, attempting to set it now...")
            if kfunc is None:
                self.Debug("No user supplied kernel function, using the default.")
                kfunc = self.Kernel
            self.CalculateGramMatrix(kfunc, pars=None)
        self.Debug("GramMatrix is now set (it might not have been before). Returning it.")
        return self.GramMatrix

    def CalculateGramMatrix(self, kfunc, pars=None):
        self.Debug("Calculating GramMatrix...")
        self.GramMatrix = sp.array(sp.empty((self.N, self.N)))
        for i in range(self.N):
            for j in range(i, self.N):
                xn = self.training_data[i,0:-1]
                xm = self.training_data[j,0:-1]
                k = kfunc(xn, xm, pars)
                self.GramMatrix[i,j] = k
                self.GramMatrix[j,i] = k
        self.IsSet_GramMatrix = True
        self.Debug("GramMatrix appears to have been calculated properly.")

    def DotProdKernel(self, x, xprim, pars=None):
        return sp.dot(x,xprim)

    def SetKernelFunction(self, func, pars=None):
        self.Debug("Setting user supplied kernel. MAKE SURE IT IS SYMMETRIC! I will not check that for you...")
        self.Kernel = func
        self.KernelParameters = pars
        self.UserSet_Kernel = True
        self.Debug("Kernel set to user supplied function.")
        if self.IsSet_GramMatrix:
            self.Debug("GramMatrix already calculated, but kernel is set by user. Will recalulate...")
            self.CalculateGramMatrix(self.Kernel, self.KernelParameters)

    def DualLagrangian(self, a, t, K):
        l1 = 0.
        l2 = 0.
        for n in range(self.N):
            for m in range(self.N):
                l2 += a[n]*a[m]*t[n]*t[m]*K[n,m]
            l1 += a[n]
        return 0.5*l2 - l1

    #def CostFuntion(self, W

    def TrainMethodDual(self):
        self.Debug("Starting training with dual Lagrangian...")
        a = sp.zeros((self.N))
        a = sp.random.uniform(0.,self.C,self.N)
        opts = {"disp":False}
        #if self.debug:
        #    opts["disp"] = True

        cons = (
                {"type":"ineq", "fun":lambda a: a},
                {"type":"ineq", "fun":lambda a: self.C - a},
                {"type":"eq", "fun":lambda a,t: sp.dot(a,t), "args":[self.training_data[:,-1]]}
                )
        func = self.DualLagrangian
        res = minimize(func, a, constraints=cons, args=(self.training_data[:,-1], self.GramMatrix), options=opts, method="SLSQP")
        if not res.success:
            self.Debug(res.message + " (Status: {:d})".format(res.status))
            self.Debug("nfev={:d}".format(res.nfev))
            self.Debug("nit={:d}".format(res.nit))
        self.a = res.x
        self.xi = sp.zeros((self.N))
        self.w = sp.zeros((self.D))
        for d in range(self.D):
            for n in range(self.N):
                self.w[d] += self.a[n]*self.training_data[n,-1]*self.training_data[n,d]
        Ns = 0
        s = 0.
        for n in range(self.N):
            if self.a[n] > self.aTol and self.a[n] < self.C:
                s2 = 0.
                Ns += 1
                for m in range(self.N):
                    if self.a[m] > self.aTol:
                        s2 += self.a[m]*self.training_data[m,-1]*self.GramMatrix[n,m]
                s += self.training_data[n,-1] - s2
        try:
            self.b = s/Ns
        except ZeroDivisionError as e:
            self.msg("ZeroDivisionError: {}".format(e))
            self.b = None
            self.msg("Ns={}".format(Ns))
            print("a=", self.a)
            pass

    def TrainMethodCanonical(self):
        self.Debug("Starting training with canonical hyperplanes...") 
        #W = sp.zeros(self.D + 1 + self.N)
        W = sp.random.uniform(0., 1., self.N + self.D + 1)
        opts = {"disp":False}
        #if self.debug:
        #    opts["disp"] = True

        cons = []
        #self.C = 
        for n in range(self.N):
            cons.append(
                    {
                        "type":"ineq",
                        "fun":lambda W,x,t,m: t*(sp.dot(W[1:self.D+1],x) + W[0]) - 1 + W[self.D+1:][m],
                        "args":[self.training_data[n,:-1], self.training_data[n,-1], n]
                        }
                    )
        cons.append(
                {
                    "type":"ineq",
                    "fun":lambda W: W[self.D+1:]
                    }
                )

        func = lambda W: 0.5*sp.dot(W[1:self.D+1],W[1:self.D+1]) + self.C*sp.sum(W[self.D+1:])
        res = minimize(func, W, constraints=cons, options=opts, method="SLSQP")
        if not res.success:
            self.Debug(res.message + " (Status: {:d})".format(res.status))
            self.Debug("nfev={:d}".format(res.nfev))
            self.Debug("nit={:d}".format(res.nit))
        self.w = res.x[1:self.D+1]
        self.xi = res.x[self.D+1:]
        self.b = res.x[0]
        self.a = sp.zeros((self.N))

    def Train(self, method="auto"):
        self.msg("Starting training...")
        tstart = time.time()
        cstart = time.clock()
        if method == "auto":
            self.Debug("Determining fastest training method...")
            if self.UserSet_Kernel:
                self.Debug("It appears the user has defined the kernel. Will train with dual Lagrangian (to be safe).")
                self.TrainMethodDual()
            elif self.D < self.N:
                self.Debug("Since D < N, I will use canonical hyperplanes to get complexity ~ O(D^3).")
                self.TrainMethodCanonical()
            else:
                self.Debug("Since D >= N, I will use dual Lagrangian to get complexity ~ O(N^3).")
                self.TrainMethodDual()
        elif method == "canonical":
            self.TrainMethodCanonical()
        elif method == "dual":
            self.TrainMethodDual()
        cstop = time.clock()
        tstop = time.time()
        elapsed = tstop-tstart
        celapsed = (cstop-cstart)
        self.msg("Training done (Real: {:.3f}s CPU: {:.3f}s ).".format(elapsed, celapsed))

    def EvalPoint(self, x):
        y = sp.dot(self.w, x) + self.b
        return y

    def EvalSet(self, X):
        Y = sp.zeros((self.testN))
        for n in range(self.testN):
            Y[n] = self.EvalPoint(X[n])
        return Y

    def Classify(self, X):
        self.msg("Classifying data set...")
        Y = self.EvalSet(X)
        self.msg("Classification done.")
        return Y
    
    def Test(self):
        self.msg("Testing classifier...")
        Y = self.EvalSet(self.testing_data[:,:-1])
        self.missClassRateS = 0.
        self.missClassRateB = 0.
        self.missClassRate = 0.
        Nsignal = 0
        for n in range(self.testN):
            if self.testing_data[n,-1] == 1:
                Nsignal += 1
            if Y[n]*self.testing_data[n,-1] < 0:
                self.missClassRate += 1.
                if self.testing_data[n,-1] == 1:
                    self.missClassRateS += 1.
                else:
                    self.missClassRateB += 1.
        self.missClassRateS = self.missClassRateS/Nsignal
        self.missClassRateB = self.missClassRateB/(self.testN - Nsignal)
        self.missClassRate = self.missClassRate/self.testN
        self.msg("Testing done with missclassifitation rate (S,B,Tot) = ({}, {}, {})"
                .format(self.missClassRateS, self.missClassRateB, self.missClassRate))

        return Y
