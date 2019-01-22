import numpy as np
from math import exp
from scipy.optimize import minimize
import numdifftools as nd
import time
from numpy import linalg as la
from matplotlib import cm
import Functions_homework1_question3_23 as F


def data_Gen(N):
    np.random.seed(1738522)
    X1 = np.random.uniform(0, 1, N)
    X2 = np.random.uniform(0, 1, N)
    y = []
    e_values = np.random.uniform(-10**-1, 10**-1,1738522)
    for i in range(N):
        term1 =  0.75 * exp(-((9*X1[i]-2)**2)/4 - ((9*X2[i]-2)**2)/4)
        term2 = 0.75 * exp(-((9*X1[i]+1)**2)/49 - ((9*X2[i]+1))/10)
        term3 = 0.5 * exp(-((9*X1[i]-7)**2)/4 - ((9*X2[i]-3)**2)/4)
        term4 = -0.2 * exp(-((9*X1[i]-4)**2) - ((9*X1[i]-7)**2))
        terms = term1 + term2+term3+term4 + e_values[i]
        y.append(terms)
    data = [[X1[x], X2[x], y[x]] for x in range(N)]
    return data
data = data_Gen(100)
X = np.array([[i[0],i[1]] for i in data])
y = np.array([[i[2]] for i in data])
X_train = X[:70, ]
y_train = y[:70]
X_test = X[70:, ]
y_test = y[70:]



if __name__ == '__main__':
    y = y_train
    x = X_train
    x_t = X_test
    y_t = y_test
    eta = [10**-3]
    units = [2]
    sigma = [0.7]
    start_time = time.time()
    for unit in units:
        for sig in sigma:
            for e in eta:
                rbf = F.RBF_Decomp(2, unit, 1)
                all = rbf.get_init()
                W, C =  rbf.extract_data(all)
                maxIter = 1000

                while(True):
                   res = minimize(F.RBF_Decomp.func_loss1, W, args=(C, x, y, e, sig), method='BFGS',options={'disp': False, 'maxiter':maxIter})
                   W = np.array(res.x.reshape(rbf.numCenters, rbf.outdim))

                   jac = nd.Jacobian(F.RBF_Decomp.func_loss2)
                   jac_vals = jac(C.flatten(), W, x, y, e, sig)
                   jac_norm = la.norm(jac_vals)
                   if((jac_norm >= 10**-6) and (jac_norm <= 10**-3)):
                       predic =F.RBF_Decomp.func_rbf(x_t, C, W, sig)
                       MSE_Train = res.fun
                       MSE_test = np.sum((predic - y_t) ** 2)
                       #print(res)
                       print("")
                       print("values for Eta = " + str(e) + ", num units = " + str(unit) + " and sigma = "+str(sig))
                       print("Training MSE: " + str(MSE_Train))
                       print("Test MSE: " + str(MSE_test))
                       print("function evaluations: " + str(res.nfev))
                       print("gradient evaluations: " + str(res.njev))
                       print("Number of Iteration : " + str(res.nit))
                       print(res)
                       F.RBF_Decomp.plots(predic, x_t)
                       F.RBF_Decomp.plots(y_t, x_t)
                       break
                   else:
                       res2 = minimize(F.RBF_Decomp.func_loss2, C, args=(W, x, y, e, sig), method='BFGS', options={'disp': False, 'maxiter': maxIter})
                       C = np.array(res2.x.reshape(rbf.numCenters, rbf.indim))
    print("MLP Computing time is: %s seconds " % (time.time() - start_time))


