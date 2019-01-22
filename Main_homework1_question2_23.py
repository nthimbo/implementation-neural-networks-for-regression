import numpy as np
from math import exp
import time
from scipy.optimize import minimize
import Functions_homework1_question2_23 as F

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
    eta = 10**-6
    eta1 = 10 ** -5
    units = 11
    units1 = 6
    sigma = 0.3
    #Execution of MLP network in extreme learning
    start_time = time.time()
    mlp_EL = F.MLP_EL(2, units, 1)
    V = np.zeros((units, 1))
    w_all = F.MLP_EL.initial_weights(mlp_EL)
    wh, bh = F.MLP_EL.get_weights(w_all, mlp_EL)
    maxIter = 1000

    res = minimize(F.MLP_EL.func_loss, V, args=(w_all, mlp_EL, x, y, eta), method='BFGS',options={'disp': False, 'maxiter': maxIter})
    V = res.x
    predic = F.MLP_EL.func_activation(x_t, wh, bh, V)
    MSE_Train = res.fun
    MSE_test = np.sum((predic-y_t)**2)
    print("values for Eta = "+str(eta)+" and num units = "+ str(units))
    print("Training MSE: " +str(MSE_Train))
    print("Test MSE: "+str(MSE_test))
    print("function evaluations: " + str(res.nfev))
    print("gradient evaluations: " + str(res.njev))
    print("Number of Iteration : " + str(res.nit))
    print("")
    print("")
    F.MLP_EL.plots(predic, x_t)
    F.MLP_EL.plots(y_t, x_t)
    print("MLP Computing time is: %s seconds " % (time.time() - start_time))

    start_time1 = time.time()
    #Executions for RBF function in extreme learning
    rbf_EL = F.RBF_EL(2, units1, 1)
    all = F.RBF_EL.get_init(rbf_EL, x)
    W, C = F.RBF_EL.extract_data(all, rbf_EL)
    maxIter = 1000
    res1 = minimize(F.RBF_EL.func_loss, W, args=(x, rbf_EL, C, y, eta1, sigma), method='bfgs', options={'disp': True, 'maxiter': maxIter})
    W_final = res1.x #np.array(res.x).T
    pred = np.array(F.RBF_EL.func_rbf(x_t, C, W_final, sigma))
    MSE_Train = res1.fun
    MSE_test = np.sum((pred - y_t) ** 2)
    print("values for Eta = " + str(eta1) + " , num units = " + str(units1) + " and sigma = " + str(sigma))
    print("Training MSE: " + str(MSE_Train))
    print("Test MSE: " + str(MSE_test))
    print("function evaluations: " + str(res1.nfev))
    print("gradient evaluations: " + str(res1.njev))
    print("Number of Iteration : " + str(res1.nit))
    print("")
    print("")
    print("MLP Computing time is: %s seconds " % (time.time() - start_time1))
    F.RBF_EL.plots(pred, x_t)
    F.RBF_EL.plots(y, x)


