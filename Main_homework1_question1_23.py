import numpy as np
from math import exp
from scipy.optimize import minimize
import Functions_homework1_question1_23 as F
import time

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
    length_data  = x.shape[0]
    eta = [10**-6]
    eta1 = [10 ** -5]
    units = [11]
    units1 = [6]
    sigma = [0.3]

    start_time = time.time()
    print("**EXECUTION DATA FOR PART 01 OF QUESTION 01")
    for e in eta:
        for unit in units:
            mlp = F.MLP(2, unit, 1)
            all = F.MLP.getInitialWeights(mlp)
            maxIter = 10000
            res = minimize(F.MLP.func_loss, all, args=(x,y,mlp, e), method='BFGS',options={'disp': False, 'maxiter':maxIter})
            wh, V, bh = mlp.extractWeightMatrices(res.x)
            pred = mlp.func_activation(x_t, wh, bh, V)
            MSE_Train = res.fun
            MSE_test = np.sum((pred-y_t)**2)
            print("values for Eta = "+str(e)+" and num units = "+ str(unit))
            print("Training MSE: " +str(MSE_Train))
            print("Test MSE: "+str(MSE_test))
            print("function evaluations: " + str(res.nfev))
            print("gradient evaluations: " + str(res.njev))
            print("Number of Iteration : " + str(res.nit))
            print("")
            print("")
            F.MLP.plots(np.array(pred), x_t)
            F.MLP.plots(y_t, x_t)
    print("MLP Computing time is: %s seconds " % (time.time() - start_time))

    #
    #


    start_time1 = time.time()
    print("EXECUTION DATA FOR PART 02 OF QUESTION 01")
    for num_centers in units1:
        for sig in sigma:
            for e in eta1:
                rbf = F.RBF(2,num_centers , 1)
                all = rbf.get_init()
                W, C = rbf.extract_data(all)
                #F.RBF.func_loss(all, x, y, rbf, sig, e)
                maxIter = 1000
                res1 = minimize(F.RBF.func_loss, all, args=(x, y, rbf, sig, e), method='BFGS',
                               options={'disp': False, 'maxiter': maxIter})
                W_data, C_data = rbf.extract_data(res1.x)
                predic = np.array(F.RBF.func_rbf(x_t, C_data, W_data, sig))
                MSE_Train = res1.fun
                MSE_test = np.sum((predic - y_t) ** 2)
                print("values for Eta = " + str(e) + " , num units = " + str(num_centers)+ " and sigma = "+str(sig))
                print("Training MSE: " + str(MSE_Train))
                print("Test MSE: " + str(MSE_test))
                print("function evaluations: " + str(res1.nfev))
                print("gradient evaluations: " + str(res1.njev))
                print("Number of Iteration : " + str(res1.nit))
                print("")
                print("")
                print(res1)
                F.RBF.plots(pred, x_t)
                F.RBF.plots(y_t, x_t)
    print("RBF Computing time is: %s seconds " % (time.time() - start_time1))

