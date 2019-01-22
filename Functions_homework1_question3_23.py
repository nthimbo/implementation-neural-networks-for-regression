import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from matplotlib import cm


class RBF_Decomp(object):
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        #self.centers = [random.uniform(0, 1, indim) for i in range(numCenters)]
        self.W = random.random((self.numCenters, self.outdim))

    def get_init(self):
        centers = [random.uniform(0, 1, self.indim) for i in range(self.numCenters)]
        W = np.array(random.random((self.numCenters, self.outdim)))
        return np.append(W.ravel(order="A"), np.array(centers).ravel(order="A"))

    def extract_data(self, w_all):
        W_size = self.numCenters * self.outdim
        W = np.reshape(w_all[:W_size], (self.numCenters, self.outdim), order='A')
        C = np.reshape(w_all[W_size:], (self.numCenters, self.indim), order='A')
        return W, C

    def func_rbf(x_data, c_data, v_data, sig):
        sigma = sig
        n_rows = np.shape(x_data)[0]
        n_cols = np.shape(c_data)[0]
        mat_d = np.zeros((np.shape(x_data)[0], np.shape(c_data)[0]))
        #print(mat_d.shape)
        for row in range(n_rows):
            num1 = x_data[row]
            for col in range(n_cols):
                # ff = c_data[0] - x_data[0]
                mat_d[row, col] = np.exp(-(np.sum((x_data[row] - c_data[col]) ** 2)) / sigma ** 2)
        #mat_d = np.dot(np.matrix(mat_d), v_data)
        out_put = mat_d
        #print(mat_d.shape)
        #print(v_data.shape)
        return out_put

    def func_loss1(W, C, x_data, y_data, eta, sig):
        q = eta
        fx = RBF_Decomp.func_rbf(x_data, C, W, sig)
        p = len(x_data)
        rr = np.array(fx - y_data)
        reg_loss = np.sum((rr) ** 2) / (2 * p) + q * ((np.sum(W ** 2) + np.sum(C ** 2)))
        return reg_loss

    def func_loss2(C, W, x_data, y_data, eta, sig):
        q = sig
        fx = RBF_Decomp.func_rbf(x_data, C, W, sig)
        p = len(x_data)
        rr = np.array(fx - y_data)
        reg_loss = np.sum((rr) ** 2) / (2 * p) + q * ((np.sum(W ** 2) + np.sum(C ** 2)))
        return reg_loss

    def plots(y_train, X_train):
        fig = plt.figure("plot1.png")
        ax = fig.gca(projection='3d')
        x1 = X_train[0:, :1]
        x2 = X_train[0:, 1:2]
        # x1, x2 = np.meshgrid(x1, x2)
        # ax.contour3D(x1.flatten(), x2.flatten(), y_train.flatten(), 50, cmap='binary')
        surf = ax.plot_trisurf(x1.flatten(), x2.flatten(), y_train.flatten(), cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=10, aspect=3)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()