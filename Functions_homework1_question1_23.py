import numpy as np
import matplotlib.pyplot as plt
from numpy import random


class MLP():
    def __init__(self, num_inputs, num_units, num_oput):
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_oput = num_oput

    def extractWeightMatrices(self, w_all):
        wh_size = (self.num_inputs) * self.num_units
        H_W_size = wh_size + self.num_units
        wh = np.reshape(w_all[:wh_size], (self.num_inputs, self.num_units), order='A')
        V = np.reshape(w_all[wh_size:H_W_size], (1, self.num_units), order='A')
        bh = np.reshape(w_all[H_W_size:], (1, self.num_units), order='A')
        return wh, V, bh

    def getInitialWeights(self):
        wh = np.random.rand(self.num_inputs, self.num_units)
        V = np.random.rand(self.num_units, self.num_oput)
        bh = np.zeros((self.num_oput, self.num_units))
        wh_and_V = np.append(wh.ravel(order='A'), V.ravel(order='A'))
        return np.append(wh_and_V, bh.ravel(order='A'))

    def func_activation(self, x_data, wh, bh, V):
        # perform forward propagation
        HLinput = np.dot(x_data, wh) - bh
        a1 = np.tanh(HLinput)
        OLinput = np.dot(a1, V.T)
        Oput = np.tanh(OLinput)
        return Oput

    def func_loss(init_data, x_data, y_data, obj, e):
        q = e
        size_input = obj.num_inputs
        size_H_units = obj.num_units
        num_outs = obj.num_oput
        wh, V, bh = MLP.extractWeightMatrices(obj, init_data)
        p = len(x_data)
        fx = obj.func_activation(x_data, wh, bh, V)
        rr = fx - y_data
        reg_loss = (np.sum((rr) ** 2) / (2 * p)) + q * ((np.sum(V ** 2) + np.sum(wh ** 2) + np.sum(bh ** 2)))
        return reg_loss
    def plots(y_train, X_train):
        y_train = y_train
        X_train = X_train
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        fig = plt.figure("plot1.png")
        ax = fig.gca(projection='3d')
        x1 = X_train[0:, :1]
        x2 = X_train[0:, 1:2]
        # x1, x2 = np.meshgrid(x1, x2)
        # surf = ax.plot_surface(x1, x2, y_train,cmap=cm.coolwarm, linewidth=0, antialiased=False)
        surf = ax.plot_trisurf(x1.flatten(), x2.flatten(), y_train.flatten(), cmap=cm.jet, linewidth=0.1)
        # fig.colorbar(surf, shrink=10, aspect=3)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()


class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(0, 1, indim) for i in range(numCenters)]
        self.W = random.random((self.numCenters, self.outdim))
        # print(self.W)

    def get_init(self):
        centers = [random.uniform(0, 1, self.indim) for i in range(self.numCenters)]
        W = np.array(random.random((self.numCenters, self.outdim)))
        return np.append(W.ravel(order="A"), np.array(centers).ravel(order="A"))

    def extract_data(self, w_all):
        W_size = self.numCenters * self.outdim
        W = np.reshape(w_all[:W_size], (self.numCenters, self.outdim), order='A')
        C = np.reshape(w_all[W_size:], (self.numCenters, self.indim), order='A')
        return W, C

    def func_rbf(x_data, c_data, v_data, sigma):
        n_rows = np.shape(x_data)[0]
        n_cols = np.shape(c_data)[0]
        mat_d = np.zeros((np.shape(x_data)[0], np.shape(c_data)[0]))
        for row in range(n_rows):
            num1 = x_data[row]
            for col in range(n_cols):
                # ff = c_data[0] - x_data[0]
                mat_d[row, col] = np.exp(-(np.sum((x_data[row] - c_data[col]) ** 2)) / sigma ** 2)
        mat_d = np.dot(np.matrix(mat_d), v_data)
        return mat_d

    def func_loss(ini_data, x_data, y_data, rbf, sigma, eta):
        q = eta
        W, C = rbf.extract_data(ini_data)
        fx = RBF.func_rbf(x_data, C, W, sigma)
        p = len(x_data)
        rr = np.array(fx - y_data)
        reg_loss = np.sum((rr) ** 2) / (2 * p) + q * ((np.sum(W ** 2) + np.sum(C ** 2)))
        return reg_loss

    def plots(y_train, X_train):
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

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

