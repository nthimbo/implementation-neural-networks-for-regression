import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from sklearn.cluster import KMeans

class MLP_EL(object):
    def __init__(self, num_inputs, num_units, num_oput):
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_oputs = num_oput

    def get_weights(w_all, c_object):
        wh_size = (c_object.num_inputs) * c_object.num_units
        wh = np.reshape(w_all[:wh_size], (c_object.num_inputs, c_object.num_units), order='A')
        bh = np.reshape(w_all[wh_size:], (c_object.num_oputs, c_object.num_units), order='A')
        return wh, bh

    def initial_weights(c_object):
        wh = np.random.rand(c_object.num_inputs, c_object.num_units)
        bh = np.random.rand(c_object.num_oputs, c_object.num_units)
        wh_and_V = np.append(wh.ravel(order='A'), bh.ravel(order='A'))
        return wh_and_V

    def func_activation(x_data, wh, bh, V):
        # perform forward propagation
        HLinput = np.dot(x_data, wh) - bh
        a1 = np.tanh(HLinput)
        OLinput = np.dot(a1, V.T)
        Oput = np.reshape(np.tanh(OLinput), (len(x_data), 1))
        return Oput

    def func_loss(V, w_all, EL, x, y_train, eta):
        q = eta
        p = len(x)
        wh, bh = MLP_EL.get_weights(w_all, EL)
        # print(wh)
        fx = MLP_EL.func_activation(x, wh, bh, V)
        rr = fx - y_train
        # print(y_train)
        reg_loss = (np.sum((y_train - fx) ** 2) / (2 * p)) + q * ((np.sum(V ** 2) + np.sum(wh ** 2) + np.sum(bh ** 2)))
        # print(V)
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
        fig.savefig('plot.png')
        plt.show()


class RBF_EL(object):
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        # self.centers = [random.uniform(-1, 1, self.indim) for i in range(self.numCenters)]
        # self.W = random.random((self.numCenters, self.outdim))

    def get_init(self, x):
        kmeans = KMeans(n_clusters=self.numCenters, random_state=0).fit(x)
        centers = kmeans.cluster_centers_
        # [random.uniform(0, 1, self.indim) for i in range(self.numCenters)]
        W = np.array(random.rand(self.numCenters, self.outdim))
        return np.append(W.ravel(order="A"), np.array(centers).ravel(order="A"))

    def extract_data(w_all, self):
        W_size = self.numCenters * self.outdim
        W = np.reshape(w_all[:W_size], (self.numCenters, self.outdim), order='A')
        C = np.reshape(w_all[W_size:], (self.numCenters, self.indim), order='A')
        return W, C

    def func_rbf(x_data, c_data, W, sig):
        sigma = sig
        n_rows = np.shape(x_data)[0]
        n_cols = np.shape(c_data)[0]
        mat_H = np.zeros((np.shape(x_data)[0], np.shape(c_data)[0]))  ##create a matrix H with zeros
        for row in range(n_rows):
            for col in range(n_cols):
                mat_H[row, col] = np.exp(-((np.sum((x_data[row] - c_data[col]) ** 2)) / sigma ** 2))
        out_put =  np.dot(np.matrix(mat_H), W)
        return out_put.T

    def func_loss(W, x_data, EL, C, y_train, eta, sigma):
        q = eta
        # W, C = RBF_EL.extract_data(ini_data,EL)
        fx = RBF_EL.func_rbf(x_data, C, W, sigma)
        p = len(x_data)
        rr = np.array(fx - y_train)
        reg_loss = sum((rr)**2)/(2*p)+ q*((np.sum(W**2)+np.sum(C**2)))
        return reg_loss[0]

    def plots(y_train, X_train):
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

        fig = plt.figure("plot1.png")
        ax = fig.gca(projection='3d')
        x1 = X_train[0:, :1]
        x2 = X_train[0:, 1:2]
        # x1, x2 = np.meshgrid(x1, x2)
        ax.plot_trisurf(x1.flatten(), x2.flatten(), y_train.flatten(), cmap=cm.jet, linewidth=0.1)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
