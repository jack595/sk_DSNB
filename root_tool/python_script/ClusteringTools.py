# -*- coding:utf-8 -*-
# @Time: 2021/6/10 9:26
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: ClusteringTools.py
import matplotlib.pylab as plt
import numpy as np
import sys
from collections import Counter
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/ProtonDecayML/spherical-k-means/spherecluster/")
from spherecluster import SphericalKMeans

plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

class Clustering_SKM3D:
    def __init__(self):
        self.skm:SphericalKMeans = SphericalKMeans(n_clusters=1, init='k-means++', n_init=20)

    def SetDataset(self,event2dimage:np.ndarray, x:np.ndarray, y:np.ndarray, z:np.ndarray):
        index_non_zeros = (event2dimage!=0)
        self.X = np.concatenate((x[index_non_zeros],y[index_non_zeros],z[index_non_zeros]), axis=1)
        self.event2dimage = event2dimage[index_non_zeros]
        return self.X

    def GetSKMLabel(self):
        return self.skm.labels_

    def GetNClusterList(self, n_clusters_range=range(1, 15)):
        self.Sum_of_squared_distances = []
        for k in n_clusters_range:
            km = SphericalKMeans(n_clusters=k)
            km.fit(self.X, sample_weight=self.event2dimage)
            self.Sum_of_squared_distances.append(km.inertia_)

    def OptimalNumberOfClusters(self):
        x1, y1 = 2, self.Sum_of_squared_distances[0]
        x2, y2 = 20, self.Sum_of_squared_distances[len(self.Sum_of_squared_distances)-1]

        distances = []
        for i in range(len(self.Sum_of_squared_distances)):
            x0 = i+2
            y0 = self.Sum_of_squared_distances[i]
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(numerator/denominator)
            self.n_optimal_cluster = distances.index(max(distances)) + 2
        return self.n_optimal_cluster

    def FitOptimalCluster(self):
        skm = SphericalKMeans(n_clusters=self.n_optimal_cluster)
        self.skm = skm.fit(self.X)
        return self.skm

    def WorkFlowKMeans(self,event2dimage:np.ndarray, x:np.ndarray, y:np.ndarray, z:np.ndarray):
        self.SetDataset(event2dimage, x, y, z)
        self.GetNClusterList()
        self.OptimalNumberOfClusters()
        self.FitOptimalCluster()
        return self.GetSKMLabel()

    def PlotClusteredData(self):
        label = self.GetSKMLabel()
        self.counter_labels = Counter(label)
        v_labels = list(self.counter_labels.keys())
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection="3d")
        for i in range(len(v_labels)):
            ax1.scatter(self.X[label==v_labels[i],0], self.X[label==v_labels[i],1],
                        self.X[label==v_labels[i],2], s=1)
        plt.show()







