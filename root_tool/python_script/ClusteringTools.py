# -*- coding:utf-8 -*-
# @Time: 2021/6/10 9:26
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: ClusteringTools.py
import matplotlib.pylab as plt
import matplotlib.cm as cm
import numpy as np
import sys
from collections import Counter
from sklearn.metrics import silhouette_score, silhouette_samples
sys.path.append("/afs/ihep.ac.cn/users/l/luoxj/ProtonDecayML/spherical-k-means/spherecluster/spherecluster")
from spherecluster import VonMisesFisherMixture
from spherical_kmeans import SphericalKMeans
from sklearn.cluster import KMeans
# from von_mises_fisher_mixture import VonMisesFisherMixture


from mpl_toolkits.mplot3d.axes3d import Axes3D
from collections import Counter
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")

def SinusoidaProjection_xyz(x:np.ndarray, y:np.ndarray, z:np.ndarray):
    theta = np.arctan2((x**2+y**2)**0.5,z)
    # theta[theta>np.pi/2] = theta[theta>np.pi/2]-np.pi
    phi = np.arctan2(y, x)
    cos_theta = (x**2+y**2)**0.5/((x**2+y**2+z**2)**0.5)
    x_projection = phi * cos_theta
    # x_projection = phi * np.cos(theta)
    y_projection = theta
    return (x_projection, y_projection)

def SinusoidaProjection_theta_phi(theta:np.ndarray, phi:np.ndarray):
    x_projection = phi * np.sin(theta)
    y_projection = theta
    return (x_projection, y_projection)

class Clustering_SKM3D:
    def __init__(self):
        # self.skm:SphericalKMeans = SphericalKMeans(n_clusters=1, init='k-means++', n_init=20)
        self.plot_find_optimal_cluster = True
        self.use_vonMiseFisherMixture = False
        self.use_kmeans_in_projection = True
        self.use_Silhouetee_score_get_optimal = False
        self.with_weight_by_repeat_vertex = False
        if (self.use_kmeans_in_projection or self.use_vonMiseFisherMixture) and self.with_weight_by_repeat_vertex:
            print("With weight by repeat vertex is not supported , so turn it off!!!!!")
            self.with_weight_by_repeat_vertex = False

    def ShrinkDataToPlot(self):
        v_label = self.GetSKMLabel()
        self.X_shrink = []
        self.label_shrink = []
        self.X_shrink.append(self.X[0])
        self.label_shrink.append(v_label[0])
        for i in range(1, len(v_label)):
            if np.all(self.X[i] == self.X_shrink[-1]):
                # if v_label[i] != self.label_shrink[-1]:
                #     print(self.X_shrink[-5:], self.label_shrink[-5:])
                continue
            else:
                self.X_shrink.append(self.X[i])
                self.label_shrink.append(v_label[i])
        self.X_shrink = np.array(self.X_shrink)
        self.label_shrink = np.array(self.label_shrink)

    def SetDataset(self,event2dimage:np.ndarray, x:np.ndarray, y:np.ndarray, z:np.ndarray):
        """

        :param event2dimage: 2 layer of data, event2dimage[0] is for the energy, event2dimage[1]
            is for the hittime.
        :param x:
        :param y:
        :param z:
        :return:
        """
        index_non_zeros = (event2dimage[0] != 0)
        if self.with_weight_by_repeat_vertex:
            self.X_plot = np.array([x[index_non_zeros],y[index_non_zeros],z[index_non_zeros]]).T.reshape((-1,3))
            self.eventimage = (event2dimage[0]/10)[index_non_zeros].astype(int)
            self.X = np.repeat(self.X_plot, self.eventimage,axis=0)
            return self.X
        else:
            if self.use_kmeans_in_projection:
                (x_projection, y_projection) = SinusoidaProjection_xyz(x[index_non_zeros], y[index_non_zeros], z[index_non_zeros])
                self.X = np.array([x_projection, y_projection]).T.reshape((-1, 2))
                self.eventimage = event2dimage[0][index_non_zeros]
                self.X_plot = self.X
                return self.X
            else:
                # self.X = np.concatenate([x[index_non_zeros],y[index_non_zeros],z[index_non_zeros]], axis=0)
                self.X = np.array([x[index_non_zeros],y[index_non_zeros],z[index_non_zeros]]).T.reshape((-1,3))
                self.eventimage = event2dimage[0][index_non_zeros]
                self.X_plot = self.X
                return self.X

    def GetSKMLabel(self):
        return self.skm.labels_
    def GetSKMCenter(self):
        return self.skm.cluster_centers_

    def GetNClusterList(self, n_clusters_range=range(1, 7)):
        if self.use_Silhouetee_score_get_optimal and n_clusters_range[0]==1:
            n_clusters_range = n_clusters_range[1:]
        self.n_clusters_range = n_clusters_range
        self.Sum_of_squared_distances = []
        self.list_Silhouette = []
        for k in n_clusters_range:
            if len(self.X)< k:
                self.n_clusters_range = range(1, k)
                break
            # print("processing n_cluster:\t",k )
            if self.use_vonMiseFisherMixture:
                km = VonMisesFisherMixture(n_clusters=k, n_jobs=-1)
                cluster_labels = km.fit_predict(self.X)
            else:
                if not self.use_kmeans_in_projection:
                    km = SphericalKMeans(n_clusters=k, n_init=40)
                    if self.clustering_with_weight:
                        cluster_labels = km.fit_predict(self.X, sample_weight=self.eventimage)
                    else:
                        cluster_labels = km.fit_predict(self.X)
                else:
                    km = KMeans(n_clusters=k)
                    if self.clustering_with_weight:
                        cluster_labels = km.fit_predict(self.X, sample_weight=self.eventimage)
                    else:
                        cluster_labels = km.fit_predict(self.X)

            self.Sum_of_squared_distances.append(km.inertia_)

            if self.use_Silhouetee_score_get_optimal:
                fig, (ax1, ax2) = plt.subplots(1, 2)

                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                silhouette_avg = silhouette_score(self.X, cluster_labels)
                print("For n_clusters =", k,
                      "The average silhouette_score is :", silhouette_avg)

                # Compute the silhouette scores for each sample
                sample_silhouette_values = silhouette_samples(self.X, cluster_labels)

                # self.list_Silhouette.append(silhouette_score(self.X, km.labels_, metric="euclidean"))
                # plt.plot(self.n_clusters_range, self.list_Silhouette)

                y_lower = 10
                for i in range(k):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = \
                        sample_silhouette_values[cluster_labels == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = cm.nipy_spectral(float(i) / k)
                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                      0, ith_cluster_silhouette_values,
                                      facecolor=color, edgecolor=color, alpha=0.7)

                    # Label the silhouette plots with their cluster numbers at the middle
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                ax1.set_title("The silhouette plot for the various clusters.")
                ax1.set_xlabel("The silhouette coefficient values")
                ax1.set_ylabel("Cluster label")

                # The vertical line for average silhouette score of all the values
                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                # 2nd Plot showing the actual clusters formed
                colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
                ax2.scatter(self.X[:, 0], self.X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                            c=colors, edgecolor='k')

                # Labeling the clusters
                centers = km.cluster_centers_
                # Draw white circles at cluster centers
                ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c="white", alpha=1, s=200, edgecolor='k')

                for i, c in enumerate(centers):
                    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                s=50, edgecolor='k')

                ax2.set_title("The visualization of the clustered data.")
                ax2.set_xlabel("Feature space for the 1st feature")
                ax2.set_ylabel("Feature space for the 2nd feature")

                plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                              "with n_clusters = %d" % k),
                             fontsize=14, fontweight='bold')

        plt.show()

    def PlotSumOfDistance(self):
        plt.figure()
        plt.plot(self.n_clusters_range,self.Sum_of_squared_distances)
        plt.xlabel("n clusters")
        plt.ylabel("Sum of distance to closest cluster center")


    def OptimalNumberOfClusters(self):
        x1, y1 = self.n_clusters_range[0], self.Sum_of_squared_distances[0]
        x2, y2 = self.n_clusters_range[-1], self.Sum_of_squared_distances[len(self.Sum_of_squared_distances)-1]

        distances = []
        for i in range(len(self.Sum_of_squared_distances)):
            x0 = i+self.n_clusters_range[0]
            y0 = self.Sum_of_squared_distances[i]
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(numerator/denominator)
            self.n_optimal_cluster = distances.index(max(distances)) + self.n_clusters_range[0]
        return self.n_optimal_cluster

    def OptimalNumberOfClusters_SilMethod(self):
        self.n_optimal_cluster = self.n_clusters_range[np.argmax(self.list_Silhouette)]
        return self.n_optimal_cluster

    def FitOptimalCluster(self):
        if self.use_vonMiseFisherMixture:
            skm = VonMisesFisherMixture(n_clusters=self.n_optimal_cluster, n_jobs=-1)
            # skm = VonMisesFisherMixture(n_clusters=2)
            self.skm = skm.fit(self.X)
        else:
            if not self.use_kmeans_in_projection:
                skm = SphericalKMeans(n_clusters=self.n_optimal_cluster, verbose=0, n_init=40)
                # skm = SphericalKMeans(n_clusters=2)
                if self.clustering_with_weight:
                    self.skm = skm.fit(self.X, sample_weight=self.eventimage)
                else:
                    self.skm = skm.fit(self.X)
            else:
                self.skm = KMeans(n_clusters=self.n_optimal_cluster)
                if self.clustering_with_weight:
                    self.skm.fit(self.X, sample_weight=self.eventimage)
                else:
                    self.skm.fit(self.X)

        return self.skm

    def WorkFlowKMeans(self,event2dimage:np.ndarray, x:np.ndarray, y:np.ndarray, z:np.ndarray, clustering_with_weight= False):
        self.clustering_with_weight = clustering_with_weight
        self.R = (x[0]**2 + y[0]**2 + z[0]**2)**0.5
        self.SetDataset(event2dimage, x/self.R, y/self.R, z/self.R)
        self.GetNClusterList()
        if self.use_Silhouetee_score_get_optimal:
            self.OptimalNumberOfClusters_SilMethod()
        else:
            self.OptimalNumberOfClusters()
        self.FitOptimalCluster()
        return self.GetSKMLabel()

    def SetDatasetInterp(self,event2dimg_interp, x, y, z):
        index_non_zeros = (event2dimg_interp[0]!=0)
        self.eventimage_interp = event2dimg_interp[0][index_non_zeros]
        self.x_interp = x[index_non_zeros]
        self.y_interp = y[index_non_zeros]
        self.z_interp = z[index_non_zeros]
    def PlotBaseSphere(self, ax:Axes3D, R=17.7*1000):
        # draw sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)*R
        y = np.sin(u)*np.sin(v)*R
        z = np.cos(v)*R
        ax.plot_wireframe(x, y, z, color="black", linewidth=0.8, ls="--")
    def PlotClusteredData(self, pdf_out=None):
        if pdf_out==None and self.plot_find_optimal_cluster:
            self.PlotSumOfDistance()
        if self.with_weight_by_repeat_vertex:
            self.ShrinkDataToPlot()
        label = self.GetSKMLabel()
        v_centers = self.GetSKMCenter()
        v_labels = np.unique(label)
        fig = plt.figure()
        if not self.use_kmeans_in_projection:
            ax1 = fig.add_subplot(121, projection="3d")
        else:
            ax1 = fig.add_subplot(121)

        if not self.with_weight_by_repeat_vertex:
            if not self.use_kmeans_in_projection:
                for i in range(len(v_labels)):
                    ax1.scatter(self.X[label==v_labels[i],0]*self.R, self.X[label==v_labels[i],1]*self.R,
                                self.X[label==v_labels[i],2]*self.R, s=3)
            else:
                for i in range(len(v_labels)):
                    ax1.scatter(self.X[label == v_labels[i], 0] , self.X[label == v_labels[i], 1] ,
                                 s=3)
        else:
            for i in range(len(v_labels)):
                ax1.scatter(self.X_shrink[self.label_shrink==v_labels[i],0]*self.R, self.X_shrink[self.label_shrink==v_labels[i],1]*self.R,
                            self.X_shrink[self.label_shrink==v_labels[i],2]*self.R, s=3 )
        ax1.set_title("$n_{clusters}=$"+str(self.n_optimal_cluster))
        if not self.use_kmeans_in_projection:
            for center in v_centers:
                ax1.scatter(center[0] * self.R, center[1] * self.R, center[2] * self.R, s=60, marker="v")
            self.PlotBaseSphere(ax1)
        else:
            for center in v_centers:
                ax1.scatter(center[0] , center[1], s=40, marker="v")

        if not self.use_kmeans_in_projection:
            ax2 = fig.add_subplot(122, projection="3d")
            img = ax2.scatter(self.X_plot[:, 0] * self.R, self.X_plot[:, 1] * self.R, self.X_plot[:, 2] * self.R,
                              c=self.eventimage, cmap=plt.cm.hot, s=3)
            self.PlotBaseSphere(ax2)
        else:
            ax2 = fig.add_subplot(122)
            img = ax2.scatter(self.X_plot[:,0] ,self.X_plot[:,1], c=self.eventimage, cmap=plt.cm.hot, s=3 )

        ax2.set_title("$E_{quench}$")
        # if len(self.eventimage_interp) != 0:
        #     ax2.scatter(self.x_interp, self.y_interp, self.z_interp, c=self.eventimage_interp, s=2, cmap=plt.cm.hot)

        fig.colorbar(img, orientation="horizontal")
        if pdf_out == None:
            plt.show()
        else:
            pdf_out.savefig()
            plt.close()







