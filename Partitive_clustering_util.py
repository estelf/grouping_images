"""
非階層的クラスタリング補助モジュール
---
"""
import numpy as np
import tqdm
import umap
from matplotlib import pyplot as plt


class Visualization:
    def calc_silhouette(data, clusters):
        """
        クラスタリングのシルエット係数の計算
        
        Parameters
        ----------
        data : array-like
            データサンプル
        clusters : array-like
            各データサンプルが属するクラスタ
        """
        n = len(data)
        num_clusters = len(np.unique(clusters))
        cl_ids = []
        centroids = []
        print("  データ成型")
        for c in tqdm.tqdm(range(num_clusters)):
            ids = np.where(clusters == c)[0]
            cl_ids.append(ids)
            centroids.append(np.average(data[ids], axis=0))
        silhouette = np.zeros(n)

        print("  シルエット分析中")
        for c in tqdm.tqdm(range(num_clusters), desc="全体進捗"):
            for i in tqdm.tqdm(cl_ids[c],leave=False,colour="green", desc="クラスタ内進捗"):
                nearest_clusters = np.argsort(np.sum((data[i]-centroids)**2, axis=1))
                c_nearest = nearest_clusters[1] if nearest_clusters[0]==c else nearest_clusters[0]
                a = np.sum(np.sqrt(np.sum((data[i]-data[cl_ids[c]])**2, axis=1))) / (len(cl_ids[c])-1)
                b = np.average(np.sqrt(np.sum((data[i]-data[cl_ids[c_nearest]])**2, axis=1)))
                silhouette[i] = (b-a) / max(a, b)
        return silhouette


    def plot_silhouette(data, clusters,imgname):
        """
        シルエット係数を計算・描画
        """
        cl_size = {}
        for c in np.unique(clusters):
            cl_size[c] = np.count_nonzero(clusters == c)
        silhouette = Visualization.calc_silhouette(data, clusters)
        silhouette_ave = np.nanmean(silhouette)
        ids_sorted = np.lexsort((silhouette, clusters))

        data_sorted = np.array(data[ids_sorted])
        silhouette_sorted = np.array(silhouette[ids_sorted])
        
        plt.figure(figsize=(24,12))
        # クラスタの描画
        plt.subplot(1, 2, 1)

        plt.title('Clusters')
        start = 0
        if len(data_sorted[0])>2:
            mapper = umap.UMAP(random_state=0)
            data_sorted = mapper.fit_transform(data_sorted)


        #data_sorted = TSNE(n_components=2).fit_transform(data_sorted)

        for c in range(len(cl_size)):
            plt.scatter(data_sorted[start:start+cl_size[c],0], data_sorted[start:start+cl_size[c],1], s=20, label='Cluster {}'.format(c+1))
            start += cl_size[c]

        plt.legend(loc='upper left')
        # シルエット図の描画
        plt.subplot(1, 2, 2)
        plt.title('Silhouette')
        plt.xlabel('Silhouette Coefficient')
        plt.ylabel('Number of Data')
        plt.axvline(silhouette_ave, c='black', linestyle='dashed', label='average = {:.3f}'.format(silhouette_ave))
        start = 0
        for c in range(len(cl_size)):
            plt.barh(range(start, start+cl_size[c]), silhouette_sorted[start:start+cl_size[c]], height=1.0, label='Cluster {}'.format(c+1))
            start += cl_size[c]
        plt.legend(loc='upper left')
        plt.savefig(f"{imgname}.png")
        plt.show()
        

class data_transform:
    def cluster_to_labels(sample,clusters):
        """
        pyclusteringでsklearnフォーマットラベルデータを作る
        ---
        sample : 全データ
        clusters : pyclusteringのクラスタ値
        """
        labels=np.array([0 for i in sample])
        for i,sep in enumerate(clusters):
            for ii in sep:
                labels[ii]=i
        return  labels
