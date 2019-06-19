# ���J�M��
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import silhouette_samples, silhouette_score

np.random.seed(5)

%matplotlib inline







# �ͦ� 5 �s���
X, y = make_blobs(n_samples=500,
                  n_features=2,
                  centers=5,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=123) 

# �]�w�ݭn�p�⪺ K �ȶ��X
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]










# �p���ø�s�������R�����G
# �]�U�C���j��g�k, �L�k�A�����p����϶�, �Ш���
for n_clusters in range_n_clusters:
    # �]�w�p�ϱƪ��� 1 row 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # ���Ϭ��������R(Silhouette analysis), ���M�����Y�ƽd��b(-1,1)�϶�, ���d�Ҥ���������, �]���ڭ̧���ܽd��w�b(-0.1,1)����
    ax1.set_xlim([-0.1, 1])
    # (n_clusters+1)*10 �o�����O�ΨӦb���P�����϶���J�ť�, ���ϧάݰ_�ӧ�M��
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # �ŧi KMean ���s��, �� X �V�m�ùw��
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # �p��Ҧ��I�� silhouette_score ����
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # �p��Ҧ��˥��� The silhouette_score
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # �������s i �˥����������ơA�ù復�̶i��Ƨ�
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # �b�C�Ӷ��s�����ФW i ���ƭ�
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # �p��U�@�� y_lower ����m
        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # �N silhouette_score �����Ҧb��m, �e�W�@�������u
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # �M�� y �b����u
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # �k�ϧڭ̥Ψӵe�W�C�Ӽ˥��I�����s���A, �q�t�@�Ө����[����s�O�_����
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # �b�k�ϨC�@�s�����߳B, �e�W�@�Ӷ��üе��������s��
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()




