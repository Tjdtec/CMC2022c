import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import io


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, silhouette_score, davies_bouldin_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from PIL import Image

data_df = pd.read_excel('./data/concat_d1_d2.xlsx')
pb_ba_tuple, kkmn_tuple = data_df.groupby(['类型'])
feature_list = ['纹饰', '类型', '颜色']
# data_df = kkmn_tuple[1]
data_df = pb_ba_tuple[1]
data_df.reset_index(inplace=True, drop=True)
for f in feature_list:
    encoder = LabelEncoder()
    encoder_labels = pd.DataFrame(encoder.fit_transform(data_df[f]), columns=[f])
    if f != '类型':
        data_df.drop(columns=[f], inplace=True)
        data_df = pd.concat([data_df, encoder_labels], axis=1)
    else:
        data_df.drop(columns=[f], inplace=True)
        labels_df = encoder_labels

data_df.fillna(0, inplace=True)
# 删除信息文物编号
c_id = data_df['文物编号']
data_df.drop(columns=['文物编号'], inplace=True)
data_df.drop(columns=['表面风化'], inplace=True)
feature_names = list(data_df.columns)
data_array = data_df.values
labels = labels_df.values
scaler = StandardScaler()
data_array_transformed = scaler.fit_transform(data_array)

# for i in range(0, len(data_array_transformed)):
#     plt.scatter(c_id, data_array_transformed[:, i], c='grey')
#     # plt.savefig('./img/KKmn-' + feature_names[i] + '.png')
#     plt.savefig('./img/qb_Ba-' + feature_names[i] + '.png')
#     plt.show()

num_clusters = 2  # 设定簇的数量
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit_transform(data_array_transformed)
#  获取每个簇的中心
cluster_centers = kmeans.cluster_centers_
# 计算每个特征的贡献值
feature_contributions = np.max(cluster_centers, axis=0)
# 打印每个特征的贡献度
contributions_list = []
for idx, contribution in enumerate(feature_contributions):
    print(f"{feature_names[idx]}的贡献度: {contribution}")
    contributions_list.append(contribution)

print('hello')
sorted_indices = sorted(range(len(contributions_list)), key=lambda k: contributions_list[k], reverse=True)
new_data_tr = np.zeros_like(data_array_transformed)
count = 0
for i in sorted_indices:
    temp_array = data_array_transformed[:, i]
    new_data_tr[:, count] = temp_array
    print(f"排序后：{feature_names[i]}的贡献度:{contributions_list[i]}")
    count += 1

num_clusters = 2  # 设定簇的数量
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit_transform(new_data_tr[:, :5])

# 获取每个样本的所属簇
cluster_labels = kmeans.labels_
c0, c1, c = 0, 0, 0
for i, label in enumerate(cluster_labels):
    print(f"样本{i}属于簇{label}")
    if label == 0:
        c0 += 1
    elif label == 1:
        c1 += 1
    c += 1

print(f"簇0的比例：{c0 / c: .4f}")
print(f"簇1的比例：{c1 / c: .4f}")
# 计算
silhouette_avg = silhouette_score(new_data_tr[:, :5], cluster_labels)
davies_bouldin_index = davies_bouldin_score(new_data_tr[:, :5], cluster_labels)
print("轮廓系数:", silhouette_avg)
print("Davies-Bouldin指数:", davies_bouldin_index)