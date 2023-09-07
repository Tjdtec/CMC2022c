import numpy as np
import pandas as pd
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
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
data_df_list = [kkmn_tuple[1], pb_ba_tuple[1]]
# data_df = kkmn_tuple[1]
# data_df = pb_ba_tuple[1]
type_mean_list = []
for data_df in data_df_list:
    data_df.reset_index(inplace=True, drop=True)
    print('hello')
    feature_list = ['纹饰', '类型', '颜色', '表面风化']
    for f in feature_list:
        encoder = LabelEncoder()
        if f != '类型':
            data_df.drop(columns=[f], inplace=True)
            continue
        else:
            encoder_labels = pd.DataFrame(encoder.fit_transform(data_df[f]), columns=[f])
            labels_df = encoder_labels
            data_df.drop(columns=[f], inplace=True)

    data_df.fillna(0, inplace=True)
    # 删除信息文物编号
    c_id = data_df['文物编号']
    data_df.drop(columns=['文物编号'], inplace=True)
    feature_names = ['SiO2', 'Na2O', 'K2O', 'CaO', 'MgO', 'Al2O3', 'Fe2O3', 'CuO', 'PbO', 'BaO',
                     'P2O5', 'SrO', 'SnO2', 'SO2']

    data_array = data_df.values
    type_mean_list.append(np.mean(data_array, axis=0))


# 进行t检验
glass_type_a, glass_type_b = type_mean_list[0], type_mean_list[1]
scale = StandardScaler()
glass_type_a_t = scale.fit_transform(glass_type_a.reshape(-1, 1)).ravel()
glass_type_b_t = scale.fit_transform(glass_type_b.reshape(-1, 1)).ravel()
t_statistic, p_value = ttest_ind(glass_type_a_t, glass_type_b_t)

# 绘制箱线图
data = [glass_type_a_t, glass_type_b_t]
labels = ['KKMn Type', 'Pb-Ba Type']

plt.boxplot(data, labels=labels)
plt.title('Chemical Composition Comparison')
plt.ylabel('Value')
plt.xlabel('Glass Type')

plt.ylim(-1, 1.0)
plt.savefig('./img/box-plot.png')
plt.show()

# 输出t检验结果
print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")

# 判断差异性显著性
alpha = 0.05  # 显著性水平
if p_value < alpha:
    print("The difference is significant.")
else:
    print("The difference is not significant.")
