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
# qb_ba_tuple, kkmn_tuple = data_df.groupby(['类型'])
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
# 高钾：1,  铅钡:0
labels_names = ['Pb-Ba', 'KKMn']
data_array = data_df.values
labels = labels_df.values
min_max = MinMaxScaler()
data_array_transformed = min_max.fit_transform(data_array)
rf_model = RandomForestClassifier()
param_grid = {
    'n_estimators': [5, 10, 15, 20]
}
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(data_array_transformed, labels)
# 打印最佳参数和对应的性能
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

best_rf_model = grid_search.best_estimator_
feature_importance = best_rf_model.feature_importances_

# output_dir = "./img/"
# os.makedirs(output_dir, exist_ok=True)
# # 遍历每棵树并添加到合并图中
# for idx, tree in enumerate(best_rf_model.estimators_):
#     dot_data = export_graphviz(tree, out_file=None, feature_names=feature_names,
#                                class_names=labels_names, filled=True, rounded=True,
#                                special_characters=True)
#     graph = graphviz.Source(dot_data)
#     # 保存单棵树的图像到内存中
#     image_stream = io.BytesIO(graph.pipe(format='jpg'))
#     image_path = os.path.join(output_dir, f"tree_{idx}.png")  # 使用.png格式保存图像
#     graph.format = 'png'
#     graph.render(filename=image_path, cleanup=True)
# print("Images saved to 'img' directory.")
test_df = pd.read_excel('./data/data3.xlsx')
test_df.drop(columns=['表面风化'], inplace=True)
c_id = test_df['文物编号']
test_df.drop(columns=['文物编号'], inplace=True)
test_df.fillna(0, inplace=True)
test_values = test_df.values
labels_hat = grid_search.predict(X=test_values)

for i in range(0, len(c_id)):
    print(f"{c_id[i]} 可能是 {labels_names[labels_hat[i]]} 类别")

# # 可视化特征重要性
# plt.figure(figsize=(10, 8))
# plt.rcParams['font.sans-serif'] = 'SimHei'
# plt.bar(range(len(feature_importance)), feature_importance, color=['r', 'g', 'b', 'y'], alpha=0.3)
# plt.xticks(range(len(feature_importance)), feature_names, rotation=45)
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.title('Feature Importance in Random Forest')
# plt.savefig('./img/feature_importance.png')
# plt.show()
for feature, importance in zip(feature_names, feature_importance):
    print(f"{feature}: {importance: .4f}")
