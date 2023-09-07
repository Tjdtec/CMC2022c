import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

data_df = pd.read_excel('./data/concat_d1_d2.xlsx')
# qb_ba_tuple, kkmn_tuple = data_df.groupby(['类型'])
print('hello')
feature_list = ['纹饰', '类型', '颜色', '表面风化']
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
feature_names = data_df.columns
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
#  特征的直观图
# 获取特征重要性
best_rf_model = grid_search.best_estimator_
feature_importance = best_rf_model.feature_importances_

# 可视化特征重要性
plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.bar(range(len(feature_importance)), feature_importance, color=['r', 'g', 'b', 'y'], alpha=0.3)
plt.xticks(range(len(feature_importance)), feature_names, rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Random Forest')
plt.savefig('./img/feature_importance.png')
plt.show()
for feature, importance in zip(feature_names, feature_importance):
    print(f"{feature}: {importance: .4f}")
