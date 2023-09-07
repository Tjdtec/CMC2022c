import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder


def data_predict():
    data_df = pd.read_excel('./data/concat_d1_d2.xlsx')

    qb_Ba_no_weathering, qb_Ba_weathering, K_no_weathering, K_weathering = data_df.groupby(['类型', '表面风化'])

    qb_Ba_no_weathering = qb_Ba_no_weathering[1].drop(columns=['文物编号'])
    qb_Ba_weathering = qb_Ba_weathering[1].drop(columns=['文物编号'])
    K_no_weathering = K_no_weathering[1].drop(columns=['文物编号'])
    K_weathering = K_weathering[1].drop(columns=['文物编号'])
    list = []
    for df in [qb_Ba_no_weathering, qb_Ba_weathering, K_no_weathering, K_weathering]:
        for feature in ['纹饰', '类型', '颜色', '表面风化']:
            encoder = LabelEncoder()
            encoder_df = pd.DataFrame(encoder.fit_transform(df[feature]), columns=[feature])
            df.drop(columns=[feature], inplace=True)
            df = pd.concat([df, encoder_df], axis=1)
        list.append(df)
    qb_Ba_no_weathering = list[0].fillna(0).values
    qb_Ba_weathering = list[1].fillna(0).values
    K_no_weathering = list[2].fillna(0).values
    K_weathering = list[3].fillna(0).values

    pca = PCA(n_components=3)
    k_weathering_transformed = pca.fit_transform(K_weathering)
    qb_Ba_weathering_transformed = pca.fit_transform(qb_Ba_weathering)

    # # 降维坐标图
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # dx = dy = 0.8  # 柱子的宽度和深度
    # dz = 0
    x_grid, y_grid = k_weathering_transformed[:, 0], k_weathering_transformed[:, 1]
    z_grid = k_weathering_transformed[:, 2]
    # ax.grid(False)
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # ax.set_title('Three-Dimensional Scatter Plot')
    #
    # plt.show()

    # # # 创建三维散点图
    # fig = px.scatter_3d(x=x_grid, y=y_grid, z=z_grid, opacity=0.7)
    #
    # # fig = go.Figure(data=[go.Contour(z=z_grid, x=x_grid, y=y_grid)])
    # fig.update_layout(title="PCA Denominational Plot",
    #                   scene=dict(xaxis_title='PCA1', yaxis_title='PCA2', zaxis_title='PCA3'))

    # # 显示图形
    # fig.show()

    k_model = LinearRegression()
    qb_model = LinearRegression()
    k_model.fit(k_weathering_transformed, K_no_weathering[:-1])
    qb_model.fit(qb_Ba_weathering_transformed[:43], qb_Ba_no_weathering)

    k_no_weathering_pred = k_model.predict(k_weathering_transformed)
    qb_Ba_no_weathering_pred = qb_model.predict(qb_Ba_weathering_transformed)
    qb_Ba_weathering_transformed_2 = np.pad(qb_Ba_weathering_transformed[43:56], ((0, 5), (0, 0)), mode='constant',
                                            constant_values=0)
    qb_Ba_no_weathering_pred_2 = qb_model.predict(qb_Ba_weathering_transformed_2)

    # 计算均方误差（Mean Squared Error）
    mse = mean_squared_error(K_no_weathering[:-1], k_no_weathering_pred)
    print("Mean Squared Error:", mse)
    mse = mean_squared_error(qb_Ba_no_weathering, qb_Ba_no_weathering_pred[:43])
    print("Mean Squared Error:", mse)

    # 计算均绝对误差（Mean Absolute Error）
    mae = mean_absolute_error(K_no_weathering[:-1], k_no_weathering_pred)
    print("Mean Absolute Error:", mae)
    mae = mean_absolute_error(qb_Ba_no_weathering, qb_Ba_no_weathering_pred[:43])
    print("Mean Absolute Error:", mae)

    coefficients_1 = qb_model.coef_
    intercept = qb_model.intercept_
    # coefficients_1 = k_model.coef_
    # intercept = k_model.intercept_
    print('系数权重:\n', coefficients_1)
    print('偏置：\n', intercept)
    print('hello')


if __name__ == '__main__':
    data_predict()
