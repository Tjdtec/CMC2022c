import pandas as pd

from scipy.stats import chi2_contingency


def data_process():
    total_df = pd.read_excel('./data/data1.xlsx')
    total_df.dropna(inplace=True)
    # 构建交叉表
    column_names = ['纹饰', '类型', '颜色']
    cross_tab_dict = {}
    for name in column_names:
        cross_tab = pd.crosstab(index=total_df[name], columns=total_df['表面风化'])
        print(name + '&&表面风化')
        cross_tab_dict = {name: cross_tab}
        cross_value = cross_tab.values
        chi2, p, dof, expected = chi2_contingency(cross_value)
        print("Chi-Square Statistic:", chi2)  # 卡方统计值
        print("P-value:", p)  # p值
        print("Degrees of Freedom:", dof)  # 自由度
        print("Expected Frequencies:\n", expected)  # 期望频数
        print('--------------------------------------------------------')

    return cross_tab_dict


def main():
    cross_tab_dict = data_process()


if __name__ == '__main__':
    main()
