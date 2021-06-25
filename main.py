import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

os.chdir(r'E:\百度下载\测测你的一见钟情指数')
df = pd.read_csv('speed_dating_train.csv', encoding='gbk')
dating_test = pd.read_csv('speed_dating_test.csv', encoding='gbk')

percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({
    'column_name': df.columns,
    'percent_missing': percent_missing
})
missing_value_df.sort_values(by='percent_missing')

# 多少人通过Speed Dating找到了对象
plt.subplots(figsize=(3, 3), dpi=110, )
# 构造数据
size_of_groups = df.match.value_counts().values

single_percentage = round(size_of_groups[0] / sum(size_of_groups) * 100, 2)
matched_percentage = round(size_of_groups[1] / sum(size_of_groups) * 100, 2)
names = [
    'Single:' + str(single_percentage) + '%',
    'Matched' + str(matched_percentage) + '%']

# 多少女生通过Speed Dating找到了对象
plt.subplots(figsize=(3, 3), dpi=110, )
# 构造数据
size_of_groups = df[df.gender == 0].match.value_counts().values  # 男生只需要吧0替换成1即可

single_percentage = round(size_of_groups[0] / sum(size_of_groups) * 100, 2)
matched_percentage = round(size_of_groups[1] / sum(size_of_groups) * 100, 2)

# 年龄分布
age = df[np.isfinite(df['age'])]['age']

date_df = df[[
    'iid', 'gender', 'pid', 'match', 'int_corr', 'samerace', 'age_o',
    'race_o', 'pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb',
    'pf_o_sha', 'dec_o', 'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'like_o',
    'prob_o', 'met_o', 'age', 'race', 'imprace', 'imprelig', 'goal', 'date',
    'go_out', 'career_c', 'sports', 'tvsports', 'exercise', 'dining',
    'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv',
    'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'attr1_1',
    'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'attr3_1', 'sinc3_1',
    'fun3_1', 'intel3_1', 'dec', 'attr', 'sinc', 'intel', 'fun', 'like',
    'prob', 'met'
]]
dating_test = dating_test[[
    'iid', 'gender', 'pid', 'int_corr', 'samerace', 'age_o',
    'race_o', 'pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb',
    'pf_o_sha', 'dec_o', 'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'like_o',
    'prob_o', 'met_o', 'age', 'race', 'imprace', 'imprelig', 'goal', 'date',
    'go_out', 'career_c', 'sports', 'tvsports', 'exercise', 'dining',
    'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv',
    'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'attr1_1',
    'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'attr3_1', 'sinc3_1',
    'fun3_1', 'intel3_1', 'dec', 'attr', 'sinc', 'intel', 'fun', 'like',
    'prob', 'met'
]]
# clean_df = date_df[['attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'match']]
date_df.dropna(inplace=True)
dating_test = dating_test.fillna(0)
x = date_df.drop(columns=['match'])
y = date_df['match']

# oversample = imblearn.over_sampling.SVMSMOTE()
# x, y = oversample.fit_resample(X, y)

# 做训练集和测试集分割
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
# logistic regression classification model
model = ExtraTreesClassifier()
model.fit(X_train, y_train)
predict_train_lrc = model.predict(X_train)
predict_test_lrc = model.predict(X_test)

print('Training Accuracy:', metrics.accuracy_score(y_train, predict_train_lrc))
print('Validation Accuracy:', metrics.accuracy_score(y_test, predict_test_lrc))
predict_test = model.predict(dating_test)
print(predict_test)
np.savetxt('predict_test.csv', predict_test)
