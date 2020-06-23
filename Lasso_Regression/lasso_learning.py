"""
Lasso_regulation_program

- train_data = 21대 총선 자료,
    d = 더불어 민주당
    m = 미래통합당
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

# Make graph font English to Korean
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


# Training & Test Data Load
train_data = pd.read_csv('C:/Users/khw08/Desktop/OSBAD_Project/Regression/KoNLPY_M&D_train_CSV_2.csv')

# Na Data dop
train_data.dropna()

# Arrange Data Set
x = train_data.drop(['d'], axis=1)
x_2 = x.drop(['m'], axis=1)
y = train_data.loc[:, ['d']]
# 내가 한 태깅 789까지임
x_train = x_2.loc[:4000, :]
y_train = y.loc[:4000, :]

x_test = x_2.loc[4000:5500, :]
y_test = y.loc[4000:5500, :]
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)    # Split Data Set
predictors = x_train.columns                                                # Save tarin_X attributes

"""
Lasso Regression
lassoReg   = Coefficient List
predictors = Attributes List
coef       = DataFrame with Attributes & Coefficient
pre_coef   = Except Zero Coefficient Value List
"""
lassoReg = Lasso(alpha=0.0001)                                          # Call Lasso Regression Function
lassoReg.fit(x_train, y_train)                                              # Fit Data in Lasso function
print(lassoReg.coef_)
coef = Series(lassoReg.coef_, predictors).sort_values()
print(coef)
# Save Coefficient

print(np.sum(lassoReg.coef_ != 0))                                          # Check the number of valid Coefficient
coef_pre = coef[coef != 0.0][coef != -0.0]                                  # Except Zero Coefficient


coef_pre.plot(kind='bar')                                                   # Show Graph Coefficient formed by Bar
plt.show()


alpha_set = [0.00001, 0.0001, 0.005, 0.001, 0.01, 0.1, 1]
max_inter_set = [100000000, 10000000, 5000000, 1000000, 100000, 10000, 1000]

train_score = []
test_score = []
used_feature = []

for a, m in zip(alpha_set, max_inter_set):

    lasso = Lasso(alpha=a, max_iter=m).fit(x_train, y_train)

    la_tr_score = round(lasso.score(x_train, y_train), 3)

    la_te_score = round(lasso.score(x_test, y_test), 3)

    number_used = np.sum(lasso.coef_ != 0)



    train_score.append(la_tr_score)

    test_score.append(la_te_score)

    used_feature.append(number_used)



index = np.arange(len(alpha_set))

bar_width = 0.35

plt.bar(index, train_score, width=bar_width, label='train')

plt.bar(index+bar_width, test_score, width=bar_width, label='test')

plt.xticks(index+bar_width/2, alpha_set) # bar그래프 dodge를 하기 위해 기준값에 보정치를 더해줍니다.



for i, (ts, te) in enumerate(zip(train_score, test_score)):

    plt.text(i, ts+0.01, str(ts), horizontalalignment='center')

    plt.text(i+bar_width, te+0.01, str(te), horizontalalignment='center')



plt.legend(loc=1)

plt.xlabel('alpha')

plt.ylabel('score')

plt.show()

'''
y_pred = lassoReg.predict(x_test)
print(y_pred)

arr = []
for idx, val in enumerate(y_pred):
    if val >= 0.5:
        val = 1
        arr.append(val)
    elif val < 0.5:
        val = 0
        arr.append(val)

print(arr)

dataframe_y = pd.DataFrame(arr)
dataframe_y.to_csv('C:/Users/khw08/Desktop/Lasso_y_pred.csv', encoding='utf-8-sig')

'''
# Predict Target value


"""Testing dummy Codes.."""

'''
for idx, val in enumerate(coef):
    print(val)
'''
'''
#print(y_pred)
arr = []
for idx, val in enumerate(y_pred):
    if val >= 0.5:
        val = 1
        arr.append(val)
    elif val < 0.5:
        val = 0
        arr.append(val)

print(arr)
'''
'''
# x_test 값을 Dataframe 형식으로 저장
dataframe_x = pd.DataFrame(x_test)
dataframe_x.to_csv('C:/Users/khw08/Desktop/Lasso_x_test_21.csv', encoding='utf-8-sig')

# y_pred2 값을 Dataframe 형식으로 저장
dataframe_y = pd.DataFrame(y_pred)
dataframe_y.to_csv('C:/Users/khw08/Desktop/Lasso_y_pred_21.csv', encoding='utf-8-sig')
'''

# 계수
# Rcoef = pd.read_csv('C:/Users/khw08/Desktop/Lasso_coef.csv')


#dataframe_x = pd.DataFrame(coef)
#dataframe_x.to_csv('C:/Users/khw08/Desktop/Lasso_coef.csv', encoding='utf-8-sig')


"""arr = []
for idx, val in enumerate(coef):
    if val != 0.0 or -0.0:
        arr.append(val)

print(arr)

a = Series(arr).sort_values()"""