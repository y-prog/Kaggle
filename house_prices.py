import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from scipy import stats
from sklearn.metrics import mean_squared_error
import seaborn as sns
import itertools
from sklearn.metrics import mean_squared_log_error
from scipy.stats import pearsonr
import xgboost as xgb



train_data=pd.read_csv(r'train.csv')
test_data=pd.read_csv(r'test.csv')
print(type(train_data.iloc[7][3]),type(train_data.iloc[7][6]),'afdsadsafafafdsadfadf')
print((train_data.iloc[7][3]),(train_data.iloc[7][6]),'afdsadsafafafdsadfadf')

print(train_data.shape)

sum_of_null_values= ((train_data.isnull().sum()))   # categories with n null values
print(sum_of_null_values.values.tolist())
print (sum(sum_of_null_values.values.tolist()),'sum of null values')


"""display and drop categories with more than 1000 null values"""
null_values_names=[]
remaining_null_values=[]
for i in range(len(sum_of_null_values.values.tolist())):
    if sum_of_null_values.values.tolist()[i]>1000:
        null_values_names.append(train_data.columns[i])
    else:
        remaining_null_values.append(train_data.columns[i])  # check this line

print(len(null_values_names),'len null values >1000')
train_data.drop(null_values_names, axis=1, inplace=True)
test_data.drop(null_values_names, axis=1, inplace=True)

print('****************************')
print(remaining_null_values)
print(len(remaining_null_values),'remaining null values length')
print('*****************************')

total_null_values=(train_data.isna().sum(axis=0).values.tolist())

print(train_data['LotFrontage'].values.tolist())
print(train_data['LotFrontage'].mode())

numbers=[]
def avg(num):
    for i in num:
        if type(i) !=str:
            numbers.append(i)
    return sum(numbers)/len(numbers)

print('==========================================================================================================')
x=float('nan')
for i in remaining_null_values:
    if type(train_data[i].all()) == int or float:
        #print('string values=',i)
        train_data[i].replace(x,avg(train_data[i].values.tolist())) # replace with mean if entry is numerical
    else:
        train_data[i].fillna(train_data[i].value_counts()[:].index.tolist()[0],inplace=True) #replace with mode if categorical

remaining_null_values.remove('SalePrice')

for i in remaining_null_values:
    if type(test_data[i].all()) == int or float:
        #print('string values=',i)
        test_data[i].replace(x,avg(test_data[i].values.tolist())) # replace with mean
    else:
        test_data[i].fillna(test_data[i].value_counts()[:].index.tolist()[0],inplace=True) #replace with mode

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',1500)

print(len(train_data.isnull().sum().values.tolist()) )
print((train_data.columns.values.tolist()))
remaining_null_values_2=[]

for i in range(len(train_data.isnull().sum().values.tolist())):
    if train_data.isnull().sum().values.tolist()[i]>0:
        remaining_null_values_2.append(train_data.columns.values.tolist()[i])
print(remaining_null_values_2)

for i in remaining_null_values_2:
    train_data[i].fillna(train_data[i].value_counts()[:].index.tolist()[0], inplace=True)

for i in remaining_null_values_2:
    test_data[i].fillna(test_data[i].value_counts()[:].index.tolist()[0], inplace=True)

print(train_data.columns.values.tolist())
categorical_values=[]
for i in train_data.columns.values.tolist():
    if (type(train_data[i].all())) == str:
        categorical_values.append(i)

print(categorical_values)

print(len(categorical_values))
#print (train_data.info())
train_data.drop(['Id'],axis=1,inplace=True)
test_data.drop(['Id'],axis=1,inplace=True)

condition_pivot_list_names=[]
pivot_values_list=[]
for i in categorical_values:
    condition_pivot = train_data.pivot_table(index=i, values='SalePrice', aggfunc=np.mean)
    pivot_names = (condition_pivot.index.values.tolist())
    condition_pivot_list_names.append(pivot_names)
    pivot_values_draft = ((condition_pivot.values.tolist()))
    pivot_values = [i[0] for i in pivot_values_draft]
    pivot_values_list.append(pivot_values)
print(condition_pivot_list_names, 'condition pivot list names')
print(pivot_values_list,'pivot values list')


sublist_names=[(sublists) for sublists in condition_pivot_list_names]
print(sublist_names)


sublist_values=[(sublists1) for sublists1 in pivot_values_list]
print(sublist_values)

sub_names=[]
sub_values=[]

for i in sublist_names:
    sub_names.extend(i)
print(sub_names,'subnames')

for i in sublist_values:
    sub_values.extend(i)

def myfunc(x):
    if x in sub_names:
        index=sub_names.index(x)
        return sub_values[index]
    return x

for i in categorical_values:
    train_data[i] = train_data[i].apply(lambda x: myfunc(x))
    test_data[i] = test_data[i].apply(lambda x: myfunc(x))

"""===========VISUAL REPRESENTATION OF CORRELATION=============="""
sns.set(font_scale=0.5)
heatmap = sns.heatmap(train_data.corr(),xticklabels=True, yticklabels=True, vmin=-1, vmax=1,annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':0.2}, pad=0.2)
plt.show()

correlation_to_sale_price=(train_data.corr()["SalePrice"]) #.sort_values(ascending=False))
print(correlation_to_sale_price)
print(correlation_to_sale_price.values.tolist())
print (len(train_data.columns.values.tolist()))
print(train_data.shape)

weak_correlation_variables=[]
for i in (correlation_to_sale_price.values.tolist()):
    if i <= 0.5 :
        index=correlation_to_sale_price.values.tolist().index(i)
        weak_correlation_variables.append((train_data.columns.values.tolist())[index])
print(weak_correlation_variables, 'weak correlation variables')

train_data.drop(weak_correlation_variables, axis=1, inplace=True)
test_data.drop(weak_correlation_variables , axis=1, inplace=True)

print(train_data.columns, "strong correlation variables")

#-----------check correlation of remaining variables------------------

a=train_data.columns.values.tolist()
a.remove('SalePrice')

for i in a:
    print(pearsonr(train_data[i], train_data['SalePrice']),i)

corr_ExterQual_1= pearsonr(train_data['ExterQual'], train_data['SalePrice'])
print(corr_ExterQual_1, 'corr_ExterQual_1')

"""============IDENTIFY AND REMOVE OUTLIERS========================"""

outliers_ExterQual= train_data['SalePrice'].between(600000, 800000, inclusive=False)  & train_data['ExterQual'].between(200000, 250000, inclusive=False)
outliers_ExterQual_location=train_data[outliers_ExterQual].index.values.tolist()
train_data.drop(outliers_ExterQual_location, inplace=True)
#test_data.drop(outliers_ExterQual_location, inplace=True)

"""x=(train_data['ExterQual'])
y=train_data['SalePrice']
plt.plot(x,y,'o')
m,b=np.polyfit(x,y,1)
plt.plot(x,m*x+b)
plt.show()"""

"""ANALYZE SCATTERPLOTS OF REMAINING VARIABLES"""

corr_ExterQual_1= pearsonr(train_data['ExterQual'], train_data['SalePrice'])
print(corr_ExterQual_1, 'corr_ExterQual_1')

count=1
plt.subplots(figsize=(7, 5))
sns.set(font_scale=0.8)
for i in a[0:4]:
    plt.subplot(2,2,count)#,count)
    sns.scatterplot(x=train_data[i], y=train_data['SalePrice'])
    m, b = np.polyfit(train_data[i], train_data['SalePrice'], 1)
    sns.regplot(x=train_data[i], y=train_data['SalePrice'])
    count+=1

plt.show()

count2=1
for i in a[4:8]:
    plt.subplot(2,2,count2)#,count)
    sns.scatterplot(x=train_data[i], y=train_data['SalePrice'])
    m, b = np.polyfit(train_data[i], train_data['SalePrice'], 1)
    sns.regplot(x=train_data[i], y=train_data['SalePrice'])
    count2+=1
plt.show()

count3=1
for i in a[8:12]:
    plt.subplot(2,2,count3)#,count)
    sns.scatterplot(x=train_data[i], y=train_data['SalePrice'])
    m, b = np.polyfit(train_data[i], train_data['SalePrice'], 1)
    sns.regplot(x=train_data[i], y=train_data['SalePrice'])
    count3+=1
plt.show()


count4=1
for i in a[12:16]:
    plt.subplot(2,2,count4)#,count)
    sns.scatterplot(x=train_data[i], y=train_data['SalePrice'])
    m, b = np.polyfit(train_data[i], train_data['SalePrice'], 1)
    sns.regplot(x=train_data[i], y=train_data['SalePrice'])
    count4+=1
plt.show()

#yearbuilt 200-600,1875-1900 //600-800,1975-2020, yearremodadd 600-800,1990-2000 , totalbsmtsf0-600, 2500-7000,
# 1stfloorsf LIKE BEFORE,grlivarea 100-800,4000-6000,totrmsabvgrd 600-800,9-12

train_data.drop('GarageArea', axis=1, inplace=True)
test_data.drop('GarageArea', axis=1, inplace=True)

#========OUTLIERS=YearBuilt=================================

outliers_YearBuilt_1= train_data['SalePrice'].between(200000, 600000, inclusive=False)  & train_data['YearBuilt'].between(1875, 1900, inclusive=False)
outliers_YearBuilt_2= train_data['SalePrice'].between(500000, 900000, inclusive=False)  & train_data['YearBuilt'].between(1975, 2020, inclusive=False)

outliers_YearBuilt_1_location=train_data[outliers_YearBuilt_1].index.values.tolist()
outliers_YearBuilt_2_location=train_data[outliers_YearBuilt_2].index.values.tolist()
total_outliers_YearBuilt=outliers_YearBuilt_1_location+outliers_YearBuilt_2_location
#print(outliers_YearBuilt_1_location)
#print(outliers_YearBuilt_2_location)

corr_YearBuilt_1= pearsonr(train_data['YearBuilt'], train_data['SalePrice'])
print(corr_YearBuilt_1, 'corr_YearBuilt_1')


"""sns.scatterplot(x=train_data['YearBuilt'], y=train_data['SalePrice'])
m, b = np.polyfit(train_data['YearBuilt'], train_data['SalePrice'], 1)
sns.regplot(x=train_data['YearBuilt'], y=train_data['SalePrice'])
plt.show()
"""
train_data.drop(index=total_outliers_YearBuilt,axis=0, inplace=True)

"""sns.scatterplot(x=train_data['YearBuilt'], y=train_data['SalePrice'])
m, b = np.polyfit(train_data['YearBuilt'], train_data['SalePrice'], 1)
sns.regplot(x=train_data['YearBuilt'], y=train_data['SalePrice'])
plt.show()"""

corr_YearBuilt_1= pearsonr(train_data['YearBuilt'], train_data['SalePrice'])
print(corr_YearBuilt_1, 'corr_YearBuilt_1')

#========REMOVE=YearRemodAdd=================================

train_data.drop('YearRemodAdd', axis=1, inplace=True)
test_data.drop('YearRemodAdd', axis=1, inplace=True)

#========OUTLIERS=TotalBsmtSF=================================

var='TotalBsmtSF'
outliers_TotalBsmtSF_1= train_data['SalePrice'].between(0, 500000, inclusive=False)  & train_data[var].between(3000,7000, inclusive=False)
outliers_TotalBsmtSF_1_location=train_data[outliers_TotalBsmtSF_1].index.values.tolist()

corr_TotalBsmtSF_1= pearsonr(train_data[var], train_data['SalePrice'])
print(corr_TotalBsmtSF_1, 'corr_',var,'_1')

"""sns.scatterplot(x=train_data[var], y=train_data['SalePrice'])
m, b = np.polyfit(train_data[var], train_data['SalePrice'], 1)
sns.regplot(x=train_data[var], y=train_data['SalePrice'])
plt.show()"""

train_data.drop(index=outliers_TotalBsmtSF_1_location,axis=0, inplace=True)

"""sns.scatterplot(x=train_data[var], y=train_data['SalePrice'])
m, b = np.polyfit(train_data[var], train_data['SalePrice'], 1)
sns.regplot(x=train_data[var], y=train_data['SalePrice'])
plt.show()"""

corr_TotalBsmtSF_1= pearsonr(train_data[var], train_data['SalePrice'])
print(corr_TotalBsmtSF_1, 'corr_',var,'_1')

#========UNTOUCHED=1stFlrSF=================================

corr_1stFlrSF_1= pearsonr(train_data['1stFlrSF'], train_data['SalePrice'])
print(corr_1stFlrSF_1, 'corr_1stFlrSF_1')
#========OUTLIERS=GrLivArea=================================

var='GrLivArea'
outliers_GrLivArea_1= train_data['SalePrice'].between(350000, 400000, inclusive=False)  & train_data[var].between(1300,1600, inclusive=False)
outliers_GrLivArea_1_location=train_data[outliers_GrLivArea_1].index.values.tolist()

corr_GrLivArea_1= pearsonr(train_data[var], train_data['SalePrice'])
print(corr_GrLivArea_1, 'corr_',var,'_1')

"""sns.scatterplot(x=train_data[var], y=train_data['SalePrice'])
m, b = np.polyfit(train_data[var], train_data['SalePrice'], 1)
sns.regplot(x=train_data[var], y=train_data['SalePrice'])
plt.show()"""

train_data.drop(index=outliers_GrLivArea_1_location,axis=0, inplace=True)

"""sns.scatterplot(x=train_data[var], y=train_data['SalePrice'])
m, b = np.polyfit(train_data[var], train_data['SalePrice'], 1)
sns.regplot(x=train_data[var], y=train_data['SalePrice'])
plt.show()"""

corr_GrLivArea_1= pearsonr(train_data[var], train_data['SalePrice'])
print(corr_GrLivArea_1, 'corr_',var,'_1')

#========OUTLIERS=TotRmsAbvGrd=================================

var='TotRmsAbvGrd'
outliers_TotRmsAbvGrd_1= train_data['SalePrice'].between(100000, 300000, inclusive=False)  & train_data[var].between(11.5,15, inclusive=False)
outliers_TotRmsAbvGrd_1_location=train_data[outliers_TotRmsAbvGrd_1].index.values.tolist()

corr_TotRmsAbvGrd_1= pearsonr(train_data[var], train_data['SalePrice'])
print(corr_TotRmsAbvGrd_1, 'corr_',var,'_1')

"""sns.scatterplot(x=train_data[var], y=train_data['SalePrice'])
m, b = np.polyfit(train_data[var], train_data['SalePrice'], 1)
sns.regplot(x=train_data[var], y=train_data['SalePrice'])
plt.show()"""

train_data.drop(index=outliers_TotRmsAbvGrd_1_location,axis=0, inplace=True)

"""sns.scatterplot(x=train_data[var], y=train_data['SalePrice'])
m, b = np.polyfit(train_data[var], train_data['SalePrice'], 1)
sns.regplot(x=train_data[var], y=train_data['SalePrice'])
plt.show()"""

corr_TotRmsAbvGrd_1= pearsonr(train_data[var], train_data['SalePrice'])
print(corr_TotRmsAbvGrd_1, 'corr_',var,'_1')

print('===============================REMAINING VARIABLES =================================================')
print(train_data.columns.values.tolist())
correlation_to_sale_price=(train_data.corr()["SalePrice"]) #.sort_values(ascending=False))
print(correlation_to_sale_price)
print(correlation_to_sale_price.values.tolist())
print (len(train_data.columns.values.tolist()))
print(train_data.shape)

weak_correlation_variables=[]
for i in (correlation_to_sale_price.values.tolist()):
    if i < 0.6 :
        index=correlation_to_sale_price.values.tolist().index(i)
        weak_correlation_variables.append((train_data.columns.values.tolist())[index])
print(weak_correlation_variables, 'weak correlation variables')

train_data.drop(weak_correlation_variables, axis=1, inplace=True)
test_data.drop(weak_correlation_variables , axis=1, inplace=True)

print(train_data.columns, "strong correlation variables")

print(train_data.corr()["SalePrice"])

#=========================PREDICTIONS============================================
y = (train_data.SalePrice)
X=train_data.drop(['SalePrice'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
                          X, y, random_state=30, test_size=0.25)

"""PREDICT WITH LINEAR REGRESSION AND XGB"""
x_reg=xgb.XGBClassifier()
lr=linear_model.LinearRegression()

m_1 = lr.fit(X_train, y_train)
m_reg=x_reg.fit(X_train,y_train)

print("R^2 is: \n", m_1.score(X_test, y_test))
print("R^2 xreg is: \n", m_reg.score(X_test, y_test))


prediction_Y_train=m_1.predict(X_train)
prediction_Y_train_xreg=m_reg.predict(X_train)


plt.scatter(prediction_Y_train,y_train)
m, b = np.polyfit(prediction_Y_train,y_train, 1)
plt.plot(prediction_Y_train, m*prediction_Y_train + b)
plt.xlabel('linreg')
plt.show()
#200-400/50-175,350-500/225-350

plt.scatter(prediction_Y_train_xreg,y_train)
m, b = np.polyfit(prediction_Y_train_xreg,y_train, 1)
plt.plot(prediction_Y_train_xreg, m*prediction_Y_train_xreg + b)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()

print(len(   np.array(y_train) ))
print(len(prediction_Y_train_xreg) )
print('=============TYPES==================')

preds=m_1.predict(X_test)

preds_xreg=m_reg.predict(X_test)

"""==============================mean squared error==================="""
#print('mean squared error =', mean_squared_error(preds, y_test))
#print('mean squared error SQRT =', np.sqrt (mean_squared_error(preds, y_test)) )
print  (np.sqrt(mean_squared_log_error(y_test,preds)),'root mean squared log error')
print  (np.sqrt(mean_squared_log_error(y_test,preds_xreg)),'root mean squared log error')

#print  ((mean_squared_log_error(y_test,preds)),'mean squared log error')

test_data['TotalBsmtSF'] = test_data['TotalBsmtSF'].fillna(0)
test_data['KitchenQual'] = test_data['KitchenQual'].fillna(0)
test_data['GarageCars'] = test_data['GarageCars'].fillna(0)

print(test_data.isnull().sum())

print(train_data.shape)
print(test_data.shape)

#plt.scatter(preds,y_test)
#plt.show()

#plt.scatter(preds_xreg,y_test)
#plt.show()

#pred_test=m_1.predict(test_data)
#print(pred_test)
id_list=list(range(1461,2920))

#print(len(pred_test))
#print(len( list(range(1461,2920))))

"""SUBMIT PREDICTIONS TO NEWLY GENERATED CSV FILE"""
submit = pd.DataFrame()
submit['Id'] = id_list
submit['SalePrice']=pred_test
print(submit.head())
submit.to_csv('submit_92.csv', index=False)


