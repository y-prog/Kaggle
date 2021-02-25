import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import seaborn as sns
import itertools
import xlrd




train_data=pd.read_csv(r'train.csv')

test_data=pd.read_csv(r'test.csv')


print('==================================================================')
print(train_data.shape)
print(test_data.shape)
print('==================================================================')
train_data_list= train_data.values.tolist()

list1=list(train_data.shape)
rows=list1[0]
cols=list1[1]



my_array=np.array(train_data_list)
shape=(rows,cols)

my_array.reshape(shape)


def column(matrix, i):
    return [row[i] for row in matrix]

print (type(column(my_array,2)[1]))
print(type(column(my_array,6)[1]))


"""def file_values(x):
    counterlist = []
    newlist=[]
    k = 0
    my_counter = 0
    for i in range(cols):
        for j in range(rows):
            k +=1
            if column(my_array, i)[j] == x:
                my_counter += 1
                if k % rows == 0:
                    counterlist.append(my_counter)

                    if my_counter > 1000:
                        newlist.append(list(train_data.columns)[i])
                        break
    return(newlist)


null_values='nan'
print((file_values(null_values)))"""




"""if column(my_array, i)[j] == np.nan:
my_counter +=1
print(my_counter)"""

null_values = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']


train_data.drop(null_values, axis=1, inplace=True)
test_data.drop(null_values,axis=1,inplace=True)
# print(train_data.head())

print('after dropping null values')

print(train_data.shape)
print(test_data.shape)

train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())
train_data['BsmtCond'] = train_data['BsmtCond'].fillna('TA')
train_data['BsmtQual'] = train_data['BsmtQual'].fillna('TA')
train_data['BsmtExposure'] = train_data['BsmtExposure'].fillna('No')
train_data['BsmtFinType1'] = train_data['BsmtFinType1'].fillna('Unf')
train_data['BsmtFinType1'] = train_data['BsmtFinType1'].fillna('Unf')
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].fillna('Unf')

train_data['GarageType'] = train_data['GarageType'].fillna('Attchd')
train_data['GarageFinish'] = train_data['GarageFinish'].fillna('Unf')
train_data['GarageQual'] = train_data['GarageQual'].fillna('TA')
train_data['GarageCond'] = train_data['GarageCond'].fillna('TA')




train_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(2005)



test_data['LotFrontage'] = test_data['LotFrontage'].fillna(test_data['LotFrontage'].mean())
test_data['BsmtCond'] = test_data['BsmtCond'].fillna('TA')
test_data['BsmtQual'] = test_data['BsmtQual'].fillna('TA')
test_data['BsmtExposure'] = test_data['BsmtExposure'].fillna('No')
test_data['BsmtFinType1'] = test_data['BsmtFinType1'].fillna('Unf')
test_data['BsmtFinType1'] = test_data['BsmtFinType1'].fillna('Unf')
test_data['BsmtFinType2'] = test_data['BsmtFinType2'].fillna('Unf')

test_data['GarageType'] = test_data['GarageType'].fillna('Attchd')
test_data['GarageFinish'] = test_data['GarageFinish'].fillna('Unf')
test_data['GarageQual'] = test_data['GarageQual'].fillna('TA')
test_data['GarageCond'] = test_data['GarageCond'].fillna('TA')

test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna(2005)

# print(train_data.shape)
# print(train_data.info())
print('====================================================================')

print('the mode is =-==--=-=-=-=-=-=-=-=-=-', train_data['MasVnrType'].mode())
print('the mode is =-==--=-=-=-=-=-=-=-=-=-', train_data['MasVnrArea'].mode())
print('the mode is =-==--=-=-=-=-=-=-=-=-=-', train_data['Electrical'].mode())
print('the mode is =-==--=-=-=-=-=-=-=-=-=-', train_data['GarageCond'].mode())
# print(train_data['BsmtFinType1'].head(100).to_string())


# sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False) #check remaining null values

# train_data['GarageYrBlt'].value_counts().plot(kind='bar');

plt.show()

#null_cols = train_data.columns[train_data.isnull().any()]
# print(train_data[null_cols].isnull().sum())            #print cols with n null values

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',1500)
numeric_features = train_data.select_dtypes(include=[np.number])
# print(numeric_features)

#corr = numeric_features.corr()
#print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
#print(corr['SalePrice'].sort_values(ascending=False)[-5:])

#categoricals = train_data.select_dtypes(exclude=[np.number])  # categorical features
# print(categoricals)

#print(train_data.Street.value_counts())


condition_pivot = train_data.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

#print(pd.pivot_table(train_data, index='SaleCondition', values="SalePrice"))  # check features related to sales price
#categoricals = train_data.select_dtypes(exclude=[np.number])
#print(categoricals.describe())

print('before encoding')
print(train_data.shape)
print(test_data.shape)

def encode(x):
    return 1 if x == 'Partial' else 0  # 'partial' has the highest price
train_data['SaleCondition'] = train_data['SaleCondition'].apply(encode)
test_data['SaleCondition'] = test_data['SaleCondition'].apply(encode)


def encode_1(x):
    return 1 if x == 'FV' or x == 'RL' else 0  # 'FV' has the highest price
train_data['MSZoning'] = train_data['MSZoning'].apply(encode)
test_data['MSZoning'] = test_data['MSZoning'].apply(encode)


def encode(x):
    return 1 if x == 'Pave' else 0  # 'pave' has the highest price
train_data['Street'] = train_data['Street'].apply(encode)
test_data['Street'] = test_data['Street'].apply(encode)


def encode(x):
    return 1 if x == 'IR2' or x == 'IR3' else 0  # feature highest price return 1
train_data['LotShape'] = train_data['LotShape'].apply(encode)
test_data['LotShape'] = test_data['LotShape'].apply(encode)


def encode(x):
    return 1 if x == 'HLS' or x == 'Low' else 0  # feature highest price return 1
train_data['LandContour'] = train_data['LandContour'].apply(encode)
test_data['LandContour'] = test_data['LandContour'].apply(encode)


def encode(x):
    return 1 if x == 'AllPub' else 0  # feature highest price return 1
train_data['Utilities'] = train_data['Utilities'].apply(encode)
test_data['Utilities'] = test_data['Utilities'].apply(encode)


def encode(x):
    return 1 if x == 'CulDSac' or x == 'FR3' else 0  # feat highest price return 1
train_data['LotConfig'] = train_data['LotConfig'].apply(encode)
test_data['LotConfig'] = test_data['LotConfig'].apply(encode)

train_data.drop(["LandSlope"], axis=1, inplace=True)
test_data.drop(["LandSlope"], axis=1, inplace=True)

print('while encoding')

print(train_data.shape)
print(test_data.shape)

def encode(x):
    return 1 if x == 'NoRidge' or x == 'NridgHt' or x == 'StoneBr' else 0  # feat highest price return 1
train_data['Neighborhood'] = train_data['Neighborhood'].apply(encode)
test_data['Neighborhood'] = test_data['Neighborhood'].apply(encode)


def encode(x):
    return 1 if x == 'PosA' or x == 'PosN' or x == 'RRNn' else 0  # feat highest price return 1
train_data['Condition1'] = train_data['Condition1'].apply(encode)
test_data['Condition1'] = test_data['Condition1'].apply(encode)

def encode(x):
    return 1 if x == 'PosA' or x == 'PosN' else 0  # feat highest price return 1
train_data['Condition2'] = train_data['Condition2'].apply(encode)
test_data['Condition2'] = test_data['Condition2'].apply(encode)


def encode(x):
    return 1 if x == '1Fam' or x == 'TwnhsE' else 0  # feat highest price return 1
train_data['BldgType'] = train_data['BldgType'].apply(encode)
test_data['BldgType'] = test_data['BldgType'].apply(encode)

def encode(x):
    return 1 if x == '2.5Fin' or x == '2Story' else 0  # feat highest price return 1
train_data['HouseStyle'] = train_data['HouseStyle'].apply(encode)
test_data['HouseStyle'] = test_data['HouseStyle'].apply(encode)


train_data.drop(["RoofStyle"], axis=1, inplace=True)
test_data.drop(["RoofStyle"], axis=1, inplace=True)


def encode(x):
    return 1 if x == 'WdShngl' else 0  # feat highest price return 1
train_data['RoofMatl'] = train_data['RoofMatl'].apply(encode)
test_data['RoofMatl'] = test_data['RoofMatl'].apply(encode)

def encode(x):
    return 1 if x == 'CemntBd' or x == 'ImStucc' or x == 'Stone' else 0  # feat highest price return 1
train_data['Exterior1st'] = train_data['Exterior1st'].apply(encode)
test_data['Exterior1st'] = test_data['Exterior1st'].apply(encode)

def encode(x):
    return 1 if x == 'Other' else 0  # feat highest price return 1
train_data['Exterior2nd'] = train_data['Exterior2nd'].apply(encode)
test_data['Exterior2nd'] = test_data['Exterior2nd'].apply(encode)

def encode(x):
    return 1 if x == 'WD' else 0  # feat highest price return 1
train_data['SaleType'] = train_data['SaleType'].apply(encode)
test_data['SaleType'] = test_data['SaleType'].apply(encode)

def encode(x):
    return 1 if x == 'TA' or x == 'Gd' else 0  # feat highest price return 1
train_data['ExterQual'] = train_data['ExterQual'].apply(encode)
test_data['ExterQual'] = test_data['ExterQual'].apply(encode)

def encode(x):
    return 1 if x == 'TA' or x == 'Gd' else 0  # feat highest price return 1
train_data['ExterCond'] = train_data['ExterCond'].apply(encode)
test_data['ExterCond'] = test_data['ExterCond'].apply(encode)

def encode(x):
    return 1 if x == 'PConc' or x == 'CBlock' else 0  # feat highest price return 1
train_data['Foundation'] = train_data['Foundation'].apply(encode)
test_data['Foundation'] = test_data['Foundation'].apply(encode)

def encode(x):
    return 1 if x == 'TA' or x == 'Gd' else 0  # feat highest price return 1
train_data['BsmtQual'] = train_data['BsmtQual'].apply(encode)
test_data['BsmtQual'] = test_data['BsmtQual'].apply(encode)

def encode(x):
    return 1 if x == 'TA' else 0  # feat highest price return 1
train_data['BsmtCond'] = train_data['BsmtCond'].apply(encode)
test_data['BsmtCond'] = test_data['BsmtCond'].apply(encode)

def encode(x):
    return 1 if x == 'No' else 0  # feat highest price return 1
train_data['BsmtExposure'] = train_data['BsmtExposure'].apply(encode)
test_data['BsmtExposure'] = test_data['BsmtExposure'].apply(encode)

def encode(x):
    return 1 if x == 'Unf' or x=='GLQ' else 0  # feat highest price return 1
train_data['BsmtFinType1'] = train_data['BsmtFinType1'].apply(encode)
test_data['BsmtFinType1'] = test_data['BsmtFinType1'].apply(encode)

def encode(x):
    return 1 if x == 'Unf' else 0  # feat highest price return 1
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].apply(encode)
test_data['BsmtFinType2'] = test_data['BsmtFinType2'].apply(encode)

def encode(x):
    return 1 if x == 'GasA' else 0  # feat highest price return 1
train_data['Heating'] = train_data['Heating'].apply(encode)
test_data['Heating'] = test_data['Heating'].apply(encode)

def encode(x):
    return 1 if x == 'Ex' or x=='TA' or x=='Gd' else 0  # feat highest price return 1
train_data['HeatingQC'] = train_data['HeatingQC'].apply(encode)
test_data['HeatingQC'] = test_data['HeatingQC'].apply(encode)

def encode(x):
    return 1 if x == 'Y' else 0  # feat highest price return 1
train_data['CentralAir'] = train_data['CentralAir'].apply(encode)
test_data['CentralAir'] = test_data['CentralAir'].apply(encode)

def encode(x):
    return 1 if x == 'TA' or x=='Gd' else 0  # feat highest price return 1
train_data['KitchenQual'] = train_data['KitchenQual'].apply(encode)
test_data['KitchenQual'] = test_data['KitchenQual'].apply(encode)


def encode(x):
    return 1 if x == 'Typ'  else 0  # feat highest price return 1
train_data['Functional'] = train_data['Functional'].apply(encode)
test_data['Functional'] = test_data['Functional'].apply(encode)

def encode(x):
    return 1 if x == 'Attchd' or x=='Detchd'  else 0  # feat highest price return 1
train_data['GarageType'] = train_data['GarageType'].apply(encode)
test_data['GarageType'] = test_data['GarageType'].apply(encode)

def encode(x):
    return 1 if x == 'Unf' or x=='RFn'  else 0  # feat highest price return 1
train_data['GarageFinish'] = train_data['GarageFinish'].apply(encode)
test_data['GarageFinish'] = test_data['GarageFinish'].apply(encode)


def encode(x):
    return 1 if x == 'TA'  else 0  # feat highest price return 1
train_data['GarageQual'] = train_data['GarageQual'].apply(encode)
test_data['GarageQual'] = test_data['GarageQual'].apply(encode)

def encode(x):
    return 1 if x == 'TA'  else 0  # feat highest price return 1
train_data['GarageCond'] = train_data['GarageCond'].apply(encode)
test_data['GarageCond'] = test_data['GarageCond'].apply(encode)


def encode(x):
    return 1 if x == 'Y'  else 0  # feat highest price return 1
train_data['PavedDrive'] = train_data['PavedDrive'].apply(encode)
test_data['PavedDrive'] = test_data['PavedDrive'].apply(encode)

def encode(x):
    return 1 if x == 'SBrkr'  else 0  # feat highest price return 1
train_data['Electrical'] = train_data['Electrical'].apply(encode)
test_data['Electrical'] = test_data['Electrical'].apply(encode)

def encode(x):
    return 1 if x == 'BrkFace' or x=='Stone' else 0  # feat highest price return 1
train_data['MasVnrType'] = train_data['MasVnrType'].apply(encode)
test_data['MasVnrType'] = test_data['MasVnrType'].apply(encode)

print('after encoding')

print(train_data.shape)
print(test_data.shape)
"""def missingValuesInfo(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100, 2)
    temp = pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
    return temp.loc[(temp['Total'] > 0)]

print(missingValuesInfo(train_data))"""


train_data['MasVnrType'] = train_data['MasVnrType'].fillna(0)
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(0)
train_data['Electrical'] = train_data['Electrical'].fillna(0)


test_data['MasVnrType'] = test_data['MasVnrType'].fillna(0)
test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(0)
test_data['Electrical'] = test_data['Electrical'].fillna(0)

test_data['BsmtFinSF1'] = test_data['BsmtFinSF1'].fillna(0)
test_data['BsmtFinSF2'] = test_data['BsmtFinSF2'].fillna(0)
test_data['BsmtUnfSF'] = test_data['BsmtUnfSF'].fillna(0)
test_data['TotalBsmtSF'] = test_data['TotalBsmtSF'].fillna(0)

test_data['BsmtFullBath'] = test_data['BsmtFullBath'].fillna(0)
test_data['BsmtHalfBath'] = test_data['BsmtHalfBath'].fillna(0)
test_data['GarageCars'] = test_data['GarageCars'].fillna(0)
test_data['GarageArea'] = test_data['GarageArea'].fillna(0)



"""plt.scatter(train_data['GarageArea'], train_data['SalePrice'])
plt.xlabel('living area square feet')
plt.ylabel('sale price')
plt.show()"""
train_data.drop(['Id'], axis=1, inplace=True)
test_data.drop(['Id'], axis=1, inplace=True)

print('after dropping id')
print(train_data.shape)
print(test_data.shape)



y = (train_data.SalePrice)
X=train_data.drop(['SalePrice'], axis=1)
#X = train_data




X_train, X_test, y_train, y_test = train_test_split(
                          X, y, random_state=30, test_size=0.3)

lr=linear_model.LinearRegression()

#print(train_data.head(), '===================HEAD================')

print('=============================================================================')


m_1 = lr.fit(X_train, y_train)
print("R^2 is: \n", m_1.score(X_test, y_test))

preds=m_1.predict(X_test)
#print(y_test)
#print(preds)
preds_list=preds.tolist()
y_test_list=y_test.values.tolist()
print((preds_list))
print((y_test_list))
width1 = 6                #.5
X = np.arange(0,len(preds_list)*width1*3,width1*3)
plt.bar(X, preds_list, color = 'b', width = width1)
plt.bar(X+width1, y_test_list , color = 'r', width = width1) #.3
plt.show()
plt.scatter(preds_list,y_test_list)
plt.show()

#m_2=lr.fit(X,y)

#print(test_data.info())

#print(len(X),len(y))

print(train_data.shape)
print(test_data.shape)
pred_test=m_1.predict(test_data)
print(pred_test)

id_list=list(range(1,1460))
submit = pd.DataFrame()
submit['Id'] = id_list
submit['SalePrice']=pred_test
print(submit.head())
submit.to_csv('submit_1.csv', index=False)



