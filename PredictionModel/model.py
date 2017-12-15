import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets, linear_model, preprocessing, svm
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
import matplotlib
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib


def preprocess(dataFile):
    df = pd.read_csv(dataFile, sep=',', header=0, encoding='cp1252')

    print len(df)
    df = df[df.odometer != 'None']
    df['odometer'] = df['odometer'].apply(pd.to_numeric)

    print("Too new: %d" % df.loc[df.year > 2017].count()['price'])
    print("Too old: %d" % df.loc[df.year < 1990].count()['price'])
    print("Too cheap: %d" % df.loc[df.price < 100].count()['price'])

    print("Too expensive: %d" % df.loc[df.price > 150000].count()['price'])
    print("Too few km: %d" % df.loc[df.odometer < 1000].count()['price'])
    print("Too many km: %d" % df.loc[df.odometer > 300000].count()['price'])

    df = df[
        (df.year <= 2017)
        & (df.year >= 1990)
        & (df.price > 100)
        & (df.price < 150000)
        & (df.odometer > 1000)
        & (df.odometer < 300000)
        ]

    print df.describe()
    df['VIN'].fillna(value='None', inplace=True)
    df['VIN'] = df['VIN'].replace(to_replace='^((?!None).)*$', value='Yes', regex=True)
    print df['VIN'].unique()
    df['make and model'] = df['make and model'].str.lower()
    df['make'], df['model'] = df['make and model'].str.split(pat=None, n=1).str
    df['model'] = df['model'].str.replace('-', '')

    df['make'].fillna(value='None', inplace=True)
    df['model'].fillna(value='None', inplace=True)
    df = df[df['make'].isin(df['make'].value_counts().index.tolist()[:50]) &
            df['model'].isin(df['model'].value_counts().index.tolist()[:100])]
    # replace values
    df['make'].replace('vw', 'volkswagen', inplace=True)
    df['make'].replace('chevy', 'chevrolet', inplace=True)
    df['make'].replace('cheverolet', 'chevrolet', inplace=True)
    df['model'].replace('camry le', 'camry', inplace=True)

    print df['make'].value_counts()
    print df['model'].value_counts()
    print df.isnull().sum()
    labels = ['make', 'model', 'VIN', 'condition', 'cylinders', 'drive', 'fuel', 'color', 'size', 'title',
              'transmission', 'type']
    les = {}

    ''' l in labels:
        les[l] = preprocessing.LabelBinarizer()
        les[l].fit(df[l])
        tr = les[l].transform(df[l])
        df.loc[:, l + '_feat'] = pd.Series(tr, index=df.index)'''

    labeled = df[['price'
                     , 'odometer'
                     , 'year'
                  ]
                 + [x  for x in labels]]
    print len(labeled)
    return labeled


def stat():
    print '-'


def model(dataset):
    Y = dataset['price'].as_matrix()
    #X = dataset['year'].as_matrix()
    #X = np.append(X, dataset['odometer'].as_matrix())
    labels = ['make', 'model', 'VIN', 'condition', 'cylinders', 'drive', 'fuel', 'color', 'size', 'title',
              'transmission', 'type', 'year']
    les = {}
    vecs = None
    for l in labels:
        les[l] = preprocessing.LabelBinarizer()
        les[l].fit(dataset[l])
        with open(l+'_encoder', 'wb') as handle:
            pickle.dump(les[l], handle, protocol=pickle.HIGHEST_PROTOCOL)
        if vecs is None:
            vecs = les[l].transform(dataset[l])
        else:
            vecs = np.hstack((vecs,les[l].transform(dataset[l])))
    #vecs= np.hstack((vecs, dataset['year'].values.reshape(-1,1)))
    X= np.hstack((vecs, dataset['odometer'].values.reshape(-1,1)))

    '''matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    #  plt.figure()
    prices = pd.DataFrame({"1. Original": Y, "2.Log": np.log1p(Y)})
    prices.hist()
    plt.show()'''

    Y = np.log1p(Y)
    # Percent of the X array to use as training set. This implies that the rest will be test set
    test_size = .25

    # Split into train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=3)
    print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

    lr = LinearRegression()
    scores = cross_val_score(lr, X, Y ,cv=5)
    print scores
    lr.fit(X_train, Y_train)
    #joblib.dump(lr, 'model')
    print ('-----Linear Regression-----')
    print 'Training Data R2:',
    print lr.score(X_train, Y_train)
    print 'Test Data R2:',
    print lr.score(X_val, Y_val)

    lr = LinearRegression()
    lr.fit(X, Y)
    joblib.dump(lr, 'model')

    param_grid = {"alpha": [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 50]}
    trg = GridSearchCV(estimator=Ridge(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    trg.fit(X_train,Y_train)
    bp= trg.best_params_
    rg = Ridge(alpha=bp['alpha'])
    rg.fit(X_train,Y_train)
    print ('-----Ridge Regression-----')
    print 'Training Data R2:',
    print rg.score(X_train,Y_train)
    print 'Test Data R2:',
    print rg.score(X_val,Y_val)

    '''param_grid = {"alpha": [1e-6, 1e-5,1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 50]}
    tlo = GridSearchCV(estimator=Lasso(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    tlo.fit(X_train,Y_train)
    bp= tlo.best_params_
    lo = Lasso(alpha=bp['alpha'])
    lo.fit(X_train,Y_train)
    print ('-----Lasso-----')
    print 'Training Data R2:',
    print lo.score(X_train,Y_train)
    print 'Test Data R2:',
    print lo.score(X_val,Y_val)'''


    '''param_grid = {"C": [1e0,1e1,1e2,1e3]
        , "gamma": np.logspace(-2,2,5)}

    tsvr = GridSearchCV(estimator=SVR(kernel='rbf'), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    tsvr.fit(X_train,Y_train)
    bp= tsvr.best_params_
    svr= SVR(kernel='rbf', C=bp['C'], gamma=bp['gamma'])
    print ('-----Support Vector-----')
    print 'Training Data R2:',
    print svr.score(X_train,Y_train)
    print 'Test Data R2:',
    print svr.score(X_val,Y_val)'''

    '''rf = RandomForestRegressor()
    param_grid = {"min_samples_leaf": xrange(3, 4)
        , "min_samples_split": xrange(3, 4)
        , "max_depth": xrange(12, 13)
        , "n_estimators": [500]}

    gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1)
    gs = gs.fit(X_train, Y_train)
    bp = gs.best_params_'''
    forest = RandomForestRegressor(criterion='mse',
                                   min_samples_leaf=6,
                                   min_samples_split=6,
                                   max_depth=12,
                                   n_estimators=500)
    forest.fit(X_train, Y_train)
    print ('-----Random Forest -----')
    print 'Training Data R2:',
    print forest.score(X_train, Y_train)
    print 'Test Data R2:',
    print forest.score(X_val, Y_val)


if __name__ == '__main__':
    dataset = preprocess('data/all.csv')
    model(dataset)