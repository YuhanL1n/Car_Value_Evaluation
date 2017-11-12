import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNet, Lasso, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets, linear_model, preprocessing, svm
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def preprocess(dataFile):
    df = pd.read_csv(dataFile, sep=',', header=0, encoding='cp1252')
    #print df.describe()
    df.drop(['seller', 'offerType', 'abtest', 'dateCrawled', 'nrOfPictures', 'lastSeen', 'postalCode', 'dateCreated', 'name'],
            axis='columns', inplace=True)
    dedups = df.drop_duplicates(['price', 'vehicleType', 'yearOfRegistration'
                                    , 'gearbox', 'powerPS', 'model', 'kilometer', 'monthOfRegistration', 'fuelType'
                                    , 'notRepairedDamage'])

    #### Removing the outliers
    dedups = dedups[
        (dedups.yearOfRegistration <= 2017)
        & (dedups.yearOfRegistration >= 1950)
        & (dedups.price >= 100)
        & (dedups.price <= 100000)
        & (dedups.powerPS >= 10)
        & (dedups.powerPS <= 500)
        & (pd.notnull(dedups.model))]
    #print("-----------------\nData kept for analisys: %d percent of the entire set\n-----------------" % (
    #100 * dedups['name'].count() / df['name'].count()))
    dedups['notRepairedDamage'].fillna(value=' nein', inplace=True)
    dedups['fuelType'].fillna(value='benzin', inplace=True)
    dedups['gearbox'].fillna(value='manuell', inplace=True)
    dedups['vehicleType'].fillna(value='not-declared', inplace=True)
    print dedups.isnull().sum()
    labels = ['gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']
    les = {}

    for l in labels:
        les[l] = preprocessing.LabelEncoder()
        les[l].fit(dedups[l])
        tr = les[l].transform(dedups[l])
        dedups.loc[:, l + '_feat'] = pd.Series(tr, index=dedups.index)

    labeled = dedups[['price'
                         , 'yearOfRegistration'
                         , 'powerPS'
                         , 'kilometer'
                         , 'monthOfRegistration']
                     + [x + "_feat" for x in labels]]
    return labeled

def stat():
    print '-'

def model(dataset):
    Y = dataset['price']
    X = dataset.drop(['price'], axis='columns', inplace=False)
    Y = np.log1p(Y)
    # Percent of the X array to use as training set. This implies that the rest will be test set
    test_size = .25

    # Split into train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=3)
    print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)
    lr = LinearRegression()
    lr.fit(X_train,Y_train)
    print ('-----Linear Regression-----')
    print 'Training Data R2:',
    print lr.score(X_train,Y_train)
    print 'Test Data R2:',
    print lr.score(X_val,Y_val)
    rg = Ridge()
    rg.fit(X_train,Y_train)
    print ('-----Ridge Regression-----')
    print 'Training Data R2:',
    print rg.score(X_train,Y_train)
    print 'Test Data R2:',
    print rg.score(X_val,Y_val)
    lo = Lasso()
    lo.fit(X_train,Y_train)
    print ('-----Lasso-----')
    print 'Training Data R2:',
    print lo.score(X_train,Y_train)
    print 'Test Data R2:',
    print lo.score(X_val,Y_val)
    rf = RandomForestRegressor()

    param_grid = {"criterion": ["mse"]
        , "min_samples_leaf": [3]
        , "min_samples_split": [3]
        , "max_depth": [10]
        , "n_estimators": [500]}

    gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1)
    gs = gs.fit(X_train, Y_train)
    bp = gs.best_params_
    forest = RandomForestRegressor(criterion=bp['criterion'],
                                   min_samples_leaf=bp['min_samples_leaf'],
                                   min_samples_split=bp['min_samples_split'],
                                   max_depth=bp['max_depth'],
                                   n_estimators=bp['n_estimators'])
    forest.fit(X_train, Y_train)
    print ('-----Random Forest-----')
    print 'Training Data R2:',
    print forest.score(X_train,Y_train)
    print 'Test Data R2:',
    print forest.score(X_val,Y_val)



if __name__=='__main__':
    dataset = preprocess('data/autos.csv')
    model(dataset)