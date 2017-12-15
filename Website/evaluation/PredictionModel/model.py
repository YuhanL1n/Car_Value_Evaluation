import csv
import collections
import scipy
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

def most_frequent_model(dataFile):
    #print the most frequent make-model pairs in dataset
    models=[]
    columns={}
    f = open(dataFile)
    reader = csv.reader(f)
    row1 = next(reader)

    i=0
    for col in row1:
       columns[col] = i
       i+=1

    counts=0
    for row in reader:
        models.append((row[columns['brand']], row[columns['model']]))
        counts+=1
        if counts%10000==0:
            print counts

    counter=collections.Counter(models)
    print counter.most_common(10)

def vectorize(dataFile):
    dicts=[]

    x=[]
    y=[]

    columns={}
    f = open(dataFile)
    reader = csv.reader(f)
    row1 = next(reader)

    i = 0
    for col in row1:
        columns[col] = i
        i += 1

    for row in reader:
        if float(row[columns['yearOfRegistration']])<2000 or float(row[columns['yearOfRegistration']])>=2017:
            continue

        y.append(float(row[columns['price']]))
        curlist=[]
        #curlist.append(float(row[columns['yearOfRegistration']]))
        #curlist.append(float(row[columns['kilometer']]))
        #curlist.append(float(row[columns['powerPS']]))
        x.append(curlist)
        curdict={}
        curdict[row[columns['kilometer']]] = 1
        curdict[row[columns['yearOfRegistration']]] = 1
        curdict[row[columns['brand']]] = 1
        curdict[row[columns['model']]] = 1
        if row[columns['notRepairedDamage']] is None:
            curdict['nein'] = 1
        else:
            curdict[row[columns['notRepairedDamage']]] = 1

        if row[columns['vehicleType']] is None:
            curdict['kleinwagen'] = 1
        else:
            curdict[row[columns['vehicleType']]] = 1
        dicts.append(curdict)
    v=DictVectorizer(sparse=False)
    X=v.fit_transform(dicts)
    for i in xrange(len(X)):
        for ele in list(X[i]):
            x[i].append(ele)
    print x[0]
    return x,y

def model(x,y):
    lineReg=LinearRegression()
    lineReg.fit(x,y)
    print lineReg.score(x,y)






if __name__ == "__main__":
    x, y = vectorize('data/autos.csv')
    model(x,y)
    #most_frequent_model('data/autos.csv')