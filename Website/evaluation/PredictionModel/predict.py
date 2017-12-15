import pandas as pd
import numpy as np
import argparse
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
import os.path

def evaluate(make, model, year, odometer, title, condition):
    labels = ['make', 'model', 'VIN', 'condition', 'cylinders', 'drive', 'fuel', 'color', 'size', 'title',
              'transmission', 'type']
    inputs={}
    inputs['make'] = make
    inputs['model'] = model
    inputs['odometer'] = odometer
    inputs['year'] = year
    inputs['title'] = title
    inputs['condition'] = condition
    inputs['cylinders'] = 'None'
    inputs['drive'] = 'None'
    inputs['fuel'] = 'gas'
    inputs['color'] = 'None'
    inputs['size'] = 'None'
    inputs['VIN'] = 'None'
    inputs['transmission'] = 'automatic'
    inputs['type'] = 'None'

    X = np.array([])
    BASE = os.path.dirname(os.path.abspath(__file__))
    for l in labels:
        with open(os.path.join(BASE, l+'_encoder')) as handle:
            encoder=pickle.load(handle)
            X = np.append(X, encoder.transform([inputs[l]])[0])
    X = np.append(X, inputs['year'])
    X = np.append(X, inputs['odometer'])

    model = joblib.load(os.path.join(BASE, 'model'))
    return np.exp(model.predict([X])[0])-1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input car info then get predicted value')
    parser.add_argument('-make', type=str,
                        help="Make of the car",
                        required=True)
    parser.add_argument("-model", type=str,
                        help="Model of the car",
                        required=True)
    parser.add_argument("-year", type=int,
                        help="Year of the car",
                        required=True)
    parser.add_argument("-odometer", type=int,
                        help="Odometer of the car",
                        required=True)
    parser.add_argument("-title", type=str,
                        help="Title status of the car",
                        choices=['salvage', 'rebuilt', 'clean', 'parts only', 'lien', 'missing'],
                        default='clean',
                        required=False)
    parser.add_argument("-condition", type=str,
                        help="Condition of the car",
                        choices=['fair', 'good', 'excellent', 'like new', 'new'],
                        default='None',
                        required=False)

    args = parser.parse_args()
    value = evaluate(args.make, args.model, args.year, args.odometer, args.title, args.condition)
    print value

