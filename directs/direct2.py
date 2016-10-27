import dota2api
api = dota2api.Initialise('***REMOVED***')

from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

dataset = pd.read_csv('MatchOverviewTraining.csv')
testset = pd.read_csv('MatchOverviewTest.csv')
details = pd.read_csv('MatchDetail.csv')

X = dataset.ix[: , :11]
Y = dataset.ix[: , 11:]
X_t = testset.ix[: , :11]
Y_t = testset.ix[: , 11:]
det = np.array(details)


