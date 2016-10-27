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



"""
print("Done loading.", len(det))

players = {}
for row in det:
  key = int(row[0]);
  if key in players:
    players[key].update({row[1]:row[2:]})
  else:
    players.update({key:{row[1]:row[2:]}})

clean = []
for row in np.array(X):
    insert = []
    key = int(row[0])
    if key not in players:
        player = {}
    else:
        player = players[key]
    # print(match)
    for i in range(1, 11):
        replace = [0] * 21;
        if row[i] in player: replace = list(player[row[i]])
        if len(replace) < 21: 
            replace.extend([0] * (21 - len(replace)))
        replace.insert(row[i], 0)
        insert.extend(replace)
    if len(insert) != 220: print(len(insert))
    clean.append(insert)
X = np.array(clean)

print("Done cleaning training")

clean = []
for row in np.array(X_t):
    insert = []
    key = int(row[0])
    if key not in players:
        player = {}
    else:
        player = players[key]
    # print(match)
    for i in range(1, 11):
        replace = [0] * 21;
        if row[i] in player: replace = list(player[row[i]])
        if len(replace) < 21: 
            replace.extend([0] * (21 - len(replace)))
        replace.insert(row[i], 0)
        insert.extend(replace)
    if len(insert) != 220: print(len(insert))
    clean.append(insert)

X_t = np.array(clean)

print("Done cleaning test")

# split data into train and test sets
seed = 7
test_size = 0.33
 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier(
 learning_rate =0.05,
 n_estimators=1000,
 max_depth=9,
 min_child_weight=5,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=16,
 scale_pos_weight=1,
 seed=27,
 silent=1
)

y_train = (np.array(y_train)).flatten()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

y_t_pred = model.predict(X_t)
predictions = [round(value) for value in y_t_pred]
# evaluate predictions

#import sys
#sys.stdout = open('pred.out', 'w')
"""
labels = np.array(testset.ix[:,:1])
for k, label in enumerate(labels):
#    if k < 3000: continue
    match = api.get_match_details(match_id=int(labels[k][0]))
    print(str(labels[k][0]) + "," + ("TRUE" if match['radiant_win'] else "FALSE"), flush=True)

"""


Y = np.array(Y).flatten()
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, min_child_weight = min_child_weight)
kfold = StratifiedKFold(Y, n_folds=5, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=10)

result = grid_search.fit(X, Y)

# summarize results
print("Best: %f using %s" % (result.best_score_, result.best_params_))
print("Best: {} using {}".format(result.best_score_, result.best_params_))
means, stdevs = [], []
for params, mean_score, scores in result.grid_scores_:
    stdev = scores.std()
    means.append(mean_score)
    stdevs.append(stdev)
    print("%f (%f) with: %r" % (mean_score, stdev, params))


"""
