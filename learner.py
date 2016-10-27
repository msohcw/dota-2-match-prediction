from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

dataset = pd.read_csv('MatchOverviewTraining.csv')
details = pd.read_csv('MatchDetail.csv')
X = dataset.ix[: , :11]
Y = dataset.ix[: , 11:]
det = np.array(details)
print("Done loading.", len(det))
players = {}
for row in det:
  key = int(row[0]);
  if key in players:
    players[key].update({row[1]:row[2:]})
  else:
    players.update({key:{row[1]:row[2:]}})

clean = []
index = 0
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
# split data into train and test sets
seed = 7
test_size = 0.33
X = np.array(clean)
 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier(
 learning_rate =0.2,
 n_estimators=750,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=64,
 scale_pos_weight=1,
 seed=27,
 silent=1
)
"""
y_train = (np.array(y_train)).flatten()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


model = XGBClassifier()
"""
Y = np.array(Y).flatten()
#learning_rate = [0.05, 0.15, 0.3]
#n_estimators = [100, 300, 700, 1000]
max_depth = [3, 5, 7, 9]
min_child_weight = [1, 3, 5]
param_grid = dict(max_depth=max_depth, min_child_weight = min_child_weight)
kfold = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=7)
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
