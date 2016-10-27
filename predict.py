from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

dataset = pd.read_csv('data/MatchOverviewTraining.csv')
testset = pd.read_csv('data/MatchOverviewTest.csv')
details = pd.read_csv('data/MatchDetail.csv')

X = dataset.ix[: , :11]
Y = dataset.ix[: , 11:]
X_t = testset.ix[: , :11]

det = np.array(details)
print("Done loading data.")
print(len(dataset), "rows in dataset.")
print(len(det), "rows in details.")

games = {}
for row in det:
  key = int(row[0]); # match_id
  if key in games:
    # if match already in games
    games[key].update({row[1]:row[2:]}) 
  else:
    # add match to games
    games.update({key:{row[1]:row[2:]}})

fullset = []
for row in np.array(X):
    insert = []
    key = int(row[0]) # match_id
    if key not in games:
        player = {}
    else:
        player = games[key]
    for i in range(1, 11): # hero_1 - 10
        replace = [0] * 21; # default null values
        if row[i] in player: replace = list(player[row[i]])
        # add null values
        if len(replace) < 21: replace.extend([0] * (21 - len(replace)))
        
        replace.insert(row[i], 0)
        insert.extend(replace)
    fullset.append(insert)

X = np.array(fullset) 

fullset = []
for row in np.array(X_t):
    insert = []
    key = int(row[0]) # match_id
    if key not in games:
        player = {}
    else:
        player = games[key]
    for i in range(1, 11): # hero_1 - 10
        replace = [0] * 21; # default null values
        if row[i] in player: replace = list(player[row[i]])
        # add null values
        if len(replace) < 21: replace.extend([0] * (21 - len(replace)))
        
        replace.insert(row[i], 0)
        insert.extend(replace)
    fullset.append(insert)

X_t = np.array(fullset)
Y = (np.array(Y)).flatten()

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

model.fit(X, Y)

y_pred = model.predict(X_t)
predictions = [round(value) for value in y_pred]

labels = np.array(testset.ix[:,:1])
for k, prediction in enumerate(predictions):
    print(str(labels[k][0]) + "," + ("TRUE" if prediction else "FALSE"))
