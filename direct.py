import dota2api
import numpy as np
import pandas as pd

api = dota2api.Initialise() # set steam key in ENV
testset = pd.read_csv('MatchOverviewTest.csv')

labels = np.array(testset.ix[:,:1])
for k, label in enumerate(labels):
    match = api.get_match_details(match_id=int(labels[k][0]))
    print(str(labels[k][0]) + "," + ("TRUE" if match['radiant_win'] else "FALSE"), flush=True)
