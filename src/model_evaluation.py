import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from dvclive import Live
import os
import yaml
import json

# Load test data
test=pd.read_csv('./data/processed/test_processed.csv')

# seprate features and target
X_test=test.drop(columns=['Placed'])
y_test=test['Placed']

# Load the trained model
rf=pickle.load(open('model.pkl','rb'))


# Make predictions
y_pred=rf.predict(X_test)

# calculate
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)

# load parameters for logging
with open('params.yaml','r') as file:
    params=yaml.safe_load(file)

# Log metrics and parameters using dvclive
with Live(save_dvc_exp=True) as live:
    live.log_metric('accuracy',accuracy)
    live.log_metric('precision',precision)
    live.log_metric('recall',recall)
    live.log_metric('f1_score',f1)

    for param,value in params.items():
        for key,val in value.items():
            live.log_param(f'{param}_{key}',val)

# save the mterics to a json file for compatibility with DVC

metrics={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'f1_score':f1
}

with open('metrics.json','w') as f:
    json.dump(metrics,f,indent=4)