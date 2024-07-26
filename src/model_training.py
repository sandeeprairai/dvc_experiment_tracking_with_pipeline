import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml
import os


# Load paramters
with open('params.yaml','r')as file:
    params=yaml.safe_load(file)

n_estimators=params['model_training']['n_estimators']
max_depth=params['model_training']['max_depth']
boostrap=params['model_training']['bootstrap']
criterion=params['model_training']['creterion']

# Load processed data
train=pd.read_csv('./data/processed/train_processed.csv')

# seprate features and target
X_train=train.drop(columns=['Placed'])
y_train=train['Placed']

# Train the RandomForest model
rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=boostrap,criterion=criterion)
rf.fit(X_train,y_train)

# save the model
pickle.dump(rf,open('model.pkl','wb'))


