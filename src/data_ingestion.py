import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

#URL to the dataset
url='https://raw.githubusercontent.com/campusx-official/toy-datasets/main/student_performance.csv'

# Read the dataset
df=pd.read_csv(url)

# Split the data into training and test sets
train,test=train_test_split(df,test_size=0.2,random_state=42)

# save the splits
train.to_csv('./data/raw/train.csv',index=False)
test.to_csv('./data/raw/test.csv',index=False)

