import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle
#importing the data
df = pd.read_csv('C:/Users/Ijaz10/CSV/winequality-red.csv')

df['quality'] = df['quality'] > 5
labeler = LabelBinarizer()
df['quality'] = labeler.fit_transform(df['quality'])

X = df.drop(['residual sugar', 'free sulfur dioxide', 'pH', 'quality'], axis = 1)
y = df['quality']

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

tree_clf = DecisionTreeClassifier(max_depth = 6)
tree_clf.fit(train_x, train_y)


joblib.dump(tree_clf, open("wine_quality_predictor.pkl", 'wb'))
model= joblib.load(open("wine_quality_predictor.pkl",'rb'))