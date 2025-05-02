import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("coord.csv")

#Features
x = df.drop("class", axis=1)
#Target value
y = df["class"]

x_train , x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state=1234)

#Pipelines for models we tested
pipelines = {
        #"lr" : make_pipeline(StandardScaler(), LogisticRegression()),
        #"rc": make_pipeline(StandardScaler(), RidgeClassifier()),
        #"gb": make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        "rf" : make_pipeline(StandardScaler(), RandomForestClassifier()),}

fit_models = {}
for algo, pipeline in pipelines.items():
        model = pipeline.fit(x_train, y_train)
        fit_models[algo] = model

for algo, model in fit_models.items():
    yhat = model.predict(x_test)
    print(algo, accuracy_score(y_test, yhat))

with open('body_language.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)
