import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import altair as alt
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle 
import streamlit as st




s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
   y = np.where((x==1),1,0)
   return y

ss = s

ss = pd.DataFrame({
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan,s["educ2"]),
    "parent":clean_sm(s["par"]),
    "married":clean_sm(s["marital"]),
    "female":np.where(s["gender"]==2,s["gender"],0),
    "age":np.where(s["age"]>98, np.nan, s["age"]),
     "sm_li": clean_sm(s["web1h"])})

ss = ss.dropna()

y = ss["sm_li"]
X= ss[["income","education","parent","married","female","age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) 


lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)


y_pred = lr.predict(X_test)


pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(lr, pickle_out) 
pickle_out.close()




person = [8,7,0,1,1,42]
predicted_class = lr.predict([person])
probs = lr.predict_proba([person])



print(f"Predicted class: {predicted_class[0]}") # 0=not pro-environment, 1=pro-envronment
print(f"Probability that this person is pro-environment: {probs[0][1]}")