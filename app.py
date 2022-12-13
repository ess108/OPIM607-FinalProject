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
from PIL import Image
import streamlit as st
import plotly.graph_objects as go




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


#Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.
lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

#Format header of page
col1, mid, col2 = st.columns([1,1,20])
with col1:
    st.image("LinkedIn.png", width=60)
with col2:
	st.markdown("# Using Machine Learning to Predict LinkedIn Users") 

st.markdown("### Fill out the following information about yourself, \
	and our algorithm will predict if you are a LinkedIn user")

# Quesiton 1
income = st.selectbox("What Is Your Income Range?", 
              options = ["$<$10k",
                         "$10k-$20k",
                         "$20k-$30k",
                         "$30k-$40k",
                         "$40k-$50k",
                         "$50k-$75k",
                         "$75k-$100k",
                         "$100k-$150k",
                         "$150k+"])
if income == "$<$10k":
    income = 1
elif income == "$10k-$20k":
    income = 2
elif income == "$20k-$30k":
    income = 3
elif income == "$30k-$40k":
    income = 4
elif income == "$40k-$50k":
    income = 5
elif income == "$50k-$75k":
    income = 6
elif income == "$75k-$100k":
    income = 7
elif income == "$100k-$150k":
    income = 8
else:
    income = 9
# Quesiton 2
education= st.selectbox("What Is Your Education Level?", 
              options = ["Less Than Highschool",
                         "High School - No Diploma",
                         "High School Graduate",
                         "Some College",
                         "Two Year Degree",
                         "Four-Year Degree",
                         "Some Postgraduate",
                         "Postgraduate Degree"])
if education == "Less Than Highschool":
    education = 1
elif education == "High School - No Diploma":
    education = 2
elif education == "High School Graduate":
    education = 3
elif education == "Some College":
    education = 4
elif education == "Two Year Degree":
    education = 5
elif education == "Four-Year Degree":
    education = 6
elif education == "Some Postgraduate":
    education = 7
elif education == "Postgraduate Degree":
    education = 8
else:
    education = 9

# Quesiton 3
parent = st.selectbox("Are You a Parent?", 
              options = ["Yes",
                         "No",])
if parent == "Yes":
    parent = 1
else:
    parent = 0

# Quesiton 3
married = st.selectbox("Are You Married", 
              options = ["Married",
                         "Not Married",])
if married == "Married":
    married = 1
else:
    married = 0

 # Quesiton 4
female = st.selectbox("Do You Identify as Male or Female", 
              options = ["Female",
                         "Male",])
if female == "Female":
    female = 1
else:
    female= 0   

 # Quesiton 5

age = st.number_input('Enter your age: ', min_value=1,max_value=97)


#Prediction Button
result = ""
if st.button("Predict If LinkedIn User"):
    person = [income, education, parent, married, female, age]
    result = lr.predict_proba([person])[0][1]
    if result >=.5:
        st.success(f"Probability your are a LinkedIn user: {result}")
    else:
        st.error(f"Probability your are a LinkedIn user:  {result}")


#Gauge visual

    if result > .5:
        label = "LinkedIn User? Yes"
    else: 
        result < .5
        label = "LinkedIn User? No"
    

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = result,
        title = {'text': f"{label}"},
        gauge = {"axis": {"range": [0, 1]},
                "steps": [
                    {"range": [0, .5], "color":"red"},
                    {"range": [.50, 1], "color":"lightgreen"},
                    # {"range": [.15, 1], "color":"lightgreen"}
                ],
                "bar":{"color":"yellow"}}
    ))

    st.plotly_chart(fig)

