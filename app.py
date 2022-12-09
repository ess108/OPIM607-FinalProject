
import pandas as pd
import numpy as np
import pickle as pickle
from PIL import Image
import streamlit as st
import plotly.graph_objects as go


# loading in the model to predict on the data
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)
  
#st.title("Using Machine Learning to Predict LinkedIn Users")
# Title
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

age = st.number_input('Enter your age: ', min_value=18,max_value=98)





def prediction(income, education, parent, married,female):  
   
    prediction = classifier.predict_proba(
        [[income, education, parent, married,female, age]])[0][1]
    return prediction

# if prediction == 1:
#  x = classifier.predict(
#         [[income, education, parent, married, female, age]])[:,1])
# else:
#  x = classifier.predict(
#         [income, education, parent, married, female, age]])[:,0])
result = ""
if st.button("Predict If LinkedIn User"):
    result = prediction(income, education, parent, married,female) 
    if result >=.5:
        st.success(f"Probability the person IS a LinkedIn {result}")
        score = result
    else:
        st.error(f"Probability the person is NOT a LinkedIn User {result}")
        score = result

# result


# if result < .5:
#     score = result
# else: float(result)
# #score = float(result)

#### Create label (called sent) from TextBlob polarity score to use in summary below
    if score > .5:
        label = "LinkedIn User? Yes"
    else: 
        score < .5
        label = "LinkedIn User? No"
    

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
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


