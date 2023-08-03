import streamlit as st
import numpy as np

#load ML package
import joblib
import os

dep = {'Sales & Marketing':1, 'Operations':2, 'Technology':3, 'Analytics':4,
       'R&D':5, 'Procurement':6, 'Finance':7, 'HR':8, 'Legal':9}
edu = {'Below Secondary':1, "Bachelor's":2, "Master's & above":3}
rec = {'referred':1, 'sourcing':2, 'other':3}
gen = {'m':1, 'f':2}
reg = {'region_1':1,'region_2':2,'region_3':3,'region_4':4,'region_5':5,
       'region_6':6,'region_7':7,'region_8':8,'region_9':9,'region_10':10,
       'region_11':11,'region_12':12,'region_13':13,'region_14':14,'region_15':15,
       'region_16':16,'region_17':17,'region_18':18,'region_19':19,'region_20':20,
       'region_21':21,'region_22':22,'region_23':23,'region_24':24,'region_25':25,
       'region_26':26,'region_27':27,'region_28':28,'region_29':29,'region_30':30,
       'region_31':31,'region_32':32,'region_33':33,'region_34':34}

attribute_info = """
                 - Gender: Male and Female
                 - Age: 32-70
                 - Education: postgraduate, primaryschool, uneducated, graduate
                 - Current Smoker : 1 Smoke, 0 Not Smoke
                 - CigsPerDay : Berapa puntung rokok dalam sehari (1-70)
                 - BPMeds : 1 Yes, 0 No
                 - PrevalentStroke : 1 Yes, 0 No
                 - PrevalentHyp : 1 Yes, 0 No
                 - Diabetes : 1 Yes, 0 No
                 - TotChol : Total cholesterol (107 - 696)
                 - SysBP : 83.5-295
                 - DiaBP : 48-142.5
                 - BMI : Mengukur berat badan ideal (15.54-56.8)
                 - HearthRate : Denyut Jantung (44-143)
                 - Glucose : Kadar Glukosa dalam tubuh (40-394)
                 - HeartStroke : 1 Yes, 0 No
                 """

# Mengambil valuenya, key dan value
def get_value(val,my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

# Baca model        
@st.cache
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),'rb'))
    return loaded_model

def run_ml_app():
    st.subheader("ML section")

    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    st.subheader("Input Your Data")
    gender = st.radio('Gender', ['m','f'])
    age = st.number_input("Age",32,70)
    education = st.selectbox('education',["postgraduate","primaryschool", "Bachelor's", "Master's & above"])
    currentsmoker = st.radio('Current Smoke', ['Yes','No'])
    cigsperday = st.number_input("Ciggate Per Day",32,70)
    bpmeds = st.radio('BPM Eds', ['Yes','No'])
    prevalentstroke = st.radio('Prevalent Stroke', ['Yes','No'])
    prevalenthyp = st.radio('Prevalent Hypertensi', ['Yes','No'])
    diabetes = st.radio('Diabetes', ['Yes','No'])
    totchol = st.number_input("Total Chalorie",107,696)
    sysBP = st.number_input("System BP",83.5,295.0)
    diaBP = st.number_input("Daifragma BP",48.0,142.5)
    bmi = st.number_input("BMI",15.54,56.8)
    hearthrate = st.number_input("Hearth Rate",44,143)
    glucose = st.number_input("Glucose",40,394)
    hearhtstroke = st.radio('Hearth Stroke', ['Yes','No'])

    with st.expander("Your Selected Options"):
        result = {
            'gender':gender,
            'age':age,
            'education':education,
            'currentsmoker':currentsmoker,
            'cigsperday':cigsperday,
            'bpmeds':bpmeds,
            'prevalentstroke':prevalentstroke,
            'prevalenthyp':prevalenthyp,
            'diabetes':diabetes,
            'totchol':totchol,
            'sysBP':sysBP,
            'diaBP':diaBP,
            'bmi':bmi,
            'hearthrate':hearthrate,
            'glucose':glucose,
            'hearthstroke':hearhtstroke
        }

    # st.write(result)

    encoded_result = []
    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)
        elif i in ["Sales & Marketing", "Operations", "Technology", "Analytics", "R&D", "Procurement", "Finance", "HR", "Legal"]:
            res = get_value(i, dep)
            encoded_result.append(res)
        elif i in ['region_1','region_2','region_3','region_4','region_5', 'region_6','region_7',
                                     'region_8','region_9','region_10','region_11','region_12',
                                     'region_13','region_14','region_15','region_16','region_17','region_18','region_19',
                                     'region_20','region_21','region_22','region_23','region_24','region_25','region_26',
                                     'region_27','region_28','region_29','region_30','region_31','region_32','region_33',
                                     'region_34']:
            res = get_value(i, reg)
            encoded_result.append(res)
        elif i in ["Below Secondary", "Bachelor's", "Master's & above"]:
            res = get_value(i, edu)
            encoded_result.append(res)
        elif i in ['m','f']:
            res = get_value(i, gen)
            encoded_result.append(res)
        elif i in ["referred", "sourcing", "others"]:
            res = get_value(i, rec)
            encoded_result.append(res)

    # st.write(encoded_result)

    ## prediction section
    st.subheader('Prediction Result')
    single_sample = np.array(encoded_result).reshape(1,-1)
    # st.write(single_sample)

    model = load_model("model_grad.pkl")

    prediction = model.predict(single_sample)
    pred_proba = model.predict_proba(single_sample)
    # st.write(prediction)
    # st.write(pred_proba)

    pred_probability_score = {'Promoted':round(pred_proba[0][1]*100,4),
                                'Not Promoted':round(pred_proba[0][0]*100,4)}

    if prediction == 1:
        st.success("Besar kemungkinan Anda bakal terkena serangan jantung")
        
        st.write(pred_probability_score)
    else:
        st.warning("Kecil kemungkinan Anda bakal terkena serangan jantung")
        st.write(pred_probability_score)
