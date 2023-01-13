import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px  

st.set_page_config(layout="wide", initial_sidebar_state="expanded",page_title="Diabetes Disease Monitoring App",)

# Setting page mode
app_mode = st.sidebar.selectbox('Select Page',['Home','Diabetes Prediction'])

# Setting data sensor
df_sensor = pd.read_csv('./PBL_data/weight_data_0.csv')
measuredt = df_sensor['Measure Date Time'].str.split('T', expand = True)
df_sensor[['date', 'time']] = measuredt

df_pred = pd.read_csv('./prediction_result.csv')

if app_mode=='Home':
    st.title("Patients cefoxSR Dashboard")

    # top-level filters
    patient_filter = st.selectbox("Select the patient", pd.unique(df_sensor["subject_id"]))

    # dataframe filter
    df_sensor = df_sensor[df_sensor["subject_id"] == patient_filter]
    df_pred = df_pred[df_pred["subject_id"] == patient_filter]

    n_age = len(df_sensor['age'])
    gender = df_sensor['gender'][:1].values
    if gender == 'F':
        gender_values = 'Female'
    else:
        gender_values = 'Male'

    # create three columns
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    # fill in those three columns with respective metrics or KPIs
    kpi1.metric(
        label="Age",
        value=int(df_sensor['age'][n_age-1:n_age].values),
    )
    kpi2.metric(
        label="Gender",
        value=gender_values,
    )
    kpi3.metric(
        label="Average BMI",
        value=int(df_sensor['BMI'].mean()),
    )
    kpi4.metric(
        label="Average Body Age",
        value=int(df_sensor['Body Age'].mean()),
    )

    gauge1,gauge2= st.columns(2)
    with gauge1:
        st.markdown("### Diabetes Disease Risk")
        fig = go.Figure()

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            number = {'suffix': "%" },
            value = float(df_pred['risk_score_percent']),
            title = {'text': "Dibetes Disease Risk"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'shape':'angular',
                # 'steps':[
                #     {'range':[0,25],'color':'lightgray'},
                #     {'range':[25,50],'color':'gray'},
                #     {'range':[50,100],'color':'white'}
                # ],
                'bar':{'color':'green','thickness':0.5},
                'threshold':{'line':{'color':'red','width':4},
                            'thickness':0.8, 'value':50},
                'axis':{'range':[None,100]},
            }
        ))

        st.plotly_chart(fig, width=350,height=350)
    with gauge2:
        st.markdown("### Weight")
        fig1 = px.line(data_frame=df_sensor, x="Measure Date Time", y="Weight")
        fig1.update_traces(line_color='#0000ff', line_width=3)
        st.write(fig1)

    # create two columns for charts
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        st.markdown("### Body Fat Percentage")
        fig2 = px.line(data_frame=df_sensor, x="Measure Date Time", y="Body Fat Percentage")
        fig2.update_traces(line_color='#FF5733', line_width=3)
        st.write(fig2)
    
    with fig_col2:
        st.markdown("### Basal Metabolism")
        fig3 = px.line(data_frame=df_sensor, x="Measure Date Time", y="Basal Metabolism")
        fig3.update_traces(line_color='#41C835', line_width=3)
        st.write(fig3)
    

    fig_col3, fig_col4 = st.columns(2)
    with fig_col3:
        st.markdown("### Skeletal Muscle Percentage")
        fig4 = px.line(data_frame=df_sensor, x="Measure Date Time", y="Skeletal Muscle Percentage")
        fig4.update_traces(line_color='#C135C8', line_width=3)
        st.write(fig4)
    with fig_col4:
        st.markdown("### Visceral Fat Level")
        fig5 = px.line(data_frame=df_sensor, x="Measure Date Time", y="Visceral Fat Level")
        fig5.update_traces(line_color='#FFE930', line_width=3)
        st.write(fig5)

elif app_mode=='Diabetes Prediction':
    st.subheader('Fill in patient details to get predict')

    col1,col2,col3 = st.columns(3)
    with col1:
        gender_option = st.selectbox(
                'Gender',
                ('Male', 'Female'))
    if gender_option == 'Male':
        gender = 1
    elif gender_option == 'Female':
        gender = 0
    with col2:
        age = st.number_input('Age',value=0)
    with col3:
        numResident = st.number_input('Number of residents',value=0)

    col4,col5,col6 = st.columns(3)
    with col4:
        height = st.number_input('Height',value=1)
    with col5:
        bodyWeight = st.number_input('Weight',value=1)
    with col6:
        strides = st.number_input('Strides',value=0)

    col7,col8,col9 = st.columns(3)   
    with col7: 
        bfp = st.number_input('Body Fat Percentage')
    with col8: 
        bm = st.number_input('Basal Metabolism')
    with col9: 
        smp = st.number_input('Skeletal Muscle Percentage')

    col10,col11,col12 = st.columns(3) 
    with col10:
        vfl = st.number_input('Visceral Fat Level')
    with col11:
        bodyage = st.number_input('Body Age')
    with col12:
        antidepressant_op = st.selectbox(
                    'Ever patient use antidepressant?',
                    ('Yes', 'No'))
        if antidepressant_op == 'Yes':
            antidepressant = 1
        else:
            antidepressant = 0

    col13,col14,col15 = st.columns(3)
    with col13:
        osteoporosis_op = st.selectbox(
                    'Ever patient use osteoporosis?',
                    ('Yes', 'No'))
        if osteoporosis_op == 'Yes':
            osteoporosis = 1
        else:
            osteoporosis = 0
    with col14:
        antidiabetic_op = st.selectbox(
                    'Ever patient use antidiabetic?',
                    ('Yes', 'No'))
        if antidiabetic_op == 'Yes':
            antidiabetic = 1
        else:
            antidiabetic = 0
    with col15:
        drinking_op = st.selectbox(
                    'Ever patient drinking?',
                    ('Yes', 'No'))
        if drinking_op == 'Yes':
            drinking = 1
        else:
            drinking = 0

    col16,col17,col18 = st.columns(3)
    with col16:
        preference_op = st.selectbox(
                    'Eating habit',
                    ('I like light taste', 'I like sweets'))
        if preference_op == 'I like light taste':
            preference = 0
        else:
            preference = 1
    with col17:
        anxietyHealth_op = st.selectbox(
                    'Anxiety health',
                    ('A lot', 'Some','None'))
        if anxietyHealth_op == 'A lot':
            anxietyHealth = 0
        elif anxietyHealth_op == 'Some':
            anxietyHealth = 2
        else:
            anxietyHealth = 1
    with col18:
        anxietyForgetful_op = st.selectbox(
                    'Anxiety forgetful',
                    ('A lot', 'Some','None'))
        if anxietyForgetful_op == 'A lot':
            anxietyForgetful = 0
        elif anxietyForgetful_op == 'Some':
            anxietyForgetful = 2
        else:
            anxietyForgetful = 1
    
    bmi = bodyWeight/((height/100)**2)
    features = [gender,age,numResident,height,bodyWeight,strides,bfp,bm,smp,vfl,bodyage,bmi,antidepressant
                ,osteoporosis,antidiabetic,drinking,preference
                ,anxietyHealth,anxietyForgetful]

    results = np.array(features).reshape(1, -1)
    if st.button("Predict"):
    
        picklefile = open("xgb_oversampling2.pkl", "rb")
        model = pickle.load(picklefile)

        prediction = model.predict_proba(results)
        label0 = prediction[0][0]
        label1 = prediction[0][1]


        if label1 > label0:
            string = 'Patient have risk diabetes: '+str(round(label1*(100),2))+'%'
            st.error(string)
        else:
            string = "Patient don't have risk diabetes"
            st.success(string)
