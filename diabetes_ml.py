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
        st.markdown("### Dibetes Disease Risk")
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
        # fig.update_layout(width=500,height=500)
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

    gender_option = st.selectbox(
                'Gender',
                ('Male', 'Female'))
    if gender_option == 'Male':
        gender = 1
    elif gender_option == 'Female':
        gender = 0

    age = st.number_input('Age')
    numResident = st.number_input('Number of residents')
    height = st.number_input('Height',value=1)
    bodyWeight = st.number_input('Weight',value=1)
    strides = st.number_input('Strides')
    bfp = st.number_input('Body Fat Percentage')
    bm = st.number_input('Basal Metabolism')
    smp = st.number_input('Skeletal Muscle Percentage')
    vfl = st.number_input('Visceral Fat Level')
    bodyage = st.number_input('Body Age')
    
    antihypertensive_op = st.selectbox(
                'Ever patient use antihypertensive?',
                ('Yes', 'No'))
    if antihypertensive_op == 'Yes':
        antihypertensive = 1
    else:
        antihypertensive = 0

    antidepressant_op = st.selectbox(
                'Ever patient use antidepressant?',
                ('Yes', 'No'))
    if antidepressant_op == 'Yes':
        antidepressant = 1
    else:
        antidepressant = 0

    osteoporosis_op = st.selectbox(
                'Ever patient use osteoporosis?',
                ('Yes', 'No'))
    if osteoporosis_op == 'Yes':
        osteoporosis = 1
    else:
        osteoporosis = 0

    antidiabetic_op = st.selectbox(
                'Ever patient use antidiabetic?',
                ('Yes', 'No'))
    if antidiabetic_op == 'Yes':
        antidiabetic = 1
    else:
        antidiabetic = 0

    smoking_op = st.selectbox(
                'Ever patient smoking?',
                ('Yes', 'No'))
    if smoking_op == 'Yes':
        smoking = 1
    else:
        smoking = 0
    
    drinking_op = st.selectbox(
                'Ever patient drinking?',
                ('Yes', 'No'))
    if drinking_op == 'Yes':
        drinking = 1
    else:
        drinking = 0

    MesurementBloodPressure_op = st.selectbox(
                'Everyday With or without blood pressure measurement',
                ('With', 'Without'))
    if MesurementBloodPressure_op == 'With':
        MesurementBloodPressure = 1
    else:
        MesurementBloodPressure = 0
    
    eatingHabbit_op = st.selectbox(
                'Eating habit',
                ('3 meals a day', '2 meals a day'))
    if eatingHabbit_op == '3 meals a day':
        eatingHabbit = 1
    else:
        eatingHabbit = 0

    preference_op = st.selectbox(
                'Eating habit',
                ('I like light taste', 'I like sweets'))
    if preference_op == 'I like light taste':
        preference = 0
    else:
        preference = 1

    sleep_op = st.selectbox(
                'Sleep state',
                ('Sleeping well', 'Sleeping','Not sleep'))
    if sleep_op == 'Sleeping':
        sleep = 1
    elif sleep_op == 'Sleeping well':
        sleep = 2
    else:
        sleep = 3

    anxietyHealth_op = st.selectbox(
                'Anxiety health',
                ('A lot', 'Some','None'))
    if anxietyHealth_op == 'A lot':
        anxietyHealth = 0
    elif anxietyHealth_op == 'Some':
        anxietyHealth = 2
    else:
        anxietyHealth = 1

    anxietyForgetful_op = st.selectbox(
                'Anxiety forgetful',
                ('A lot', 'Some','None'))
    if anxietyForgetful_op == 'A lot':
        anxietyForgetful = 0
    elif anxietyForgetful_op == 'Some':
        anxietyForgetful = 2
    else:
        anxietyForgetful = 1

    stairWithoutTransmitted_op = st.selectbox(
                'Can up and down stairs without transmitted?',
                ('Yes', 'No'))
    if stairWithoutTransmitted_op == 'Yes':
        stairWithoutTransmitted = 1
    else:
        stairWithoutTransmitted = 0

    walk15m_op = st.selectbox(
                'Can walk for more than 15 minutes?',
                ('Yes', 'No'))
    if walk15m_op == 'Yes':
        walk15m = 1
    else:
        walk15m = 0

    walkWithoutCane_op = st.selectbox(
                'Can walk without a cane?',
                ('Yes', 'No'))
    if walkWithoutCane_op == 'Yes':
        walkWithoutCane = 1
    else:
        walkWithoutCane = 0

    goingout_op = st.selectbox(
                'Actively going out',
                ('Yes', 'No'))
    if goingout_op == 'Yes':
        goingout = 1
    else:
        goingout = 0

    eatHardFood_op = st.selectbox(
                'Can patient eat hard food?',
                ('Yes', 'No'))
    if eatHardFood_op == 'Yes':
        eatHardFood = 1
    else:
        eatHardFood = 0

    hbp_op = st.selectbox(
                'Have High blood pressure?',
                ('Yes', 'No'))
    if hbp_op == 'Yes':
        hbp = 1
    else:
        hbp = 0
    
    bmi = (height*0.01)/bodyWeight
    features = [gender,age,numResident,height,bodyWeight,strides,bfp,bm,smp,vfl,bodyage,bmi,antihypertensive,antidepressant
                ,osteoporosis,antidiabetic,smoking,drinking,MesurementBloodPressure,eatingHabbit,preference
                ,sleep,anxietyHealth,anxietyForgetful,stairWithoutTransmitted,walk15m,walkWithoutCane,
                goingout,eatHardFood,hbp]

    results = np.array(features).reshape(1, -1)
    if st.button("Predict"):
        # lda = LDA(n_components=1)
        # features_results = lda.transform(results)
        picklefile = open("xgb_oversampling.pkl", "rb")
        model = pickle.load(picklefile)

        prediction = model.predict_proba(results)
        label0 = prediction[0][0]
        label1 = prediction[0][1]

        # string = 'Patient have risk diabetes: '+str(round(label1*(100),2))+'%'
        # if label1 > 0.5:
        #     st.error(string)
        # else:
        #     st.success(string)

        if label1 > label0:
            string = 'Patient have risk diabetes: '+str(round(label1*(100),2))+'%'
            st.error(string)
        else:
            string = "Patient don't have risk diabetes" #: '+str(round(label0*(100),2))+'%'
            st.success(string)
