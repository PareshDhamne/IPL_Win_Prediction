import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

teams=[
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad',
    'Delhi Capitals',
    'Chennai Super Kings',
    'Gujarat Titans',
    'Lucknow Super Giants',
    'Kolkata Knight Riders',
    'Punjab Kings',
    'Mumbai Indians'
]

cities = [
    'Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai',
       'Sharjah', 'Abu Dhabi', 'Delhi', 'Chennai', 'Hyderabad',
       'Visakhapatnam', 'Chandigarh', 'Bengaluru', 'Jaipur', 'Indore',
       'Bangalore', 'Raipur', 'Ranchi', 'Cuttack', 'Dharamsala', 'Nagpur',
       'Johannesburg', 'Centurion', 'Durban', 'Bloemfontein',
       'Port Elizabeth', 'Kimberley', 'East London', 'Cape Town'
]

team_colors = {
    'Chennai Super Kings': 'yellow',
    'Mumbai Indians': 'blue',
    'Rajasthan Royals':'pink',
    'Royal Challengers Bangalore':'red',
    'Sunrisers Hyderabad':'orange',
    'Delhi Capitals':'blue',
    'Gujarat Titans':'#000080',
    'Lucknow Super Giants':'#00FFFF',
    'Kolkata Knight Riders':'purple',
    'Punjab Kings':'red'
}

pipe=pickle.load(open('pipe.pkl','rb'))
st.title('IPL Match Win Predictor')

col1,col2=st.columns(2)

with col1:
    batting_team=st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team=st.selectbox('Select the bowling team',sorted(teams))

slected_city =st.selectbox('Select Host City',sorted(cities))

target=st.number_input('Target')

col3,col4,col5=st.columns(3)

with col3:
    score=st.number_input('Score')
with col4:
    overs=st.number_input('Overs completed')
with col5:
    wickets=st.number_input('Total Wickets')

if st.button('Predict Probability'):
    runs_left=target-score
    balls_left=120-(overs*6)
    wickets=10-wickets
    crr=round(score/overs,2)
    rrr=round((runs_left*6)/balls_left,2)

    input_df=pd.DataFrame({'BattingTeam':[batting_team],'BowlingTeam':[bowling_team],
    'City':[slected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wickets],
    'target':[target],'current_run_rate':[crr],'required_run_rate':[rrr]})

    result = pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.header(batting_team+": "+str(round(win*100)) +"%")
    st.header(bowling_team+": "+str(round(loss*100)) +"%")
   
    colors = [team_colors.get(label, 'grey') for label in [batting_team, bowling_team]]

    # Create a pie chart
    labels = [batting_team, bowling_team]
    sizes = [win, loss]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
    ax.axis('equal')  
    # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)

