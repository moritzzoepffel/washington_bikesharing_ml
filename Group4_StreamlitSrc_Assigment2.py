import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import os

# Read the data
path = os.path.dirname(__file__)
raw_data = pd.read_csv(f"{path}/data/bike-sharing_hourly.csv")
data = pd.read_csv(f"{path}/data/bike-sharing_hourly_cleaned.csv")

data["dteday"] = pd.to_datetime(data["dteday"], format="%Y-%m-%d").dt.date

st.markdown(
    """
            # Bike Sharing in Washington, D.C.
            
            Welcome to our bike sharing dashboard. We have created this dashboard to show you the data of the bike sharing in Washington, D.C. and to show you the insights we have found during the EDA process. Furthermore, we have created a prediction model which predicts the total count of bikes for a given day and other parameters. You can find the prediction model at the bottom of the page. The date range you can define in the sidebar applies on the whole data set and therefore on almost every figure in this dashboard. The other filters only apply to the corresponding figure.
            """
)

st.sidebar.title("Filters:")

startDate = st.sidebar.date_input("Start date", value=pd.to_datetime("2011-01-01"))
endDate = st.sidebar.date_input("End date", value=pd.to_datetime("2012-12-31"))

data = data.loc[(data["dteday"] >= startDate) & (data["dteday"] <= endDate)]

################# RAW DATA #################

st.header("Raw Data")
st.dataframe(raw_data.head(5))

################# EDA PROCESS #################

st.markdown(
    """
    # EDA Process
    1. Understanding the data
    2. Feature Engineering
    3. Outlier Detection
    4. Plotting clear and meaningful figures
    5. Insights on relevant columns for prediction
    6. Data after cleaning
"""
)
# Boxplos of Bike Sharing Demand
st.header("1. Understanding the data")
# only get numerical columns

st.dataframe(raw_data.describe())

fig = plt.figure(figsize=(10, 4))
sns.heatmap(
    raw_data[
        ["temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
    ].corr(),
    annot=True,
    cmap="coolwarm",
)
st.pyplot(fig)

# Conclusion
st.write("TODO: Conclusion")

#############

st.header("2. Feature Engineering")

st.code(
    """
        dict_daylight = {
    1: 9.8,
    2: 10.82,
    3: 11.98,
    4: 13.26,
    5: 14.34,
    6: 14.93,
    7: 14.68,
    8: 13.75,
    9: 12.5,
    10: 11.25,
    11: 10.12,
    12: 9.5
}

daylight_hrs = pd.DataFrame(dict_daylight.items(), columns=['mnth', 'daylight_hrs'])

data = data.join(daylight_hrs.set_index('mnth'), on='mnth')

data['daylight_hrs'] = data['daylight_hrs'].astype('float64')
        """
)

st.code(
    """
        season = pd.get_dummies(data['season'], prefix='season')
data = pd.concat([data, season], axis=1)
weather=pd.get_dummies(data['weathersit'],prefix='weathersit')
data=pd.concat([data,weather],axis=1)
weekday=pd.get_dummies(data['weekday'],prefix='weekday')
data=pd.concat([data,weekday],axis=1)
month = pd.get_dummies(data['mnth'], prefix='mnth')
data = pd.concat([data, month], axis=1)
data.drop(['season','weathersit', "weekday", "registered", "mnth", "dteday", "atemp", "instant", "casual"],inplace=True,axis=1)
"""
)

st.write(
    """
         For the feature engineering we have used the following steps:
         - We have created a new column called daylight_hrs which contains the daylight hours for each month.
         - We have created dummy variables for the categorical columns.
         - We have dropped the columns which are not needed for the prediction.
         - We have dropped the registered and casual column because we want to predict the total count of bikes.
         - We have dropped the instant column because it is just an index.
         - We have dropped the atemp column because it is highly correlated with the temp column.
         - We have dropped the dteday column because we have already the weekday and the month column.
         - We have dropped the season column because we have already the dummy variables for the season.
         - We have dropped the weathersit column because we have already the dummy variables for the weathersit.
         - We have dropped the weekday column because we have already the dummy variables for the weekday.
         - We have dropped the mnth column because we have already the dummy variables for the mnth.
         """
)

############

st.header("3. Outlier Detection")

data_outliers = raw_data[
    np.abs(raw_data["cnt"] - raw_data["cnt"].mean()) >= (2.5 * raw_data["cnt"].std())
    ]

col1, col2, col3 = st.columns(3)

fig = plt.figure(figsize=(10, 8))
sns.boxplot(
    data=raw_data,
    y="cnt",
)
col1.pyplot(fig)

fig2 = plt.figure(figsize=(10, 8))
sns.boxplot(data=data_outliers, y="cnt", x="season", orient="v")
col2.pyplot(fig2)

fig3 = plt.figure(figsize=(10, 8))
sns.boxplot(data=raw_data, y="cnt", x="season", orient="v")
col3.pyplot(fig3)

col1.markdown(
    """
              #### Raw Data
              Boxplot for the cnt column showing that there are a lot outliers.
              """
)

col2.markdown(
    """
              #### Outliers - Season Analysis
              These are the cnt values by season for the outliers.
              """
)

col3.markdown(
    """
              #### Data - Season Analysis
              These are the cnt values by season for the whole dataset.
              """
)

"""
We have seen, that there is no difference between the outliers and the cnt values by season. Therefore we decided to remove the outliers. To do that we have used the following code:
"""

st.code(
    """data = data[np.abs(data["cnt"]-data["cnt"].mean())<=(2.5*data["cnt"].std())]"""
)

"""
This code removes outliers which are more than 2.5 standard deviations away from the mean.

### After outlier removal
"""

fig = plt.figure(figsize=(10, 4))
sns.boxplot(data=data, y="cnt")
st.pyplot(fig)

"""
As we can see, there are less outliers now. We have removed 3% of the data.
"""

##########


# Boxplos of Bike Sharing Demand
st.header("4. Plotting clear and meaningful figures")

"""
    Please use the selectbox to choose a column to plot. The plot and descriptions will be shown below.
"""

# only get numerical columns
options = {
    "Daylight Hours": "daylight_hrs",
    "Holiday": "holiday",
    "Hour": "hr",
    "Season": "season",
    "Temperature": "temp",
    "Year": "yr",
    "Weather": "weathersit",
    "Working Day": "workingday",
}

column = st.selectbox(
    "Select Column",
    options=options.keys(),
)

column_chosen = options[column]

# fig = plt.figure(figsize=(10, 4))
# sns.boxplot(
#    data=data, y="cnt", x=column_chosen, orient="v"
# )  # TODO: Maybe change to plotly
# st.pyplot(fig)

fig = px.box(data, y="cnt", x=column_chosen, orientation="v")
st.plotly_chart(fig)

descriptions = {
    "Daylight Hours": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse vehicula dolor in auctor interdum. Vestibulum arcu sapien, efficitur sit amet varius nec, pharetra non nunc. Nunc vitae eros non sapien vulputate rhoncus ac id nibh. Praesent ac leo ac purus blandit blandit. Nam sit amet lorem et odio laoreet maximus. Duis gravida, nisi sed finibus efficitur, magna ligula sodales sapien, at pellentesque augue dolor sed eros. Mauris porttitor nibh nec odio tincidunt dictum. Maecenas lectus nulla, auctor in ornare in, pulvinar vel libero. Donec tincidunt sem at ex dapibus, sed semper justo condimentum. In hac habitasse platea dictumst. Etiam ac.",
    "Holiday": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. efficitur sit amet varius nec, pharetra non nunc. Nunc vitae eros non sapien vulputate rhoncus ac id nibh. Praesent ac leo ac purus blandit blandit. Nam sit amet lorem et odio laoreet maximus. Duis gravida, nisi sed finibus efficitur, magna ligula sodales sapien, at pellentesque augue dolor sed eros. Mauris porttitor nibh nec odio tincidunt dictum. Maecenas lectus nulla, auctor in ornare in, pulvinar vel libero. Donec tincidunt sem at ex dapibus, sed semper justo condimentum. In hac habitasse platea dictumst. Etiam ac.",
    "Hour": " Suspendisse vehicula dolor in auctor interdum. Vestibulum arcu sapien, efficitur sit amet varius nec, pharetra non nunc. Nunc vitae eros non sapien vulputate rhoncus ac id nibh. Praesent ac leo ac purus blandit blandit. Nam sit amet lorem et odio laoreet maximus. Duis gravida, nisi sed finibus efficitur, magna ligula sodales sapien, at pellentesque augue dolor sed eros. Mauris porttitor nibh nec odio tincidunt dictum. Maecenas lectus nulla, auctor in ornare in, pulvinar vel libero. Donec tincidunt sem at ex dapibus, sed semper justo condimentum. In hac habitasse platea dictumst. Etiam ac.",
    "Season": "efficitur sit amet varius nec, pharetra non nunc. Nunc vitae eros non sapien vulputate rhoncus ac id nibh. Praesent ac leo ac purus blandit blandit. Nam sit amet lorem et odio laoreet maximus. Duis gravida, nisi sed finibus efficitur, magna ligula sodales sapien, at pellentesque augue dolor sed eros. Mauris porttitor nibh nec odio tincidunt dictum. Maecenas lectus nulla, auctor in ornare in, pulvinar vel libero. Donec tincidunt sem at ex dapibus, sed semper justo condimentum. In hac habitasse platea dictumst. Etiam ac.",
    "Temperature": " amet, consectetur adipiscing elit. Suspendisse vehicula dolor in auctor interdum. Vestibulum arcu sapien, efficitur sit amet varius nec, pharetra non nunc. Nunc vitae eros non sapien vulputate rhoncus ac id nibh. Praesent ac leo ac purus blandit blandit. Nam sit amet lorem et odio laoreet maximus. Duis gravida, nisi sed finibus efficitur, magna ligula sodales sapien, at pellentesque augue dolor sed eros. Mauris porttitor nibh nec odio tincidunt dictum. Maecenas lectus nulla, auctor in ornare in, pulvinar vel libero. Donec tincidunt sem at ex dapibus, sed semper justo condimentum. In hac habitasse platea dictumst. Etiam ac.",
    "Year": "Lorem ipsum dolorcing elit. Suspendisse vehicula dolor in auctor interdum. Vestibulum arcu sapien, efficitur sit amet varius nec, pharetra non nunc. Nunc vitae eros non sapien vulputate rhoncus ac id nibh. Praesent ac leo ac purus blandit blandit. Nam sit amet lorem et odio laoreet maximus. Duis gravida, nisi sed finibus efficitur, magna ligula sodales sapien, at pellentesque augue dolor sed eros. Mauris porttitor nibh nec odio tincidunt dictum. Maecenas lectus nulla, auctor in ornare in, pulvinar vel libero. Donec tincidunt sem at ex dapibus, sed semper justo condimentum. In hac habitasse platea dictumst. Etiam ac.",
    "Weather": "Lorem ipsum dolor sit amet, consecteturhicula dolor in auctor interdum. Vestibulum arcu sapien, efficitur sit amet varius nec, pharetra non nunc. Nunc vitae eros non sapien vulputate rhoncus ac id nibh. Praesent ac leo ac purus blandit blandit. Nam sit amet lorem et odio laoreet maximus. Duis gravida, nisi sed finibus efficitur, magna ligula sodales sapien, at pellentesque augue dolor sed eros. Mauris porttitor nibh nec odio tincidunt dictum. Maecenas lectus nulla, auctor in ornare in, pulvinar vel libero. Donec tincidunt sem at ex dapibus, sed semper justo condimentum. In hac habitasse platea dictumst. Etiam ac.",
    "Working Day": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse vehicula lum arcu sapien, efficitur sit amet varius nec, pharetra non nunc. Nunc vitae eros non sapien vulputate rhoncus ac id nibh. Praesent ac leo ac purus blandit blandit. Nam sit amet lorem et odio laoreet maximus. Duis gravida, nisi sed finibus efficitur, magna ligula sodales sapien, at pellentesque augue dolor sed eros. Mauris porttitor nibh nec odio tincidunt dictum. Maecenas lectus nulla, auctor in ornare in, pulvinar vel libero. Donec tincidunt sem at ex dapibus, sed semper justo condimentum. In hac habitasse platea dictumst. Etiam ac.",
}

st.write(f"Description: ")
st.write(descriptions[column])

##########

st.header("5. Insights on relevant columns for prediction")

st.markdown(
    """
            ### We have a lot of columns in our dataset. We will focus on the following columns for our prediction:
            - **Season**: 1 = spring, 2 = summer, 3 = fall, 4 = winter
            - **Year**: 0 = 2011, 1 = 2012
            - **Month**: 1 to 12
            - **Hour**: 0 to 23
            - **Holiday**: whether the day is considered a holiday
            - **Working Day**: whether the day is neither a weekend nor holiday
            - **Weather**: 1: Clear, Few clouds, Partly cloudy, Partly cloudy
            - **Temperature**: temperature in Celsius
            - **Humidity**: relative humidity
            - **Windspeed**: wind speed
            
            ### Further we have the following insights from the charts above:
            - The bike sharing demand is highest in the summer and lowest in the winter
            - The bike sharing demand is highest in 2012
            - The bike sharing demand is highest in the months of July and August
            - The bike sharing demand is highest in the hours of 7am and 8am and 5pm and 6pm
            - The bike sharing demand is highest on working days
            - The bike sharing demand is highest when the weather is clear
            - The bike sharing demand is highest when the temperature is between 20 and 30 degrees Celsius
            - The bike sharing demand is highest when the humidity is between 40 and 60 percent
            - The bike sharing demand is highest when the windspeed is between 0 and 20 km/h
            """
)

############


# Cleaned data

st.header("6. Data after cleaning")

st.dataframe(data.head(5))
st.dataframe(data.describe())

###########

# Line Chart of Bike Sharing Demand
st.write("### Bike Sharing Demand by Date")

# use ff to create timeseries plot
fig = px.line(data, x="dteday", y="cnt")
st.plotly_chart(fig)

################# PREDCITION PROCESS #################

st.write("# Prediction Process")

st.header("RMSLE")

rmsle_frame = pd.read_csv(f"{path}/data/rmsle_frame.csv")

st.dataframe(rmsle_frame.sort_values(by="RMSLE", ascending=True).reset_index(drop=True))

st.header("Feature Importance")
fig = plt.figure(figsize=(10, 5))
feat_importances = pd.read_csv(f"{path}/data/feat_importances.csv")
most_important = feat_importances.sort_values(by="0", ascending=False).head(10)
fig = px.bar(most_important, x="0", y="index", orientation="h")
st.plotly_chart(fig)

######### PREDICTION #########

dict_daylight = {
    1: 9.8,
    2: 10.82,
    3: 11.98,
    4: 13.26,
    5: 14.34,
    6: 14.93,
    7: 14.68,
    8: 13.75,
    9: 12.5,
    10: 11.25,
    11: 10.12,
    12: 9.5,
}

st.header("Predictions")

col1, col2, col3 = st.columns(3)

yr = col1.selectbox("Year", options=[0, 1], index=1)
hr = col3.selectbox(
    "Hour",
    options=[
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
    ],
    index=8,
)
mnth = col2.selectbox("Month", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], index=5)
holiday = col1.selectbox("Holiday", options=[0, 1], index=0)
workingday = col2.selectbox("Working Day", options=[0, 1], index=1)
weekday = col3.selectbox("Weekday", options=[0, 1, 2, 3, 4, 5, 6], index=2)
hum = col1.slider("Humidity", min_value=0.0, max_value=1.0, value=0.5)
windspeed = col2.slider("Windspeed", min_value=0.0, max_value=1.0, value=0.5)
daylight_hrs = dict_daylight[mnth]
season = (
    1
    if mnth in [3, 4, 5]
    else 2
    if mnth in [6, 7, 8]
    else 3
    if mnth in [9, 10, 11]
    else 4
)
temp = col3.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)
weathersit = col1.selectbox("Weather Situation", options=[1, 2, 3, 4], index=0)

values = [yr, hr, holiday, workingday, temp, hum, windspeed, daylight_hrs]
values_cat = [season, weathersit, weekday, mnth]

columns = [
    "yr",
    "hr",
    "holiday",
    "workingday",
    "temp",
    "hum",
    "windspeed",
    "daylight_hrs",
    "season_1",
    "season_2",
    "season_3",
    "season_4",
    "weathersit_1",
    "weathersit_2",
    "weathersit_3",
    "weathersit_4",
    "weekday_0",
    "weekday_1",
    "weekday_2",
    "weekday_3",
    "weekday_4",
    "weekday_5",
    "weekday_6",
    "mnth_1",
    "mnth_2",
    "mnth_3",
    "mnth_4",
    "mnth_5",
    "mnth_6",
    "mnth_7",
    "mnth_8",
    "mnth_9",
    "mnth_10",
    "mnth_11",
    "mnth_12",
]

x_test = {}

i = 0

for col in columns:
    x_test[col] = values[i]
    i += 1
    if i == 8:
        break

for i in range(4):
    if season == i:
        x_test["season_" + str(i + 1)] = 1
    else:
        x_test["season_" + str(i + 1)] = 0

for i in range(4):
    if weathersit == i:
        x_test["weathersit_" + str(i + 1)] = 1
    else:
        x_test["weathersit_" + str(i + 1)] = 0

for i in range(7):
    if weekday == i:
        x_test["weekday_" + str(i)] = 1
    else:
        x_test["weekday_" + str(i)] = 0

for i in range(12):
    if mnth == i:
        x_test["mnth_" + str(i + 1)] = 1
    else:
        x_test["mnth_" + str(i + 1)] = 0

x_test = pd.DataFrame(x_test, index=[0])

st.dataframe(x_test)

loaded_model = pickle.load(open(f"{path}/models/rf_model.sav", "rb"))

result = loaded_model.predict(x_test)

st.write(f"""
We're predicting :red[{result[0].round(0)}] rentals for the given parameters.
""")
