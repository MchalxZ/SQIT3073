# importing the necessary libraries
import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import folium
import numpy as np
import pandas as pd
import plost
import pytz
import requests
import streamlit as st
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from streamlit_folium import st_folium

# setting the global variable to make sure the variable can be accessed any part
st.session_state.navigation = False
st.session_state.validity = False

# initialize first page
if 'page' not in st.session_state:
    st.session_state.page = 0

# initialize the page name
pages = ['Home', 'Forecast', 'Flight']


# moving to next page
def next_page():
    st.session_state.page += 1


# moving to previous page
def prev_page():
    st.session_state.page -= 1


# this function is used for retrieving the historical data of weather by using API service
def fetch_historical_weather(lat, long, api_key, date):
    url = f'https://api.weatherapi.com/v1/history.json?key={api_key}&q={lat},{long}&dt={date}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f'HTTP Request failed for date {date}: {e}')
        return None


# The function converts location into latitude and longitude
def get_lat_long(location, api_key):
    url = f'https://api.opencagedata.com/geocode/v1/json?q={location}&key={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['results'][0]['geometry']['lat'], data['results'][0]['geometry']['lng']
    except requests.exceptions.RequestException as e:
        st.error(f'HTTP Request failed for geocoding {location}: {e}')
        return None, None


# The function returns the api_key
def get_api_key(service_name):
    api_keys = {
        'GEOCODE API': '3f34588e34254d84b706dae3e247814a',
        'WEATHER API': '65ade77659214036bec132718240107',
        'FLIGHT API': 'aWrrXWTDigGPGzMMqVIh3VVZnJVVHBEh',
        'SECRET API': 'fWn1JsgRWHvGjBsq',
    }
    return api_keys.get(service_name)


# The function is used to connect to cassandra
def connect_to_cassandra():
    try:
        # using localhost address as cluster
        cluster = Cluster(['127.0.0.1'])
        # connect to keyspace
        session = cluster.connect('sqit3073')
        return session
    except Exception as e:
        # exception might occur when keyspace not found
        st.error(f'Failed to connect to Cassandra: {e}')
        return None


# The function is used for checking the data into cassandra to decide retrival of data
def check_existing_data(session, place, date):
    query = 'SELECT COUNT(*) FROM WEATHER WHERE place = %s AND date = %s'
    result = session.execute(query, (place, date)).one()
    return result.count > 0


# The function is used for fetch and store historical data into cassandra
def fetch_and_store_weather_data(session, lat, long, api_key, date, place):
    # check if data is existing in the cassandra
    if check_existing_data(session, place, date):
        return None
    # fetch historical weather through api service
    weather_data = fetch_historical_weather(lat, long, api_key, date)
    if weather_data and 'forecast' in weather_data and 'forecastday' in weather_data['forecast']:
        # assigning it to each corresponding variable
        forecast_day = weather_data['forecast']['forecastday'][0]
        date = forecast_day['date']
        temperature = forecast_day['day']['avgtemp_c']
        weather_condition = forecast_day['day']['condition']['text']
        humidity = forecast_day['day']['avghumidity']
        windspeed = forecast_day['day']['maxwind_kph']
        place = weather_data['location']['name']
        result = (place, date, humidity, temperature, weather_condition, windspeed)
        # check if all the data is acquired then return the result
        if len(result) == 6:
            return result
    # if necessary result does not occur, then return nothing
    return None


# The function is fetch the weather data from cassandra between a certain duration.
def fetch_weather_data(session, place, start_date, end_date):
    query = '''
    SELECT * FROM weather
    WHERE place = %s AND date >= %s AND date <= %s
    '''
    # result will be obtained after executing the queries from cassandra
    try:
        result = session.execute(query, (place, str(start_date), str(end_date)))
        return pd.DataFrame(list(result))
    except Exception as e:
        print(f'Error executing query: {e}')
        return pd.DataFrame()


# The function is used for assigning the target and feature data, performing oversampling method to handle imbalance class
def preprocess_data(df):
    # use label_encoder function to encoding the categorical variable to numeric value
    label_encoder = LabelEncoder()
    df['weather_condition'] = label_encoder.fit_transform(df['weather_condition'])
    class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    # print the all available classes after mapping
    print(f'Class mapping: {class_mapping}')
    # defining the X (features) and y (target)
    X = df[['temperature', 'humidity', 'windspeed']]
    y = df['weather_condition']
    # concatenating both by columns
    data = pd.concat([X, y], axis=1)
    # print the count in each of the classes
    class_counts = data['weather_condition'].value_counts()
    print(f'Class counts:\n{class_counts}')
    # determining the min count among the classes
    min_size = data['weather_condition'].value_counts().min()
    # determining the max count among the classes
    max_size = data['weather_condition'].value_counts().max()
    print(f'Min size (imbalanced value): {min_size}')
    print(f'Max size (balanced value): {max_size}')  # Print the balanced value
    # insert the dataframe into list
    lst = [data]
    # iteration to make all the data follow the highest count so all the data will be trained well without bias
    for class_index, group in data.groupby('weather_condition'):
        lst.append(group.sample(max_size - len(group), replace=True))
    # concatenating the dataframe and put it into a new variable
    data_resampled = pd.concat(lst)
    # removing the weather_condition since it is our target variable, will not be feature variable
    X_resampled = data_resampled.drop('weather_condition', axis=1)
    y_resampled = data_resampled['weather_condition']
    # Verification step to check the value of all classes to make sure all be the highest count
    print('Verification of resampled: ' + str(y_resampled.value_counts()))
    # return the resampled values and label encoder
    return X_resampled, y_resampled, label_encoder


# The function is used for training the decision tree for 70 train and 30 test set
def train_decisiontree_70(X, y):
    print('Starting DecisionTree with 70 train and 30 test...')
    # print the total count
    print('Total count: ' + str(len(X)))
    class_distribution = Counter(y)
    class_distribution_y = dict(class_distribution)
    # print the count in each of the class
    print('balanced y: ' + str(class_distribution_y))
    # X_train, X_test are representing the features train and features test while y_train and y_test are representing the target train and target test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    class_distribution = Counter(y_train)
    class_distribution_ytrain = dict(class_distribution)
    # print the count in the train set to see if the data is 70% from the overall data
    print('y-train: ' + str(class_distribution_ytrain))
    class_distribution = Counter(y_test)
    class_distribution_ytest = dict(class_distribution)
    # print the count in the train set to see if the data is 30% from the overall data
    print('y-test: ' + str(class_distribution_ytest))
    # using standard scalar to fit the feature train and doing transformation using feature test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print('Training set: ' + str(len(X_train)))  # Number of training samples
    print('Testing set: ' + str(len(X_test)))  # Number of testing samples

    # param_grid contains all the possible search attribute
    param_grid = {
        'criterion': ['gini', 'entropy'],  # the criterion that used in decision tree
        'max_depth': [None, 10, 20, 30, 40, 50],  # the max depth used in decision tree
        'min_samples_split': [2, 10, 20],  # the min samples split used in decision tree
        'min_samples_leaf': [1, 5, 10]  # the min samples leaf used in decision tree
    }

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, verbose=2, n_jobs=-1)

    # Fit the model
    grid_search.fit(X_train_scaled, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters of decision tree with 70:30 : ", grid_search.best_params_)

    # Make predictions on the test set
    y_pred = grid_search.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    print_evaluation('Decision Tree 70:30', y_test, y_pred)
    # return the best result, transformed scaler, accuracy and precision
    return grid_search.best_estimator_, scaler, accuracy, precision


# The function is used for training the decision tree for 80 train and 20 test set
def train_decisiontree_80(X, y):
    print('Starting DecisionTree with 80 train and 20 test...')
    # print the total count
    print('Total count: ' + str(len(X)))
    class_distribution = Counter(y)
    class_distribution_y = dict(class_distribution)
    # print the count in each of the class
    print('balanced y: ' + str(class_distribution_y))
    # X_train, X_test are representing the features train and features test while y_train and y_test are representing the target train and target test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    class_distribution = Counter(y_train)
    class_distribution_ytrain = dict(class_distribution)
    # print the count in the train set to see if the data is 80% from the overall data
    print('y-train: ' + str(class_distribution_ytrain))
    class_distribution = Counter(y_test)
    class_distribution_ytest = dict(class_distribution)
    # print the count in the train set to see if the data is 20% from the overall data
    print('y-test: ' + str(class_distribution_ytest))
    # using standard scalar to fit the feature train and doing transformation using feature test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print('Training set: ' + str(len(X_train)))  # Number of training samples
    print('Testing set: ' + str(len(X_test)))  # Number of testing samples

    # param_grid contains all the possible search attribute
    param_grid = {
        'criterion': ['gini', 'entropy'],  # the criterion that used in decision tree
        'max_depth': [None, 10, 20, 30, 40, 50],  # the max depth used in decision tree
        'min_samples_split': [2, 10, 20],  # the min samples split used in decision tree
        'min_samples_leaf': [1, 5, 10]  # the min samples leaf used in decision tree
    }

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, verbose=2, n_jobs=-1)

    # Fit the model
    grid_search.fit(X_train_scaled, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters of decision tree 80:20 : ", grid_search.best_params_)

    # Make predictions on the test set
    y_pred = grid_search.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    print_evaluation('Decision Tree 80:20', y_test, y_pred)
    # return the best result, transformed scaler, accuracy and precision
    return grid_search.best_estimator_, scaler, accuracy, precision


# The function prints the result for evaluation
def print_evaluation(name, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy for ' + name + ': ', accuracy)
    print('Confusion Matrix ' + name + ':\n', confusion_matrix(y_test, y_pred))
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Sum up TP, TN, FP, FN for all classes
    TP = np.diag(conf_matrix)
    FP = conf_matrix.sum(axis=0) - TP
    FN = conf_matrix.sum(axis=1) - TP

    # Calculate TN for all classes
    total_instances = conf_matrix.sum()
    TN = total_instances - (TP + FP + FN)

    # Calculate accuracy using formula
    accuracy_manual = TP.sum() / total_instances

    # Calculate accuracy using sklearn's accuracy_score for comparison
    accuracy_sklearn = accuracy_score(y_test, y_pred)

    print(f"Manual Accuracy: {accuracy_manual:.4f}")
    print(f"Sklearn Accuracy: {accuracy_sklearn:.4f}")
    print('Classification Report ' + name + ':\n', classification_report(y_test, y_pred))


# The function fetch the real world data to make example prediction through API
def fetch_forecast_data(place, api_key):
    url = f'https://api.weatherapi.com/v1/forecast.json?key={api_key}&q={place}&days=7'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f'HTTP Request failed for forecast data: {e}')
        return None


# The function is used to predict the forecast value
def make_forecasts(model, scaler, place, api_key):
    # Fetch forecast data if data not exist, returning five empty list
    forecast_data = fetch_forecast_data(place, api_key)
    if not forecast_data:
        return [], [], [], [], []
    forecast_days = forecast_data['forecast']['forecastday']
    forecast_dates = []
    temperature_data = []
    humidity_data = []
    windspeed_data = []
    # assigning the value into the variable if data is existed
    for day in forecast_days:
        date = day['date']
        temperature = day['day']['avgtemp_c']
        humidity = day['day']['avghumidity']
        windspeed = day['day']['maxwind_kph']
        # append the variable into the expected data
        forecast_dates.append(date)
        temperature_data.append(temperature)
        humidity_data.append(humidity)
        windspeed_data.append(windspeed)
    # Prepare the data for prediction
    forecast_features = pd.DataFrame({
        'temperature': temperature_data,
        'humidity': humidity_data,
        'windspeed': windspeed_data
    })
    # doing transformation using scaler
    forecast_features_scaled = scaler.transform(forecast_features)
    # prediction starts
    forecast_conditions_numeric = model.predict(forecast_features_scaled)
    forecast_conditions = label_encoder.inverse_transform(forecast_conditions_numeric)
    # Ensure all forecast conditions are strings
    forecast_conditions = [str(condition) for condition in forecast_conditions]
    return forecast_dates, forecast_conditions, temperature_data, humidity_data, windspeed_data


# The function is used for training the K-Nearest Neighbours for 80 train and 20 test set
def train_knn_80(X, y):
    print('Starting KNN with 80 train and 20 test...')
    # print the total count
    print('Total count: ' + str(len(X)))
    class_distribution = Counter(y)
    class_distribution_y = dict(class_distribution)
    # print the count in each of the class
    print('balanced y: ' + str(class_distribution_y))
    # X_train, X_test are representing the features train and features test while y_train and y_test are representing the target train and target test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    class_distribution = Counter(y_train)
    class_distribution_ytrain = dict(class_distribution)
    # print the count in the train set to see if the data is 80% from the overall data
    print('y-train: ' + str(class_distribution_ytrain))
    class_distribution = Counter(y_test)
    class_distribution_ytest = dict(class_distribution)
    # print the count in the train set to see if the data is 20% from the overall data
    print('y-test: ' + str(class_distribution_ytest))
    # using standard scalar to fit the feature train and doing transformation using feature test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print('Training set: ' + str(len(X_train)))  # Number of training samples
    print('Testing set: ' + str(len(X_test)))  # Number of testing samples

    # param_grid contains all the possible attribute search
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],  # neighbours count of KNN
        'weights': ['uniform', 'distance'],  # the weights of KNN
        'metric': ['euclidean', 'manhattan', 'minkowski']  # the metric of KNN
    }

    # Initialize the GridSearchCV object
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=2, cv=5)

    # Fit the model
    grid.fit(X_train_scaled, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters of KNN 80:20 : ", grid.best_params_)

    # Make predictions on the test set
    y_pred = grid.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    print_evaluation('KNN 80:20', y_test, y_pred)
    # return the best result, transformed scaler, accuracy and precision
    return grid.best_estimator_, scaler, accuracy, precision


# The function is used for training the K-Nearest Neighbours for 70 train and 30 test set
def train_knn_70(X, y):
    print('Starting KNN with 70 train and 30 test...')
    # print the total count
    print('Total count: ' + str(len(X)))
    class_distribution = Counter(y)
    class_distribution_y = dict(class_distribution)
    # print the count in each of the class
    print('balanced y: ' + str(class_distribution_y))
    # X_train, X_test are representing the features train and features test while y_train and y_test are representing the target train and target test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    class_distribution = Counter(y_train)
    class_distribution_ytrain = dict(class_distribution)
    # print the count in the train set to see if the data is 70% from the overall data
    print('y-train: ' + str(class_distribution_ytrain))
    class_distribution = Counter(y_test)
    class_distribution_ytest = dict(class_distribution)
    # print the count in the train set to see if the data is 30% from the overall data
    print('y-test: ' + str(class_distribution_ytest))
    # using standard scalar to fit the feature train and doing transformation using feature test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print('Training set: ' + str(len(X_train)))  # Number of training samples
    print('Testing set: ' + str(len(X_test)))  # Number of testing samples

    # param_grid contains all the possible attribute search
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],  # neighbours count of KNN
        'weights': ['uniform', 'distance'],  # the weights of KNN
        'metric': ['euclidean', 'manhattan', 'minkowski']  # the metric of KNN
    }

    # Initialize the GridSearchCV object
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=2, cv=5)

    # Fit the model
    grid.fit(X_train_scaled, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters of KNN 70:30 : ", grid.best_params_)

    # Make predictions on the test set
    y_pred = grid.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    print_evaluation('KNN 70:30', y_test, y_pred)
    # return the best result, transformed scaler, accuracy and precision
    return grid.best_estimator_, scaler, accuracy, precision


# The function is used for training the Support Vector Machine for 80 train and 20 test set
def train_svm_80(X, y):
    print('Starting SVM with 80 train and 20 test...')
    # print the total count
    print('Total count: ' + str(len(X)))
    class_distribution = Counter(y)
    class_distribution_y = dict(class_distribution)
    # print the count in each of the class
    print('balanced y: ' + str(class_distribution_y))
    # X_train, X_test are representing the features train and features test while y_train and y_test are representing the target train and target test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    class_distribution = Counter(y_train)
    class_distribution_ytrain = dict(class_distribution)
    # print the count in the train set to see if the data is 80% from the overall data
    print('y-train: ' + str(class_distribution_ytrain))
    class_distribution = Counter(y_test)
    class_distribution_ytest = dict(class_distribution)
    # print the count in the train set to see if the data is 20% from the overall data
    print('y-test: ' + str(class_distribution_ytest))
    # using standard scalar to fit the feature train and doing transformation using feature test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print('Training set: ' + str(len(X_train)))  # Number of training samples
    print('Testing set: ' + str(len(X_test)))  # Number of testing samples

    # param_grid contains all the possible attribute search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    # Initialize the GridSearchCV object
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)

    # Fit the model
    grid.fit(X_train_scaled, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters of SVM 80:20 : ", grid.best_params_)

    # Make predictions on the test set
    y_pred = grid.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    print_evaluation('SVM 80:20', y_test, y_pred)
    # return the best result, transformed scaler, accuracy and precision
    return grid.best_estimator_, scaler, accuracy, precision


# The function is used for training the Support Vector Machine for 70 train and 30 test set
def train_svm_70(X, y):
    print('Starting SVM with 70 train and 30 test...')
    # print the total count
    print('Total count: ' + str(len(X)))
    class_distribution = Counter(y)
    class_distribution_y = dict(class_distribution)
    # print the count in each of the class
    print('balanced y: ' + str(class_distribution_y))
    # X_train, X_test are representing the features train and features test while y_train and y_test are representing the target train and target test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    class_distribution = Counter(y_train)
    class_distribution_ytrain = dict(class_distribution)
    # print the count in the train set to see if the data is 80% from the overall data
    print('y-train: ' + str(class_distribution_ytrain))
    class_distribution = Counter(y_test)
    class_distribution_ytest = dict(class_distribution)
    # print the count in the train set to see if the data is 20% from the overall data
    print('y-test: ' + str(class_distribution_ytest))
    # using standard scalar to fit the feature train and doing transformation using feature test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print('Training set: ' + str(len(X_train)))  # Number of training samples
    print('Testing set: ' + str(len(X_test)))  # Number of testing samples

    # param_grid contains all the possible attribute search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    # Initialize the GridSearchCV object
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)

    # Fit the model
    grid.fit(X_train_scaled, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters of SVM 70:30 : ", grid.best_params_)

    # Make predictions on the test set
    y_pred = grid.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    print_evaluation('SVM 70:30', y_test, y_pred)
    # return the best result, transformed scaler, accuracy and precision
    return grid.best_estimator_, scaler, accuracy, precision


# The function determines whether user should visit the place in next week
def should_visit(forecast_conditions):
    # use list to store the possible bad weather keywords
    bad_weather_keywords = ['cloudy', 'rain', 'rainy', 'thunderstorm', 'snow', 'sleet']
    # use for loop to increase the sum of bad weather count
    bad_weather_count = sum(
        1 for weather in forecast_conditions if any(keyword in weather.lower() for keyword in bad_weather_keywords))
    if bad_weather_count >= 4:  # if bad weather is 4 days or more then
        return 'It is not recommended to visit this place due to bad weather conditions on multiple days.'
    else:
        return 'The weather looks good! It is recommended to visit this place.'


# The function is used to fetch flight details using API service
def get_flight_offers(origin, destination, departure_date, access_token):
    url = 'https://test.api.amadeus.com/v2/shopping/flight-offers'
    params = {
        'originLocationCode': origin,
        'destinationLocationCode': destination,
        'departureDate': departure_date,
        'adults': 1,  # Number of adult passengers
        'currencyCode': 'MYR'  # Currency for prices
    }
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    return data


# use dictionaries to map the string into the picture
image_mapping = {
    'sunny': '/Users/mchalxz/Downloads/sun.png',
    'rainy': '/Users/mchalxz/Downloads/rainy-day.png',
    'partly cloudy': '/Users/mchalxz/Downloads/cloud.png',
    'cloudy': '/Users/mchalxz/Downloads/cloudy.png',
    'heavy rain': '/Users/mchalxz/Downloads/heavy-rain-2.png',
    'patchy rain': '/Users/mchalxz/Downloads/nature.png',
    'rain': '/Users/mchalxz/Downloads/heavy-rain.png',
    'wind': '/Users/mchalxz/Downloads/wind.png',
    'overcast': '/Users/mchalxz/Downloads/overcast.png',
    'light drizzle': '/Users/mchalxz/Downloads/drizzle.png',
    'fog': '/Users/mchalxz/Downloads/fog.png',
    'thunder': '/Users/mchalxz/Downloads/lightning-bolt-.png',
    'snow': '/Users/mchalxz/Downloads/snow.png',
    'showers': '/Users/mchalxz/Downloads/shower.png'
}


# The function is used to get the image path through the image mapping
def get_image_path(condition):
    # make the string into lower case to match with the image mapping
    condition_lower = condition.lower()
    # using for loop to match the keyword with image mapping
    for keyword, image_path in image_mapping.items():
        if keyword in condition_lower:
            return image_path
    return '/Users/mchalxz/Downloads/images-3.png'


# The function is used for combining the apikey and apisecret of flight API
def get_access_token(api_key, api_secret):
    url = 'https://test.api.amadeus.com/v1/security/oauth2/token'
    data = {
        'grant_type': 'client_credentials',
        'client_id': api_key,
        'client_secret': api_secret
    }
    response = requests.post(url, data=data)
    access_token = response.json().get('access_token')
    return access_token


# basic markdown of design
st.markdown(
    '''
    <div style='display: flex; align-items: center; justify-content: center;'>
        <span style='font-size: 2em; color: green; margin-right: 10px;'>🌤️</span>
        <h1 style='color: green; margin: 0 10px; text-align: center;'>
            Vacation Decision Maker with <br> Weather Classification
        </h1>
        <span style='font-size: 2em; color: green; margin-left: 10px;'>🌧️</span>
    </div>
    ''',
    unsafe_allow_html=True
)

# If the page is 0/initial page
if st.session_state.page == 0:
    st.session_state.navigation = True
    # Retrieve the location using the API key
    rows = connect_to_cassandra().execute('SELECT location FROM iata')
    # store the resultset into the list
    destinations = [(row.location) for row in rows]

    # Creating a list of destinations for the combobox
    destination_list = [f'{city}' for city in destinations]
    # initializing the default value
    destination_list.insert(0, 'None')
    # select box is displayed for user to choose destination
    selected_destination = st.selectbox('Select a destination', destination_list)
    # if the selected destination is the default value then will stop the program for a while until user choose destination
    if selected_destination == 'None':
        st.error('Please select a destination.')
        st.stop()
    # when selected destination has value only will proceed
    if selected_destination != 'None':
        # get the latitude and longitude based on the destination
        lat, lon = get_lat_long(selected_destination, get_api_key('GEOCODE API'))
        # using map function to display the chosen destination
        my_map = folium.Map(location=[lat, lon], zoom_start=12)

        # Add a marker at the location
        folium.Marker([lat, lon], popup='Selected Location').add_to(my_map)

        # Display the map in Streamlit
        st_folium(my_map, width=700, height=500)
    # store the value into the global variable
    st.session_state.place = selected_destination
    # create a 'Continue' button and assign it to a variable
    process = st.button('Continue')
    # if button is clicked
    if process:
        # process started by displaying 'processing'
        with st.spinner('Processing...'):
            # get the latitude and longitude based on the destination
            lat, long = get_lat_long(selected_destination, get_api_key('GEOCODE API'))
            # check the existence of latitude and longitude
            if lat is None or long is None:
                st.error(f'Failed to get latitude and longitude for {selected_destination}.')
            else:  # if latitude and longitude is existed
                # retrieving data of last date is set as today
                end_date_obj = datetime.date.today()
                # the start date is determined by deducting 365 days from today
                start_date_obj = end_date_obj - datetime.timedelta(days=365)
                # start the cassandra connection
                session = connect_to_cassandra()
                # check validity of the session
                if session is None:
                    # if session not found, prompt error message
                    st.error('Failed to connect to Cassandra. Exiting.')
                else:
                    # use list to store the date
                    dates = [(start_date_obj + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(365)]
                    # initialize a empty list for storing result in the future
                    results = []
                    # using thread for faster processing
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        # submitting fetch and store weather data task to thread
                        future_to_date = {executor.submit(fetch_and_store_weather_data, session, lat, long,
                                                          get_api_key('WEATHER API'), date, selected_destination): date
                                          for date in
                                          dates}
                        # future is initialized and for loop to iterate the result
                        for future in as_completed(future_to_date):
                            data = future.result()
                            if data:
                                results.append(data)

                    # Prepare statement and batch insert into Cassandra
                    insert_query = 'INSERT INTO WEATHER (place, date, humidity, temperature, weather_condition, windspeed) VALUES (?, ?, ?, ?, ?, ?)'
                    prepared_stmt = session.prepare(insert_query)
                    batch = BatchStatement()
                    for result in results:
                        # check if all the required data are obtained only will proceed for inserting query into cassandra
                        if isinstance(result, tuple) and len(result) == 6:
                            batch.add(prepared_stmt, result)
                        else:
                            # if not then will print the invalid result format
                            print(f'Invalid result format: {result}')
                    try:
                        # execute the batch insert
                        session.execute(batch)
                        # sending the message to alert users
                        st.success('Process completed! Click Next to proceed!')
                    except Exception as e:
                        st.error(f'Error executing batch insert: {e}')

                    # Save the place to session state
                    st.session_state['place'] = selected_destination
                    st.session_state['data_fetched'] = True
                    st.session_state.validity = True


elif st.session_state.page == 1:
    # set navigation as false to reset the usage
    st.session_state.navigation = False

    # Default styles
    st.markdown('''
    <style>
    .main {
            background-color: #000000;
    }
    .sidebar .sidebar-content {
          background-color: #f0f0f5;
     }
    </style>
    ''', unsafe_allow_html=True)
    # store in the place
    place = st.session_state.place
    # if place is not being picked
    if not place:
        st.warning('Please go to the Home page and fetch the weather data first.')
    else:
        # initialize session
        session = connect_to_cassandra()
        # check if session is valid
        if session is None:
            st.error('Failed to connect to Cassandra. Exiting.')
        else:
            # initialize both end date and start date
            end_date = datetime.datetime.today().date()
            start_date = end_date - datetime.timedelta(days=365)
            # the historical result is stored into weather_data dataframe
            weather_data = fetch_weather_data(session, place, start_date, end_date)
            # print the data frame before handling missing value
            print('Before handling missing value:')
            print(weather_data)

            # Handle missing values: fill empty humidity and wind speed with mean values
            weather_data['humidity'] = weather_data['humidity'].fillna(weather_data['humidity'].mean())
            weather_data['windspeed'] = weather_data['windspeed'].fillna(weather_data['windspeed'].mean())
            weather_data['temperature'] = weather_data['temperature'].fillna(weather_data['temperature'].mean())
            # print the data frame after handling missing value
            print('After handling missing value:')
            print(weather_data)
            # check if data is empty
            if weather_data.empty:
                st.error('No data found for the specified place and date range.')
            else:
                # if data is not empty then start preprocess data
                X_resampled, y_resampled, label_encoder = preprocess_data(weather_data)

                # Train model using six different combination
                model_decisiontree_70, scaler_decisiontree_70, accuracy_decisiontree_70, precision_decisiontree_70 = train_decisiontree_70(
                    X_resampled, y_resampled)
                model_decisiontree_80, scaler_decisiontree_80, accuracy_decisiontree_80, precision_decisiontree_80 = train_decisiontree_80(
                    X_resampled, y_resampled)
                model_svm_70, scaler_svm_70, accuracy_svm_70, precision_svm_70 = train_svm_70(X_resampled, y_resampled)
                model_svm_80, scaler_svm_80, accuracy_svm_80, precision_svm_80 = train_svm_80(X_resampled, y_resampled)
                model_knn_70, scaler_knn_70, accuracy_knn_70, precision_knn_70 = train_knn_70(X_resampled, y_resampled)
                model_knn_80, scaler_knn_80, accuracy_knn_80, precision_knn_80 = train_knn_80(X_resampled, y_resampled)

                # Make forecasts of that six combination to get the accuracy
                forecast_dates_decisiontree_70, forecast_conditions_decisiontree_70, temperature_data_decisiontree_70, humidity_data_decisiontree_70, windspeed_data_decisiontree_70 = make_forecasts(
                    model_decisiontree_70, scaler_decisiontree_70,
                    place, get_api_key('WEATHER API'))

                forecast_dates_decisiontree_80, forecast_conditions_decisiontree_80, temperature_data_decisiontree_80, humidity_data_decisiontree_80, windspeed_data_decisiontree_80 = make_forecasts(
                    model_decisiontree_80, scaler_decisiontree_80,
                    place, get_api_key('WEATHER API'))

                forecast_dates_svm_70, forecast_conditions_svm_70, temperature_data_svm_70, humidity_data_svm_70, windspeed_data_svm_70 = make_forecasts(
                    model_svm_70, scaler_svm_70,
                    place, get_api_key('WEATHER API'))

                forecast_dates_svm_80, forecast_conditions_svm_80, temperature_data_svm_80, humidity_data_svm_80, windspeed_data_svm_80 = make_forecasts(
                    model_svm_80, scaler_svm_80,
                    place, get_api_key('WEATHER API'))

                forecast_dates_knn_70, forecast_conditions_knn_70, temperature_data_knn_70, humidity_data_knn_70, windspeed_data_knn_70 = make_forecasts(
                    model_knn_70, scaler_knn_70,
                    place, get_api_key('WEATHER API'))

                forecast_dates_knn_80, forecast_conditions_knn_80, temperature_data_knn_80, humidity_data_knn_80, windspeed_data_knn_80 = make_forecasts(
                    model_knn_80, scaler_knn_80,
                    place, get_api_key('WEATHER API')
                )
                # print each of the accuracy
                print('Accuracy of Decision Tree (70:30): ' + str(accuracy_decisiontree_70))
                print('Accuracy of Decision Tree (80:20): ' + str(accuracy_decisiontree_80))
                print('Accuracy of SVM (70:30): ' + str(accuracy_svm_70))
                print('Accuracy of SVM (80:20): ' + str(accuracy_svm_80))
                print('Accuracy of KNN (70:30): ' + str(accuracy_knn_70))
                print('Accuracy of KNN (80:20): ' + str(accuracy_knn_80))
                # store all of the accuracy into list
                accuracy_list = [accuracy_decisiontree_70, accuracy_decisiontree_80, accuracy_svm_70, accuracy_svm_80,
                                 accuracy_knn_70, accuracy_knn_80]
                # get the highest accuracy through the list
                highest_accuracy = max(accuracy_list)
                # get the index of that
                max_index = accuracy_list.index(highest_accuracy)

                # Update the dictionary to match all indices
                accuracy_dict = {
                    0: 'Decision tree (70:30)',
                    1: 'Decision tree (80:20)',
                    2: 'SVM (70:30)',
                    3: 'SVM (80:20)',
                    4: 'KNN (70:30)',
                    5: 'KNN (80:20)'
                }

                print(f'The best technique is {accuracy_dict[max_index]} with an accuracy of {highest_accuracy}')

                # check max_index and assign the value according to accuracy_dict
                if max_index == 0:  # decision tree
                    forecast_dates = forecast_dates_decisiontree_70
                    forecast_conditions = forecast_conditions_decisiontree_70
                    temperature_data = temperature_data_decisiontree_70
                    humidity_data = humidity_data_decisiontree_70
                    windspeed_data = windspeed_data_decisiontree_70
                elif max_index == 1:  # decision tree
                    forecast_dates = forecast_dates_decisiontree_80
                    forecast_conditions = forecast_conditions_decisiontree_80
                    temperature_data = temperature_data_decisiontree_80
                    humidity_data = humidity_data_decisiontree_80
                    windspeed_data = windspeed_data_decisiontree_80
                elif max_index == 2:  # SVM
                    forecast_dates = forecast_dates_svm_70
                    forecast_conditions = forecast_conditions_svm_70
                    temperature_data = temperature_data_svm_70
                    humidity_data = humidity_data_svm_70
                    windspeed_data = windspeed_data_svm_70
                elif max_index == 3:  # SVM
                    forecast_dates = forecast_dates_svm_80
                    forecast_conditions = forecast_conditions_svm_80
                    temperature_data = temperature_data_svm_80
                    humidity_data = humidity_data_svm_80
                    windspeed_data = windspeed_data_svm_80
                elif max_index == 4:  # KNN
                    forecast_dates = forecast_dates_knn_70
                    forecast_conditions = forecast_conditions_knn_70
                    temperature_data = temperature_data_knn_70
                    humidity_data = humidity_data_knn_70
                    windspeed_data = windspeed_data_knn_70
                else:  # KNN
                    forecast_dates = forecast_dates_knn_80
                    forecast_conditions = forecast_conditions_knn_80
                    temperature_data = temperature_data_knn_80
                    humidity_data = humidity_data_knn_80
                    windspeed_data = windspeed_data_knn_80
                # Validate lengths
                if len(forecast_dates) == len(forecast_conditions) == len(temperature_data) == len(
                        humidity_data) == len(windspeed_data):
                    # Display forecasts

                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Weather Condition': forecast_conditions,
                        'Temperature': temperature_data,
                        'Humidity': humidity_data,
                        'Windspeed': windspeed_data
                    })
                    # get image by mapping the string value
                    forecast_df['Image'] = forecast_df['Weather Condition'].apply(get_image_path)

                    # Display the images in a grid
                    st.markdown('### Weather condition prediction for ' + place + ' (One Week)')
                    cols = st.columns(len(forecast_df))
                    # using for loop to iterate the image path
                    for idx, col in enumerate(cols):
                        image_path = forecast_df.iloc[idx]['Image']
                        # Ensure the image path is valid before displaying
                        if isinstance(image_path, str):
                            col.write(forecast_df['Date'][idx])
                            col.image(image_path, width=100)
                            col.write(forecast_df['Weather Condition'][idx])
                        else:
                            col.write('No image available')

                    # Calculate averages
                    avg_temp = sum(temperature_data) / len(temperature_data)
                    avg_humidity = sum(humidity_data) / len(humidity_data)
                    avg_windspeed = sum(windspeed_data) / len(windspeed_data)
                    # Display all of it into streamlit
                    st.markdown('### Metrics📊')
                    col1, col2, col3 = st.columns(3)
                    col1.metric('Average Temperature🌡️', f'{avg_temp:.1f} °C')
                    col2.metric('Average Wind Speed💨', f'{avg_windspeed:.1f} mph')
                    col3.metric('Average Humidity💧', f'{avg_humidity:.1f}%')
                    # display line charts
                    st.markdown('### Temperature Line chart')
                    plost.line_chart(
                        data=forecast_df,
                        x='Date',
                        y='Temperature',
                        legend=None,
                        height=345,
                        use_container_width=True
                    )

                    c1, c2 = st.columns((3, 3))
                    # aligning the two charts in one row
                    with c1:
                        st.markdown('### Wind Speed Line Chart')
                        plost.line_chart(
                            data=forecast_df,
                            x='Date',
                            y='Windspeed',
                            legend=None,
                            height=345,
                            use_container_width=True
                        )

                    with c2:
                        st.markdown('### Humidity Line chart')
                        plost.line_chart(
                            data=forecast_df,
                            x='Date',
                            y='Humidity',
                            legend=None,
                            height=345,
                            use_container_width=True
                        )

                    # Decision on visiting
                    decision = should_visit(forecast_conditions)
                    st.subheader('Recommendation💡')
                    st.write(decision)

                    # Save recommendation to session state
                    st.session_state['recommendation'] = decision
                else:
                    st.error('Data mismatch: Length of temperature, humidity, or windspeed data does not match the '
                             'length of forecast dates.')
                # set validity true to have 'Next'
                st.session_state.validity = True
                # check to have 'Next' button or not to navigate to get flight details based on the decision
                if st.session_state[
                    'recommendation'] is 'The weather looks good! It is recommended to visit this place.':
                    st.session_state.navigation = True
# showing the flight details page
elif st.session_state.page == 2:
    st.session_state.validity = True
    place = st.session_state.get('place', None)
    recommendation = st.session_state.get('recommendation', None)
    # check if place is chosen or recommendation is provided
    if not place or not recommendation:
        st.warning('Please complete the steps in the previous pages first.')
    else:
        # Establish connection
        session = connect_to_cassandra()
        # retrieve the required data from cassandra
        rows = session.execute('SELECT codes, location FROM iata')
        # use list to store the result
        destinations = [(row.codes, row.location) for row in rows]

        # Origin selection
        destination_list_origin = [f'{city}' for _, city in destinations]
        # initialize the default value
        destination_list_origin.insert(0, 'None')
        # put the columns to put content
        col1, col2, col3, col4 = st.columns([19, 1, 6, 19])
        with col1:
            origin = st.selectbox('Origin🛫', destination_list_origin)
        with col3:
            st.image('/Users/mchalxz/Downloads/arrow-3.png')
        with col4:
            st.text_input('Destination🛬', place, disabled=True)
        if origin == 'None':
            st.warning('Please select an origin.')
            st.stop()
        col5, col6 = st.columns([1, 1])
        with col5:
            dates = st.date_input('Departure Date🗓️:')
        with col6:
            back_dates = st.date_input('Return Date🗓️:')
        if dates > back_dates:
            st.error('Return date cannot be earlier than departure date')
            st.stop()
        # provide filter options
        filter_list = ['None', 'Price Range', 'Only exact day', 'Cheapest']
        # use select box to get the list
        filters = st.selectbox('Filter🔍', filter_list)
        # differentiate each different filter option
        if filters == 'Price Range':
            col9, col10 = st.columns(2)
            with col9:
                try:
                    min_price = int(st.text_input('Minimum price:'))
                except ValueError:
                    st.stop()
            with col10:
                try:
                    max_price = int(st.text_input('Maximum price:'))
                except ValueError:
                    st.stop()

        # Find the IATA code for the selected origin
        origin_iata = None
        for code, location in destinations:
            if location == origin:
                origin_iata = code
                break

        if not origin_iata:
            st.error(f'Could not find IATA code for the selected origin: {origin}')
        else:
            # Find the IATA code for the selected destination
            selected_iata_code = None
            # get the iata code based on the location
            for code, location in destinations:
                if location == place:
                    selected_iata_code = code
                    break
            # if the selected iata code is not existed then prompt an error
            if not selected_iata_code:
                st.error(f'Could not find IATA code for {place}.')
            else:
                st.write("")
                st.session_state.temporigin = origin
                # get the access token
                access_token = get_access_token(get_api_key('FLIGHT API'), get_api_key('SECRET API'))
                # if access token is valid then get the flight offers details
                if access_token:
                    flight_offers = get_flight_offers(origin_iata, selected_iata_code, dates,
                                                      access_token)
                    flight_offers_limited = flight_offers.get('data', [])[:6]
                    # if flight offers obtained from origin to destination
                    if flight_offers_limited:
                        st.subheader(f'Flight Offers from {origin} to {place} on ' + str(dates))
                        # initialize a variable for getting the exact min price in future
                        minprice = 10000000
                        # for loop to iterate and display
                        for i in range(0, len(flight_offers_limited), 3):
                            cols = st.columns(3)
                            for idx, flight in enumerate(flight_offers_limited[i:i + 3]):
                                with cols[idx]:
                                    price = flight['price']['total']
                                    priceint = float(price.replace('RM', ''))
                                    if minprice > priceint:
                                        minprice = priceint
                                    segments = flight['itineraries'][0]['segments']
                                    plane_number = segments[0]['carrierCode'] + segments[0]['number']
                                    airline_company = segments[0]['carrierCode']
                                    origin = segments[0]['departure']['iataCode']
                                    destination = segments[-1]['arrival']['iataCode']
                                    departure_time = segments[0]['departure']['at']

                                    # Handle both formats for departure time
                                    try:
                                        # Attempt to parse with 'Z' at the end
                                        departure_time_utc = datetime.datetime.strptime(departure_time,
                                                                                        '%Y-%m-%dT%H:%M:%SZ')
                                    except ValueError:
                                        # Parse without 'Z'
                                        departure_time_utc = datetime.datetime.strptime(departure_time,
                                                                                        '%Y-%m-%dT%H:%M:%S')

                                    # Convert departure time to Malaysia time
                                    departure_time_malaysia = departure_time_utc.replace(
                                        tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Kuala_Lumpur'))
                                    departure_time_malaysia_str = departure_time_malaysia.strftime(
                                        '%Y-%m-%d %H:%M:%S')

                                    date = departure_time_malaysia_str.split(' ')[0]
                                    time = departure_time_malaysia_str.split(' ')[1]
                                    # other filter
                                    if filters == 'Price Range':  # result with green highlighted
                                        if min_price <= priceint <= max_price:
                                            background_color = 'background-color: rgb(34, 139, 34)};'
                                        else:
                                            background_color = 'background-color: ;'
                                    # if not applying filter makes no effects
                                    elif filters == 'None':
                                        background_color = 'background-color: ;'
                                    elif filters == 'Cheapest':  # result with green highlighted
                                        if priceint == minprice:
                                            background_color = 'background-color: rgb(34, 139, 34)};'
                                        else:
                                            background_color = 'background-color: ;'
                                    else:
                                        if date == str(dates):  # result with green highlighted
                                            background_color = 'background-color: rgb(34, 139, 34)};'
                                        else:
                                            background_color = 'background-color: ;'
                                    # build frames to frame all the flight details
                                    st.markdown(
                                        f'''
                                                                                            <div style='border: 1px solid #e1e1e1; border-radius: 10px; padding: 10px; margin-bottom: 10px; {background_color}'>
                                                                                                <h4><u>Flight {i + idx + 1}✈️:</u></h4>
                                                                                                <p><strong>Price:</strong> RM{price}</p>
                                                                                                <p><strong>Airline Company:</strong> {airline_company}</p>
                                                                                                <p><strong>Plane Number:</strong> {plane_number}</p>
                                                                                                <p><strong>Origin:</strong> {origin}</p>
                                                                                                <p><strong>Destination:</strong> {destination}</p>
                                                                                                <p><strong>Date (M'sia):</strong> {date}</p>
                                                                                                <p><strong>Time (M'sia):</strong> {time}</p>
                                                                                            </div>
                                                                                            ''', unsafe_allow_html=True
                                    )

                    else:  # if flight details is not available
                        st.error('No flight offers available.')
                    st.write('')

                    if access_token:  # check if access token available
                        # get the flight offers using API service
                        flight_offers = get_flight_offers(selected_iata_code, origin_iata, back_dates,
                                                          access_token)
                        # get for 6 flight details
                        flight_offers_limited = flight_offers.get('data', [])[:6]
                        # flight details from place to origin
                        if flight_offers_limited:
                            st.subheader(
                                f'Flight Offers from {place} to {st.session_state.temporigin} on ' + str(back_dates))
                            # initialize higher value to a variable to get real minimum number in future
                            minprice = 10000000
                            for i in range(0, len(flight_offers_limited), 3):
                                cols = st.columns(3)
                                for idx, flight in enumerate(flight_offers_limited[i:i + 3]):
                                    with cols[idx]:
                                        price = flight['price']['total']
                                        priceint = float(price.replace('RM', ''))
                                        if minprice > priceint:
                                            minprice = priceint
                                        segments = flight['itineraries'][0]['segments']
                                        plane_number = segments[0]['carrierCode'] + segments[0]['number']
                                        airline_company = segments[0]['carrierCode']
                                        origin = segments[0]['departure']['iataCode']
                                        destination = segments[-1]['arrival']['iataCode']
                                        departure_time = segments[0]['departure']['at']

                                        # Handle both formats for departure time
                                        try:
                                            # Attempt to parse with 'Z' at the end
                                            departure_time_utc = datetime.datetime.strptime(departure_time,
                                                                                            '%Y-%m-%dT%H:%M:%SZ')
                                        except ValueError:
                                            # Parse without 'Z'
                                            departure_time_utc = datetime.datetime.strptime(departure_time,
                                                                                            '%Y-%m-%dT%H:%M:%S')

                                        # Convert departure time to Malaysia time
                                        departure_time_malaysia = departure_time_utc.replace(
                                            tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Kuala_Lumpur'))
                                        departure_time_malaysia_str = departure_time_malaysia.strftime(
                                            '%Y-%m-%d %H:%M:%S')

                                        date = departure_time_malaysia_str.split(' ')[0]
                                        time = departure_time_malaysia_str.split(' ')[1]
                                        # apply filter
                                        if filters == 'Price Range':  # result with green highlighted
                                            if min_price <= priceint <= max_price:
                                                background_color = 'background-color: rgb(34, 139, 34)};'
                                            else:
                                                background_color = 'background-color: ;'
                                        elif filters == 'None':  # make no effects if choosing None
                                            background_color = 'background-color: ;'
                                        elif filters == 'Cheapest':  # result with green highlighted
                                            if priceint == minprice:
                                                background_color = 'background-color: rgb(34, 139, 34)};'
                                            else:
                                                background_color = 'background-color: ;'
                                        else:
                                            if date == str(back_dates):  # result with green highlighted
                                                background_color = 'background-color: rgb(34, 139, 34)};'
                                            else:
                                                background_color = 'background-color: ;'
                                        # create frames to frame all the flight details
                                        st.markdown(
                                            f'''
                                                                                                                <div style='border: 1px solid #e1e1e1; border-radius: 10px; padding: 10px; margin-bottom: 10px; {background_color}'>
                                                                                                                    <h4><u>Flight {i + idx + 1}✈️:</u></h4>
                                                                                                                    <p><strong>Price:</strong> RM{price}</p>
                                                                                                                    <p><strong>Airline Company:</strong> {airline_company}</p>
                                                                                                                    <p><strong>Plane Number:</strong> {plane_number}</p>
                                                                                                                    <p><strong>Origin:</strong> {origin}</p>
                                                                                                                    <p><strong>Destination:</strong> {destination}</p>
                                                                                                                    <p><strong>Date (M'sia):</strong> {date}</p>
                                                                                                                    <p><strong>Time (M'sia):</strong> {time}</p>
                                                                                                                </div>
                                                                                                                ''',
                                            unsafe_allow_html=True
                                        )




                else:
                    st.error('Failed to obtain access token.')

# make next page on right side, previous page on left side
col1, col2, col3 = st.columns([4, 5, 1])

# left bottom 'Previous'
with col1:
    if st.session_state.page > 0:
        st.button('Previous', on_click=prev_page)

# right bottom 'Next'
with col3:
    if st.session_state.page < len(pages) - 1 and st.session_state.validity and st.session_state.navigation:
        st.button('Next', on_click=next_page)
