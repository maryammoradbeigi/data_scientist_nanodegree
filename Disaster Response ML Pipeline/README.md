# Disaster Response Pipeline Project

## Motivation

Project Overview In this course, you've learned and built on your data engineering skills to expand your opportunities and potential as a data scientist. In this project, you'll apply these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

Efficiency and fast decision making under natural distater situations can save many lives. One can take advantage of machine learning models to achive this objectives. This is the motivation fordoing this project.

In this project, there is a dataset containing real messages that people had sent during disasters . My main goal is to predict which aid a person needs based on the sent message. To this end, I assess and clean the data frist and then create a machine learning classifiers to group the tokenized messages

## Files

- **process_data.py** It is a Python script for ETL Pipeline. It gets the data from CSV files, clean it and save it in SQLite table. 
- **train_classifier.py** It is a Python script for ML Pipeline. It reads the data from SQLite database, split it into the train and test set and fit the random forest model to do the prediction. It also contains model performance metrics. 
- **Flask Web App** It is a web in which one can type the message as an input and the classification willWe will be shown.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

[Here](https://github.com/maryammoradbeigi/data_scientist_nanodegree/tree/master/Disaster%20Response%20ML%20Pipeline) is the github link to the source codes. 
