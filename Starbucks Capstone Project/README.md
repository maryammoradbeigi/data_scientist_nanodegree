# Starbucks Capstone Challenge


**Udacity Data Scientist Nanodegree**

---

## Python Packeges used

---

This project is executed in a Jupyter Notebook with the help of below packages

- Numpy
- Pandas
- Matplotlib
- Seaborn
- Sklearn
- json

## Project Motivation

---

This project is the capstone project for Udacity Data Scientist Nanodegree. Three datasources, i.e. Portfolio which contains features of . different offers sent by Starbucks, Profile which which is the datasource for Satrbucks’ customers and their soci-economic characteristics and transcript which is the log table for different offers’ status at any given time per customer and the transactions that they make.
These datasources provide great opportunity for variety of studies such as market segmentation, the amount a person might spend, forecasting the future customers, etc.. The main objective of the current study is to predict which offer will be completed after viewing by the customer.

Starbucks send out different offers randomly to customers. Some of these offers are only informational and some of them come with reward upon the completion. One of the marketing question that can help with optimizing the number of offers to send out is whether an offer will be completed based on offer’s features and the customer’s soci-economics characteristics. Therefore, this study aims to create a classifier using which we can predict whether an offer will be competed.

## Data Sources

---

The data is contained in three files:

- portfolio.json 
- profile.json
- transcript.json 

## ML models and Results

---

I use two classifiers, namely random forest and logistic regression, to make the prediction about whether an offer will be completed. Then, I will compare the result and pick the best model. 

the model’s performance metrics fir the randome forest classifier are:

- **Accuracy:** 0.75
- **F1:** 0.75

while these metrics for logistic regression are:

- **Accuracy:** 0.72
- **F1:** 0.72

Therefore, using either accuracy or F1 to choose the best performed model to predict whether an offer gets completed, random forest is a better predictor in classifying the data points.


Results A summary of the results of the analysis can be found in the Jupyter notebook or in this [blog](https://medium.com/@maryammoradbeygi/prediction-starbucks-offer-completion-with-machine-learning-model-ae53a19305e0) on medium .
