# Hotel Reservation Demand

## Motivation

Getting more reservations on one hand and preventing cancelation are two main factors determining a hotel’s revenue. Therefore, My main goal in this project is to get more insights about these two metrics. To this end, I used hotel demand data from Kaggel, to take a closer look at the following questions:

* What is the cancelation difference between hotels and resorts?
* Where do guests who canceled come from?
* Which month does have the highest net reservations?

The dataset has 119,390 records (each of them is one reservation) and 32 features. I focus only on the hotel types, guest countries as well as arrival month in this study to investigate the demand for hotels as well as cancelation rate.

# Python Packeges used
This project is executed in a Jupyter Notebook with the help of below packages

- Numpy
- Pandas
- Matplotlib
- Seaborn

## Files

- **photel_booking_demand.ipynb** It is a Jupyter notebook that contains the analysis for EDA 

## Summary of the results

In this simple data exploration, I looked into three factors that can impact reservations as well as cancelation. Although city hotels receive more demands, they have higher cancelation rate in compare with resort hotels. Portugal with 62% rate of cancelation has the highest rate, come after it Britain and Spain with 5.5% and 4.9% cancelation rate, respectively. Both city and resort hotels have the lowest demand in winter, however, March is the pick season for resort hotels and city hotels receive the highest reservation in May.

## Licensing, Authors, Acknowledgements

The appreciation goes for Kaggel for providing the dataset. One can find the Licensing for the data as well as the description of variables in [this](https://www.kaggle.com/jessemostipak/hotel-booking-demand) kaggel link.

[Here](https://medium.com/@maryammoradbeygi/hotel-reservation-demand-dea2754f9e98) is the blog for this project. 
