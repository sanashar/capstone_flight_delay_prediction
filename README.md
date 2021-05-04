# Flight Delay Prediction - Using Machine Learning Algorithms 
![Delays](Pictures/delayed.jpeg)

# Overview 
Over the last two decades, the popularity of air travel has increased significantly among travelers, mostly because of its speed in comparison to other modes of transportation. This has led to increase in traffic in the air and on the ground, which further has resulted in massive levels of aircraft delays [1]. Flight delays cost billions of dollars and have a huge impact on the US economy, causing a stain on the air travel system, passengers and society [2]. 

In this Capstone project my aim is to apply machine learning algorithms like decision tree, random forest and logistic regression to predict flight delays. I will train the models on one dataset and test on another, to check the accuracy of the models in predicting the flight delays. 

# Repository Navigation 

Table of Contents -

Code Notebooks               : [Link to GitHub Folder](Notebook/eda_phase1.ipynb)

Report       : [Link](https://sites.google.com/umbc.edu/data606/spring-2021-section-1/sana-sharma?authuser=0)

Tableau       : [Link](https://public.tableau.com/profile/sana.sharma#!/vizhome/FlightDelay2015-2018/FlightDelay2015-2018)

Presentation Phase 1      : [Link](https://drive.google.com/file/d/1zIyrIUfUie7ceeJ0vsogHz07Q7NRyv5z/view?usp=sharing)

Presentation Phase 2      : [Link](https://drive.google.com/file/d/12y9ae-DkaJI4pXcGxMU3jtgcqvAnvbuP/view?usp=sharing)

Presentation Phase 3      : [Link](https://drive.google.com/file/d/1jdAvQWvtvjlNeS35OGU8_ihLal8fy5eI/view?usp=sharing)

<p align="center">
 Phase 1 -
</p> 


# Data Description

The datasets I am using were acquired from United States Department of Transportation’s (DOT) Bureau of Transportation Statistics (BTS) website. BTS tracks the on-time performance of domestic flights operated by large air carriers. It provides datasets that are focused on the number of on-time, delayed, canceled and diverted flights that appear in DOT's monthly Air Travel Consumer Report, published about 30 days after the month's end. The website contains data starting from June 2003 till November 2020 [3]. 

For my analysis, I will use 5 datasets. All 5 datasets contain the same fields but represent different time slices. I will use 4 datasets for Phase 1 & 2 for Exploratory Data Analysis and Training my models. For Phase 3, I will use the 5th dataset to test my models on and check their accuracies. These datasets contain flight delay statistics ranging from January 2015 – January 2019. After concatenating January 2015 – January 2018 datasets into 1 dataframe, the dataset includes 1,935,930 rows and 51 columns for my Phase 1 & 2 analysis. After further exploration of the data on the BTS website, I have decided to use 29 out of the 51 columns for my analysis [4] –

1.	YEAR – Year                 
2.	DAY_OF_MONTH – Day of the Month
3.	DAY_OF_WEEK – Day of the Week         
4.	FL_DATE – Flight Date (yyyymmdd)            
5.	OP_UNIQUE_CARRIER – Reporting Airline
6.	ORIGIN – Origin Airport             
7.	ORIGIN_WAC – Origin Airport, World Area Code                   
8.	DEST – Destination Airport                        
9.	DEST_WAC – Destination Airport, World Area Code                   
10.	DEP_TIME – Actual Departure Time (local time: hhmm)                              
11.	DEP_DELAY – Difference in minutes between scheduled and actual departure time. Early departures show negative numbers 
12.	DEP_DEL15 – Departure Delay Indicator, 15 Minutes or More (1=Yes)
13.	TAXI_OUT – Taxi out Time, in Minutes           
14.	TAXI_IN – Taxi in Time, in Minutes         
15.	WHEELS_OFF – Wheels Off Time (local time: hhmm)
16.	WHEELS_ON – Wheels on Time (local time: hhmm)       
17.	ARR_TIME – Actual Arrival Time (local time: hhmm)                  
18.	ARR_DELAY – Difference in minutes between scheduled and actual arrival time. Early arrivals show negative numbers
19.	ARR_DEL15 – Arrival Delay Indicator, 15 Minutes or More (1=Yes)
20.	CANCELLED – Cancelled Flight Indicator (1=Yes)    
21.	CANCELLATION_CODE – Specifies the Reason for Cancellation
22.	DIVERTED – Diverted Flight Indicator (1=Yes)     
23.	AIR_TIME – Flight Time, in Minutes       
24.	DISTANCE – Distance between airports (miles)           
25.	CARRIER_DELAY – Carrier Delay, in Minutes  
26.	WEATHER_DELAY – Weather Delay, in Minutes
27.	NAS_DELAY – National Air System Delay, in Minutes         
28.	SECURITY_DELAY – Security Delay, in Minutes   
29.	LATE_AIRCRAFT_DELAY – Late Aircraft Delay, in Minutes

# Implementation Details
The aim of this Capstone project is to predict flight delays using machine learning algorithms like decision tree, random forest and logistic regression. Thus, I chose 29 out of 51 features out of which, some are usually known in advance like – Day, Day of the week, Carrier, Origin airport, Destination Airport, Scheduled departure, Departure delay, taxi-out/in, Distance, Scheduled arrival etc.
 
My plan is to train the models on one dataset and test on another, to check the accuracy of the models in predicting flight delays. 

1. Data Cleaning and Exploratory Data Analysis (EDA) –
I will use Pandas, Matplotlib, NumPy and Seaborn libraries, to name a few for my initial EDA. I will also use Tableau to create a data story with interactive visuals that provide useful insights to anyone viewing them.

2. Machine Learning – 
I will be using decision tree, random forest and logistic regression to predict flight delays. To apply the algorithms, I will use SciKit-Learn library in Python for testing and training and SK-Learn library to import all the methods of classification algorithms. After training my models on January 2015 – January 2018 dataset, I will test the models on the January 2019 dataset. And finally, can also use Confusion Matrix to check the accuracy as confusion matrix is a way of tabulating the number of misclassifications.

<p align="center">
 Phase 2 -
</p> 

During the second phase of the project, I have carried out –

1. Exploratory Data Analysis (EDA)

- Data Loading  
- Data Cleaning  
- Data Visualizations

2. Initial Machine Learning Analysis

<p align="center">
 Phase 3 -
</p> 

By using Machine Learning (ML) Algorithms we can try to predict if the flight will be delayed. While using different algorithms, I did face undeniable challenges and a certain degree of accuracy, which is associated to the data that they are fed. In this phase, I looked at different ML techniques/algorithms to try to predict if a flight will be delayed. Along with getting the highest accuracy, my results are focused on the top 3 airports and top 3 airlines during January 2015 – 2019.

# Machine Learning Methodology
## Approach 1 –
My first approach was to test and train on the same dataset i.e., January 2015 – 2018 concatenated data. After preprocessing the data, I used the test train split method and applied 3 machine learning algorithms - Logistic Regression, GaussianNB and Random Forest. Along with their accuracies, I also plotted their precision, recall and f1 score. 

## Approach 2 –
In my second approach, I decided to take the training and testing data separately, i.e., training on January 2015 - 2018 data and testing on January 2019 data and applied 4 machine learning algorithms - Logistic Regression, GaussianNB, Random Forest and Decision Tree.
 
## Approach 3 –
Now that I know which model is best trained on my dataset, in my third approach, I focused the test data on top 3 airlines and applied the best model i.e. Logistic Regression to predict which airline according to the top 3 has the most delays.  
 
## Approach 4 –
In my fourth approach, I applied Logistic Regression for predicting the delay on one airline i.e. DL (Delta Airline), for a particular origin i.e. ATL (Atlanta International Airport) and a particular destination i.e. DFW (Dallas/Fort Worth International Airport).

# Results

## Exploratory Data Analysis Results (January 2015 – 2018)

- Maximum number of flights were in 2018
- WN, DL and OO, were the top 3 airlines with the highest flight count each year
- WN, DL and OO, had the maximum delay % as well
- There is a strong correlation between distance and airtime, as well as late aircraft delay and departure/arrival delay
- ATL, ORD and DFW, were the busiest airports in this dataset

## Approach 1 Result
My models are giving the following accuracy in predicting flight delays on the same testing and training data. We can see that the models are overfitting when testing and training on the same dataset.

### Logistic Regression - 99.007

### GaussianNB -
accuracy - 0.97

precision - 0.85

recall - 0.81

f1 - 0.83

### Random Forest -
accuracy - 1.0

precision - 0.85

recall - 0.81

f1 - 0.83

## Approach 2 Result
My models are giving the following accuracy in predicting flight delays on different testing and training data. 

### Logistic Regression - 0.83
### GaussianNB - 0.83
### Random Forest - 0.75
### Decision Tree - 0.78

## Approach 3 Result
Since we can see above that Logistic is trained well on my data, in approach 3 I used Logistic Regression for predicting top 3 airline delay accuracies on different testing and training data. 

### Southwest Airline - 80.80
### Delta Airline - 87.73
### SkyWest Airline - 72.20

# Neural Networks
I applied fully connected Neural Network using 3 dense and then 5 dense layers. The results of the Deep Neural Network has high accuracies in flight delay prediction. Thus, the model is overfitting. Furthermore, application of this method drops memory space and time during the training.

## Approach 4 Result
I tried Logistic Regression on one airline and a particular origin and particular destination –

### DL – ATL to DFW - 82.27

# Conclusion
Predicting flight delays was a challenging but interesting capstone research topic. My research focused on develop, grow and comparing the models in order to increase the precision and accuracy of predicting flight delays. Since the issue of flights being on-time is very important, flight delay prediction models must have high precision and accuracy.

This project is done in three parts – project information, exploratory data analysis and machine learning. In phase 3 i.e., machine learning – I experimented using 4 approaches by using machine learning algorithms like Logistic Regression, Decision Tree, Random Forest and GaussianNB.

Comparing the four models through various approaches, I conclude that Logistic Regression predicts the best accuracy for flights delays on my chosen dataset. I was best able to train logistic regression model to give 83% accuracy on flight delays. After further exploration and narrowing down the data, the logistic regression model predicts 87% delays in Delta Airlines. And lastly, the same model an 82.27% accuracy when focused on one airline and a particular origin and particular destination i.e., DL airline – from ATL to DFW.

# Project Info & Software Requirements
Capstone Project - Sana Sharma

Languages    : Python 2.7

Tools/IDE    : Google Collab

Libraries    : pandas, matplotlib, statsmodels, sklearn, seaborn
