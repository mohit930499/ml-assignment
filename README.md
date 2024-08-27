# ml-assignment

Iris dataset 



# Dataset Exploration: Iris Dataset

## Overview

This project demonstrates how to load, explore, and analyze the famous Iris dataset using Python's sklearn and pandas libraries. The Iris dataset is a popular dataset in machine learning and statistics, containing 150 samples of iris flowers from three different species (setosa, versicolor, virginica), with four features (sepal length, sepal width, petal length, petal width) that describe each flower. The goal of this project is to perform some basic data exploration techniques to understand the dataset.


pip install numpy pandas scikit-learn


## Steps

1. Load the Iris dataset: 
   - The Iris dataset is included in the sklearn.datasets module and can be easily loaded using the load_iris() function.
   
2. Display the first five rows of the dataset: 
   - After loading the data into a pandas DataFrame, the first five rows will be displayed using the head() method.

3. Display the dataset’s shape: 
   - The shape of the dataset (i.e., the number of rows and columns) will be printed using the shape attribute.

4. Summary statistics: 
   - Calculate and display summary statistics such as the mean, standard deviation, minimum, maximum, and other descriptive statistics for each feature in the dataset using the describe() method.

## Code Example

python

Import necessary libraries

import pandas as pd

import numpy as np

from sklearn.datasets import load_iris

Load the Iris dataset

iris = load_iris()

# Create a DataFrame from the dataset
df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])

# Display the first five rows of the dataset
print("First five rows of the Iris dataset:")

print(df.head())

# Display the shape of the dataset
print("\nShape of the dataset (rows, columns):")

print(df.shape)

# Display summary statistics for the dataset
print("\nSummary statistics for the Iris dataset:")

print(df.describe())


## Expected Output

1. First Five Rows:
   The first five rows of the dataset will display the features for each iris sample (sepal length, sepal width, petal length, petal width).

2. Dataset Shape:
   The output will show the shape of the dataset, which should be (150, 4) indicating 150 samples and 4 features.

3. Summary Statistics:
   The summary statistics will display the mean, standard deviation, minimum, maximum, and percentiles for each feature. For example:


salary 

# Salary Prediction Using Linear Regression

## Overview

This project aims to predict an individual's salary based on their years of experience using a simple Linear Regression model. By providing years of experience as input, the model outputs the predicted salary. Linear Regression is a statistical method that models the relationship between a dependent variable (Salary) and one or more independent variables (YearsExperience).


You can install the necessary packages using pip:

bash
pip install numpy pandas matplotlib scikit-learn


## Project Structure


salary_prediction/
│
├── data/
│   └── salary_data.csv         # The dataset with years of experience and corresponding salaries
│
├── src/
│   └── salary_prediction.py    # Script for loading the data, training, and predicting salary
│
├── README.md                   # Project description and instructions
└── requirements.txt            # List of required Python libraries


## Dataset

The dataset salary_data.csv contains two columns:
- YearsExperience: Number of years of professional experience
- Salary: Annual salary in dollars

The dataset is used to train and validate the model.

## Workflow

1. Data Loading: The dataset is loaded from a CSV file using Pandas.
2. Exploratory Data Analysis (EDA): The data is explored visually and statistically to understand trends and relationships. We plot Years of Experience vs. Salary using Matplotlib.
3. Model Training: A Linear Regression model is trained using the scikit-learn library.
4. Prediction: The trained model is used to predict salaries for new input values of years of experience.
5. Evaluation: The performance of the model is evaluated using metrics such as Mean Squared Error (MSE) or R-squared.

## How to Run the Project

### Step 1: Prepare the Data
Ensure you have the dataset (salary_data.csv) stored in the data/ directory.

### Step 2: Run the Prediction Script
Navigate to the src/ directory and execute the Python script:

bash
python salary_prediction.py


The script will train the Linear Regression model on the dataset and display the predicted salaries for new input values.

### Step 3: Visualize the Results
The script also provides a visual plot of the best fit line for the Linear Regression model over the dataset points.


## Model Performance

The performance of the Linear Regression model can be evaluated using:
- Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.
- R-squared Score: Indicates how well the independent variable explains the variability in the dependent variable.

These metrics will be printed when running the script.

## Conclusion

This project demonstrates how to build a simple machine learning model using Linear Regression to predict salaries based on years of experience. With just a few lines of code, you can train a predictive model and use it to make forecasts for new data.
