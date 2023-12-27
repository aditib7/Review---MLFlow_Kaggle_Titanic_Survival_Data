import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import plotnine
from plotnine import (
    ggplot,
    aes,
    geom_boxplot,
    geom_jitter,
    scale_x_discrete,
    coord_flip
)


# Algorithms
import scipy
from scipy import stats
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow import log_metric,log_param,log_artifacts


if __name__ == "__main__":
  np.random.seed(40)

  data_path = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Titanic+Data+Set (1).csv")
  data = pd.read_csv(data_path, index_col = 0)
  print('There are {0} rows and {1} columns in given dataset.'.format(data.shape[0], data.shape[1]))
  print('Checking the information about different columns in given dataset')
  print('\n')
  print(data.info())

  # removing duplicate rows, if any in given dataset
  data = data.drop_duplicates()

  # The columns, 'Survived' is a label and 'Pclass' is feature of categorical data type in given dataset. Thus, it can be converted 
  # into 'object' data type as numeric computations cannot be carried out on these columns
  data['Survived'] = data['Survived'].astype('object')
  data['Pclass'] = data['Pclass'].astype('object')

  # checking the statistical summary of columns with quantitative/non-categorical data - it is observed that in 
  # column, 'Age' median is close to the mean value and thus, the data is normally distributed. But, it is observed that 
  # there is significant difference between median and mean value in 'Fare' column. Thus, the data in 'Fare' column is 
  # skewed and does not follow normal distribution. 

  data[['Age', 'Fare']].describe()
  # it can be observed below in the distribution plot that the data in 'Age' column follows normal distribution
  sns.distplot(data['Age'])
  plt.show()
  print('\n')
  # it can be observed in this distribution plot that the distribution of data in 'Fare' column is right-skewed and thus, it does 
  # not follow normal distribution
  sns.distplot(data['Fare'])
  plt.show()
  print('\n')
  # Checking for Outliers in 'Fare' column:
  # Inferences: 
  # In Fare there are a lots of outliers but considering the dataset the first class tickets price is really high, 
  # so outliers can be visible as per statistics but domain knowledge they are acceptable.

  fig1, ax1 = plt.subplots()
  ax1.set_title('BoxPlot for Fare')
  ax1.boxplot(data['Fare'])
  plt.show()

  
  


  
