import os
import sys
import warnings
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
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn
from mlflow import log_metric,log_param,log_artifacts


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Titanic+Data+Set (1).csv")
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
   
    # It can be observed in this distribution plot that the distribution of data in 'Fare' column is right-skewed and thus, it does not follow normal distribution.
    print('\n')
    sns.distplot(data['Fare'])
    
    # Checking for Outliers in 'Fare' column:
    # Inferences: 
    # In Fare there are a lots of outliers but considering the dataset the first class tickets price is really high, 
    # so outliers can be visible as per statistics but domain knowledge they are acceptable.

    fig1, ax1 = plt.subplots()
    ax1.set_title('BoxPlot for Fare')
    ax1.boxplot(data['Fare'])
    plt.show()

    # Checking for Outliers in 'Age' column w.r.t Titanic Survivors:
    # Inferences: 
    # There are some outliers in Age w.r.t. Survivors and Non-survivors and it is common. These extreme data points cannot 
    # affect the predictions of dependent variable, 'Survived' so leaving them be.

    ggplot(data, aes(x='Survived', y = 'Age')) + \
    geom_boxplot()

    # checking the null values in data - it is found that columns, 'Age', 'Cabin' and 'Embarked' have null values

    data.isnull().sum().sort_values(ascending = False)

    # first plotting the null values in the Titanic dataset
    sns.heatmap(data.isnull(), cbar = False).set_title("Missing values heatmap")

    # It is observed above that column, Cabin, which is a categorical feature, has greater than 50% or around 77% of null values.
    # Thus, column, 'Cabin' will be dropped from the dataset
    data = data.drop(['Cabin'], axis = 1)

    # dropping the 'PassengerId' index column from the given dataset
    data.reset_index(drop = True, inplace = True)

    # checking non-numeric values if there are any in columns i.e., Age and Fare - it is found that there are no such values
    data[['Age', 'Fare', 'SibSp', 'Parch']][~data[['SibSp', 'Parch', 'Age', 'Fare']].applymap(np.isreal).all(1)]

    # checking invalid values. for e.g., if there are rows where the value of Age is 0 - it is found that there are no such values
    data[data['Age'].isin([0])]

    # imputing missing values in 'Age' column with median values of age by salutation of Titanic passengers since Age differs 
    # by salutations even for same gender

    data['Salutation'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0])
    data['Salutation'] = data['Salutation'].str.replace('Mlle', 'Miss')
    data['Salutation'] = data['Salutation'].str.replace('Ms', 'Miss')
    data['Salutation'] = data['Salutation'].str.replace('Mme', 'Mrs')

    # imputing the missing values in 'Age' column by Salutation information obtained from the name of passengers
    data['Age'] = data.groupby(['Salutation'])['Age'].transform(lambda x: x.fillna(x.median()))

    # imputing the missing values in 'Embarked' column, which is a categorical column,  with most frequently occurring values or mode
    data['Embarked']= data['Embarked'].fillna(data['Embarked'].mode()[0])

    # column, 'Name' will also be dropped now from the dataset
    data = data.drop(['Name'], axis = 1)

    # we can also drop the 'Salutation' column from the dataset since we have completed the imputation of null values in 'Age' column

    data = data.drop(['Salutation'], axis = 1)

    # checking whether there is imbalanced distribution of classes 0 and 1 for non-survived and survived passengers in 
    # target column, 'Survived' - it is found that distribution for positive or Survived passenger class is just 38% 
    # while the distribution for negative or non-survived passenger class is around 62%.
    # Thus, the ratio of classes in the dataset is close to 60:40, which is considered fairly balanced 
    # distribution of classes in the dataset

    # Please refer to this link for information: 
    # https://thesai.org/Downloads/Volume13No6/Paper_27-Survey_on_Highly_Imbalanced_Multi_class_Data.pdf
    # Please also check: https://rpubs.com/DeclanStockdale/799284

    pd.value_counts(data['Survived'])

    # checking the no. of unique values in each column
    data.nunique()

    ### EDA to check the patterns in the titanic dataset for the survived and non-survived passengers in the target column, "Survived"

    plt.figure(figsize=(11,5))

    plt.subplot(1,2,1)
    sns.countplot(x=data['Survived'])
    plt.title('Count of passengers survived')
    plt.subplot(1,2,2)
    sns.countplot(x=data['Sex'],palette='rainbow')
    plt.title('Count of gender')


    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    sns.countplot(x=data['Pclass'])
    plt.subplot(1,3,2)
    sns.countplot(x=data['SibSp'])
    plt.subplot(1,3,3)
    sns.countplot(x=data['Parch'])

    # checking the proportion of Survived passengers by Passenger Class - There is a higher chance of survival if you have a first class ticket 
    # than having a second or third class ticket

    sns.barplot(x='Pclass', y='Survived', data=data);

    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=14)
    plt.figure()
    fig = data.groupby('Survived')['Pclass'].plot.hist(histtype= 'bar', alpha = 0.8)
    plt.legend(('Not Survived','Survived'), fontsize = 12)
    plt.xlabel('Pclass', fontsize = 18)
    plt.show()

    sns.countplot( x='Survived', data=data, hue="Embarked")

    # creating a column, 'Family Members' to capture the no. of family members traveling with passengers
    data['Family Members'] = data['SibSp'].astype(int) + data['Parch'].astype(int)

    # it can be observed in below plot that it is more likely for a passenger to survive if the passenger is traveling with 1 to 3 people than if the passenger is traveling with 0 or more than three people.
    sns.countplot(x = 'Family Members', hue = 'Survived', data= data)

    survived = 'survived'
    not_survived = 'not survived'
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))
    women = data[data['Sex']=='female']
    men = data[data['Sex']=='male']
    ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False, color="green")
    ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False, color="red")
    ax.legend()
    ax.set_title('Female')
    ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False, color="green")
    ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False, color="red")
    ax.legend()
    _ = ax.set_title('Male');

    # converting the data type of 'SibSp' and 'Parch' columns to 'object' data types as these are categorical columns in 
    # given dataset
    data['SibSp'] = data['SibSp'].astype('object')
    data['Parch'] = data['Parch'].astype('object')

    '''
    1. Does column, 'Pclass' has an impact on label, 'Survived' in given dataset?
    Since they both are categorical variables, we can use chi-squared test to konwn if there is any noticeable relation between them.
    Hypothesis

    Null hypothesis (H0): There is No relationship between 'Pclass' and 'Survived'
    Alternate Hypothesis (H1) : There is relationship between both.
    Assuming alpha is 0.05
    '''
    matrix = pd.crosstab(data['Pclass'],data['Survived'], margins = True)
    val = stats.chi2_contingency(matrix)
    val

    # getting the expected frequency values from the chi-squared test
    val.expected_freq

    critical_value = stats.chi2.ppf(q=1-0.05,df=val.dof)
    print("Critical value:", critical_value)

    if val.pvalue < 0.05 and val.statistic > critical_value:
        print("We reject the Null hypothesis. There is relationship between Pclass and Survived")
    else:
        print("We fail to reject the Null hypothesis. There is no relationship between Pclass and Survived")

    '''
    Similarly, List of other hypotheses to check:

    Does Sex have an impact on Survived?
    Does Sibsp have an impact on Survived?
    Does Parch have an impact on Survived?
    Does Embarked have an impact on Survived?

    Hypothesis:

    Null Hypothesis(H0): No Relationship
    Alternate Hypothesis(H1): There is relationship
    Alpha value is 0.05

    '''

    def chisquared_test(data, var, dependent):
        data_matrix = pd.crosstab(data[var],data[dependent])
        result = stats.chi2_contingency(data_matrix)
        if result.pvalue < 0.05:
            print(" p value:", round(result.pvalue,7), '\n', var, "has significant impact on Survived.","\n")
        else:
            print(" p value:", round(result.pvalue,7), '\n', var,  "has no significant impact, can be ignored if needed in feature engineering.","\n")


    chisquared_test(data, 'Sex', 'Survived')
    chisquared_test(data, 'SibSp', 'Survived')
    chisquared_test(data, 'Parch', 'Survived')
    chisquared_test(data, 'Embarked', 'Survived')

    '''
    Does Age have a significant impact on label, 'Survived'?
    Hypothesis:
    Null hypothesis (H0): There is No relationship between Age and Survived
    Alternate Hypothesis (H1) : There is relationship between both.
    Assuming alpha is 0.05 
    using one-way ANOVA test and Krushkal test to find the relationship.
    '''
    def annova_krushkal(data, cont):
        annova = stats.f_oneway(data[cont][data['Survived'] == 0], data[cont][data['Survived'] == 1])
        krushkal = stats.kruskal(data[cont][data['Survived'] == 0], data[cont][data['Survived'] == 1])
    
        if (annova.pvalue and krushkal.pvalue) < 0.05:
            print(" annova pvalue:",annova.pvalue,"\n","krushkal pvalue:",krushkal.pvalue,"\n", "Null hypothesis is rejected and it can be concluded that there is relationship between both.")
        else:
            print(" annova pvalue:",annova.pvalue,"\n","krushkal pvalue:",krushkal.pvalue,"\n", "We fail to reject the null hypothesis and there is no statistical significance between both variables.")
        
    annova_krushkal(data, 'Age')
    ## Both One-way ANOVA test and Krushkal test show the p value > 0.05. Thus, there is NO strong relationship between Age and Survived.
    
    annova_krushkal(data, 'Fare')
    ## The above tests suggest that Fare has an strong impact on Survived variable.

    # Column, Age can be dropped from the dataset based on the statistical tests.
    data = data.drop(['Age'], axis = 1)

    # Feature Engineering

    # One-Hot Encoding with multiple categories
    # since column, 'Ticket' has a lot of different categories, encoding the top 10 most frequent categories in a variable eliminates the other categories.
    # This ensembling method has been found by reasearchers.
    data['Ticket'].value_counts().count()
    
    # as there are 681 different categories present in 'Ticket' variable, one-hot encoding on this variable will further 
    # give 680 more variables to deal with and in order to avoid such a circumstance, using the method which 
    # has been used by researchers.

    # making a copy of original data before making changes

    data_d = data.copy(deep = True)

    # the values in Ticket column will be replaced with the most frequently occurring 10 categories in 'Ticket' column with random sample imputation with replacement.
    # creating a dataframe to be used in random sample imputation of values in 'Ticket' column
    df = pd.DataFrame(data=data_d['Ticket'].value_counts().head(10).index, index=range(0,10))
    ticket_top10 = df.sample(data_d['Ticket'].count(), replace=True, random_state=42)
    ticket_top10.index = data_d['Ticket'].index
    data_d['Ticket'] = ticket_top10

    '''
    One Hot Encoding:-
    Encoding the categorical feature into numerical values by performing one hot encoding on Pclass, Sex, Parch, Ticket and Embarked
    Since 'Survived' is dependent variable, that variable will be dropped and one hot encoding will be carried out on other variables availabe.

    '''
    data_d = data_d.drop(['Survived'], axis = 1)

    # dropping columns, 'Family Members' from dataset
    data = data.drop(['Family Members'], axis = 1)

    cleaned_data = pd.get_dummies(data_d, dtype=int)
    cleaned_data['Survived'] = data['Survived']
    cleaned_data = cleaned_data.drop(['Family Members'], axis = 1)

    # DIVIDING DATA INTO TARGET AND LABEL 
    X = cleaned_data.drop(["Survived"], axis=1)
    Y = cleaned_data["Survived"]

    # changing the data type of lable, 'Survived' to categorical data type
    Y = Y.astype('category')

    metrics = {}
    params = {}

    with mlflow.start_run():
        
        # splitting the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25, random_state=1)
        mlflow.log_params({"Train_shape": X_train.shape, "Test_shape": X_test.shape})

        # Stochastic Gradient Descent (SGD) Classifier
        sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
        sgd.fit(X_train, Y_train)

        sgd_predictions = sgd.predict(X_test)
        print(f"SGD Classifier Score: {accuracy_score(Y_test, sgd_predictions)}")
        acc_sgd = round(accuracy_score(Y_test, sgd_predictions) * 100, 2)

        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(X_train, Y_train)
        rf_predictions = random_forest.predict(X_test)
        print(f"Random Forest Classifier Score: {accuracy_score(Y_test, rf_predictions)}")
        acc_random_forest = round(accuracy_score(Y_test, rf_predictions) * 100, 2)

        logreg = LogisticRegression()
        logreg.fit(X_train, Y_train)
        logreg_predictions = logreg.predict(X_test)
        print(f"Logistic Regression Score: {accuracy_score(Y_test, logreg_predictions)}")
        acc_log = round(accuracy_score(Y_test, logreg_predictions) * 100, 2)

        knn = KNeighborsClassifier(n_neighbors = 3)
        knn.fit(X_train, Y_train)  
        knn_predictions = knn.predict(X_test.values) 
        print(f"K-Nearest Neighbors Classifier Score: {accuracy_score(Y_test, knn_predictions)}")
        acc_knn = round(accuracy_score(Y_test, knn_predictions) * 100, 2)
    

        linear_svc = LinearSVC()
        linear_svc.fit(X_train, Y_train)
        svc_predictions = linear_svc.predict(X_test)
        print(f"Linear Support Vector Machine Classifier Score: {accuracy_score(Y_test, svc_predictions)}")
        acc_linear_svc = round(accuracy_score(Y_test, svc_predictions) * 100, 2)


        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, Y_train)  
        dectree_predictions = decision_tree.predict(X_test)  
        print(f"Decision Tree Classifier Score: {accuracy_score(Y_test, dectree_predictions)}")
        acc_decision_tree = round(accuracy_score(Y_test, dectree_predictions) * 100, 2)

        gaussian = GaussianNB() 
        gaussian.fit(X_train, Y_train)
        gaussian_predictions = gaussian.predict(X_test)
        print(f"Naive Bayes Score: {accuracy_score(Y_test, gaussian_predictions)}")
        acc_gaussian = round(accuracy_score(Y_test, gaussian_predictions) * 100, 2)

        perceptron = Perceptron(max_iter=5)
        perceptron.fit(X_train, Y_train)
        perceptron_predictions = perceptron.predict(X_test)
        print(f"Perceptron Score: {accuracy_score(Y_test, perceptron_predictions)}")
        acc_perceptron = round(accuracy_score(Y_test, perceptron_predictions) * 100, 2)

        metrics['Support Vector Machines Score'] = acc_linear_svc
        metrics['KNN Score'] = acc_knn
        metrics['Logistic Regression Score'] = acc_log
        metrics['Random Forest Score'] = acc_random_forest
        metrics['Naive Bayes Score'] = acc_gaussian
        metrics['Perceptron Score'] = acc_perceptron
        metrics['Stochastic Gradient Decent Score'] = acc_sgd
        metrics['Decision Tree Score'] = acc_decision_tree

        params['Support Vector Machines Params'] = linear_svc.get_params()
        params['KNN Params'] = knn.get_params()
        params['Logistic Regression Params'] = logreg.get_params()
        params['Random Forest Params'] = random_forest.get_params()
        params['Naive Bayes Params'] = gaussian.get_params()
        params['Perceptron Params'] = perceptron.get_params()
        params['Stochastic Gradient Decent Params'] = sgd.get_params()
        params['Decision Tree Params'] = decision_tree.get_params()
        
        results = pd.DataFrame({
            'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
                      'Random Forest', 'Naive Bayes', 'Perceptron', 
                      'Stochastic Gradient Decent', 'Decision Tree'],
            'Score': [acc_linear_svc, acc_knn, acc_log, 
                      acc_random_forest, acc_gaussian, acc_perceptron, 
                      acc_sgd, acc_decision_tree]})
            
        result_df = results.sort_values(by='Score', ascending=False)
        result_df.head(9)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sgd, 'sgd_model')
        mlflow.sklearn.log_model(knn, 'knn_model')
        mlflow.sklearn.log_model(logreg, 'logreg_model')
        mlflow.sklearn.log_model(random_forest, 'random_forest_model')
        mlflow.sklearn.log_model(gaussian, 'gaussian_model')
        mlflow.sklearn.log_model(perceptron, 'perceptron_model')
        mlflow.sklearn.log_model(linear_svc, 'linear_svc_model')
        mlflow.sklearn.log_model(decision_tree, 'decision_tree')
        


    


  
  


  
