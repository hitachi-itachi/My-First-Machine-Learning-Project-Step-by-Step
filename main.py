# Check the versions of libraries

# Python version
import sys

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
...
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv" #access the github data via github
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] #Define the names of the columns in Iris dataset
dataset = read_csv(url, names=names) #passes url, names variable as argument to get stored in a variable "called dataset"

#this are are all pandas function
# shape
print(dataset.shape) #Pandas lib function to check dimensions

# looking at first 20 rows of the data
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size()) #group the dataset by group columns and then find out the size of the column


# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()