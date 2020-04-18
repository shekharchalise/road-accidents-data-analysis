import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil

def clear_output_files():
    if os.path.exists('./results-without-weather'):
       shutil.rmtree('./results-without-weather')
    try:
        os.makedirs('./results-without-weather/confusion matrix')
        os.makedirs('./results-without-weather/classifiction report')
    except OSError:
        pass

clear_output_files()

input_dataframe = pd.read_csv('preprocessed-file.csv')
# not using following columns
#"Temperature(F)","Wind_Chill(F)","Humidity(%)","Pressure(in)","Visibility(mi)","Wind_Speed(mph)","Precipitation(in)","Weather_Condition",
#input_dataframe= input_dataframe[:5000] #keep 5000
input_dataframe= input_dataframe[["Severity","Start_Lat","Start_Lng","Zipcode","State","Amenity","Bump","Crossing","Give_Way","Junction","No_Exit","Railway","Roundabout","Station","Stop","Traffic_Calming","Traffic_Signal","Turning_Loop","Start_Time_Year","Start_Time_Month","Start_Time_Day","Start_Time_Hour","Start_Time_Minute","End_Time_Year","End_Time_Month","End_Time_Day","End_Time_Hour","End_Time_Minute"]]


Y = input_dataframe.Severity.values

cols = input_dataframe.shape[1]
X = input_dataframe.loc[:, input_dataframe.columns != 'Severity']
X.columns

X_train, X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.33, random_state=99)

# Support Vector Machines
# svc = SVC()
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)
# acc_svc = round(svc.score(X_test, Y_test) * 100, 2)
# sns.heatmap(confusion_matrix(Y_test, Y_pred),annot=True,fmt="d") 
# plt.savefig('results/confusion matrix/SVC.png')
# plt.clf()
# pd.DataFrame(classification_report(Y_test,Y_pred, output_dict=True)).transpose().to_csv('results-without-weather/classifiction report/SVC.csv')
# print(acc_svc)

#KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_test, Y_test) * 100, 2)
sns.heatmap(confusion_matrix(Y_test, Y_pred),annot=True,fmt="d") 
plt.savefig('results-without-weather/confusion matrix/knn.png')
plt.clf()
pd.DataFrame(classification_report(Y_test,Y_pred, output_dict=True)).transpose().to_csv('results-without-weather/classifiction report/knn.csv')
print(acc_knn)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
sns.heatmap(confusion_matrix(Y_test, Y_pred),annot=True,fmt="d") 
plt.savefig('results-without-weather/confusion matrix/logistic_regression.png')
plt.clf()
pd.DataFrame(classification_report(Y_test,Y_pred, output_dict=True)).transpose().to_csv('results-without-weather/classifiction report/logistic_regression.csv')
print(acc_log)

# # Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_test, Y_test) * 100, 2)
sns.heatmap(confusion_matrix(Y_test, Y_pred),annot=True,fmt="d") 
plt.savefig('results-without-weather/confusion matrix/gaussian_nb.png')
plt.clf()
pd.DataFrame(classification_report(Y_test,Y_pred, output_dict=True)).transpose().to_csv('results-without-weather/classifiction report/gaussian_nb.csv')
print(acc_gaussian)

# # Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_test, Y_test) * 100, 2)
sns.heatmap(confusion_matrix(Y_test, Y_pred),annot=True,fmt="d") 
plt.savefig('results-without-weather/confusion matrix/perceptron.png')
plt.clf()
pd.DataFrame(classification_report(Y_test,Y_pred, output_dict=True)).transpose().to_csv('results-without-weather/classifiction report/perceptron.csv')
print(acc_perceptron)

# Stochastic Gradient Descent

sgd = SGDClassifier(max_iter=1200000)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_test, Y_test) * 100, 2)
sns.heatmap(confusion_matrix(Y_test, Y_pred),annot=True,fmt="d") 
plt.savefig('results-without-weather/confusion matrix/SGD.png')
plt.clf()
pd.DataFrame(classification_report(Y_test,Y_pred, output_dict=True)).transpose().to_csv('results-without-weather/classifiction report/SGD.csv')
print(acc_sgd)

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)
sns.heatmap(confusion_matrix(Y_test, Y_pred),annot=True,fmt="d") 
plt.savefig('results-without-weather/confusion matrix/decision_tree.png')
plt.clf()
pd.DataFrame(classification_report(Y_test,Y_pred, output_dict=True)).transpose().to_csv('results-without-weather/classifiction report/decision_tree.csv')
print(acc_decision_tree)

# # Random Forest

random_forest = RandomForestClassifier(n_estimators=100, random_state = 4)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)
sns.heatmap(confusion_matrix(Y_test, Y_pred),annot=True,fmt="d") 
plt.savefig('results-without-weather/confusion matrix/random_forest.png')
plt.clf()
pd.DataFrame(classification_report(Y_test,Y_pred, output_dict=True)).transpose().to_csv('results-without-weather/classifiction report/random_forest.csv')
print(acc_random_forest)


models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
models.sort_values(by='Score', ascending=False).to_csv("results-without-weather/prediction-score.csv")


