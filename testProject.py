import pandas as pd
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

# Reading CSV

hr_data = pd.read_csv('HR_Dataset.csv')

# Exploratory Data Analysis

hr_data.head()

cols = hr_data.columns.tolist()
new_position = 10

cols.insert(new_position, cols.pop(cols.index('left')))
hr_data = hr_data[cols]


hr_data.head()
print(hr_data.info())
print(hr_data.describe())


hr_data.rename(columns={'Departments ':'departments'},inplace=True)
hr_data.columns = hr_data.columns.str.strip()

# print(hr_data.groupby('Department').mean())
# print(hr_data.groupby('salary').mean())
# print(hr_data.groupby('left').mean())

# Cleaning of data

hr_data.isnull().sum()

print("Number of duplicates : ", len(hr_data[hr_data.duplicated()]))

hr_data = hr_data.drop_duplicates()
print("Number of duplicates : ", len(hr_data[hr_data.duplicated()]))




# Data Preprocessing

hr_data = pd.get_dummies(hr_data, columns=['salary'])
X = hr_data.drop(columns = ['left', 'departments','Work_accident'])
y = hr_data['left']



# Data Visualization


# sns.countplot(hr_data.left, palette="Set2")
#
# print(hr_data.columns)
# sns.countplot(x='salary', hue='left', palette="Set2", data=hr_data)
# plt.figure(figsize=(15, 7))
# sns.countplot(x='departments', hue='left', palette="Set2", data=hr_data)

#
# sns.pairplot(hr_data, hue='left')


sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Model Building

models = {
    '        Logistic Regression': LogisticRegression(),
    '        Decision Tree': DecisionTreeClassifier(),
    '        Random Forest Classifier': RandomForestClassifier(),
}

accuracy, precision, recall = {}, {}, {}

for i in models.keys():
    models[i].fit(X_train, y_train)
    y_pred = models[i].predict(X_test)

    accuracy[i] = accuracy_score(y_pred, y_test)
    precision[i] = precision_score(y_pred, y_test)

hr_data_models = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision'])
hr_data_models['Accuracy'] = accuracy.values()
hr_data_models['Precision'] = precision.values()
hr_data_models

cm = confusion_matrix(y_test, y_pred)
conf_mat = pd.DataFrame(data=cm, columns=['Predicted Not Left', 'Predicted Left'],
                        index=['Actual Not Left', 'Actual Left'])
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="YlGnBu")

TN = cm[0, 0]
TP = cm[1, 1]
FN = cm[1, 0]
FP = cm[0, 1]
sensitivity = TP / float(TP + FN)
specificity = TN / float(TN + FP)
print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ', (TP + TN) / float(TP + TN + FP + FN), '\n', '\n',
      'Sensitivity or True Positive Rate = TP/(TP+FN) = ', TP / float(TP + FN), '\n',
      'Specificity or True Negative Rate = TN/(TN+FP) = ', TN / float(TN + FP), '\n', '\n',
      'Positive Predictive value = TP/(TP+FP) = ', TP / float(TP + FP), '\n',
      'Negative predictive Value = TN/(TN+FN) = ', TN / float(TN + FN), '\n', )

mlp = MLPClassifier(max_iter=500)
mlp.fit(X_train, y_train)
mlp_y_pred = mlp.predict(X_test)

print('The accuracy score of MLP is : ', accuracy_score(mlp_y_pred, y_test))
print('The precision score of MLP is : ', precision_score(mlp_y_pred, y_test))

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size=32, epochs=10)
ann_y_pred = ann.predict(X_test)
ann_y_pred = (ann_y_pred > 0.5)

print('The accuracy score of MLP is : ', accuracy_score(y_test, ann_y_pred))
print('The precision score of MLP is : ', precision_score(y_test, ann_y_pred))

# Conclusion