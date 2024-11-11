import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
from seaborn import heatmap

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


warnings.filterwarnings('ignore')

# Reading CSV
hr_data = pd.read_csv('HR_Analytics.csv')
print(hr_data.head())
#Steps we are going to follow
# Data Exploration
# Handle missing values
# Remove duplicates
# Check for outliers
# Correct data types
# Fix inconsistent data
# Address class imbalance (if relevant)
# Feature engineering

# Data Info
print(hr_data.info())

# Shape of the hr data
print("Shape command")
print(hr_data.shape)
# Part -1 Data Cleaning

hr_data.rename(
    columns={
        "left": "retention",
        "Work_accident": "work_accident",
        "Department": "department",
        "time_spend_company": "tenure",
        "average_montly_hours": "average_monthly_hours"
    }, inplace=True
)

# Qualitative, quantitative division
qualitative = []
quantitative = []

for column in hr_data.columns:
    if pd.api.types.is_numeric_dtype(hr_data[column]):
        quantitative.append(column)
    else:
        qualitative.append(column)

# Handle missing values
print("Missing Value List")
print(hr_data.isna().sum())


# Remove duplicates
print("Number of Duplicate data")
print(hr_data.duplicated().sum())

if hr_data.duplicated().any():
    print('Duplicate data found')
    hr_data.drop_duplicates(inplace=True)
else:
    print('No duplicate data found')

print(hr_data.duplicated().sum())

# Check for outliers
columns_boxplot_to_detect_outliers = [t for t in hr_data.columns if
                                t not in ['retention', 'promotion_last_5years', 'work_accident']]
print(columns_boxplot_to_detect_outliers)

plt.figure(figsize=(10, 6))
plt.suptitle("Boxplot")
for i, feature in enumerate(columns_boxplot_to_detect_outliers):
    ax = plt.subplot(3, 3, i + 1)
    sns.boxplot(hr_data[feature], ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
plt.show()

# Some outliers in the tenure
for col in quantitative:
    print(col)
    Q1 = hr_data[col].quantile(0.25)
    Q3 = hr_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = hr_data[(hr_data['tenure'] > upper_bound) | (hr_data['tenure'] < lower_bound)]
    print(outliers)

    # Compute the mean of the column
    mean_value = hr_data['tenure'].mean()
    # Set outlier values to the mean value
    #hr_data.loc[(hr_data['tenure'] > upper_bound) | (hr_data['tenure'] < lower_bound), hr_data['tenure']] = mean_value

    print(hr_data.shape)

print("Tenure columns details")
print(hr_data['tenure'])

# Correct data types
hr_data['salary'] = hr_data['salary'].astype('category')
# Fix inconsistent data
    # No inconsistent data

# Address class imbalance (if relevant)
    # No address class imbalance
# Feature engineering
    # No need to create new feature from existing features.

# Data Visualization
# Histograms,
salary_vs_retention = pd.crosstab(hr_data['salary'], hr_data['retention'], margins=True)
department_vs_tenure = pd.crosstab(hr_data['department'], hr_data['retention'], margins=True)

# Create a figure and a set of subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot for Salary vs Retention
ax1 = axes[0]
salary_vs_retention.drop('All', axis=1).drop('All', axis=0).plot(kind="bar", ax=ax1)
ax1.set_title('Salary vs Retention')
ax1.set_xlabel('Salary')
ax1.set_ylabel('Retention Count')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

# Plot for Department vs Tenure
ax2 = axes[1]
department_vs_tenure.drop('All', axis=1).drop('All', axis=0).plot(kind="bar", ax=ax2)
ax2.set_title('Department vs Retention')
ax2.set_xlabel('Department')
ax2.set_ylabel('Retention Count')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="average_monthly_hours", y="satisfaction_level",
    hue="retention",
    data=hr_data,
    palette="Set2"
)
plt.title("Average Monthly hours vs Satisfaction Level")
plt.xlabel("Average Monthly hours")
plt.ylabel("Satisfaction Level")
plt.show()


# Box plots, Scatter plots, Bar plots, Line plots, Pair plots


# Encoding Data
encoder = LabelEncoder()

hr_data["department"] = encoder.fit_transform(hr_data["department"])
hr_data["salary"] = encoder.fit_transform(hr_data["salary"])

print("Department only unique after encoding")
print(hr_data["department"].unique())
print(hr_data.dtypes)

# Correlation heatmaps
plt.figure(figsize=(20, 12))
heatmap = sns.heatmap(
    hr_data.corr(),
    vmin=-1, vmax=1, annot=True,
    cmap=sns.color_palette("vlag", as_cmap=True),
    square=True,
    cbar_kws={'shrink': 0.8}
)

# Set title
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 16}, pad=20)

# Rotate both x and y-axis labels and adjust alignment
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=10)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=45, ha='right', fontsize=10)

plt.tight_layout()
plt.show()

# Data Preprocessing

# Dropping unnecessary columns
    # Encoding categorical variables (done for correlation of data)

# Storage for further processing
X = hr_data.drop(columns=['retention', 'department', 'work_accident'])
y = hr_data['retention']

# Standardizing the data
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Model Building

#sepearte function for this


models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
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
print(hr_data_models)

# Confusion matrix
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
print('The accuracy of the model = TP+TN/(TP+TN+FP+FN) = ', (TP + TN) / float(TP + TN + FP + FN), '\n', '\n',
      'Sensitivity or True Positive Rate = TP/(TP+FN) = ', TP / float(TP + FN), '\n',
      'Specificity or True Negative Rate = TN/(TN+FP) = ', TN / float(TN + FP), '\n', '\n',
      'Positive Predictive value = TP/(TP+FP) = ', TP / float(TP + FP), '\n',
      'Negative predictive Value = TN/(TN+FN) = ', TN / float(TN + FN), '\n')
