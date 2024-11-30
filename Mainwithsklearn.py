import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



hr_data = pd.read_csv('HR_Analytics.csv')
print(hr_data)

hr_data.rename(
    columns={
        "left": "retention",
        "Work_accident": "work_accident",
        "Department": "department",
        "time_spend_company": "tenure",
        "average_montly_hours": "average_monthly_hours"
    }, inplace=True
)

hr_data.drop_duplicates(inplace=True)

encoder = LabelEncoder()
hr_data['department'] = encoder.fit_transform(hr_data['department'])
hr_data['salary'] = encoder.fit_transform(hr_data['salary'])


X = hr_data.drop(columns=['retention', 'department', 'work_accident'])
y = hr_data['retention']

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=43)

ann = Sequential([
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


models = {
    #logistic regression
    'lr' : LogisticRegression(),
    'lr-1': LogisticRegression(penalty= 'l1',solver='liblinear', random_state= 42),
    'lr-2': LogisticRegression(penalty= 'l2',solver='liblinear', random_state=42),
    'lr-1-saga': LogisticRegression(penalty= 'l1',solver='saga', random_state=42),
    'lr-2-saga': LogisticRegression(penalty= 'l2',solver='saga', random_state=42),
    'svm': SVC(kernel='linear', random_state=43, class_weight='balanced'),
    'rdt': RandomForestClassifier(n_estimators=100, random_state=42),
    'ann': ann,
    'knn': KNeighborsClassifier(),
    'naive_bayes': GaussianNB(),
    'gradientBoost': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'decision_tree': DecisionTreeClassifier()
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    if model_name == 'ann':
        y_pred = (model.predict(X_test) >= 0.3).astype(int)
    else:
        y_pred = model.predict(X_test)

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}\n")





# pre = tp/tp+fp
# recall = tp/tp+fn

#precision is 51 percent means, 51% of the people are actually leaving out of the predicted leaving the company.

# recall is 19 percent means, model is only 19% correct that correctly detecting that people are leaving.
#
# Model: lg
# Accuracy: 0.8299291371404752
# Precision: 0.5159235668789809
# Recall: 0.19612590799031476
# F1 Score: 0.28421052631578947
#
# Model: svm
# Accuracy: 0.7603167986661109
# Precision: 0.4121475054229935
# Recall: 0.9200968523002422
# F1 Score: 0.5692883895131086
#
# Model: rdt
# Accuracy: 0.9837432263443101
# Precision: 0.9819587628865979
# Recall: 0.9225181598062954
# F1 Score: 0.951310861423221
#
#
# Model: ann
# Accuracy: 0.8345143809920801
# Precision: 0.5121580547112462
# Recall: 0.8159806295399515
# F1 Score: 0.6293183940242764


