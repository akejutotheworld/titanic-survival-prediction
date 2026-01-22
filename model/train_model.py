import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load dataset
data = pd.read_csv('data/titanic.csv')
print(data['Survived'].value_counts())

# Select features and target
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# Encode categorical column
encoder = LabelEncoder()
X['Sex'] = encoder.fit_transform(X['Sex'])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X[['Age', 'Fare']] = imputer.fit_transform(X[['Age', 'Fare']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model ✅ FIXED INDENTATION
with open('../model/titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")

