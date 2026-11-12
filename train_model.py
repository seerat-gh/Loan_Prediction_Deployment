import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read the training dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# Fill missing categorical values with the most common value
for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill missing numerical values with the median
for col in ["LoanAmount", "Loan_Amount_Term", "Credit_History"]:
    df[col] = df[col].fillna(df[col].median())

# Convert text categories into numeric values
category_maps = {
    "Gender": {"Female": 0, "Male": 1},
    "Married": {"No": 0, "Yes": 1},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"No": 0, "Yes": 1},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
    "Loan_Status": {"N": 0, "Y": 1}
}

for col, mapping in category_maps.items():
    df[col] = df[col].map(mapping)

# Loan_ID is excluded because it is only an identifier.
X = df.drop(columns=["Loan_ID", "Loan_Status"])
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

with open("loan_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as loan_model.pkl")
