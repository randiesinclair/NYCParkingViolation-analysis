import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv("../NYCParkingViolations_FiscalYear_2023.csv")

# Create a binary target column: 1 for ticketed, 0 for not ticketed
data['Ticketed'] = data['Violation Code'].apply(lambda x: 1 if x > 0 else 0)

# Select the relevant features
features = ['Vehicle Make', 'Vehicle Color', 'Registration State', 'Vehicle Body Type', 'Street Name', 'Ticketed']
data = data[features]

# Encode categorical variables
encoders = {}
for col in features[:-1]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# Split the data into train and test sets
X = data.drop('Ticketed', axis=1)
y = data['Ticketed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def predict_ticket_probability(model, scaler, encoders):
    questions = [
        "What is the make of the vehicle?",
        "What is the color of the vehicle?",
        "What is the registration state?",
        "What is the vehicle body type?",
        "What is the street name you're parked on?"
    ]

    user_input = []
    for question, feature in zip(questions, features[:-1]):
        answer = input(question + " ")
        user_input.append(encoders[feature].transform([answer])[0])

    # Scale the input
    user_input_scaled = scaler.transform([user_input])[0].reshape(1, -1)


    # Get the prediction probability
    prob = model.predict_proba(user_input_scaled)[0][1]
    return prob

# Get user input and predict the chance of getting ticketed
probability = predict_ticket_probability(clf, scaler, encoders)
print(f"The probability of getting ticketed is: {probability*100:.2f}%")
