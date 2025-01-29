import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("transactions.csv")

# Features and Target
X = df[["amount", "location", "time"]]
y = df["fraud"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

def predict_fraud(person_id):
    person_data = df[df["person_id"] == person_id]
    if person_data.empty:
        return 0
    features = person_data[["amount", "location", "time"]]
    fraud_prob = model.predict_proba(features)[0][1]
    return round(fraud_prob * 100, 2)

def get_fraud_chain(person_id):
    fraud_chain = []
    visited = set()
    
    def dfs(current_id):
        if current_id in visited:
            return
        visited.add(current_id)
        fraud_percentage = predict_fraud(current_id)
        if fraud_percentage > 70:
            fraud_chain.append(current_id)
            # Get all transactions involving this person
            transactions = df[(df["person_id"] == current_id) | (df["related_person_id"] == current_id)]
            for _, row in transactions.iterrows():
                next_id = row["related_person_id"] if row["person_id"] == current_id else row["person_id"]
                dfs(next_id)
    
    dfs(person_id)
    return fraud_chain
