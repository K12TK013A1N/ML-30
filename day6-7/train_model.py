# train_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pickle

# Load dataset and train model
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier()
model.fit(X, y)

# Save model to a file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)