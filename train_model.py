import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("dataset.csv")

symptoms_list = data.iloc[:,1:-1].values.tolist()
for i,row in enumerate(symptoms_list):
    symptoms_list[i] = [s for s in row if str(s) != 'nan']

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(symptoms_list)
y = data['Disease']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump((model, mlb), "model.pkl")