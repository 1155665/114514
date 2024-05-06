
from rich import traceback, print
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

traceback.install()

model = joblib.load('model.pkl')
test_data = np.array(["我是最棒的！！！"]) # Convert test_data to a numpy array
test_data = test_data.reshape(1, -1) # Reshape test_data to have two dimensions with 1546 features

# Convert test_data to numeric values using LabelEncoder
label_encoder = LabelEncoder()
test_data = label_encoder.fit_transform(test_data.ravel())

test_data = np.array(test_data, dtype=np.float64)
test_data = test_data.reshape(-1, 1) # Convert test_data to numeric values
predictions = model.predict(test_data)
print(predictions)