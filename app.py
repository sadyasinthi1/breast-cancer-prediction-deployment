from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import os  # ✅ Import this for reading environment variables
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Prepare scaler and model
data = load_breast_cancer()
scaler = StandardScaler().fit(data.data)
INPUT_SIZE = data.data.shape[1]

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

model = NeuralNet(INPUT_SIZE)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Fetch and validate form data
            features = []
            for i in range(30):
                feature_value = request.form.get(f"f{i}")
                if feature_value:
                    features.append(float(feature_value))
                else:
                    raise ValueError(f"Missing or invalid value for f{i}")
            
            # Scale the features
            scaled = scaler.transform([features])
            
            # Make prediction using the model
            with torch.no_grad():
                input_tensor = torch.tensor(scaled, dtype=torch.float32)
                output = model(input_tensor).item()
                prediction = "Malignant" if output >= 0.5 else "Benign"
        
        except Exception as e:
            prediction = f"Error: {str(e)}"
            app.logger.error(f"Error during prediction: {str(e)}")
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    # ✅ This allows Render.com (or similar) to set the port
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
