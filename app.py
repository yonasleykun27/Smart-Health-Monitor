from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import os

app = Flask(__name__)
CORS(app)

# ---------- MODEL ----------
class HealthLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(HealthLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ⚠️ Remove dropout if your trained model didn't use it
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return self.sigmoid(out)

# ---------- LOAD MODEL ----------
MODEL_PATH = os.path.join('models', 'health_model.pth')

model = None
mean = None
std = None

try:
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')

    input_dim = checkpoint.get('input_dim', 13)

    model = HealthLSTM(input_size=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    mean = checkpoint['mean']
    std = checkpoint['std']

    print("✅ Model loaded successfully")

except Exception as e:
    print(f"❌ Model loading error: {e}")

# ---------- ROUTES ----------
@app.route('/')
def home():
    return jsonify({"message": "Health API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json(force=True)

        # Required inputs
        required = ['age', 'gender', 'bmi', 'bp', 'gluc']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # ⚠️ MUST MATCH TRAINING FEATURES (13 values)
        features = [
            float(data['age']),
            float(data['gender']),
            float(data['bmi']),
            1.0, 0.0, 0.0, 0.0,   # placeholders
            float(data['bp']),
            180.0,
            float(data['gluc']),
            float(data.get('fatigue', 0)),
            float(data.get('chest_pain', 0)),
            float(data.get('dizziness', 0))
        ]

        x = torch.tensor(features, dtype=torch.float32)

        # Safe normalization
        x_scaled = (x - mean) / (std + 1e-6)

        # Clamp extreme values
        x_scaled = torch.clamp(x_scaled, -3, 3)

        # Shape for LSTM: (batch, seq_len, input_size)
        x_input = x_scaled.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            prob = model(x_input).item()

        return jsonify({
            'risk_score': round(prob, 4),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ---------- RUN ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)