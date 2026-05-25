from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import os

app = Flask(__name__)
CORS(app)

# --- 1. LSTM ARCHITECTURE ---
class SmartHealthLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(SmartHealthLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# Load the model
MODEL_PATH = os.path.join('models', 'smart_health_lstm_model.pth')
model = SmartHealthLSTM(input_size=13)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        print("✅ DBU Adjusted Model Loaded Successfully")
    except Exception as e:
        print(f"⚠️ Load Warning: {e}")
else:
    print("⚠️ Model file not found, running with random weights for demo.")

# --- 2. IMPROVED NORMALIZATION LOGIC ---
def normalize_balanced(features):
    # Max values for normalization
    max_vals = [100, 1, 50, 3, 1, 1, 190, 350, 250, 1, 1, 1, 160]
    normalized = []
    for i, (f, m) in enumerate(zip(features, max_vals)):
        val = f / m if m != 0 else 0
        normalized.append(min(max(val, 0.0), 1.0))
    return normalized

# --- 3. PREDICTION ENDPOINT WITH CLINICAL LOGIC ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Binary factors
        smoking = 1.0 if data.get('smoking') else 0.0
        alcohol = 1.0 if data.get('alcohol') else 0.0
        chest_pain = 1.0 if data.get('chest_pain') else 0.0
        fatigue = 1.0 if data.get('fatigue') else 0.0
        dizziness = 1.0 if data.get('dizziness') else 0.0

        # Numeric factors
        age = float(data.get('age', 25))
        gender = 1.0 if str(data.get('gender')).lower() in ['female', '1'] else 0.0
        bmi = float(data.get('bmi', 22))
        bp = float(data.get('bp', 120))
        gluc = float(data.get('gluc', 90))
        hr = float(data.get('heart_rate', 72))
        chol = float(data.get('cholesterol', 180))
        ex_level = float(data.get('exercise_level', 0))

        raw = [age, gender, bmi, ex_level, smoking, alcohol, bp, chol, gluc, fatigue, chest_pain, dizziness, hr]
        
        # 1. Get AI Prediction
        scaled = normalize_balanced(raw)
        x = torch.tensor(scaled, dtype=torch.float32).view(1, 1, 13)
        with torch.no_grad():
            score = model(x).item()

        # 2. APPLY CLINICAL RULES (The "Fix")
        
        # A. Emergency Situations (Addressing image_f79456.png)
        if hr <= 0:
            score = 1 # Cardiac Arrest
            status_detail = "EMERGENCY: CARDIAC ARREST"
        elif hr < 40 or hr > 150:
            score = max(score, 0.85)
            status_detail = "Critical Heart Rate"
            
        # B. High Glucose (Addressing image_f6ab9f.png - Glucose 250)
        elif gluc > 200:
            score = max(score, 0.78)
            status_detail = "Critical Glucose"

        # C. Normalizing for "Green" (Healthy Vitals)
       
        elif (90 <= bp <= 125) and (70 <= gluc <= 105) and (60 <= hr <= 85):
           
            if (smoking + alcohol + chest_pain) == 0:
                score = 0.18 # Healthy (Green)
            else:
                score = 0.27 # Moderate (Yellow) - as requested

        # D. Other Guardrails
        else:
            if bp > 160: score = max(score, 0.70)
            if bmi > 32 and chol > 240: score = max(score, 0.65)
            if chest_pain == 1.0: score = max(score, 0.60)

        # Final score formatting
        final_score = round(min(score, 0.9999), 4)

        return jsonify({
            'risk_score': final_score,
            'status': 'Critical' if final_score > 0.6 else 'Moderate' if final_score > 0.25 else 'Stable'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)