import torch
import torch.nn as nn

class SmartHealthLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(SmartHealthLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer: Processes the input features as a sequence
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully Connected Layer: Map LSTM output to a single risk score
        self.fc = nn.Linear(hidden_size, 1)
        
        # Sigmoid: Ensures the output is a probability between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)