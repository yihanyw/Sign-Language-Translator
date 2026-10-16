"""
Lightweight model definition for the final Streamlit app.

shapes in saved_models/asl_citizen_fc_model.pth:

    lstm: input 84 -> hidden 256, 4 layers   (lstm.weight_ih_l0 = [1024, 84])
    fc1 : 256 -> 128
    fc2 : 128 -> num_classes (20)

forward() returns a 3-tuple because the app does `logits, _, _ = model(video)`.
"""
import torch


class Video_LSTM_morelayers(torch.nn.Module):
    def __init__(self, input_size=84, hidden_size=256, num_layers=4,
                 dropout: float = 0.5, num_classes=20):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc1 = torch.nn.Linear(hidden_size, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x, h0=None, c0=None):
        # x: (B, T, input_size)
        b = x.size(0)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, b, self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, b, self.hidden_size, device=x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        h = self.fc1(out[:, -1, :])  # no activation, matches Video_LSTM
        logits = self.fc2(h)
        return logits, out, (hn, cn)
