
import torch
import torch.nn as nn

class RiskModel(nn.Module):
    """
    Adaptive Autopilot Risk Model (v4).
    Estimates context risk r_t in [0, 1] from the last token hidden state.
    Architecture: 3-layer MLP with Dropout for regularization.
    """
    def __init__(self, hidden_size: int = 768, dropout: float = 0.3, input_clamp: float = 256.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_clamp = float(input_clamp)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),  # lighter dropout on narrower layer
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, H) - Last token hidden state
        Returns:
            risk: (B,)
        """
        if hidden_states.dim() != 2:
            raise ValueError(f"RiskModel expected hidden_states with shape (B, H), got {tuple(hidden_states.shape)}.")
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                f"RiskModel hidden size mismatch: expected {self.hidden_size}, got {hidden_states.size(-1)}."
            )
        if not torch.isfinite(hidden_states).all():
            raise ValueError("RiskModel received non-finite hidden states.")
        # Convert hidden_states to the same dtype as the network parameters
        target_dtype = next(self.net.parameters()).dtype
        hidden_states = hidden_states.to(target_dtype).clamp(
            min=-self.input_clamp,
            max=self.input_clamp,
        )
        risk = self.net(hidden_states).squeeze(-1)
        if not torch.isfinite(risk).all():
            raise FloatingPointError("RiskModel produced non-finite outputs.")
        return risk

# DummyRiskModel removed - Fail Early:
# Please provide real RiskModel weights for testing or use --smoke flag.
