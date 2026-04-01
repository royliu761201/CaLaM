
import torch
import torch._dynamo
import torch.nn.functional as F
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

class CaLaMController:
    """
    CaLaM: Geometric Control Theory of Generative AI Behavior.
    Implements Theorem 3.1: KL-based Logit Shift Control Law.
    """
    def __init__(
        self, 
        num_features: int = 1, 
        learning_rate: float = 0.1, 
        dual_steps: int = 5,
        lambda_init: float = 0.0,
        max_lambda: float = 100.0,
        max_shift: float = 80.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.m = num_features
        self.lr = learning_rate
        self.k = dual_steps
        self.device = device
        self.max_lambda = float(max_lambda)
        self.max_shift = float(max_shift)
        self.lambda_t = torch.zeros(self.m, device=self.device) + lambda_init

        # [Speed Opt] JIT Kernel Fusion: eliminates Python GIL overhead in dual loop
        torch._dynamo.config.suppress_errors = True
        self.solve_dual = torch.compile(self._solve_dual_internal)

    def _solve_dual_internal(
        self, 
        base_logits: torch.Tensor, 
        features: torch.Tensor, 
        constraints: torch.Tensor,
        prev_lam: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Solves the dual optimization problem via Projected Gradient Ascent (Alg. 2).
        """
        batch_size = base_logits.size(0)

        # [Speed Opt] Warm-start Momentum: inherit λ from previous token step
        if prev_lam is not None:
            lam = prev_lam.clone().to(torch.float32)
        else:
            lam = torch.zeros(batch_size, self.m, device=self.device, dtype=torch.float32, requires_grad=False)

        base_logits = base_logits.to(torch.float32)
        features = features.to(torch.float32)
        constraints = constraints.to(torch.float32)

        if not torch.isfinite(base_logits).all():
            raise ValueError("Non-finite base logits passed into CaLaM dual solver.")
        if not torch.isfinite(features).all():
            raise ValueError("Non-finite features passed into CaLaM dual solver.")
        if not torch.isfinite(constraints).all():
            raise ValueError("Non-finite constraints passed into CaLaM dual solver.")

        # Initialize PID state
        integral = torch.zeros_like(lam)
        prev_error = torch.zeros_like(lam)

        # PID Gains (Tunable, but defaults should be stable)
        Kp = self.lr
        Ki = self.lr * 0.1
        Kd = self.lr * 0.01

        for step in range(self.k):
            # Compute current q_lambda
            # q(v) \propto p_t(v) * exp(-lambda^T f(v))
            # Equivalent to shifting logits: l_new = l_old - lambda^T f

            # (B, 1, m) * (B, V, m) -> (B, V) via einsum or broadcast
            shift = torch.einsum('bm,bvm->bv', lam, features)
            if not torch.isfinite(shift).all():
                raise FloatingPointError("CaLaM dual solver produced non-finite logit shifts.")
            shift = shift.clamp(min=-self.max_shift, max=self.max_shift)

            # Unnormalized q
            log_q_unnorm = base_logits - shift
            if not torch.isfinite(log_q_unnorm).all():
                raise FloatingPointError("CaLaM dual solver produced non-finite shifted logits.")
            q_lambda = F.softmax(log_q_unnorm, dim=-1, dtype=torch.float32) # (B, V) 

            # Expected features E_q[f]
            # (B, V) * (B, V, m) -> (B, m)
            # Cast back to native bf16 for tensor ops
            mu_lambda = torch.einsum('bv,bvm->bm', q_lambda.to(features.dtype), features)
            if not torch.isfinite(mu_lambda).all():
                raise FloatingPointError("CaLaM dual solver produced non-finite expected features.")

            # Gradient for Dual Ascent: E_q[f] - b_t
            error = mu_lambda - constraints
            if not torch.isfinite(error).all():
                raise FloatingPointError("CaLaM dual solver produced non-finite dual residuals.")

            # PID Update
            integral = (integral + error).clamp_(min=-10.0, max=10.0)
            derivative = error - prev_error

            # Update Lambda
            # lam = lam + Kp * error + Ki * integral + Kd * derivative
            # We apply PID to the *update step* to smooth the trajectory
            update = Kp * error + Ki * integral + Kd * derivative
            if not torch.isfinite(update).all():
                raise FloatingPointError("CaLaM dual solver produced non-finite lambda updates.")
            lam = lam + update

            # Projection: lambda >= 0 and cap
            lam = lam.clamp_(min=0.0, max=self.max_lambda)

            prev_error = error

        return lam

    def control(
        self, 
        base_logits: torch.Tensor, 
        features: torch.Tensor, 
        constraints: torch.Tensor,
        use_geometry: bool = True,
        use_lookahead: bool = True,
        prev_lam: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies CaLaM control to logits with optional ablations.

        Args:
            base_logits: (B, V)
            features: (B, V, m)
            constraints: (B, m)
            constraint_threshold (float): Safe bound b_t
            use_geometry: If False, uses Euclidean mixing instead of Geometric
            use_lookahead: If False, uses simplified non-predictive shift
            prev_lam: (B, m) warm-start lambda from previous token step

        Returns:
            final_logits: (B, V)
            lambda_t:     (B, m)
        """
        # [Zero-Anchoring Fix for Toxicity Inversion / Lambda Flatness]
        # Shift f(v) so that min(f(v)) == 0.0 across the vocabulary.
        f_min = features.amin(dim=1)  # (B, m)
        centered_features = features - f_min.unsqueeze(1)
        effective_constraints = constraints - f_min

        # For the paper setting, infeasible centered constraints must not be silently relaxed.
        if (effective_constraints < 0).any():
            raise ValueError(
                "CaLaM control problem infeasible after zero-anchoring: "
                "raw feature floor exceeds the requested constraint."
            )

        if not use_lookahead:
            # Ablation: No Lookahead — fixed penalty (λ=λ_init, not adaptive dual solver)
            # This tests: "is adaptive λ better than fixed λ?"
            if prev_lam is not None:
                lam_star = prev_lam
            else:
                lambda_fixed = max(self.lambda_t.max().item(), 1.0)  # Fallback
                lam_star = torch.ones(base_logits.size(0), self.m, device=self.device) * lambda_fixed
        else:
            # 1. Solve for optimal Lambda via Dual Ascent
            lam_star = self.solve_dual(
                base_logits, 
                centered_features, 
                effective_constraints, 
                prev_lam=prev_lam
            )
            self.lambda_t = lam_star.detach().mean(dim=0)

        # 2. Compute intervention shift
        shift = torch.einsum('bm,bvm->bv', lam_star, centered_features)
        if not torch.isfinite(shift).all():
            raise FloatingPointError("CaLaM control produced non-finite final logit shifts.")
        shift = shift.clamp(min=-self.max_shift, max=self.max_shift)

        # 3. Apply shift — THIS IS WHERE Geometry vs No-Geometry DIVERGE
        if use_geometry:
            # Geometric Control: shift in log-space (exponential family / natural gradient)
            # l_star = l - shift  →  q(v) ∝ p(v) * exp(-λ^T f(v))
            l_star = base_logits - shift
        else:
            # [C1 FIX] Euclidean Control: shift in probability-space (flat / naive)
            # p_star = softmax(l) - λ^T f(v), then convert back to logits
            # This ignores the simplex curvature — the key ablation difference
            p_base = F.softmax(base_logits, dim=-1, dtype=torch.float32)
            p_shifted = p_base - shift * 0.01  # Scale down shift in prob space
            p_shifted = torch.clamp(p_shifted, min=1e-10)  # Stay positive
            p_shifted = p_shifted / p_shifted.sum(dim=-1, keepdim=True)  # Re-normalize
            l_star = torch.log(p_shifted)  # Back to logits

        # 4. Pure Mathematical Intervention (V2.0)
        final_logits = l_star
        if not torch.isfinite(final_logits).all():
            raise FloatingPointError("CaLaM control produced non-finite final logits.")

        return final_logits, lam_star

if __name__ == "__main__":
    # Unit Test: Zero-Features Dummy Validation (V2.0 Requirement)
    print("Testing pure mathematical bypass for safe context (features = zeros)...")
    base_l = torch.randn(2, 32000)
    feat_z = torch.zeros(2, 32000, 1)
    const  = torch.tensor([[0.05], [0.05]])
    ctrl = CaLaMController(device="cpu")
    final_l, _ = ctrl.control(base_l, feat_z, const)

    # Assert exact byte-alignment 100% losslessly
    assert torch.allclose(final_l, base_l, atol=1e-6), "Mathematical bypass failed!"
    print("✅ Dummy Test Passed! final_logits == base_logits when risk is 0.")
