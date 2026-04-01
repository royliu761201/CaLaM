import numpy as np
from scipy.optimize import minimize

def kl_projection(p, f, b, lambda_init=None):
    """
    Solves min KL(q || p) s.t. E_q[f] <= b
    q(v) propto p(v) * exp(-lambda^T f(v))
    """
    v_size = p.shape[0]
    m_size = f.shape[1] # dimension of constraints

    if lambda_init is None:
        lambda_init = np.zeros(m_size)

    def dual_objective(lam):
        # lam must be >= 0
        lam = np.maximum(lam, 0)
        # log Z(lam) = log sum p(v) exp(-lam^T f(v))
        # Use log-sum-exp for stability
        logits = -f @ lam
        max_logit = np.max(logits)
        log_z = max_logit + np.log(np.sum(p * np.exp(logits - max_logit)))
        return log_z + lam @ b

    res = minimize(dual_objective, lambda_init, bounds=[(0, None)] * m_size)
    lam_opt = np.maximum(res.x, 0)

    # Compute resulting distribution
    logits = -f @ lam_opt
    max_logit = np.max(logits)
    unnorm_q = p * np.exp(logits - max_logit)
    q_opt = unnorm_q / np.sum(unnorm_q)

    return q_opt, lam_opt

def logit_shift_control(logits, f, lam):
    """
    q = softmax(logits - lam^T f)
    """
    return logits - f @ lam

class CaLaMController:
    def __init__(self, risk_model=None):
        self.risk_model = risk_model

    def get_alpha(self, risk_score):
        # Monotone controller: maps risk to intervention strength
        # Simple sigmoid-like mapping for the toy example
        return 1 / (1 + np.exp(-10 * (risk_score - 0.5)))

    def step(self, p, f, b, risk_score):
        q_star, lam = kl_projection(p, f, b)
        alpha = self.get_alpha(risk_score)

        # Logit mixing
        logits_p = np.log(p + 1e-12)
        logits_q = np.log(q_star + 1e-12)

        mixed_logits = (1 - alpha) * logits_p + alpha * logits_q
        q_final = np.exp(mixed_logits - np.max(mixed_logits))
        q_final /= np.sum(q_final)

        return q_final, alpha
