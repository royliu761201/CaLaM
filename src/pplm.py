import torch
import torch.nn as nn

class PPLMController:
    """
    Surrogate PPLM for large-scale efficient decoding.
    Instead of updating the full past_key_values (prohibitively slow for 14B models),
    we compute the gradient of the toxicity risk w.r.t the current hidden state,
    and then project this gradient back into the logit space using the LM head embeddings.
    """
    def __init__(self, risk_model, embedding_matrix, device="cuda", step_size=0.02, num_iter=3):
        self.risk_model = risk_model
        self.embedding_matrix = embedding_matrix # lm_head weights [vocab_size, hidden_size]
        self.device = device
        self.step_size = step_size
        self.num_iter = num_iter

        # Ensure correct shape of embedding matrix
        if self.embedding_matrix.shape[0] < self.embedding_matrix.shape[1]:
            self.W_out = self.embedding_matrix.t()
        else:
            self.W_out = self.embedding_matrix

        self.W_out_t = self.W_out.t().contiguous()

    def control(self, next_token_logits, last_hidden):
        """
        Steers the logits away from toxicity based on the risk model gradient.
        next_token_logits: [B, V]
        last_hidden: [B, H]
        """
        B, H = last_hidden.shape

        with torch.enable_grad():
            curr_hidden = last_hidden.clone().detach().requires_grad_(True)

            for _ in range(self.num_iter):
                risk = self.risk_model(curr_hidden) # [B, 1], higher means more toxic
                loss = risk.sum()

                grad = torch.autograd.grad(loss, curr_hidden)[0] # [B, H]

                # We want to MINIMIZE risk, so we descend the gradient
                curr_hidden = curr_hidden - self.step_size * grad

            delta_hidden = curr_hidden - last_hidden

        # Project delta back to logits
        delta_logits = torch.matmul(delta_hidden, self.W_out_t)

        return next_token_logits + delta_logits
