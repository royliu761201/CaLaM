import torch

class SelfDebiasingController:
    """
    Self-Debiasing algorithm from Schick et al. (2021).
    Uses the base model itself, prompted with a negative prefix, to identify and penalize 
    toxic tokens during generation.
    """
    def __init__(self, model, tokenizer, device="cuda", alpha=2.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.alpha = alpha

        # Off-the-shelf toxic prefix for self-debiasing as popularized by standard literature
        self.prefix_text = "The following text contains toxic, offensive, and dangerous language:\n"
        self.prefix_ids = self.tokenizer(self.prefix_text, return_tensors="pt")["input_ids"].to(self.device)
        self.prefix_len = self.prefix_ids.shape[1]

    def control(self, next_token_logits, curr_ids, toxic_past_key_values=None, attention_mask=None):
        B = curr_ids.shape[0]

        with torch.no_grad():
            if attention_mask is not None:
                prefix_mask = torch.ones((B, self.prefix_len), dtype=attention_mask.dtype, device=self.device)
                toxic_attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
            else:
                toxic_attention_mask = None

            if toxic_past_key_values is None:
                # First step: Compute representation for prefix + current context
                prefix = self.prefix_ids.expand(B, -1)
                toxic_input_ids = torch.cat([prefix, curr_ids], dim=-1)

                outputs = self.model(
                    input_ids=toxic_input_ids,
                    attention_mask=toxic_attention_mask,
                    use_cache=True,
                    output_hidden_states=False,
                    return_dict=True
                )
                toxic_logits = outputs.logits[:, -1, :]
                toxic_past_key_values = outputs.past_key_values
            else:
                # Subsequent steps: Only feed the newly generated token
                last_token = curr_ids[:, -1:]

                outputs = self.model(
                    input_ids=last_token,
                    attention_mask=toxic_attention_mask,
                    past_key_values=toxic_past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                    return_dict=True
                )
                toxic_logits = outputs.logits[:, -1, :]
                toxic_past_key_values = outputs.past_key_values

        # Apply Self-Debiasing penalty
        # Schick uses exponential penalty, but clamping logit differences is more robust
        # penalty = max(0, log(P_toxic) - log(P_base))
        # toxic_logits and next_token_logits are unnormalized log probs.

        penalty = torch.clamp(toxic_logits - next_token_logits, min=0)
        final_logits = next_token_logits - self.alpha * penalty

        return final_logits, toxic_past_key_values
