
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ToxicityFeatureExtractor(nn.Module):
    """
    Extracts toxicity features f_t(v) for the next token v.
    In a full implementation, this might look ahead or use a value function.
    For this prototype, we use a simplified approach:
    1. Keyword matching (very fast)
    2. or a small classifier on the sequence prefix + v (slow, but accurate)
    """
    def __init__(self, tokenizer=None, device='cpu', model_name="unitary/unbiased-toxic-roberta"):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.toxic_ids = set()

        # 1. Define Bad Words List (Canonical "Seven Dirty Words" + extensions)
        # In a real paper, we use the "profanity_list" from RTP dataset.
        self.toxic_keywords = [
            "bad", "evil", "kill", "murder", "hate", "stupid", "idiot", 
            "fuck", "shit", "damn", "bitch", "crap", "sex", "rape", "death",
            "racist", "slur", "nazi", "hitler", "kill", "die", "ugly"
        ]

        # 2. Pre-compute Toxic Token IDs
        if self.tokenizer is not None:
            self._register_toxic_tokens()

    def _register_toxic_tokens(self):
        """Maps keywords to token IDs."""
        for word in self.toxic_keywords:
            # Note: space prefix is important for BPE
            for w in [word, " " + word, word.capitalize(), " " + word.capitalize()]:
                ids = self.tokenizer.encode(w, add_special_tokens=False)
                # We only care about single-token matches for now to be fast
                # Multi-token phrases require n-gram lookahead (future work)
                if len(ids) == 1:
                    self.toxic_ids.add(ids[0])

        # Convert to tensor for fast lookup
        # We create a dense vector 
        self.vocab_size = self.tokenizer.vocab_size
        self.toxic_mask = torch.zeros(self.vocab_size, device=self.device)
        self.toxic_mask[list(self.toxic_ids)] = 1.0

    def forward(self, input_ids: torch.Tensor, next_token_logits: torch.Tensor) -> torch.Tensor:
        """
        Returns features (B, V, m).
        For m=1 (toxicity), returns 1.0 if token is in toxic list, 0.0 otherwise.
        """
        B, V = next_token_logits.shape

        # Ensure mask is on correct device/size
        # Fail Early: tokenizer is strictly required, cannot be None.
        if self.tokenizer is None:
            raise RuntimeError(
                "[FAIL-FAST] ToxicityFeatureExtractor requires a tokenizer. "
                "Pass tokenizer at init time."
            )

        if not hasattr(self, 'toxic_mask'):
            raise RuntimeError("toxic_mask was not initialized.")
        if self.toxic_mask.device != next_token_logits.device:
             self.toxic_mask = self.toxic_mask.to(next_token_logits.device)

        # Expand mask: (V,) -> (B, V, 1)
        # Note: If V differs (e.g. aligned vocab), we might need slicing.
        # Assuming standard vocab.

        mask = self.toxic_mask
        if mask.numel() < V:
            import torch.nn.functional as F
            mask = F.pad(mask, (0, V - mask.numel()), value=0.0)
        elif mask.numel() > V:
            mask = mask[:V]
        features = mask.view(1, -1, 1).expand(B, V, 1)

        return features

class SemanticFeatureExtractor(nn.Module):
    """
    Extracts semantic features based on embedding similarity to a concept direction.
    Implements a "WEAT" (Word Embedding Association Test) style or "RepE" style direction.
    Direction = Mean(Toxic_Embeddings) - Mean(Safe_Embeddings).
    """
    def __init__(self, embedding_matrix: torch.Tensor, tokenizer, device='cpu'):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer

        # We need the embedding matrix to compute the similarity for *all* vocabulary items.
        # Shape: (Vocab, Hidden)
        self.embedding_matrix = embedding_matrix.detach().to(device)
        self.embedding_matrix.requires_grad_(False)
        # Normalize for cosine similarity: E_norm = E / |E|
        self.embedding_norm = F.normalize(self.embedding_matrix, p=2, dim=1)

        # Task-Specific Definitions (The "Direction" of Harm)
        # Source: LDNOOBW (List of Dirty, Naughty, Obscene, and Otherwise Bad Words)
        # — the canonical blocklist cited by Gehman et al. (2020) "RealToxicityPrompts"
        # Safe seeds: common prosocial & neutral words for contrastive direction
        self.seed_sets = {
            "toxicity": {
                "bad": [
                    # Profanity/Obscenity (LDNOOBW core)
                    "fuck", "shit", "ass", "bitch", "damn", "crap", "bastard",
                    "slut", "whore", "piss", "cock", "dick", "pussy",
                    # Violence/Harm
                    "kill", "murder", "rape", "attack", "stab", "shoot", "bomb",
                    "destroy", "torture", "abuse", "assault",
                    # Hate/Discrimination
                    "hate", "racist", "nazi", "slur", "bigot", "sexist",
                    # Derogation
                    "stupid", "idiot", "moron", "ugly", "loser", "pathetic",
                    "worthless", "disgusting", "horrible", "evil",
                    # Threat
                    "threat", "harm", "danger", "die", "death", "suicide",
                ],
                "good": [
                    # Prosocial
                    "good", "kind", "safe", "love", "happy", "friend", "help",
                    "protect", "peace", "respect", "trust", "care", "gentle",
                    "honest", "fair", "brave", "grateful", "polite", "generous",
                    # Neutral/Constructive
                    "learn", "create", "build", "grow", "share", "support",
                    "understand", "improve", "discover", "achieve",
                ]
            },
            "truthfulness": {
                "bad": [
                    "false", "lie", "deception", "fake", "wrong", "error",
                    "hallucination", "myth", "fiction", "misleading", "hoax",
                    "fabricate", "distort", "deny", "conspiracy",
                ],
                "good": [
                    "true", "fact", "correct", "real", "accurate", "proven",
                    "verified", "logic", "reality", "evidence", "honest",
                    "data", "science", "research", "valid",
                ]
            },
            "safety": {
                "bad": [
                    "danger", "unsafe", "illegal", "harmful", "risk", "attack",
                    "weapon", "bomb", "fraud", "exploit", "hack", "steal",
                    "poison", "arson", "smuggle",
                ],
                "good": [
                    "safe", "legal", "ethical", "harmless", "secure", "defend",
                    "shield", "law", "help", "protect", "guard", "comply",
                    "regulate", "prevent", "authorize",
                ]
            }
        }

        # [FAIL-FAST] Initialize default mode matrix
        self.set_mode("toxicity")

    def set_mode(self, mode: str = "toxicity"):
        """Updates the concept direction based on the task mode."""
        # Fail Early: Unknown mode is a configuration error, silent fallback not allowed.
        if mode not in self.seed_sets:
            raise ValueError(
                f"[FAIL-FAST] Unknown feature mode '{mode}'. "
                f"Valid modes: {list(self.seed_sets.keys())}"
            )

        seeds = self.seed_sets[mode]
        self.toxic_seeds = seeds["bad"]
        self.safe_seeds = seeds["good"]

        # Recompute direction for the new mode
        self.concept_direction = self._compute_direction()

        # [P1 PERF] Pre-compute vocab similarity — eliminates matmul from every forward() call
        raw_sim = torch.matmul(self.embedding_norm, self.concept_direction.unsqueeze(1)).squeeze(1)
        self._cached_vocab_sim = torch.relu(raw_sim)  # (V_emb,) with ReLU pre-applied
        print(f"✅ CaLaM Geometric Direction updated for mode: {mode} "
              f"(cached vocab_sim: {self._cached_vocab_sim.shape}, nonzero={100*(self._cached_vocab_sim > 0).float().mean():.1f}%)")

    def _compute_direction(self) -> torch.Tensor:
        """Computes the direction vector (H,) representing the concept."""
        def get_mean_embedding(words):
            indices = []
            for w in words:
                # Use first token of the word
                ids = self.tokenizer.encode(w, add_special_tokens=False)
                if ids: indices.append(ids[0])

            if not indices: return torch.zeros(self.embedding_matrix.size(1), device=self.device)

            # Look up embeddings: (N, H)
            embeds = self.embedding_matrix[torch.tensor(indices, device=self.device)]
            return torch.mean(embeds, dim=0) # (H,)

        toxic_vec = get_mean_embedding(self.toxic_seeds)
        safe_vec = get_mean_embedding(self.safe_seeds)

        direction = toxic_vec - safe_vec
        # Normalize direction
        return F.normalize(direction.unsqueeze(0), p=2, dim=1).squeeze(0) # (H,)

    def forward(self, input_ids: torch.Tensor, next_token_logits: torch.Tensor) -> torch.Tensor:
        """
        Returns features (B, V, m).
        [P1 PERF] Uses pre-cached vocab_sim from set_mode() — zero matmul cost per step.
        """
        B, V = next_token_logits.shape

        # Use cached similarity (computed once in set_mode)
        vocab_sim = self._cached_vocab_sim  # (V_emb,) — already ReLU'd

        # Handle vocab size mismatch
        V_emb = vocab_sim.size(0)
        if V_emb < V:
            vocab_sim = F.pad(vocab_sim, (0, V - V_emb), value=0.0)
        elif V_emb > V:
            vocab_sim = vocab_sim[:V]

        # Reshape to (B, V, 1)
        features = vocab_sim.view(1, -1, 1).expand(B, V, 1)
        return features

class DynamicContextFeatureExtractor(nn.Module):
    """
    V2.0 Core: Context-aware dynamic feature extraction f_t(v).
    Maps context h_t and token embedding E_v to a dynamic [0,1] risk scalar.
    f_t(v) = sigmoid(h_t * W_risk * E_v^T)
    """
    def __init__(
        self,
        hidden_size: int,
        embedding_matrix: torch.Tensor,
        device='cpu',
        logit_clamp: float = 30.0,
        projection_clamp: float = 256.0,
    ):
        super().__init__()
        if embedding_matrix.dim() != 2:
            raise ValueError(
                f"DynamicContextFeatureExtractor expects embedding_matrix with shape (V, H), got {tuple(embedding_matrix.shape)}."
            )
        if not torch.isfinite(embedding_matrix).all():
            raise ValueError("DynamicContextFeatureExtractor received non-finite embedding weights.")
        self.device = device
        self.embedding_matrix = embedding_matrix.detach().to(device)  # (V, H)
        self.embedding_matrix.requires_grad_(False)
        self.vocab_size = int(self.embedding_matrix.size(0))
        self.W_risk = nn.Linear(hidden_size, hidden_size, bias=False).to(device).to(self.embedding_matrix.dtype)
        self.logit_clamp = float(logit_clamp)
        self.projection_clamp = float(projection_clamp)

    def _project_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 2:
            raise ValueError(
                f"DynamicContextFeatureExtractor expected hidden_states with shape (B, H), got {tuple(hidden_states.shape)}."
            )
        if not torch.isfinite(hidden_states).all():
            raise ValueError("DynamicContextFeatureExtractor received non-finite hidden states.")
        # Align dtypes (important for bfloat16 LLMs vs float32 W_risk)
        hidden_states = hidden_states.to(self.embedding_matrix.dtype)
        projected = self.W_risk(hidden_states)
        projected = projected.clamp(min=-self.projection_clamp, max=self.projection_clamp)
        if not torch.isfinite(projected).all():
            raise FloatingPointError("DynamicContextFeatureExtractor produced non-finite projected hidden states.")
        return projected

    def _bounded_sigmoid(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clamp(min=-self.logit_clamp, max=self.logit_clamp)
        features = torch.sigmoid(logits)
        if not torch.isfinite(features).all():
            raise FloatingPointError("DynamicContextFeatureExtractor produced non-finite features.")
        return features

    def score_tokens(self, hidden_states: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Scores the observed next tokens used during offline training.
        Args:
            hidden_states: (B, H)
            token_ids: (B,)
        Returns:
            scores: (B,)
        """
        if token_ids.dim() != 1:
            raise ValueError(f"Expected token_ids with shape (B,), got {tuple(token_ids.shape)}.")
        if token_ids.size(0) != hidden_states.size(0):
            raise ValueError(
                f"Batch mismatch between hidden_states ({hidden_states.size(0)}) and token_ids ({token_ids.size(0)})."
            )
        h_proj = self._project_hidden(hidden_states)
        token_ids = token_ids.to(device=self.embedding_matrix.device, dtype=torch.long)
        token_min = int(token_ids.min().item())
        token_max = int(token_ids.max().item())
        if token_min < 0 or token_max >= self.vocab_size:
            raise IndexError(
                f"token_ids out of embedding range [0, {self.vocab_size - 1}]: min={token_min} max={token_max}"
            )
        e_v = self.embedding_matrix[token_ids]  # (B, H)
        logits = (h_proj * e_v).sum(dim=-1)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("DynamicContextFeatureExtractor produced non-finite token logits.")
        return self._bounded_sigmoid(logits)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Computes the full dynamic risk matrix for the next token distribution.
        Args:
            hidden_states: (B, H) The context vector h_t
        Returns:
            features: (B, V, 1) The dynamic risk mapped for each vocabulary token
        """
        # 1. Project context: h_proj = h_t * W_risk (B, H)
        h_proj = self._project_hidden(hidden_states)

        # 2. Compute bilinear scores for all V: h_proj * E_v^T  --> (B, V)
        logits = torch.matmul(h_proj, self.embedding_matrix.T)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("DynamicContextFeatureExtractor produced non-finite vocabulary logits.")

        # 3. Squashing with bounded logits to avoid silent overflow/saturation drift
        features = self._bounded_sigmoid(logits)

        # Expand to standard feature tensor format: (B, V, 1)
        return features.unsqueeze(-1)
