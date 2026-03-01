import torch
import torch.nn as nn


class HLGAN(nn.Module):
    """
    Hierarchical Local–Global Attention Network
    Improved for MNIST with normalization, dropout, and CLS token.
    """

    def __init__(
        self,
        image_size=28,
        patch_size=7,
        d_model=96,        # reduced for MNIST
        num_heads=3,
        num_classes=10,
        dropout=0.2
    ):
        super().__init__()

        assert image_size % patch_size == 0, "Image size must be divisible by patch size"

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model

        # ---- Patch Embedding ----
        self.patch_embedding = nn.Linear(
            patch_size * patch_size,
            d_model
        )

        # ---- CLS Token ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # ---- Positional Embedding (including CLS) ----
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model)
        )

        # ---- Local Self-Attention ----
        self.local_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # ---- Global Self-Attention ----
        self.global_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # ---- Normalization & Dropout ----
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # ---- Classification Head ----
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x shape: (B, 1, 28, 28)
        """
        B, C, H, W = x.shape

        # -------- Patch Extraction --------
        patches = x.unfold(2, self.patch_size, self.patch_size) \
                   .unfold(3, self.patch_size, self.patch_size)

        patches = patches.contiguous().view(
            B,
            self.num_patches,
            self.patch_size * self.patch_size
        )
        # (B, 16, 49)

        # -------- Patch Embedding --------
        tokens = self.patch_embedding(patches)
        # (B, 16, d_model)

        # -------- Add CLS Token --------
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        # (B, 17, d_model)

        # -------- Add Positional Encoding --------
        tokens = tokens + self.pos_embedding

        # -------- Local Self-Attention (Feature Extraction) --------
        N = tokens.size(1)
        radius = 1

        attn_mask = torch.full(
            (N, N),
            float('-inf'),
            device=x.device
        )

        for i in range(N):
            for j in range(max(0, i - radius), min(N, i + radius + 1)):
                attn_mask[i, j] = 0.0

        local_features, _ = self.local_attention(
            tokens,
            tokens,
            tokens,
            attn_mask=attn_mask
        )
        local_features = self.norm1(local_features)
        local_features = self.dropout(local_features)

        # -------- Global Self-Attention (Context Integration) --------
        global_features, _ = self.global_attention(
            local_features,
            local_features,
            local_features
        )
        global_features = self.norm2(global_features)
        global_features = self.dropout(global_features)

        # -------- CLS Pooling --------
        pooled = global_features[:, 0]
        # (B, d_model)

        # -------- Classification --------
        logits = self.classifier(pooled)
        # (B, 10)

        return logits


# Optional test
if __name__ == "__main__":
    model = HLGAN()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print("Output shape:", output.shape)
