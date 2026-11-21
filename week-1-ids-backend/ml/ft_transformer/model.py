import torch
from torch import nn


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num,
        cardinalities,
        n_outputs,
        d_token=64,
        n_blocks=2,
        n_heads=4,
        ff_mult=2,
        dropout=0.1,
    ):
        super().__init__()
        self.n_num = n_num
        self.cardinalities = list(cardinalities)
        self.n_outputs = n_outputs
        self.d_token = d_token
        self.config = {
            "n_num": n_num,
            "cardinalities": self.cardinalities,
            "n_outputs": n_outputs,
            "d_token": d_token,
            "n_blocks": n_blocks,
            "n_heads": n_heads,
            "ff_mult": ff_mult,
            "dropout": dropout,
        }
        self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
        self.num_bias = nn.Parameter(torch.zeros(n_num, d_token))
        nn.init.normal_(self.num_weight, std=0.02)
        total_cat = sum(self.cardinalities)
        self.cat_embedding = nn.Embedding(max(total_cat, 1), d_token)
        offsets = [0]
        for card in self.cardinalities[:-1]:
            offsets.append(offsets[-1] + card)
        self.register_buffer("cat_offsets", torch.tensor(offsets, dtype=torch.long))
        self.cat_bias = nn.Parameter(torch.zeros(len(self.cardinalities), d_token))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, n_outputs))

    def tokenize(self, x_num, x_cat):
        tokens = []
        if self.n_num:
            tokens.append(x_num.unsqueeze(-1) * self.num_weight + self.num_bias)
        if self.cardinalities:
            tokens.append(self.cat_embedding(x_cat + self.cat_offsets) + self.cat_bias)
        batch = x_num.shape[0] if self.n_num else x_cat.shape[0]
        tokens.append(self.cls.expand(batch, -1, -1))
        return torch.cat(tokens, dim=1)

    def forward(self, x_num, x_cat):
        z = self.encoder(self.tokenize(x_num, x_cat))
        return self.head(z[:, -1])


def build_model(prep, **kwargs):
    return FTTransformer(len(prep.numeric_cols), prep.cardinalities, prep.n_outputs, **kwargs)


def fit(model, prep, tensors, epochs=10, lr=1e-3, batch_size=256, seed=42):
    torch.manual_seed(seed)
    x_num, x_cat, y = tensors["train"]
    vx_num, vx_cat, vy = tensors["val"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss() if prep.task == "classification" else nn.MSELoss()
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(x_num.shape[0])
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad()
            out = model(x_num[idx], x_cat[idx])
            loss = loss_fn(out, y[idx]) if prep.task == "classification" else loss_fn(out.squeeze(-1), y[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(vx_num, vx_cat)
            if prep.task == "classification":
                val_loss = loss_fn(out, vy).item()
                val_metric = (out.argmax(dim=1) == vy).float().mean().item()
            else:
                val_loss = loss_fn(out.squeeze(-1), vy).item()
                val_metric = val_loss
        history.append({"epoch": epoch + 1, "val_loss": val_loss, "val_metric": val_metric})
    return history


def save_model(model, path):
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = FTTransformer(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
