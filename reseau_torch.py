# Module run in the PyTorch process (part 0.7): PyTorch is imported first there.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

torch.set_num_threads(2)


class ReseauPerf(nn.Module):
    '''MLP with embeddings. n_num: number of numerical variables; cardinalites: list of integers.'''

    def __init__(self, n_num, cardinalites, dim_emb=4, couches=(64, 32), dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, min(dim_emb, c)) for c in cardinalites])
        d = n_num + sum(e.embedding_dim for e in self.embs)
        blocs = []
        for h in couches:
            blocs += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        blocs.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*blocs)

    def forward(self, x_num, x_cat):
        '''x_num: (n, n_num) float32; x_cat: (n, n_cat) int64. Output: (n,).'''
        z = torch.cat([x_num] + [emb(x_cat[:, k]) for k, emb in enumerate(self.embs)], dim=1)
        return self.mlp(z).squeeze(-1)


# Continuation of the reseau_torch.py module (PyTorch process, part 0.7).
import copy


def entrainer_reseau(modele, Xn, Xc, y, Xn_val=None, Xc_val=None, y_val=None, lr=1e-3, weight_decay=1e-4,
                     epochs=100, batch_size=256, patience=None, seed=0, tol=1e-6):
    '''Trains modele in place. Xn float32 (n, p), Xc int64 (n, q), y float32 (n,): torch tensors.

    Output: dict {"train": [losses], "val": [losses]} (MSE per epoch).
    '''
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(modele.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(Xn, Xc, y), batch_size=batch_size, shuffle=True, generator=gen)
    perte = nn.MSELoss()
    hist = {"train": [], "val": []}
    meilleur, etat, attente = np.inf, None, 0
    for _ in range(epochs):
        modele.train()
        cumul = 0.0
        for xb, cb, yb in loader:
            opt.zero_grad()
            l = perte(modele(xb, cb), yb)
            l.backward()
            opt.step()
            cumul += l.item() * len(yb)
        hist["train"].append(cumul / len(y))
        if y_val is not None:
            modele.eval()
            with torch.no_grad():
                v = perte(modele(Xn_val, Xc_val), y_val).item()
            hist["val"].append(v)
            if v < meilleur - tol:
                meilleur, etat, attente = v, copy.deepcopy(modele.state_dict()), 0
            else:
                attente += 1
                if patience is not None and attente >= patience:
                    break
    if patience is not None and etat is not None:
        modele.load_state_dict(etat)
    return hist
