import math
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset, OnDiskDataset
from torch_geometric.utils import softmax, scatter

from helpers import build_ellipse_graph_features

class GraphDataset(InMemoryDataset):
    """Collated in-memory APD graphs, with legacy list-file support."""

    def __init__(self, source):
        super().__init__()

        if isinstance(source, (str, Path)):
            source = Path(source)
            payload = torch.load(source, map_location="cpu", weights_only=False)
            if isinstance(payload, tuple) and len(payload) in (2, 3):
                self.load(str(source))
                return
            graph_list = payload
        else:
            graph_list = source

        data_list = [self._to_data(graph) for graph in graph_list]
        self.data, self.slices = self.collate(data_list)

    @staticmethod
    def _to_data(graph):
        if isinstance(graph, Data):
            return graph
        return Data(x=graph["x"], ani=graph["ani"], w=graph["w"])

    @classmethod
    def save_graphs(cls, graph_list, path):
        """Save graphs in PyG's compact collated representation."""
        cls.save([cls._to_data(graph) for graph in graph_list], str(path))


class OnDiskGraphDataset(OnDiskDataset):
    """SQLite-backed APD graphs for datasets that do not fit in RAM."""

    schema = {
        "x": dict(dtype=torch.float32, size=(-1, 2)),
        "ani": dict(dtype=torch.float32, size=(-1, 2)),
        "w": dict(dtype=torch.float32, size=(-1, 1)),
    }

    def __init__(self, root):
        super().__init__(root=str(root), backend="sqlite", schema=self.schema)

    def process(self):
        # Accessing the database creates the empty SQLite file on first use.
        _ = self.db

    def serialize(self, data):
        return data.to_dict()

    def deserialize(self, data):
        return Data.from_dict(data)

    def append_graphs(self, graph_list, chunk_size=1000):
        if len(self) != 0:
            raise FileExistsError(f"On-disk dataset is not empty: {self.root}")
        self.extend_graphs(graph_list, chunk_size=chunk_size)

    def extend_graphs(self, graph_list, chunk_size=1000):
        for start in range(0, len(graph_list), chunk_size):
            chunk = graph_list[start:start + chunk_size]
            self.extend([GraphDataset._to_data(graph) for graph in chunk])


def load_graph_dataset(path):
    """Load either a collated/legacy .pt file or an on-disk dataset directory."""
    path = Path(path)
    if path.is_dir():
        return OnDiskGraphDataset(path)
    return GraphDataset(path)


def load_model_checkpoint(path, map_location="cpu"):
    """Return model weights and optional metadata from new or legacy files."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], payload
    return payload, None

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: shape [batch] or [batch, 1], values typically in [0, 1]
        returns: shape [batch, dim]
        """
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)

        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb_freq = torch.exp(
            torch.arange(half_dim, device=t.device) * -emb_scale
        )
        emb = t[:, None] * emb_freq[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb

class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t):
        return self.mlp(self.sinusoidal(t))


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim + edge_feat_dim, 3*hidden_dim),
            nn.SiLU(),
            nn.Linear(3*hidden_dim, 2*hidden_dim),
            nn.SiLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.time_emb = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )


    def forward(self, h_edge, edge_feats, t_edge, edge_index):
        """
        h_edge:             (E,hidden_dim)
        edge_feats:         (E,F)
        t_node:             (E,hidden_dim)
        edge_index:         (2,E)
        """

        src, dst = edge_index

        # Retained for compatibility with existing checkpoints. The layer-level
        # time MLP is currently unused; time is injected before the output head.
        att_scr = self.attention(torch.cat([h_edge, edge_feats], dim=-1))
        attention_weights_evec = softmax(att_scr, dst)  # [E, 1]
        weigh_h_edge = h_edge * attention_weights_evec
        num_nodes = int(edge_index.max()) + 1
        h_node = scatter(weigh_h_edge, dst, dim=0, dim_size=num_nodes, reduce="sum")

        # feature update
        h_nodei, h_nodej = h_node[src], h_node[dst]
        h_edge_new = torch.cat([h_nodei, h_nodej, edge_feats], dim=-1)
        h_edge_new = self.edge_mlp(h_edge_new)

        return h_edge_new


class FlowEGNN(nn.Module):
    def __init__(self, edge_feat_dim=15, hidden_dim=128, edge_out_dim=11, n_layers=2):
        super().__init__()

        self.time_emb = TimeEmbedding(2*hidden_dim, hidden_dim)

        self.edge_in_proj = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_feat_dim)
            for _ in range(n_layers)
        ])

        self.out_edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim+edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_out_dim)
        )

        self.time_emb_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, data, time, edge_index):
        """
        data: dictionary with:
            x:      (N,2)
            ani:    (N,2)
            w:      (N,1)
            batch:  (N,)
            ptr:    (B,)
        time:       (B,1), one time value per graph
        edge_index: (2,E)
        """

        src, dst = edge_index
        batching = data["batch"]
        N = data["x"].size(0)

        feats = build_ellipse_graph_features(data, edge_index)
        edge_feats = feats["edge"]["edge_scalar"]
        h_edge = self.edge_in_proj(edge_feats)

        t_emb_graph = self.time_emb(time)         # [B, hidden_dim]
        edge_batch = batching[dst]                # [E]
        t_edge = t_emb_graph[edge_batch]          # [E, hidden_dim]

        h_edge_new = h_edge
        for layer in self.layers:
            h_edge_new = layer(h_edge_new, edge_feats, t_edge, edge_index)

        # Add time and residual connection
        out_t_emb = self.time_emb_out(t_edge)
        h_edge_new = h_edge_new + out_t_emb
        fin_edge = torch.cat([h_edge_new, edge_feats], dim=-1)

        # Calculate attention weighting
        att_scr = self.attention(fin_edge)
        attention_weights_evec = softmax(att_scr, dst)  # [E, 1]

        # Calculate final scaling per edge
        alpha1, alpha2, alpha3, alpha4, beta1, beta2, beta3, beta4, beta5, beta6, w_edge =  torch.split(self.out_edge_mlp(fin_edge), 1, dim=-1)

        edge_feats_full = feats["edge"]

        # dx update
        xi = edge_feats_full["xi"]
        d_ij = edge_feats_full["d_ij"]
        Ai_dij = edge_feats_full["Ai_dij"]
        Aj_dij = edge_feats_full["Aj_dij"]
        coord_update_edge = ( (d_ij * alpha1) + (xi * alpha2) + (Ai_dij * alpha3) + (Aj_dij * alpha4) ) * attention_weights_evec
        dx = scatter(coord_update_edge, dst, dim=0, dim_size=N, reduce="sum")

        # dani update
        anii = edge_feats_full["anii"]
        anij = edge_feats_full["anij"]
        disp_ij = edge_feats_full["disp_ij"]
        anii_perp = edge_feats_full["anii_perp"]
        anij_perp = edge_feats_full["anij_perp"]
        disp_ij_perp = edge_feats_full["disp_ij_perp"]
        ani_update_edge = ( (beta1 * anii) + (beta2 * anij) + (beta3 * disp_ij) + (beta4 * anii_perp) + (beta5 * anij_perp) + (beta6 * disp_ij_perp) ) * attention_weights_evec
        dani = scatter(ani_update_edge, dst, dim=0, dim_size=N, reduce="sum")

        # dw update
        w_update_edge = w_edge * attention_weights_evec
        dw = scatter(w_update_edge, dst, dim=0, dim_size=N, reduce="sum")

        out = {
            "x": dx,
            "ani": dani,
            "w": dw,
        }

        return out
