import random

import numpy as np
import torch
import torch.nn.functional as F
# from torch_geometric.nn import knn_graph
from torch_cluster import knn_graph

import PyAPD


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch RNGs when a seed is provided."""
    if seed is None:
        return
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_knn_graph(x, k, batch):
    """Build kNN edges on the input device, with an MPS CPU fallback."""
    if x.device.type == "mps":
        return knn_graph(x.cpu(), k=k, batch=batch.cpu()).to(x.device)
    return knn_graph(x, k=k, batch=batch)


def spin2_perp(t):
    return torch.stack([-t[:, 1], t[:, 0]], dim=-1)

def logm_spd(A, eps=1e-8):
    """
    Safe SPD log for MPS.
    Performs eigendecomposition on CPU if needed.
    """
    A = 0.5 * (A + A.transpose(-1, -2))

    device = A.device

    # Move to CPU for stability if using MPS
    if device.type == "mps":
        A_cpu = A.cpu()
        eigvals, eigvecs = torch.linalg.eigh(A_cpu)
        eigvals = torch.clamp(eigvals, min=eps)
        log_eigvals = torch.log(eigvals)
        S_cpu = eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-1, -2)
        return S_cpu.to(device)
    else:
        eigvals, eigvecs = torch.linalg.eigh(A)
        eigvals = torch.clamp(eigvals, min=eps)
        log_eigvals = torch.log(eigvals)
        return eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-1, -2)

def expm_sym(S):
    """
    S: (..., 2, 2) symmetric
    Returns:
        A = exp(S) SPD
    """
    S = 0.5 * (S + S.transpose(-1, -2))

    eigvals, eigvecs = torch.linalg.eigh(S)

    exp_eigvals = torch.exp(eigvals)

    A = eigvecs @ torch.diag_embed(exp_eigvals) @ eigvecs.transpose(-1, -2)
    return A

def decompose_symmetric_2x2(S):
    """
    Decompose symmetric 2x2 matrix into:
        S = mu I + T
    where T = [[a, b],
               [b,-a]]

    S: (N, 2, 2)

    Returns:
        ani: (N, 2)  where t[...,0]=a, t[...,1]=b
    """
    x = S[..., 0, 0]
    y = S[..., 0, 1]

    ani = torch.stack([x, y], dim=-1)

    return ani

def reconstruct_symmetric_2x2(ani):
    """
    Reconstruct SPD matrix from (mu, ani) via:
        S = muI + T

    ani: (N, 2)

    Returns:
        A: (N, 2, 2)
    """
    a = ani[..., 0]
    b = ani[..., 1]

    S = torch.zeros((*ani.shape[:-1], 2, 2), device=ani.device)

    S[..., 0, 0] = a
    S[..., 1, 1] = -a
    S[..., 0, 1] = b
    S[..., 1, 0] = b

    return S

def preprocess_data(x, A, w):
    """
    Convert (x, A, v) into irreducible representation channels
    suitable for equivariant flow matching.

    Inputs:
        x: (N, 2)
        A: (N, 2, 2) SPD
        w: (N, 1) or (N)

    Returns:
        dict with:
            x: (N, 2)
            ani: (N, 2)
            w: (N, 1)
    """
    if w.dim() == 1:
        w = w.unsqueeze(-1)

    S = logm_spd(A)
    ani = decompose_symmetric_2x2(S)

    return {
        "x": x,
        "ani": ani,
        "w": w
    }

def compute_normalisation_stats(data):
    """
    data: list of dicts with keys:
      x:    (N,2)
      ani:  (N,2)   traceless part (a,b)
      w:    (N,) or (N,1)   scalar

    returns:
      stats dict
    """
    all_ani = []
    all_w = []

    for sample in data:
        ani = sample["ani"].float()                 # (N,2)
        w = sample["w"].reshape(-1).float()     # (N,)

        all_ani.append(ani)
        all_w.append(w)

    all_ani = torch.cat(all_ani, dim=0)   # (total_nodes, 2)
    all_w = torch.cat(all_w, dim=0)   # (total_nodes,)

    # traceless part t=(a,b): mean vector + one scalar scale
    ani_mean = all_ani.mean(dim=0)        # (2,)
    ani_centered = all_ani - ani_mean
    ani_scale = torch.sqrt((ani_centered.pow(2).sum(dim=1).mean()) / 2.0).clamp_min(1e-8)

    # scalar w
    w_mean = all_w.mean()
    w_std = all_w.std(unbiased=False).clamp_min(1e-8)

    device = ani_mean.device

    stats = {
        "x_shift": torch.tensor([0.5, 0.5], device=device),
        "x_scale": torch.tensor([2.0, 2.0], device=device),

        "ani_mean": ani_mean,     # shape (2,)
        "ani_scale": ani_scale,   # scalar

        "w_mean": w_mean,
        "w_std": w_std,
    }

    return stats

def normalise_sample(sample, stats):

    x = sample["x"].float()
    ani = sample["ani"].float()
    w = sample["w"].float()

    x_norm = (x - stats["x_shift"]) * stats["x_scale"]

    ani_norm = ani/ stats["ani_scale"] # You shouldn't subtract mean from this or you lose rotation equivariance

    w_norm = (w - stats["w_mean"]) / stats["w_std"]

    return {
        "x": x_norm,
        "ani": ani_norm,
        "w": w_norm,
    }

def denormalise_sample(sample, stats):
    x = sample["x"]
    ani = sample["ani"]
    w = sample["w"]

    x_denorm = x / stats["x_scale"] + stats["x_shift"]

    ani_denorm = ani * stats["ani_scale"]

    w_denorm = w * stats["w_std"] + stats["w_mean"]

    return {
        "x": x_denorm,
        "ani": ani_denorm,
        "w": w_denorm,
    }

def anisotropy_apply_to_vector(ani, vec):
    """
    Apply anisotropy tensor A = [[u, v], [v, -u]] to vec.

    ani: (E, 2) or (N, 2)
    vec: (E, 2) or (N, 2)

    Returns:
        A vec: same shape as vec
    """
    u = ani[:, 0:1]
    v = ani[:, 1:2]
    x = vec[:, 0:1]
    y = vec[:, 1:2]

    out = torch.cat(
        [
            u * x + v * y,
            v * x - u * y,
        ],
        dim=-1,
    )
    return out

def spin2_from_vector(vec):
    """
    Given vec = (x, y), return spin-2 quantity:
        (x^2 - y^2, 2xy)

    vec: (..., 2)
    returns: (..., 2)
    """
    x = vec[..., 0:1]
    y = vec[..., 1:2]
    return torch.cat([x * x - y * y, 2.0 * x * y], dim=-1)

def dot2(a, b):
    """
    Dot product for (...,2) tensors, kept as (...,1)
    """
    return (a * b).sum(dim=-1, keepdim=True)

def cross2(a, b):
    """
    2D scalar cross product:
        a_x b_y - a_y b_x

    Returns (...,1)
    """
    return a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1]



# ============================================================
# Feature builders
# ============================================================

def build_ellipse_graph_features(feats: dict, edge_index):
    """
    Build pairwise geometric features.

    Inputs:
        feats: dict with keys
            x:          (N,2)
            mu:         (N,1)
            ani:        (N,2)
            v:          (N,1)
            w:          (N,1)
        edge_index: (2,E)

    Convention:
        edge_index = [src, dst]
        d_ij = x_j - x_i = x[dst] - x[src]

    Returns:
        dict with node and edge features.
    """

    x = feats["x"]
    ani = feats["ani"]
    w_node = feats["w"]

    src = edge_index[0].long()
    dst = edge_index[1].long()

    # -------------------------
    # Node-level basic
    # -------------------------
    ani_norm_sq = (ani ** 2).sum(dim=-1, keepdim=True)

    #-------------------------
    # Node level used for egde features
    #-------------------------
    x2 = (x[:, 0:1] ** 2)
    y2 = (x[:, 1:2] ** 2)
    x4 = (x[:, 0:1] ** 4)
    y4 = (x[:, 1:2] ** 4)
    x_prod = x2 * y2
    x2_sum = x2 + y2
    x4_sum = x4 + y4

    # -------------------------
    # Gather endpoints
    # -------------------------
    xi = x[src]
    xj = x[dst]

    ti = ani[src]
    tj = ani[dst]

    wi = w_node[src]
    wj = w_node[dst]

    # -------------------------
    # Relative displacement
    # -------------------------
    d_ij = xj - xi
    dist2 = dot2(d_ij, d_ij)

    # spin-2 basis from displacement
    disp_ij = spin2_from_vector(d_ij)
    disp_ij_perp = spin2_perp(disp_ij)

    # -------------------------
    # Scalar edge invariants
    # -------------------------
    ti_norm_sq = dot2(ti, ti)
    tj_norm_sq = dot2(tj, tj)
    ti_dot_tj = dot2(ti, tj)
    ti_cross_tj = cross2(ti, tj)

    ti_dot_dispij = dot2(ti, disp_ij) # ti and tj are spin-2 so need to compare to spin-2 version of dij
    tj_dot_dispij = dot2(tj, disp_ij)

    xi_prod = x_prod[src]
    x2i_sum = x2_sum[src]
    x4i_sum = x4_sum[src]
    xj_prod = x_prod[dst]
    x2j_sum = x2_sum[dst]
    x4j_sum = x4_sum[dst]

    # -------------------------
    # Vector bases
    # -------------------------

    ti_perp = spin2_perp(ti)
    tj_perp = spin2_perp(tj)

    Ai_dij = anisotropy_apply_to_vector(ti, d_ij)
    Aj_dij = anisotropy_apply_to_vector(tj, d_ij)

    edge_scalar = torch.cat(
        [
            dist2,
            wi, wj,
            ti_norm_sq, tj_norm_sq,
            ti_dot_tj,
            ti_cross_tj,
            ti_dot_dispij,
            tj_dot_dispij,
            xi_prod, x2i_sum, x4i_sum,
            xj_prod, x2j_sum, x4j_sum,
        ],
        dim=-1,
    )  # (E, 15)

    return {
        "node": {
            "x": x,
            "ani": ani,
            "ani_norm_sq": ani_norm_sq,
        },
        "edge": {
            "src": src,
            "dst": dst,
            "xi": xi,
            "xj": xj,
            "d_ij": d_ij,
            "Ai_dij": Ai_dij,
            "Aj_dij": Aj_dij,
            "anii": ti,
            "anij": tj,
            "anii_perp": ti_perp,
            "anij_perp": tj_perp,
            "disp_ij": disp_ij,
            "disp_ij_perp": disp_ij_perp,
            "dist2": dist2,
            "w_i": wi,
            "w_j": wj,
            "anii_norm_sq": ti_norm_sq,
            "anij_norm_sq": tj_norm_sq,
            "anii_dot_anij": ti_dot_tj,
            "anii_cross_anij": ti_cross_tj,
            "anii_dot_dispij": ti_dot_dispij,
            "anij_dot_dispij": tj_dot_dispij,
            "edge_scalar": edge_scalar,
        },
    }

def sample_base_like(data, device=None):
    """
    data is a dict with normalised tensors:
      x: (N,2)
      ani: (N,2)
      w: (N,1)

    returns z0 with same shapes
    """
    x1 = data["x"]
    ani1 = data["ani"]
    w1 = data["w"]

    if device is None:
        device = x1.device

    # x0 = torch.rand_like(x1, device=device) * 2 - 1
    x0 = torch.randn_like(x1, device=device)
    ani0 = torch.randn_like(ani1, device=device)
    w0 = torch.randn_like(w1, device=device)


    return {
        "x": x0,
        "ani": ani0,
        "w": w0,
        "batch": data["batch"],
        "ptr": data["ptr"],
    }

def interpolate_states(z0, z1, t):
    """
    z0, z1 are dicts of tensors
    t: (B,1) tensor in [0,1]

    returns z_tau and target velocity u
    """
    x_t = (1 - t) * z0["x"] + t * z1["x"]
    ani_t = (1 - t) * z0["ani"] + t * z1["ani"]
    w_t = (1 - t) * z0["w"] + t * z1["w"]

    u = {
        "x": z1["x"] - z0["x"],
        "ani": z1["ani"] - z0["ani"],
        "w": z1["w"] - z0["w"],
        "batch": z1["batch"],
        "ptr": z1["ptr"],
    }

    z_t = {
        "x": x_t,
        "ani": ani_t,
        "w": w_t,
        "batch": z1["batch"],
        "ptr": z1["ptr"],
    }

    return z_t, u

def flow_matching_loss(pred, target, weights=None):
    if weights is None:
        weights = {"x": 1.0, "ani": 1.0, "w": 1.0}

    loss_x = F.mse_loss(pred["x"], target["x"])
    loss_ani = F.mse_loss(pred["ani"], target["ani"])
    loss_w = F.mse_loss(pred["w"], target["w"])

    loss = (
        weights["x"] * loss_x +
        weights["ani"] * loss_ani +
        weights["w"] * loss_w
    )

    metrics = {
        "loss": loss.detach(),
        "loss_x": loss_x.detach(),
        "loss_ani": loss_ani.detach(),
        "loss_w": loss_w.detach(),
    }

    return loss, metrics

def finish_optimizer_step(
    model,
    optimiser,
    scaler=None,
    gradient_multiplier=1.0,
):
    """Unscale, optionally renormalise, clip, and apply accumulated gradients."""
    if scaler is not None:
        scaler.unscale_(optimiser)

    if gradient_multiplier != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_multiplier)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if scaler is None:
        optimiser.step()
    else:
        scaler.step(optimiser)
        scaler.update()
    optimiser.zero_grad(set_to_none=True)


def training_step(
    model,
    batch,
    optimiser,
    device,
    K_knn,
    *,
    loss_divisor=1,
    scaler=None,
    amp_dtype=None,
    step_optimizer=True,
    zero_grad=True,
    non_blocking=False,
):

    data = batch

    model.train()
    if zero_grad:
        optimiser.zero_grad(set_to_none=True)

    batch = {
        k: v.to(device, non_blocking=non_blocking)
        for k, v in data.items()
    }

    z1 = batch
    z0 = sample_base_like(z1, device=device)

    B = z1["ptr"].shape[0] - 1
    t_graph = torch.rand(B, 1, device=device)      # [B, 1]
    t_node = t_graph[z1["batch"]]                # [N, 1]

    z_t, u_target = interpolate_states(z0, z1, t_node)

    edge_index = build_knn_graph(z_t["x"], k=K_knn, batch=z1["batch"])

    device_type = torch.device(device).type
    with torch.autocast(
        device_type=device_type,
        dtype=amp_dtype,
        enabled=amp_dtype is not None,
    ):
        pred = model(z_t, t_graph, edge_index)
        weights = {"x": 1.0, "ani": 1.0, "w": 1.0}
        loss, metrics = flow_matching_loss(pred, u_target, weights)

    backward_loss = loss / loss_divisor
    if scaler is None:
        backward_loss.backward()
    else:
        scaler.scale(backward_loss).backward()

    if step_optimizer:
        finish_optimizer_step(model, optimiser, scaler=scaler)

    return metrics

@torch.no_grad()
def sample_flow(
    model,
    device,
    N,
    num_nodes,
    K_knn,
    n_steps=100,
    save_every_n=None,
    noise_type="gaussian",
    knn_update_every=1,
):
    """
    Sample from FlowEGNN by Euler integration from tau=0 to tau=1.

    Parameters
    ----------
    model : FlowEGNN
    edge_index : LongTensor, shape (2, E)
    batching : LongTensor, shape (N,)
        Node-to-graph assignment.
    num_nodes : int
        Total number of nodes in the batch.
    device : torch.device
    n_steps : int
    stats : dict or None
        Normalisation stats. If provided, initial noise is drawn in normalised space.
    x_init, t_init, s_init, v_init : optional tensors
        Custom initial state. If None, standard Gaussian noise is used.

    Returns
    -------
    feats : dict
        Sample in normalised space.
    """
    model.eval()

    if num_nodes <= K_knn:
        raise ValueError("num_nodes must be greater than K_knn")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if save_every_n is not None and save_every_n <= 0:
        raise ValueError("save_every_n must be positive when provided")
    if knn_update_every <= 0:
        raise ValueError("knn_update_every must be positive")

    if noise_type is None or noise_type == "gaussian":
        x0 = torch.randn(N*num_nodes, 2, device=device)
    elif noise_type == "uniform":
        x0 = torch.rand(N*num_nodes, 2, device=device) * 2 - 1
    elif noise_type == "OT":
        x0 = PyAPD.sample_seeds_with_exclusion(
            n=N*num_nodes, dim=2, radius_prefactor=0.01
        ).to(device) * 2 - 1
    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}")
    ani0 = torch.randn(N*num_nodes, 2, device=device)
    w0 = torch.randn(N*num_nodes, 1, device=device)

    batch_idx = torch.arange(N).repeat_interleave(num_nodes).to(device)
    ptr = torch.arange(0, (N + 1) * num_nodes, step=num_nodes, device=device)
    edge_index = build_knn_graph(x0, k=K_knn, batch=batch_idx)

    z0 = {
        "x": x0,
        "ani": ani0,
        "w": w0,
        "batch": batch_idx,
        "ptr": ptr,
        "edge_index": edge_index,
    }

    zt = {
        "x": x0,
        "ani": ani0,
        "w": w0,
        "batch": batch_idx,
        "ptr": ptr,
        "edge_index": edge_index,
    }

    dt = 1.0 / n_steps
    zt_all = []

    for k in range(0, n_steps):

        t_graph = torch.full(
            (N, 1),
            fill_value=k / n_steps,
            device=device,
            dtype=x0.dtype
        )

        edge_index = zt["edge_index"]
        out = model(zt, t_graph, edge_index)

        x_new = zt["x"] + dt * out["x"]
        ani_new = zt["ani"] + dt * out["ani"]
        zt_new = zt["w"] + dt * out["w"]

        step = k + 1
        if step < n_steps and step % knn_update_every == 0:
            next_edge_index = build_knn_graph(
                x_new, k=K_knn, batch=batch_idx
            )
        else:
            next_edge_index = edge_index

        zt = {
            "x": x_new,
            "ani": ani_new,
            "w": zt_new,
            "batch": zt["batch"],
            "ptr": zt["ptr"],
            "edge_index": next_edge_index,
        }

        if save_every_n is not None:
            if step % save_every_n == 0 or step == n_steps:
                zt_all.append([zt, step / n_steps])

    return zt, z0, zt_all

@torch.no_grad()
def generate_N_graph(
    model,
    N,
    num_nodes,
    K_knn,
    stats,
    device,
    n_steps=200,
    save_every_n=None,
    noise_type="gaussian",
    knn_update_every=1,
):
    model.eval()
    model.to(device)

    # Sample in normalised space
    z1, z0, zt_all = sample_flow(
        model=model,
        device=device,
        N=N,
        num_nodes=num_nodes,
        K_knn=K_knn,
        n_steps=n_steps,
        save_every_n=save_every_n,
        noise_type=noise_type,
        knn_update_every=knn_update_every,
    )

    batch = z1["batch"]
    ptr = z1["ptr"]

    z1 = denormalise_sample(z1, stats)
    A_1 = reconstruct_symmetric_2x2(z1["ani"])
    A_1 = expm_sym(A_1)

    z0 = denormalise_sample(z0, stats)
    A_0 = reconstruct_symmetric_2x2(z0["ani"])
    A_0 = expm_sym(A_0)

    zt_out_list = []
    for zt in zt_all:
        zt, time = zt
        zt = denormalise_sample(zt, stats)
        A_t = reconstruct_symmetric_2x2(zt["ani"])
        A_t = expm_sym(A_t)
        zt_out = {
            "x": zt["x"],
            "A": A_t,
            "w": zt["w"],
            "ani": zt["ani"],
            "batch": batch,
            "ptr": ptr,
            "time": time,
        }
        zt_out_list.append(zt_out)

    z0_out = {
        "x": z0["x"],
        "A": A_0,
        "w": z0["w"],
        "ani": z0["ani"],
        "batch": batch,
        "ptr": ptr,
    }
    z1_out = {
        "x": z1["x"],
        "A": A_1,
        "w": z1["w"],
        "ani": z1["ani"],
        "batch": batch,
        "ptr": ptr,
    }

    return z1_out, z0_out, zt_out_list
