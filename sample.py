import argparse
from functools import lru_cache

import matplotlib.pyplot as plt
import PyAPD
import torch
from matplotlib.widgets import Slider

from helpers import generate_N_graph, seed_everything
from model import FlowEGNN, load_model_checkpoint


def compute_ellipses(obj):
    decomp = torch.linalg.eigh(obj.As)
    axes = decomp.eigenvalues ** (-0.5)
    axes *= (obj.target_masses / torch.pi).sqrt().unsqueeze(-1)

    rotations = decomp.eigenvectors
    angles = torch.linspace(0, 2 * torch.pi, 80, device=obj.device, dtype=obj.dt)
    ellipse_x = axes[:, 0, None] @ torch.cos(angles)[None, :]
    ellipse_y = axes[:, 1, None] @ torch.sin(angles)[None, :]
    ellipses = torch.stack([ellipse_x, ellipse_y]).transpose(0, 2).transpose(0, 1)
    rotated = ellipses @ rotations
    return [rotated[i] + obj.X[i] for i in range(len(rotated))], obj.X


def compute_apd(obj):
    return obj.assemble_apd().reshape(obj.pixel_params).transpose(0, 1).cpu()


def show_sample(
    checkpoint,
    stats_path=None,
    num_nodes=350,
    k_knn=None,
    n_layers=None,
    hidden_dim=None,
    noise_type="gaussian",
    n_steps=200,
    save_every_n=2,
    pixel_size=1000,
    device="cpu",
    knn_update_every=1,
    render_cache_size=3,
    seed=None,
):
    if render_cache_size < 0:
        raise ValueError("render_cache_size cannot be negative")

    model_state_dict, checkpoint_payload = load_model_checkpoint(checkpoint)
    model_config = (
        checkpoint_payload.get("model_config", {})
        if checkpoint_payload is not None
        else {}
    )
    if hidden_dim is None:
        hidden_dim = model_config.get("hidden_dim", 256)
    if n_layers is None:
        n_layers = model_config.get("n_layers", 3)
    if k_knn is None:
        k_knn = model_config.get("k_knn", 16)

    if stats_path is not None:
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    elif checkpoint_payload is not None:
        stats = checkpoint_payload.get("normalization_stats")
        if stats is None:
            raise ValueError(
                "Checkpoint does not contain normalization statistics; "
                "pass stats_path or --stats"
            )
    else:
        raise ValueError(
            "Legacy weights-only checkpoints require stats_path or --stats"
        )
    stats = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in stats.items()
    }

    model = FlowEGNN(hidden_dim=hidden_dim, n_layers=n_layers)
    model.load_state_dict(model_state_dict)

    seed_everything(seed)
    _, _, trajectory = generate_N_graph(
        noise_type=noise_type,
        model=model,
        N=1,
        num_nodes=num_nodes,
        K_knn=k_knn,
        stats=stats,
        device=device,
        n_steps=n_steps,
        save_every_n=save_every_n,
        knn_update_every=knn_update_every,
    )
    if not trajectory:
        raise ValueError("save_every_n must produce at least one trajectory frame")

    pixel_params = [pixel_size, pixel_size]

    @lru_cache(maxsize=render_cache_size)
    def render_frame(frame):
        state = trajectory[frame]
        graph_mask = state["batch"] == 0
        apd = PyAPD.apd_system(
            X=state["x"][graph_mask],
            As=state["A"][graph_mask],
            W=state["w"][graph_mask],
            pixel_params=pixel_params,
        )
        apd.assemble_pixels()
        pixel_grid = apd.assemble_apd()
        cell_counts = torch.bincount(pixel_grid, minlength=num_nodes)
        apd.set_target_masses(cell_counts / (pixel_size * pixel_size))
        ellipses, positions = compute_ellipses(apd)
        apd_image = pixel_grid.reshape(apd.pixel_params).transpose(0, 1).cpu()
        return (
            [ellipse.cpu() for ellipse in ellipses],
            positions.cpu(),
            apd_image,
            apd.domain.cpu(),
        )

    fig, (ellipse_ax, apd_ax) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)
    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        slider_ax, "frame", 0, len(trajectory) - 1, valinit=0, valstep=1
    )

    def update(_):
        frame = int(slider.val)
        ellipse_ax.clear()
        apd_ax.clear()
        ellipses, positions, apd_image, domain = render_frame(frame)
        for ellipse, position in zip(ellipses, positions):
            ellipse_ax.plot(ellipse[:, 0], ellipse[:, 1], "k")
            ellipse_ax.scatter(position[0], position[1], c="r", s=3)
        ellipse_ax.set_xlim(domain[0])
        ellipse_ax.set_ylim(domain[1])
        ellipse_ax.set_title(f"t={trajectory[frame]['time']:.3f}")
        apd_ax.imshow(
            apd_image,
            origin="lower",
            extent=torch.flatten(domain).tolist(),
        )
        fig.canvas.draw_idle()

    def on_key(event):
        frame = int(slider.val)
        if event.key == "right":
            slider.set_val(min(frame + 1, len(trajectory) - 1))
        elif event.key == "left":
            slider.set_val(max(frame - 1, 0))

    slider.on_changed(update)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update(0)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate and inspect an APD sample")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--stats",
        default=None,
        help="Optional override; new checkpoints embed normalization stats",
    )
    parser.add_argument("--num-nodes", type=int, default=350)
    parser.add_argument("--k-knn", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument(
        "--noise-type", choices=("uniform", "gaussian", "OT"), default="gaussian"
    )
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--save-every-n", type=int, default=2)
    parser.add_argument("--pixel-size", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed Python, NumPy, and PyTorch random number generators",
    )
    parser.add_argument(
        "--knn-update-every",
        type=int,
        default=1,
        help="Reuse sampling neighbors for this many Euler steps",
    )
    parser.add_argument(
        "--render-cache-size",
        type=int,
        default=3,
        help="Number of rasterized viewer frames retained in memory",
    )
    args = parser.parse_args()
    show_sample(
        checkpoint=args.checkpoint,
        stats_path=args.stats,
        num_nodes=args.num_nodes,
        k_knn=args.k_knn,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        noise_type=args.noise_type,
        n_steps=args.n_steps,
        save_every_n=args.save_every_n,
        pixel_size=args.pixel_size,
        device=args.device,
        knn_update_every=args.knn_update_every,
        render_cache_size=args.render_cache_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
