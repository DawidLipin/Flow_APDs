import argparse
import contextlib
import os
from pathlib import Path

import PyAPD
import torch

from helpers import (
    compute_normalisation_stats,
    normalise_sample,
    preprocess_data,
    seed_everything,
)
from model import GraphDataset, OnDiskGraphDataset


def _dataset_stem(num_images, N, N_range):
    if N_range is None:
        return f"{num_images}_N{N}"
    return f"{num_images}_N{N_range[0]}_{N_range[1]}"


def _generate_samples(num_images, N, N_range, max_attempts):
    D = 2
    tg_var = 0.8
    data_samples = []

    with open(os.devnull, "w") as f:
        with contextlib.redirect_stdout(f):
            for _ in range(num_images):
                if N_range is not None:
                    N = torch.randint(
                        low=N_range[0], high=N_range[1] + 1, size=[]
                    ).item()

                optimal = False
                attempts = 0
                while not optimal:
                    attempts += 1
                    if attempts > max_attempts:
                        raise RuntimeError(
                            f"PyAPD did not find an optimal {N}-cell system "
                            f"after {max_attempts} attempts"
                        )

                    tg = torch.randn(N) * tg_var
                    tg = torch.softmax(tg, dim=0)

                    apd_sys = PyAPD.apd_system(
                        N=N,
                        D=D,
                        target_masses=tg,
                        det_constraint=True,
                        pixel_size_prefactor=2,
                        error_tolerance=0.02,
                    )

                    apd_sys.assemble_pixels()
                    apd_sys.find_optimal_W(verbose=False)
                    apd_sys.check_optimality()
                    optimal = apd_sys.optimality
                    if not apd_sys.optimality:
                        continue
                    else:
                        # Lloyd relaxation may mutate these tensors in place.
                        X_initial = apd_sys.X.clone()
                        W_initial = apd_sys.W.clone()
                        apd_sys.Lloyds_algorithm(verbosity_level=0)
                        apd_sys.find_optimal_W(verbose=False)
                        apd_sys.check_optimality()
                        optimal = apd_sys.optimality

                X = apd_sys.X
                As = apd_sys.As
                W = apd_sys.W
                out = ((X_initial, As, W_initial), (X, As, W))
                data_samples.append(out)

    return data_samples


def generate_data(
    num_images=100,
    N=5,
    N_range=None,
    output_dir="data/generated",
    max_attempts=100,
    seed=None,
):
    seed_everything(seed)
    data_samples = _generate_samples(
        num_images=num_images,
        N=N,
        N_range=N_range,
        max_attempts=max_attempts,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _dataset_stem(num_images, N, N_range)
    torch.save(data_samples, output_dir / f"{stem}_samples.pt")

    return data_samples


def generate_data_sharded(
    num_images,
    shard_size,
    N=5,
    N_range=None,
    output_dir="data/generated",
    max_attempts=100,
    shard_index=None,
    seed=None,
):
    """Generate bounded-memory raw shards instead of one in-memory dataset."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _dataset_stem(num_images, N, N_range)
    shard_dir = output_dir / f"{stem}_raw_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_bounds = [
        (index, start, min(start + shard_size, num_images))
        for index, start in enumerate(range(0, num_images, shard_size))
    ]
    if shard_index is not None:
        if not 0 <= shard_index < len(shard_bounds):
            raise ValueError(
                f"shard_index must be in [0, {len(shard_bounds) - 1}]"
            )
        shard_bounds = [shard_bounds[shard_index]]

    shard_paths = []
    for current_shard_index, start, end in shard_bounds:
        shard_seed = (
            seed + current_shard_index if seed is not None else None
        )
        seed_everything(shard_seed)
        shard_path = shard_dir / f"samples_{start:08d}_{end:08d}.pt"
        if shard_path.exists():
            raise FileExistsError(f"Raw shard already exists: {shard_path}")
        samples = _generate_samples(
            num_images=end - start,
            N=N,
            N_range=N_range,
            max_attempts=max_attempts,
        )
        torch.save(samples, shard_path)
        shard_paths.append(shard_path)
        print(f"Saved raw shard {shard_path} ({end}/{num_images})")

    return shard_paths


def collect_raw_shards(
    num_images,
    shard_size,
    N=5,
    N_range=None,
    output_dir="data/generated",
):
    """Return all expected raw shards, failing if an array job is incomplete."""
    output_dir = Path(output_dir)
    stem = _dataset_stem(num_images, N, N_range)
    shard_dir = output_dir / f"{stem}_raw_shards"
    expected = [
        shard_dir / f"samples_{start:08d}_{min(start + shard_size, num_images):08d}.pt"
        for start in range(0, num_images, shard_size)
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} of {len(expected)} raw shards; "
            f"first missing shard: {missing[0]}"
        )
    return expected

def process_data_for_training(
    data_samples,
    num_images,
    N,
    N_range=None,
    output_dir="data/generated",
    on_disk=False,
):

    graph_list = []
    for noise, sample in data_samples:
        x_noise, A_noise, w_noise = noise
        x, A, w = sample

        add_d_noise = preprocess_data(x_noise, A_noise, w_noise)
        add_d = preprocess_data(x, A, w)

        graph_list.append([add_d_noise, add_d])

    # Normalise
    target_samples = [sample for _, sample in graph_list]
    stats = compute_normalisation_stats(target_samples)
    graph_list_norm_noise = [normalise_sample(noise, stats) for noise, _ in graph_list]
    graph_list_norm = [normalise_sample(sample, stats) for _, sample in graph_list]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _dataset_stem(num_images, N, N_range)
    torch.save(stats, output_dir / f"{stem}_stats.pt")
    torch.save(graph_list_norm_noise, output_dir / f"{stem}_norm_noise.pt")
    if on_disk:
        dataset_path = output_dir / f"{stem}_data_norm_on_disk"
        dataset = OnDiskGraphDataset(dataset_path)
        dataset.append_graphs(graph_list_norm)
        dataset.close()
    else:
        dataset_path = output_dir / f"{stem}_data_norm.pt"
        GraphDataset.save_graphs(graph_list_norm, dataset_path)

    return graph_list_norm


def _compute_stats_from_raw_shards(shard_paths):
    node_count = 0
    ani_sum = torch.zeros(2, dtype=torch.float64)
    ani_square_sum = torch.zeros((), dtype=torch.float64)
    w_sum = torch.zeros((), dtype=torch.float64)
    w_square_sum = torch.zeros((), dtype=torch.float64)

    for shard_path in shard_paths:
        samples = torch.load(shard_path, map_location="cpu", weights_only=False)
        for _, sample in samples:
            x, A, w = sample
            processed = preprocess_data(x, A, w)
            ani = processed["ani"].to(torch.float64)
            weights = processed["w"].reshape(-1).to(torch.float64)
            node_count += ani.shape[0]
            ani_sum += ani.sum(dim=0)
            ani_square_sum += ani.square().sum()
            w_sum += weights.sum()
            w_square_sum += weights.square().sum()

    if node_count == 0:
        raise ValueError("Cannot compute statistics from empty shards")

    ani_mean = ani_sum / node_count
    centered_square_sum = ani_square_sum - ani_sum.square().sum() / node_count
    ani_variance = (centered_square_sum / node_count / 2.0).clamp_min(0.0)
    ani_scale = torch.sqrt(ani_variance).clamp_min(1e-8)
    w_mean = w_sum / node_count
    w_variance = (w_square_sum / node_count - w_mean.square()).clamp_min(0.0)
    w_std = torch.sqrt(w_variance).clamp_min(1e-8)

    return {
        "x_shift": torch.tensor([0.5, 0.5]),
        "x_scale": torch.tensor([2.0, 2.0]),
        "ani_mean": ani_mean.to(torch.float32),
        "ani_scale": ani_scale.to(torch.float32),
        "w_mean": w_mean.to(torch.float32),
        "w_std": w_std.to(torch.float32),
    }


def process_sharded_data_for_training(
    shard_paths,
    num_images,
    N,
    N_range=None,
    output_dir="data/generated",
    save_noise=False,
):
    """Normalize raw shards into one SQLite-backed training dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _dataset_stem(num_images, N, N_range)
    stats = _compute_stats_from_raw_shards(shard_paths)
    torch.save(stats, output_dir / f"{stem}_stats.pt")

    dataset_path = output_dir / f"{stem}_data_norm_on_disk"
    dataset = OnDiskGraphDataset(dataset_path)
    if len(dataset) != 0:
        raise FileExistsError(f"On-disk dataset is not empty: {dataset_path}")

    for shard_index, shard_path in enumerate(shard_paths):
        samples = torch.load(shard_path, map_location="cpu", weights_only=False)
        target_graphs = []
        noise_graphs = []
        for noise, sample in samples:
            x, A, w = sample
            target_graphs.append(
                normalise_sample(preprocess_data(x, A, w), stats)
            )
            if save_noise:
                x_noise, A_noise, w_noise = noise
                noise_graphs.append(
                    normalise_sample(
                        preprocess_data(x_noise, A_noise, w_noise), stats
                    )
                )

        dataset.extend_graphs(target_graphs)
        if save_noise:
            noise_path = output_dir / (
                f"{stem}_norm_noise_shard_{shard_index:05d}.pt"
            )
            GraphDataset.save_graphs(noise_graphs, noise_path)
        print(
            f"Processed shard {shard_index + 1}/{len(shard_paths)} "
            f"({len(dataset)} graphs total)"
        )

    dataset.close()
    return dataset_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--N_range", nargs=2, type=int, default=None)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed Python, NumPy, and PyTorch random number generators",
    )
    parser.add_argument(
        "--on-disk",
        action="store_true",
        help="Store normalized targets in a SQLite-backed PyG dataset",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Generate bounded-memory raw shards; implies on-disk preprocessing",
    )
    parser.add_argument(
        "--save-sharded-noise",
        action="store_true",
        help="Also save normalized initial-state shards",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Generate only this zero-based shard, for job arrays",
    )
    parser.add_argument(
        "--process-existing-shards",
        action="store_true",
        help="Skip generation and preprocess all expected raw shards",
    )

    args = parser.parse_args()

    uses_existing_shards = (
        args.shard_index is not None or args.process_existing_shards
    )
    if uses_existing_shards and args.shard_size is None:
        parser.error("--shard-index and --process-existing-shards require --shard-size")
    if args.shard_index is not None and args.preprocess:
        parser.error(
            "do not preprocess an individual shard; after all array jobs finish, "
            "use --process-existing-shards --preprocess"
        )

    if args.shard_size is None:
        data_samples = generate_data(
            num_images=args.num_images,
            N=args.N,
            N_range=args.N_range,
            output_dir=args.output_dir,
            max_attempts=args.max_attempts,
            seed=args.seed,
        )

        if args.preprocess:
            process_data_for_training(
                data_samples,
                num_images=args.num_images,
                N=args.N,
                N_range=args.N_range,
                output_dir=args.output_dir,
                on_disk=args.on_disk,
            )
    else:
        if args.process_existing_shards:
            shard_paths = collect_raw_shards(
                num_images=args.num_images,
                shard_size=args.shard_size,
                N=args.N,
                N_range=args.N_range,
                output_dir=args.output_dir,
            )
        else:
            shard_paths = generate_data_sharded(
                num_images=args.num_images,
                shard_size=args.shard_size,
                N=args.N,
                N_range=args.N_range,
                output_dir=args.output_dir,
                max_attempts=args.max_attempts,
                shard_index=args.shard_index,
                seed=args.seed,
            )
        if args.preprocess:
            process_sharded_data_for_training(
                shard_paths,
                num_images=args.num_images,
                N=args.N,
                N_range=args.N_range,
                output_dir=args.output_dir,
                save_noise=args.save_sharded_noise,
            )
