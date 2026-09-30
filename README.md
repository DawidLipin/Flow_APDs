# Anisotropic power-diagram flow matching

This repository contains the two-dimensional flow-matching
model, synthetic data pipeline, trained checkpoint, unguided sampler, and
training-free guidance experiments.

The model was trained only on synthetic anisotropic power diagrams (APDs). No
experimental or real-world sample is present in either training dataset. The
experimental microstructure images under `examples_for_guidance/` are used only
as a qualitative final figure column; the model and guidance objectives never
load them.

## Contents

- `generate_data.py`: generate APDs with PyAPD and preprocess them into graph
  states.
- `helpers.py`: APD parameterization, normalization, graph features, losses,
  batching, and sampling utilities.
- `model.py`: the C4-equivariant graph architecture and dataset loaders.
- `train.py`: train or resume the flow-matching model.
- `sample.py`: run the unguided Euler sampler and inspect its trajectory.
- `guidance.ipynb`: reproduce the documented guidance objectives and
  experiments.
- `data_for_testing/`: normalization statistics used by the supplied model.
- `model_for_testing/`: trained weights for the reported 256-wide, three-layer,
  16-neighbour model.
- `examples_for_guidance/`: reported generated figures and the qualitative
  reference-image column, with source notes.

## Assemble the supplied checkpoint

The trained weights are stored as verified parts. Reconstruct and verify the
checkpoint before running an example:

```bash
python prepare_artifacts.py
```

This creates ignored `.pt` files beside their tracked parts. Running the command
again verifies the existing files.

## Environment

Python 3.13 was used for the supplied environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch Geometric binary packages must match the installed PyTorch and CUDA
versions. If `torch-cluster` cannot be installed directly, use the wheel index
recommended by the PyTorch Geometric installation guide for the selected
PyTorch/CUDA combination.

## Run the trained model

The supplied checkpoint is a weights-only file, so its architecture and
normalization statistics are passed explicitly:

```bash
python sample.py \
  --checkpoint model_for_testing/model_256hd_3ly_16knn_100_500_N100.pt \
  --stats data_for_testing/12k_N100_500_stats.pt \
  --hidden-dim 256 \
  --n-layers 3 \
  --k-knn 16 \
  --num-nodes 300 \
  --n-steps 60 \
  --seed 117
```

The viewer slider and left/right arrow keys move through the saved integration
frames.

## Reproduce the guidance comparison

After assembling the artifacts, launch:

```bash
python -m jupyter lab guidance.ipynb
```

Run all cells from the repository root. The notebook uses four shared seeds
`[117, 237, 377, 527]`, 300 nodes, 60 Euler steps, a 16-neighbour graph updated
at every step, and the reported 10% warm-up plus cosine ramp. It creates 20
matched trajectories: unguided plus four guided motifs for each seed. The saved
comparison is `examples_for_guidance/guidance_examples.png`.

The notebook contains only the four reported objectives: alternating build
layers, cast-slab zones, a copper-weld growth field, and heterogeneous lamellae.
The two smaller `layered_*.png` files are the reported unguided/guided
illustration of the first objective.

### Qualitative image acknowledgements

1. Unguided APD reference: M. Buze, J. Feydy, S. M. Roper, K. Sedighiani, and D. P. Bourne. Anisotropic power diagrams for polycrystal modelling: Efficient generation of curved grains via optimal transport. URL https://www.sciencedirect.com/science/article/pii/S092702562400538X.
2. 3D-printed steel: Yanis Balit, Eric Charkaluk, and Andrei Constantinescu. Digital image correlation for microstructural analysis of deformation pattern in additively manufactured 316L thin walls. URL https://linkinghub.elsevier.com/retrieve/pii/S2214860419305469.
3. Cast slab schematic: Robert E. Reed-Hill. Physical Metallurgy Principles. D. Van Nostrand Company, New York, 2nd edition, 1973.
4. Copper weld: Kati Savolainen, Tapio Saukkonen, and Hannu Hänninen. Localization of plastic deformation in copper canisters for spent nuclear fuel. URL https://www.scirp.net/journal/paperinformation?paperid=16567.
5. Heterogeneous lamella titanium: Xiaolei Wu, Muxin Yang, Fuping Yuan, Guilin Wu, Yujie Wei, Xiaoxu Huang, and Yuntian Zhu. Heterogeneous lamella structure unites ultrafine-grain strength with coarse-grain ductility. URL https://www.pnas.org/doi/abs/10.1073/pnas.1517193112.

## Recreate the full-scale synthetic training set

The training set contains 12,000 synthetic APDs with a uniformly
sampled number of generators between 100 and 500. This bounded-memory command
generates raw shards, computes normalization statistics, and creates an on-disk
dataset:

```bash
python generate_data.py \
  --num_images 12000 \
  --N_range 100 500 \
  --seed 0 \
  --shard-size 250 \
  --preprocess \
  --output-dir data/training
```

Train the supplied architecture for 100 epochs with:

```bash
python train.py \
  --data data/training/12000_N100_500_data_norm_on_disk \
  --stats data/training/12000_N100_500_stats.pt \
  --output-dir models \
  --epochs 100 \
  --hidden-dim 256 \
  --n-layers 3 \
  --k-knn 16 \
  --seed 0
```

On CUDA, add `--amp` and optionally `--max-nodes-per-batch`. The default
optimizer is AdamW. Checkpoints created by `train.py` embed the normalization
tensors and only portable source filenames, not absolute user paths.

`--resume` continues from a full checkpoint produced by `train.py`.

## Reproducibility and scope

`--seed` seeds Python, NumPy, and PyTorch. CPU runs are reproducible; exact CUDA
reproducibility can still depend on hardware and kernels. Sampling starts from
the same Gaussian position, anisotropy, and weight base used during training.
