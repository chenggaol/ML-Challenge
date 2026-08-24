# Sensor Context Encoder Challenge

Converts a window of UCI HAR inertial-sensor data into a continuous context
embedding that a frozen SmolLM2-360M-Instruct model can read directly (via
`inputs_embeds`, never as text), and compares that approach against a normal
direct sensor classifier using the same encoder architecture.

## Setup

1. Download the [UCI Human Activity Recognition Using Smartphones dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
   and extract it so the repo root contains `UCI HAR Dataset/train/` and
   `UCI HAR Dataset/test/` (each with an `Inertial Signals/` subfolder).

2. Create and activate a virtual environment:

   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install PyTorch. If you have an NVIDIA GPU (CUDA 12.6-compatible driver):

   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```

   Plain `pip install torch` also works but installs a CPU-only build on
   Windows — training will run, just much slower, especially the
   context-embedding model.

4. Install the remaining dependencies:

   ```
   pip install transformers scikit-learn numpy matplotlib
   ```

## Reproducing the results

All three runs below use `--seed 0`. Hyperparameters (`--encoder cnn --lr 3e-4
--batch_size 16 --lr_scheduler cosine`) were chosen by a greedy sequential
sweep over the direct classifier — see [Hyperparameter sweep](#hyperparameter-sweep)
below.

**1. Direct sensor classifier**

```
python train.py --name final_direct --model direct --epochs 10 --seed 0 --encoder cnn --lr 3e-4 --batch_size 16 --lr_scheduler cosine
```

Saves `checkpoints/final_direct.pt` and prints the test macro-F1 at the end.

**2. Context-embedding model**

```
python train.py --name final_context --model context --epochs 10 --seed 0 --encoder cnn --lr 3e-4 --batch_size 16 --lr_scheduler cosine
```

Saves `checkpoints/final_context.pt` and prints the test macro-F1 at the end.

**3. Sensor-dependence check (shuffled embeddings)**

```
python test.py --name final_context --model context --encoder cnn --projector linear
```

Loads the trained context model (no retraining), reports its normal test
macro-F1, then reruns evaluation after shuffling which example's projected
embedding gets used by which — while keeping every label in place — and
reports the resulting macro-F1 and the drop between the two.

Both `train.py` runs also save loss/F1 curves to `graphs/<name>/`.

## Results

| Condition | Macro-F1 |
|---|---|
| Direct sensor classifier | 0.9301 |
| Context-embedding model | 0.9050 |
| Context model with shuffled embeddings | 0.1657 |

The shuffled-embedding F1 (0.1657) is close to the chance level for 6
balanced classes (1/6 ≈ 0.167), and the drop from 0.9050 is large — evidence
that the context model's prediction genuinely depends on the specific sensor
window it's given, not on a shortcut in the fixed prompt text.

## Hyperparameter sweep

The hyperparameters used above came from a greedy sequential sweep (one
parameter group at a time, most important first, keeping whichever value won
on **validation** macro-F1 before moving to the next group):

```
python param_sweep.py --model direct --epochs 10
```

This only sweeps the direct classifier — see the technical note for why the
context model reuses these values rather than being swept independently.

## Repo structure

```
dataset.py            # HARInertialDataset + create_dataloaders (subject-wise split, optional noise augmentation)
model_arch/
  encoder.py           # CNNEncoder / TransformerEncoder: [B,128,9] -> [B,1,embedding_dim]
  projector.py          # LinearProjector / MLPProjector: [B,1,embedding_dim] -> [B,1,960]
train.py                # models, training loop, checkpointing, plotting
test.py                 # loads a checkpoint, reports test F1, runs the shuffle check
param_sweep.py          # greedy hyperparameter sweep over train.py
checkpoints/            # saved weights, <name>.pt
graphs/                 # saved training curves, <name>/*.png
```
