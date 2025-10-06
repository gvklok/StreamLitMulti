import streamlit as st
from pathlib import Path
from datetime import datetime
import random

st.set_page_config(page_title="Project2 - FashionMNIST CNN", page_icon="�", layout="wide")

ROOT = Path(__file__).parent
CNN_ROOT = ROOT / "CNN project"
RESULTS = CNN_ROOT / "results"

st.title("Project 2 - Fashion-MNIST CNN Explorer")
st.caption("Browse the project README, preview dataset images, and inspect training results and evaluation artifacts.")

with st.sidebar:
  st.header("Controls")
  sample_count = st.slider("Number of sample images", 1, 25, 8)
  random_seed = st.number_input("Random seed (0 = random)", min_value=0, value=0)
  show_results = st.checkbox("Show training results and metrics", True)
  show_readme = st.checkbox("Show project README", True)
  st.write("---")

if random_seed != 0:
  random.seed(int(random_seed))

cols = st.columns([2, 1])
with cols[0]:
  st.header("Overview")
  st.write("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
  st.write(
    "This page loads the local Fashion-MNIST dataset (if available), shows sample images and labels, and displays training/evaluation artifacts saved in `CNN project/results`.")

with cols[1]:
  st.header("Status")
  if CNN_ROOT.exists():
    st.success("Found CNN project folder")
  else:
    st.error("`CNN project` folder not found — check repository layout")

## Show README (top)
if show_readme:
  readme_path = CNN_ROOT / "README.md"
  if readme_path.exists():
    st.markdown("### Project README")
    try:
      readme_text = readme_path.read_text(encoding="utf8")
      # Remove a 'How to run' / reproducibility section for the viewer
      import re

      readme_text = re.sub(r"##\s*Reproducibility[\s\S]*?(?=##|$)", "", readme_text, flags=re.IGNORECASE)
      readme_text = re.sub(r"##\s*How to run[\s\S]*?(?=##|$)", "", readme_text, flags=re.IGNORECASE)
      st.markdown(readme_text)
    except Exception as e:
      st.error(f"Failed to read README.md: {e}")
  else:
    st.info("No README.md found in `CNN project`")

st.markdown("---")

## Try to load dataset and show previews
st.header("Dataset preview - Fashion-MNIST samples")


def _load_fashion_mnist_samples(root: Path, n: int):
  """Return list of (PIL.Image, label_index, label_name)."""
  # try torchvision first (fast when available)
  class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
  ]

  try:
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    # torchvision will look for the dataset under the provided root; try common locations
    for candidate in (root, root / "data", root / "data" / "FashionMNIST"):
      try:
        ds = datasets.FashionMNIST(root=str(candidate), train=True, download=False, transform=transform)
        # If dataset object has positive length, use it
        if len(ds) > 0:
          total = len(ds)
          indices = random.sample(range(total), k=min(n, total))
          samples = []
          for idx in indices:
            img_tensor, label = ds[idx]
            img = transforms.ToPILImage()(img_tensor)
            samples.append((img, int(label), class_names[int(label)]))
          return samples
      except Exception:
        continue
  except Exception:
    # fall through to IDX reader fallback below
    pass

  # Fallback: read raw IDX files directly (gz or uncompressed)
  import gzip
  import struct
  from PIL import Image

  data_dir = root / "data" / "FashionMNIST" / "raw"
  data_dir_alt = root / "data" / "FashionMNIST"
  # possible filenames
  img_candidates = [
    data_dir / "train-images-idx3-ubyte.gz",
    data_dir / "train-images-idx3-ubyte",
    data_dir_alt / "train-images-idx3-ubyte.gz",
    data_dir_alt / "train-images-idx3-ubyte",
    root / "data" / "train-images-idx3-ubyte.gz",
    root / "data" / "train-images-idx3-ubyte",
  ]
  lbl_candidates = [
    data_dir / "train-labels-idx1-ubyte.gz",
    data_dir / "train-labels-idx1-ubyte",
    data_dir_alt / "train-labels-idx1-ubyte.gz",
    data_dir_alt / "train-labels-idx1-ubyte",
    root / "data" / "train-labels-idx1-ubyte.gz",
    root / "data" / "train-labels-idx1-ubyte",
  ]

  img_path = next((p for p in img_candidates if p.exists()), None)
  lbl_path = next((p for p in lbl_candidates if p.exists()), None)

  if not img_path or not lbl_path:
    # no dataset available
    raise FileNotFoundError(f"Could not find Fashion-MNIST raw files. Checked locations under {root}")

  def _open_maybe_gz(p: Path):
    if str(p).endswith(".gz"):
      return gzip.open(str(p), "rb")
    return open(str(p), "rb")

  # Read labels
  with _open_maybe_gz(lbl_path) as f:
    magic, num = struct.unpack(">II", f.read(8))
    if magic != 2049:
      raise ValueError("Invalid labels IDX file")
    labels = list(f.read())

  # Read images
  with _open_maybe_gz(img_path) as f:
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    if magic != 2051:
      raise ValueError("Invalid images IDX file")
    image_data = f.read()

  total = min(len(labels), num)
  indices = random.sample(range(total), k=min(n, total))
  samples = []
  for i in indices:
    start = i * rows * cols
    pixels = image_data[start : start + rows * cols]
    img = Image.frombytes("L", (cols, rows), pixels)
    label = labels[i]
    samples.append((img, int(label), class_names[int(label)]))

  return samples

preview_cols = st.columns(4)
try:
  samples = _load_fashion_mnist_samples(CNN_ROOT, sample_count)
  for i, (img, label_idx, label_name) in enumerate(samples):
    col = preview_cols[i % 4]
    with col:
      st.image(img, width=150, caption=f"{label_idx} - {label_name}")
except ImportError:
  st.warning("torch/torchvision not installed in the environment. Install them or run the app in the `425` env used to train the model.")
  st.info("You can still view saved result images below if they exist.")
except Exception as e:
  st.error(f"Error loading dataset samples: {e}")

st.markdown("---")

## Show training results / evaluation artifacts
if show_results:
  st.header("Training results & evaluation")

  png_files = [RESULTS / "loss_pytorch.png", RESULTS / "accuracy_pytorch.png", RESULTS / "confusion_matrix.png"]
  png_files = [p for p in png_files if p.exists()]

  if png_files:
    st.subheader("Plots")
    for p in png_files:
      st.markdown(f"**{p.name}**")
      try:
        st.image(str(p), use_container_width=True)
      except Exception as e:
        st.write(f"Failed to show {p.name}: {e}")
  else:
    st.info("No PNG result plots found in `CNN project/results`")

  report_path = RESULTS / "classification_report.txt"
  if report_path.exists():
    st.subheader("Classification report")
    try:
      text = report_path.read_text(encoding="utf8")
      st.code(text)
    except Exception as e:
      st.write(f"Failed to read classification report: {e}")

  history_csv = RESULTS / "history.csv"
  if history_csv.exists():
    import pandas as pd

    st.subheader("Training history")
    try:
      df = pd.read_csv(history_csv)
      st.dataframe(df)
      with st.expander("Charts from history.csv"):
        cols = [c for c in df.columns if c.lower() in ("loss", "val_loss", "accuracy", "val_accuracy")]
        if cols:
          st.line_chart(df[cols])
        else:
          st.info("No standard loss/accuracy columns found in history.csv")
    except Exception as e:
      st.write(f"Failed to load history.csv: {e}")

  st.subheader("Other artifacts in results/")
  if RESULTS.exists():
    files = sorted(RESULTS.iterdir())
    for f in files:
      if f.is_file():
        st.write(f.name)
  else:
    st.info("No results folder found")

st.markdown("---")
st.write("If something is missing, make sure the `CNN project/results` folder contains the artifacts produced by the training run (these are displayed above when present).")

st.markdown("---")
st.header("Model report: CNN design, training & evaluation")

st.markdown("""
Short description of the dataset and recognition goal

The dataset used is Fashion-MNIST: 70,000 grayscale images of clothing items (28x28 pixels) divided into 10 classes. The CNN's goal is to classify an input 28x28 image into one of these 10 categories.

Key libraries used in the project

- torch, torchvision
- numpy
- matplotlib
- glob
- scikit-learn (sklearn)
- pandas
- PIL (Pillow)

Model initialization and example PyTorch architecture

The project uses a small PyTorch nn.Module (SimpleCNN) with three convolutional blocks and max-pooling after each block. The first conv layer example arguments:

- out_channels (filters): 32
- kernel_size: 3
- padding: 1
- activation: ReLU
- input shape: (1, 28, 28) in channels-first format

Pooling choice

Max pooling (nn.MaxPool2d) is used to downsample spatial dimensions while preserving strong local activations. This yields translation-invariant summarized features for subsequent layers.

Flattening and fully-connected layers

After the convolutional and pooling blocks the feature maps are flattened and passed through a fully-connected layer (e.g., 128 units, ReLU) and a final linear layer producing 10 logits. Training uses CrossEntropyLoss which applies softmax + log-loss.

Training, compile and evaluation settings

- Loss: CrossEntropyLoss (PyTorch)
- Optimizer: Adam (lr=1e-3)
- Metrics: accuracy
- Batch size: 64
- Epochs: 50

Outputs and plots

The training script produces `results/history.csv`, `loss_pytorch.png`, `accuracy_pytorch.png`, `confusion_matrix.png`, and `classification_report.txt` which are shown above if present.

Summary

The saved run in this repo reports strong training accuracy (~98-99%) with test/validation accuracy around ~91-92%, indicating good performance but some overfitting. Consider augmentation or regularization to improve generalization.
""")