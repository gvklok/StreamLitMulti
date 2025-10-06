# Fashion-MNIST CNN — Project README

This repository contains a small image-classification project for the course assignment: train a Convolutional Neural Network (CNN) to recognize items in the Fashion‑MNIST dataset. The work includes model code, training scripts (PyTorch and TensorFlow variants), saved artifacts (plots, models), and evaluation outputs (confusion matrix and classification report).

This README includes:
- a) Problem statement
- b) Algorithm / model description
- c) Analysis of findings (results, interpretation, suggested improvements)
- d) References

---

## a) Problem statement

Image classification is a canonical computer vision task: given an input image, the model should assign a class label describing the image content. In this assignment we use the Fashion‑MNIST dataset (70,000 28×28 grayscale images, 10 clothing-related classes) as the data source. The objective is to build a CNN that learns from the labeled training images and can accurately classify unseen test images.

Deliverables required by the assignment that this repo satisfies:
- Select a dataset (Fashion‑MNIST)
- Provide a short description of the dataset and recognition goal
- Implement a CNN with at least three convolutional layers and pooling between them
- Use a fully-connected layer and a softmax output
- Train for at least 50 epochs and report training/validation/test loss and accuracy
- Produce plots (loss and accuracy) and evaluate using confusion matrix and classification report
- Summarize model ability and performance

Hardware & environment note: this project was trained on a macOS machine with PyTorch + MPS (Metal) available. Training finished in ~23.4 minutes for 50 epochs using the PyTorch script.

---

## b) Algorithm of the solution (model architecture + training details)

Data preprocessing
- Dataset: Fashion‑MNIST (train: 60,000 images, test: 10,000 images)
- Images normalized to roughly zero mean / unit variance using torchvision transforms (ToTensor + Normalize).
- Training set split into train/validation with 90/10 split.

Model architecture (PyTorch `SimpleCNN` used for final run):
- Input: 28×28 grayscale image (1 channel)
- Conv1: 32 filters, kernel_size=3×3, padding=1, activation=ReLU
- MaxPool (2×2)
- Conv2: 64 filters, kernel_size=3×3, padding=1, activation=ReLU
- MaxPool (2×2)
- Conv3: 128 filters, kernel_size=3×3, padding=1, activation=ReLU
- MaxPool (2×2)
- Flatten
- Dense (fully connected): 128 units, ReLU
- Dropout: 0.4
- Output Dense: 10 units (raw logits) — training uses CrossEntropyLoss which applies softmax internally

Pooling choice
- Max pooling (2×2) is used after each convolutional block. Rationale: max pooling selects the most salient activation within local neighborhoods, introducing modest translation invariance and dramatically reducing spatial size (computational cost) while preserving strong features — appropriate for classification tasks like Fashion‑MNIST.

Training configuration
- Loss: CrossEntropyLoss (equivalent to categorical_crossentropy in Keras when labels are integer class indices)
- Optimizer: Adam (default lr=1e-3)
- Batch size: 64
- Epochs: 50 (assignment requirement)
- Checkpointing: best model saved based on validation loss; final model saved at the end
- Additional outputs: per-epoch training/validation loss & accuracy stored in `results/history.csv`

Scripts
- `fashion_mnist_pytorch.py` — main training & evaluation script (recommended). It: trains, saves best checkpoint, saves final model, writes `history.csv`, computes confusion matrix and classification report, and produces training plots in `results/`.
- `fashion_mnist_cnn.py` — alternative TensorFlow/Keras implementation (kept for reference). Note: TensorFlow had runtime/mutex issues on the development macOS environment; PyTorch/MPS was used to complete runs reliably.

How the model is evaluated
- After training (or on KeyboardInterrupt), the script loads the best checkpoint (by validation loss) and evaluates it on the held-out test set. The script computes:
	- Test loss and test accuracy
	- Per-class precision/recall/f1-score (classification report)
	- Confusion matrix (saved as an image)

---

## c) Analysis of the findings

Key numeric results (final run)
- Final training loss: 0.0347 — training accuracy ≈ 98.69%
- Final validation loss: 0.5967 — validation accuracy ≈ 91.90%
- Test loss: 0.6158 — test accuracy ≈ 91.64%
- Classification report (per-class f1 scores): overall accuracy ≈ 91.19% (report produced in `results/classification_report.txt`)
- Training time: ~23.4 minutes for 50 epochs on macOS with MPS

Interpretation
- The model achieves strong accuracy (~91–92%) on the Fashion‑MNIST test set. Training loss is very low while validation/test loss is substantially higher; this pattern indicates mild-to-moderate overfitting: the model fits the training set almost perfectly (98.7% acc), but is less confident and a little less accurate on unseen images.

Per-class observations
- The classification report shows variability across classes. Some classes (e.g., class 1, 5, 7, 8, 9) show very high precision/recall (≥0.96), while others (notably class 6) have lower recall and f1 (example: class 6 f1 ≈ 0.74). The confusion matrix (`results/confusion_matrix.png`) helps identify which classes are confused — often visually similar clothing items (e.g., pullover vs coat or shirt vs T‑shirt).

Possible improvements (next steps)
- Data augmentation: random rotations, small translations, and shifts to increase robustness and reduce overfitting.
- Regularization: stronger dropout, weight decay (L2), or reducing model capacity (fewer filters, smaller fc layer).
- Learning-rate schedule: ReduceLROnPlateau or StepLR to refine training near convergence.
- Transfer learning: using a small pretrained network (MobileNetV2) and fine-tuning the head often yields better accuracy quickly (useful if you need top performance in limited time).
- Class-specific fixes: reweighting classes or adding targeted augmentations where the confusion matrix pinpoints weaknesses.

Practical note about the loss vs accuracy behavior
- Accuracy reports the fraction of examples correctly classified. Cross-entropy loss additionally penalizes the model for being overconfident when wrong and for low-confidence correct predictions; because of that, you can see training loss nearly zero while validation loss remains higher even with high accuracy. Use both metrics together to reason about confidence and generalization.

---

## Reproducibility / How to run

1) Activate your Conda env (the code was developed in `conda` env named `425`):

```bash
conda activate 425
```

2) Install requirements if needed (if your env doesn't have PyTorch or sklearn):

```bash
pip install -r requirements.txt
# or install pytorch following https://pytorch.org/get-started/locally/ for best platform-specific build
```

3) Run the PyTorch training script (recommended):

```bash
python3 fashion_mnist_pytorch.py
```

This will:
- download Fashion‑MNIST into `data/` if not present
- run training for 50 epochs (or until you Ctrl+C)
- save the best checkpoint, final model, `results/history.csv`, `results/loss_pytorch.png`, `results/accuracy_pytorch.png`, `results/confusion_matrix.png`, and `results/classification_report.txt`.

Useful commands after a run

```bash
ls -la results
cat results/classification_report.txt
open results/confusion_matrix.png   # macOS: open the image in Preview
open results/loss_pytorch.png
open results/accuracy_pytorch.png
```

---

## d) References

- Artzi lecture slides (course materials)
- GitHub Copilot (code-generation assistance)


