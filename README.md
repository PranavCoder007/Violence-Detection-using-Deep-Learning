# Violence-Detection-using-Deep-Learning 

This repository contains a Jupyter Notebook implementing **Human Action Recognition** using four deep learning architectures:

* **ConvLSTM (Convolutional Long Short-Term Memory)**
* **LRCN (Long-term Recurrent Convolutional Network)**
* **Timesformer**
* **ViViT**
* **MViT**
* 
The project is developed using **TensorFlow/Keras** and trained on the **UCF50 Dataset** & **Real Life Violence Situations Dataset**. It demonstrates the complete workflow from dataset preparation and preprocessing to model training, evaluation, and action prediction on custom videos.


# PART I — ConvLSTM & LRCN
---

## 1. Dataset

### Dataset Used

**UCF50 Action Recognition Dataset**

The notebook automatically downloads the dataset and extracts it before training.

### Classes Used

The project uses the following five action classes:

* TaiChi
* Punch
* Nunchucks
* JumpingJack
* HorseRace

### Dataset Structure

```
UCF50/
├── TaiChi/
│   ├── video1.avi
│   ├── video2.avi
│   └── ...
├── Punch/
├── Nunchucks/
├── JumpingJack/
└── HorseRace/
```

---

## 2. Features

* Automatic dataset download and extraction
* Video frame extraction using OpenCV
* Frame resizing and normalization
* Sequence generation from videos
* Dataset visualization
* Train/Test dataset split
* ConvLSTM model implementation
* LRCN model implementation
* Early stopping during training
* Model evaluation
* Training history visualization
* Prediction on uploaded videos
* Processed video generation with predicted action labels

---

## 3. Libraries & Dependencies

Install the required libraries before running the notebook.

```bash
pip install tensorflow
pip install opencv-python
pip install numpy
pip install matplotlib
pip install moviepy
pip install scikit-learn
pip install pafy
pip install youtube_dl
pip install yt-dlp
```

Or install everything together:

```bash
pip install tensorflow opencv-python numpy matplotlib moviepy scikit-learn pafy youtube_dl yt-dlp
```

---

## 4. Project Workflow

### Step 1 – Install Dependencies

Install all required Python packages.

---

### Step 2 – Import Libraries

Import TensorFlow, NumPy, OpenCV, Matplotlib, MoviePy, and other required modules.

---

### Step 3 – Set Random Seed

Random seeds are fixed to ensure reproducible results.

---

### Step 4 – Download Dataset

The notebook downloads and extracts the **UCF50** dataset automatically.

---

### Step 5 – Visualize Dataset

Sample frames from each selected action class are displayed to verify the dataset.

---

### Step 6 – Configure Parameters

Important parameters include:

* Image Size: **64 × 64**
* Sequence Length: **20 frames**
* Dataset Directory
* Selected Action Classes

---

### Step 7 – Frame Extraction

Each video is processed by:

* Reading video frames
* Uniformly sampling frames
* Resizing frames
* Normalizing pixel values
* Creating fixed-length frame sequences

---

### Step 8 – Dataset Creation

The notebook creates:

* Feature sequences
* Corresponding labels
* Video file paths

Labels are then one-hot encoded for classification.

---

### Step 9 – Data Splitting

The dataset is divided into training and testing sets using Scikit-learn.

---

### Step 10 – Feature Visualization

The notebook visualizes the extracted feature vectors using dimensionality reduction techniques to better understand class separation.

---

### Step 11 – ConvLSTM Model

A ConvLSTM model is built using Keras for learning both spatial and temporal information directly from video frame sequences.

---

### Step 12 – LRCN Model

The LRCN architecture combines:

* Convolutional Neural Networks (CNN)
* Long Short-Term Memory (LSTM)

This architecture first extracts spatial features from each frame using CNN layers and then learns temporal dependencies using LSTM layers.

---

### Step 13 – Model Training

The models are trained using:

* Early Stopping
* Validation Split
* Adam Optimizer
* Categorical Crossentropy Loss

Training automatically stops if validation performance stops improving.

---

### Step 14 – Model Evaluation

After training, the notebook evaluates the model on the test dataset and reports:

* Test Loss
* Test Accuracy

---

### Step 15 – Training Curves

Training history is visualized using:

* Training Loss vs Validation Loss
* Training Accuracy vs Validation Accuracy

These plots help analyze model convergence and detect overfitting.

---

### Step 16 – Video Prediction

Users can upload their own videos.

The notebook:

1. Reads the uploaded video
2. Extracts frames
3. Predicts the action class
4. Writes the predicted label onto the video
5. Saves the processed output video

---

## 5. Model Architectures

### ConvLSTM

ConvLSTM extends traditional LSTM by replacing matrix multiplications with convolution operations, allowing it to capture both spatial and temporal information simultaneously.

### LRCN

LRCN combines CNN-based feature extraction with LSTM-based sequence modeling, making it well suited for human action recognition in videos.

---

## 6. Evaluation Metrics

The notebook evaluates the trained models using:

* Classification Accuracy
* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy

---

## 7. Running the Notebook

Execute the notebook sequentially:

1. Install dependencies.
2. Download and extract the dataset.
3. Run preprocessing cells.
4. Create the dataset.
5. Train the ConvLSTM model.
6. Train the LRCN model.
7. Evaluate model performance.
8. Visualize training metrics.
9. Upload a custom video.
10. Predict the action and generate the annotated output video.

---

## 8. Notes

* The notebook is designed for **Google Colab**.
* GPU acceleration is recommended for faster training.
* Ensure sufficient storage for downloading and extracting the UCF50 dataset.
* Update file paths if running outside Google Colab.
* Custom videos should contain one of the supported action classes for reliable predictions.

---

## 9. Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* MoviePy

---

## 10. Future Improvements

* Train on all 50 UCF50 classes.
* Improve performance using larger frame resolutions.
* Integrate attention mechanisms.
* Support real-time webcam-based action recognition.
* Deploy the trained model as a web application using Flask or Streamlit.

# PART II — Transformer Models
--
# Video-Based Violence Detection — TimeSformer, ViViT & MViT

This repository contains three independent Jupyter notebooks that fine-tune different video-classification architectures to detect **Violence vs. NonViolence** in short video clips. Each notebook covers data loading, model fine-tuning, evaluation, and single-video inference.

| Notebook | Architecture | Backbone / Source |
|---|---|---|
| `timesformer-v1.ipynb` | TimeSformer (HuggingFace) | `facebook/timesformer-base-finetuned-k400` |
| `vivit-final-v3.ipynb` | ViViT (HuggingFace) | `google/vivit-b-16x2-kinetics400` |
| `mvit-v1.ipynb` | MViT v2-S (TorchVision) | `torchvision.models.video.mvit_v2_s` (Kinetics-400 weights) |

All three notebooks were authored for a **Kaggle** environment (paths such as `/kaggle/input/...` and `/kaggle/working/...` appear throughout) and target the **Real Life Violence Situations Dataset**.

---

## 1. Dataset

- **Name:** Real Life Violence Situations Dataset
- **Expected structure:**
  ```
  Real Life Violence Dataset/
  ├── Violence/
  │   ├── video_001.mp4
  │   └── ...
  └── NonViolence/
      ├── video_001.mp4
      └── ...
  ```
- The notebooks read this folder structure directly to infer class labels (`label2id` / `id2label` are built from the sub-folder names: `Violence` and `NonViolence`).
- On Kaggle the dataset is mounted at:
  `/kaggle/input/real-life-violence-situations-dataset/Real Life Violence Dataset`
- If running outside Kaggle, download the dataset (e.g., from Kaggle Datasets) and update this path in each notebook.
- A 70% / 15% / 15% train / validation / test split is created with `train_test_split` (stratified on labels, `random_state=42`).

---

## 2. Libraries & Dependencies

### Common to all notebooks
- `torch`, `torchvision`, `torchaudio`
- `numpy`
- `scikit-learn` (train/test split, metrics)
- `matplotlib` (loss/accuracy/confusion-matrix plots)
- `av` (PyAV — frame-accurate video decoding)
- `opencv-python` (`cv2`) — used for video frame extraction in the MViT notebook
- `tqdm`

### TimeSformer notebook (`timesformer-v1.ipynb`)
- `transformers` (`TimesformerForVideoClassification`, `VideoMAEImageProcessor`, `Trainer`, `TrainingArguments`, `EarlyStoppingCallback`)
- `accelerate`
- `evaluate` (HuggingFace `evaluate` library — accuracy metric)
- `einops`
- `decord` (installed but optional; PyAV is used for decoding)
- `pandas`

### ViViT notebook (`vivit-final-v3.ipynb`)
- `transformers` (`VivitConfig`, `VivitForVideoClassification`, `Trainer`, `TrainingArguments`, `EarlyStoppingCallback`, `TrainerCallback`)
- `accelerate`
- `evaluate`
- `einops`
- `decord`
- `pandas`
- `torchvision.transforms` (custom augmentation pipeline: random resized crop, flip, color jitter, rotation)

### MViT notebook (`mvit-v1.ipynb`)
- `torchvision.models.video` (`mvit_v2_s`, `MViT_V2_S_Weights`) — no `transformers` dependency, pure PyTorch/TorchVision
- `cv2` (OpenCV) for frame extraction
- `scikit-learn` (`f1_score`, `accuracy_score`, `precision_recall_fscore_support`, `confusion_matrix`, `classification_report`)

### Install command (covers TimeSformer / ViViT notebooks)
```bash
pip install transformers accelerate einops opencv-python decord torch torchvision torchaudio matplotlib tqdm scikit-learn pandas av evaluate
pip install --upgrade transformers
```

### Install command (covers MViT notebook)
```bash
pip install torch torchvision opencv-python scikit-learn matplotlib tqdm
```

> **GPU:** All three notebooks assume CUDA availability (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`) and use mixed precision (`fp16=True`) for the HuggingFace `Trainer` runs. A GPU with at least 16GB VRAM is recommended for batch sizes used in the notebooks.

---

## 3. Step-Wise Execution

### A. TimeSformer (`timesformer-v1.ipynb`)
1. **Install dependencies** (Cell 1–2): installs/upgrades `transformers` and supporting libraries.
2. **Imports** (Cell 3): loads `torch`, `av`, `TimesformerForVideoClassification`, `VideoMAEImageProcessor`, `Trainer`, etc.
3. **Helper functions & Dataset class** (Cell 4):
   - `sample_frame_indices` — uniformly/randomly samples frame indices from a video.
   - `read_video_pyav` — decodes selected frames with PyAV.
   - `get_video_paths_and_labels` — walks the dataset folder and builds `(video_paths, labels, label2id, id2label)`.
   - `VideoDataset` — a `torch.utils.data.Dataset` that samples 8 frames per clip and runs them through the HuggingFace image processor.
4. **Model initialization & training config** (Cell 5):
   - `initialise_model()` loads `TimesformerForVideoClassification` from `facebook/timesformer-base-finetuned-k400`, replacing the classification head for 2 labels (`NonViolence`, `Violence`).
   - Defines accuracy metric via `evaluate.load("accuracy")`.
   - Configures `TrainingArguments` (3 epochs, batch size 4/2, cosine LR schedule, fp16, early stopping on accuracy).
5. **Data prep + training run** (Cell 6):
   - Builds video paths/labels, splits into train/val/test, instantiates `VideoDataset` objects.
   - Initializes `Trainer` with `EarlyStoppingCallback` and calls `trainer.train()`.
6. **Evaluation & plotting** (Cell 7):
   - Plots training/validation loss and validation accuracy from `trainer.state.log_history`.
   - Runs `trainer.predict()` on the held-out test set and prints a `classification_report`.
   - Saves the fine-tuned model with `trainer.save_model(...)`.
7. **Label maps** (Cell 8): re-declares `label2id` / `id2label` for downstream inference cells.
8. **Batch inference over a directory** (Cell 9): loads a saved model checkpoint and iterates over a validation video folder, running predictions on each `.mp4/.avi/.mov` file.
9. **Single-video inference** (Cell 10): loads the fine-tuned model from `/kaggle/working/...`, processes one sample video, and prints the predicted label.

> **To run:** update `MODEL_PATH`, the dataset root path, and any sample video paths to match your environment before re-executing Cells 9–10.

---

### B. ViViT (`vivit-final-v3.ipynb`)
1. **Install dependencies** (Cell 1).
2. **Imports** (Cell 2): `torch`, `av`, `VivitConfig`, `VivitForVideoClassification`, `Trainer`, `TrainingArguments`.
3. **Helper functions** (Cell 3): `generate_all_files`, `sample_frame_indices`, `read_video_pyav` (same pattern as TimeSformer notebook, video decoded as PIL images — no resize at this stage).
4. **Dataset class** (Cell 4): `get_video_paths_and_labels` plus a `VideoDataset` that:
   - Samples 16 frames per clip.
   - Applies a `torchvision.transforms` augmentation pipeline when training (`RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, `RandomRotation`) and a deterministic resize for validation/test.
5. **Metric function** (Cell 5): accuracy via `evaluate.load("accuracy", trust_remote_code=True)`.
6. **Model initialization** (Cell 6): `initialise_model()` configures `VivitConfig` (16 frames, 224×224 images, increased dropout 0.3) and loads `google/vivit-b-16x2-kinetics400` with a 2-class head.
7. **Training arguments** (Cell 7): 10 epochs, batch size 4, gradient accumulation (2 steps), cosine LR schedule, label smoothing, fp16.
8. **Data split + model init** (Cell 8): builds train/val/test datasets and initializes the model.
9. **Trainer setup & training** (Cell 9): custom `MetricsCallback` (prints val loss/accuracy each eval step) + `EarlyStoppingCallback`; calls `trainer.train()`.
10. **Loss/accuracy plots** (Cell 10).
11. **Test-set predictions** (Cells 11–13): `trainer.predict()` on the test set, prints `classification_report`.
12. **Save model** (Cell 14): `trainer.save_model("vivit-violence-detection-model-v5")`.
13. **Standalone inference** (Cells 15–17): loads a saved checkpoint, defines a manual frame-extraction + transform pipeline (different from training-time processor — uses `torchvision.transforms` directly), and predicts on a sample video.
14. **Relabeling** (Cell 18): overrides `model.config.id2label` / `label2id` with `non-violent` / `violent` keys.
15. **Print prediction** (Cell 19).
16. **Archive & download model** (Cells 20–21): zips the saved model directory with `shutil.make_archive` and exposes a download link via `IPython.display.FileLink` (Kaggle-specific).

> **To run:** update the dataset root path, `checkpoint_path`, and sample inference video path for your environment.

---

### C. MViT (`mvit-v1.ipynb`)
1. **Imports** (Cell 1): `torch`, `torchvision.models.video.mvit_v2_s`, `MViT_V2_S_Weights`, `cv2`.
2. **Dataset class** (Cell 2): `ViolenceDataset` extracts a fixed number of frames per clip (`frames_per_clip=16`), pads short videos, randomly samples a continuous 16-frame window from longer videos, applies transforms, and permutes to `(C, T, H, W)` for MViT's expected input shape.
3. **Data preparation** (Cell 3):
   - Defines an augmentation pipeline (`Resize`, `RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`, `RandomRotation`, `Normalize`).
   - Lists video paths/labels directly from the `Violence` and `NonViolence` sub-folders.
   - Splits into train (70%) / val (15%) / test (15%).
4. **Model setup** (Cell 4):
   - Loads `mvit_v2_s` with pretrained Kinetics-400 weights.
   - Replaces the classification head with `Dropout(0.6) + Linear(in_features, 2)`.
   - Freezes all transformer blocks except the last 3 for fine-tuning.
5. **Loss/optimizer/scheduler** (Cell 5): custom `LabelSmoothingCrossEntropy`, `AdamW` optimizer, `CosineAnnealingLR` scheduler.
6. **Early stopping utility** (Cells 6–7): a simple `EarlyStopping` class tracking validation loss.
7. **Evaluation utilities** (Cell 8): `evaluate()` function computing accuracy/precision/recall/F1, and `plot_confusion_matrix()` for visualizing results.
8. **Training run** (Cell 9): calls `train_model(model, train_loader, val_loader, epochs=20, accumulation_steps=2)` *(defined earlier in the notebook’s training-loop cell — ensure this cell is executed before training starts)*.
9. **Final test evaluation** (Cell 10):
   - Loads the best checkpoint (`/kaggle/working/best_mvit_model.pth`).
   - Computes test accuracy/precision/recall/F1, prints a `classification_report`, and plots a confusion matrix and loss/accuracy curves.
10. **Inference utility class** (Cell 11): `VideoProcessor` — loads the trained MViT weights, extracts frames with OpenCV, applies the same normalization used in training, and exposes a method to classify a new video file.

> **To run:** ensure `train_loader`/`val_loader`/`test_loader` and the `train_model` function (built earlier in the notebook from the `ViolenceDataset` + `DataLoader`) are defined before Cell 9; update dataset/model paths for your environment.

---

## 4. General Execution Order (applies to all three notebooks)

1. Install dependencies (first cell of each notebook).
2. Mount/point to the **Real Life Violence Situations Dataset**.
3. Run all data-preparation and helper-function cells top-to-bottom.
4. Run the model-initialization and `TrainingArguments` / optimizer cells.
5. Run the training cell (`trainer.train()` for TimeSformer/ViViT, custom `train_model(...)` loop for MViT).
6. Run evaluation cells to generate loss/accuracy plots and the classification report on the held-out test set.
7. Save/export the fine-tuned model.
8. Run the inference cell(s) on new/sample video(s), updating file paths as needed.

---

## 5. Notes & Caveats

- All notebooks were built for **Kaggle Notebooks** — file paths under `/kaggle/input` and `/kaggle/working` must be changed to local/cloud equivalents if running elsewhere (e.g., Colab, local GPU machine).
- Some inference cells reference model checkpoints saved from a *previous* training run (e.g., `/kaggle/input/timesformer-1/...`, `/kaggle/input/vivit-model-v4/...`) — these are pre-trained artifacts uploaded back into Kaggle as a dataset, not auto-generated by the notebook itself. You will need your own checkpoint or path if reproducing from scratch.
- The MViT notebook's `train_model` function and `DataLoader` instantiations are referenced but not shown in the inspected cells above — verify they exist earlier in your copy of the notebook before running the training cell.
- Class labels are binary in all three notebooks: `0 = NonViolence`, `1 = Violence`.
