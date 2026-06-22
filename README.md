# Violence-Detection-using-Deep-Learning using ConvLSTM and LRCN

This repository contains a Jupyter Notebook implementing **Human Action Recognition** using two deep learning architectures:

* **ConvLSTM (Convolutional Long Short-Term Memory)**
* **LRCN (Long-term Recurrent Convolutional Network)**

The project is developed using **TensorFlow/Keras** and trained on the **UCF50 Dataset**. It demonstrates the complete workflow from dataset preparation and preprocessing to model training, evaluation, and action prediction on custom videos.

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
