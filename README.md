# EEG Seizure Detection and Prediction

## Détection et Prédiction des Crises d'Épilepsie à partir de Signaux EEG : Approche Hybride LSTM-Transformer avec Analyse Temps-Fréquence (STFT)

This repository contains code and documentation for a project focused on detecting and predicting epileptic seizures using EEG signals. The approach leverages a hybrid LSTM-Transformer model combined with Short-Time Fourier Transform (STFT) analysis, with a particular emphasis on identifying the preictal phase to anticipate seizures.

### Features
- **Data Preprocessing**: Processes EEG data from the CHB-MIT dataset, validates channels, extracts STFT spectrograms, labels segments, balances classes, and saves to HDF5.
- **Model Architecture**: Implements a hybrid LSTM-Transformer model optimized for EEG signals, capturing temporal dynamics and global relationships.
- **Custom Training**: Includes a custom loss function and callback to prioritize preictal sensitivity.
- **Evaluation & Visualization**: Provides metrics (accuracy, F1-score, sensitivity, etc.) and visualizations (confusion matrix, learning curves, ROC curves).
- **Report**: Includes a detailed PDF report (in French) summarizing the methodology, results, and visualizations.

## Requirements
- Python 3.8+
- Libraries: `numpy`, `scipy`, `mne`, `wfdb`, `h5py`, `tensorflow`, `sklearn`, `matplotlib`, `seaborn`, `tqdm`
- Dataset: Download the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) and place in `./chbmit_dataset`.

Install dependencies:
```
pip install -r requirements.txt
```

## Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/eeg-seizure-prediction.git
   cd eeg-seizure-prediction
   ```
2. Download and extract the CHB-MIT dataset into `./chbmit_dataset`.
3. Ensure the HDF5 file path in `predection_model.py` matches your setup .

## Usage
### 1. Data Preprocessing
Run `data_preprocess.py` to process EEG files, generate spectrograms, balance classes, and save to HDF5:
```
python data_preprocess.py
```
- Output: `spectrograms_labels_balanced.h5` (balanced dataset with spectrograms and labels).
- Visualizations: Plots raw signals and spectrograms for sample classes (e.g., ictal).

### 2. Model Training and Evaluation
Run `predection_model.py` to load data, train the model, evaluate on test set, and generate visualizations:
```
python predection_model.py
```
- Output: Trained model (`hybrid_model_tf.h5`), metrics printed to console, visualizations in `./output/visualization1/` and `./visualization1/`.
- Key Metrics (example from report):
  - Preictal Sensitivity: 0.9128
  - Preictal Precision: 0.9120
  - Preictal F1-Score: 0.9128
  - Overall Accuracy: 0.9887
  - AUC: 0.9887

## Dataset
- **Source**: CHB-MIT (Children's Hospital Boston EEG recordings).
- **Format**: .edf files.
- **Channels**: 19 bipolar electrodes (e.g., FP1-F7, F7-T7, ..., CZ-PZ).
- **Sampling Frequency**: 256 Hz.
- **Window**: 1 second (256 samples), overlap 0.5 seconds.
- **Classes**: 
  - Class 0 (Interictal): Normal activity without seizure.
  - Class 1 (Preictal): 30 minutes preceding a seizure.
  - Class 2 (Ictal): During an ongoing seizure.
- **Balancing**: Targets 25,000 segments per class using undersampling and augmentation (amplitude scaling 0.8–1.2, frequency jitter, simulation).

## Model Details
- **Input**: Tensor (30 freqs × 18 times × channels), reshaped for temporal processing.
- **LSTM Block**: 96 neurons, dropout 0.25, recurrent dropout, BatchNormalization.
- **Transformer Block**: Dense projection (128 dims), MultiHeadAttention (4 heads, key_dim=32), Add+LayerNormalization, Feed-Forward (Dense+Dropout).
- **Output**: Dense(3, softmax) for [interictal, preictal, ictal].
- **Loss**: Custom (weighted cross-entropy + focal loss, gamma=1.5, alpha=1.5).
- **Callback**: ImprovedPreictalSensitivityCallback (baseline=10 epochs, patience=15, min_delta=0.01).
- **Training Split**: 68% train, 15% validation, 20% test (stratified).
- **Optimizer**: Adam (initial LR=0.0005), with learning rate scheduling.

## Results
Refer to the PDF report (`Rapport_Prediction_Epilepsie_EEG.pdf`) for detailed results, including:
- Spectrogram visualizations (e.g., ictal).
- Confusion matrices (2D/3D).
- Learning curves (loss, accuracy).
- ROC curves per class.

Sample Visualizations:
- Confusion Matrix: High diagonal values, low off-diagonals.
- ROC: AUC ~0.9887 for all classes.
- Preictal Focus: Achieves ~91% sensitivity, critical for prediction.

## Contributing
Contributions are welcome! Open an issue or submit a pull request for improvements, bug fixes, or new features.

## License
MIT License – feel free to use and modify.

## Acknowledgments
- Built using TensorFlow, MNE, and PhysioNet datasets.
- Inspired by epilepsy prediction research focusing on preictal detection.

