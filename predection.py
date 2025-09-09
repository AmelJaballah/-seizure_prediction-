import os
import numpy as np
import h5py
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score, roc_curve, auc, 
    recall_score, precision_score, precision_recall_fscore_support
)
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Settings ===
ch_labels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
             'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
             'FZ-CZ', 'CZ-PZ']
fs = 256
time_window = 2
nperseg = 18  
noverlap = 9
# Adjust to match actual data shape (30, 18, 10)
num_freqs = 30 
num_time = 18  
num_channels = 10 
vis_dir = './visualization1'
os.makedirs(vis_dir, exist_ok=True)
min_segments_per_class = 2000
dtype = np.float32

# Model Definition 
def create_improved_lstm_transformer_model(input_shape=(num_freqs, num_time, num_channels)):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Reshape((input_shape[1], input_shape[0] * input_shape[2]))(inputs)
    
    x = LSTM(units=96, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    num_heads = 4  
    d_model = 128 
    x = Dense(d_model)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=0.1)(x, x)
    x = tf.keras.layers.Add()([x, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    x = Dense(d_model, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(3, activation='softmax', name='predictions')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

#  Loss Function 
def enhanced_custom_loss(y_true, y_pred):
    class_weights = tf.constant([1.0, 2.5, 1.5], dtype=tf.float32)
    
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    weighted_ce = ce_loss * tf.reduce_sum(y_true * class_weights, axis=-1)
    
    preictal_true = y_true[:, 1]
    preictal_pred = y_pred[:, 1]
    alpha = 1.5  
    gamma = 1.5  
    focal_weight = alpha * tf.pow(1 - preictal_pred, gamma)
    focal_loss = focal_weight * preictal_true * (-tf.math.log(preictal_pred + 1e-8))
    
    total_loss = weighted_ce + 0.7 * focal_loss
    return tf.reduce_mean(total_loss)

#  Callback 
class ImprovedPreictalSensitivityCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, label_encoder, patience=20, min_delta=0.005, 
                 restore_best_weights=True, baseline_epochs=15):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.label_encoder = label_encoder
        self.best_sensitivity = 0.0
        self.best_weights = None
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.restore_best_weights = restore_best_weights
        self.baseline_epochs = baseline_epochs
        self.epoch_count = 0
        self.sensitivity_history = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(self.y_val, axis=1)
        
        preictal_idx = self.label_encoder.transform(['preictal'])[0]
        TP = np.sum((y_true_labels == preictal_idx) & (y_pred_labels == preictal_idx))
        FN = np.sum((y_true_labels == preictal_idx) & (y_pred_labels != preictal_idx))
        sensitivity = TP / (TP + FN + 1e-6)
        
        precision = TP / (TP + np.sum((y_true_labels != preictal_idx) & (y_pred_labels == preictal_idx)) + 1e-6)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-6)
        
        self.sensitivity_history.append(sensitivity)
        print(f"Epoch {epoch + 1} - Preictal Sensitivity: {sensitivity:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")
        
        if self.epoch_count <= self.baseline_epochs:
            if sensitivity > self.best_sensitivity:
                self.best_sensitivity = sensitivity
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            return
        
        if sensitivity > (self.best_sensitivity + self.min_delta):
            self.best_sensitivity = sensitivity
            self.patience_counter = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            print(f"New best preictal sensitivity: {self.best_sensitivity:.4f}")
        else:
            self.patience_counter += 1
            
        if len(self.sensitivity_history) >= 3:
            recent_trend = np.mean(self.sensitivity_history[-3:]) - np.mean(self.sensitivity_history[-6:-3])
            if recent_trend > 0.01:
                self.patience_counter = max(0, self.patience_counter - 1)
        
        if self.patience_counter >= self.patience:
            print(f"Early stopping: No improvement in preictal sensitivity for {self.patience} epochs")
            print(f"Best sensitivity achieved: {self.best_sensitivity:.4f}")
            if self.restore_best_weights and self.best_weights is not None:
                print("Restoring best weights...")
                self.model.set_weights(self.best_weights)
            self.model.stop_training = True

# Visualization
def visualize_multi_channel_spectrograms(spectrograms, labels, save_path_prefix):
    class_names = ['interictal', 'preictal', 'ictal']
    class_names_display = ['Interictal', 'Preictal', 'Ictal']
    channel_to_plot = 0  # Just the first channel
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    
    # Only generate a single combined plot for efficiency
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    label_indices = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    for cls in range(3):
        indices = np.where(label_indices == cls)[0]
        if len(indices) == 0 or cls >= len(axes):
            logging.warning(f"No samples for {class_names_display[cls]}. Skipping visualization.")
            continue
        
        idx = indices[0]
        spec = spectrograms[idx]  # Shape: (num_freqs, num_time, channels)
        
        # Spectrogram for the selected channel
        freqs = np.linspace(0, fs / 2, spec.shape[0])
        times = np.linspace(0, time_window, spec.shape[1])
        
        im = axes[cls].imshow(spec[:, :, channel_to_plot], aspect='auto', origin='lower',
                           extent=[times[0], times[-1], freqs[0], freqs[-1]], cmap='viridis')
        axes[cls].set_title(f"{class_names_display[cls]} - {ch_labels[channel_to_plot]}")
        axes[cls].set_xlabel("Time (s)")
        axes[cls].set_ylabel("Frequency (Hz)")
    
    plt.tight_layout()
    plot_file = f"{save_path_prefix}_class_spectrograms.png"
    plt.savefig(plot_file, dpi=200)
    logging.info(f"✅ Saved simplified spectrograms to '{plot_file}'.")
    plt.close(fig)

# === Data Loading and Splitting ===
def load_data(file_path='', test_size=0.2, val_size=0.15, random_state=42):#data path after preprocessing
    try:
        with h5py.File(file_path, 'r') as f:
            spectrograms = f['spectrograms'][:]
            labels = f['labels'][:]
        logging.info(f"Loaded HDF5: Spectrograms shape={spectrograms.shape}, Labels shape={labels.shape}")
    except Exception as e:
        logging.error(f"Error loading HDF5 file {file_path}: {str(e)}")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), None
    
    spectrograms = np.transpose(spectrograms, (0, 2, 3, 1))
    logging.info(f"Transposed spectrograms shape: {spectrograms.shape}")
    
    label_encoder = LabelEncoder()
    label_encoder.fit(['interictal', 'preictal', 'ictal'])
    labels_decoded = [l.decode('utf-8') if isinstance(l, (bytes, np.bytes_)) else l for l in labels]
    labels_encoded = label_encoder.transform(labels_decoded)
    labels_categorical = to_categorical(labels_encoded, num_classes=3)
    
    # Stratified split: train+val (80%) and test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        spectrograms, labels_categorical, test_size=test_size, stratify=labels_categorical, random_state=random_state
    )
    
    # Split train+val into train (85% of 80% = 68%) and val (15% of total)
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for train+val portion
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
    )
    
    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Train data shape: {X_train.shape}, Validation data shape: {X_val.shape}, Test data shape: {X_test.shape}")
    print(f"Label distribution: {np.bincount(labels_encoded)}")
    
    # Log class distribution for each split
    for name, y in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]:
        counts = np.bincount(np.argmax(y, axis=1), minlength=3)
        logging.info(f"{name} class distribution: Interictal={counts[1]}, Preictal={counts[2]}, Ictal={counts[0]}")
    
    visualize_multi_channel_spectrograms(X_test, y_test, os.path.join(vis_dir, 'test_sample'))
    
    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder


def enhanced_train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, label_encoder):
    os.makedirs('./output/visualization1/', exist_ok=True)
    class_weights = {0: 1.0, 1: 3.0, 2: 1.8}
    
    def lr_schedule(epoch, lr):
        if epoch < 20:
            return lr
        elif epoch < 50:
            return lr * 0.8
        elif epoch < 100:
            return lr * 0.5
        else:
            return lr * 0.2
    
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,     
        patience=5,     
        min_lr=1e-6,    
        verbose=1
    )
    
    preictal_callback = ImprovedPreictalSensitivityCallback(
        validation_data=(X_val, y_val), 
        label_encoder=label_encoder, 
        patience=15,     
        min_delta=0.01,   
        baseline_epochs=10 
    )
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        './best_preictal_model.h5', 
        monitor='val_loss',
        save_best_only=True, 
        save_weights_only=False, 
        verbose=1
    )
    
    
    # Compile the model 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005, 
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=1e-7
        ),
        loss=enhanced_custom_loss, 
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
           
        ]
    )
    
    # Fit the model 
    history = model.fit(
        X_train, y_train, 
        batch_size=32,    
        epochs=150,        
        validation_data=(X_val, y_val),
        callbacks=[
            lr_scheduler, 
            lr_plateau, 
            preictal_callback, 
            checkpoint
        ], 
        class_weight=class_weights, 
        verbose=1
    )
    
    if os.path.exists('./best_preictal_model.h5'):
        model = tf.keras.models.load_model('./best_preictal_model.h5', custom_objects={'enhanced_custom_loss': enhanced_custom_loss})
        print("Loaded best model weights.")
    
    test_pred = model.predict(X_test, verbose=0)
    test_pred_labels = np.argmax(test_pred, axis=1)
    test_true_labels = np.argmax(y_test, axis=1)
    class_names = label_encoder.classes_.tolist()
    
    f1_weighted = f1_score(test_true_labels, test_pred_labels, average='weighted')
    f1_macro = f1_score(test_true_labels, test_pred_labels, average='macro')
    accuracy = accuracy_score(test_true_labels, test_pred_labels)
    
    preictal_idx = label_encoder.transform(['preictal'])[0]
    preictal_sensitivity = recall_score(test_true_labels, test_pred_labels, labels=[preictal_idx], average=None, zero_division=0)
    preictal_sensitivity = preictal_sensitivity[0] if len(preictal_sensitivity) > 0 else 0.0
    preictal_mask_true = (test_true_labels == preictal_idx)
    preictal_mask_pred = (test_pred_labels == preictal_idx)
    preictal_tp = np.sum(preictal_mask_true & preictal_mask_pred)
    preictal_fp = np.sum(~preictal_mask_true & preictal_mask_pred)
    preictal_precision = preictal_tp / (preictal_tp + preictal_fp + 1e-6)
    preictal_f1 = 2 * (preictal_precision * preictal_sensitivity) / (preictal_precision + preictal_sensitivity + 1e-6)
    
    print(f"\n{'='*50}\nFINAL TEST METRICS:\n{'='*50}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print(f"Macro F1 Score: {f1_macro:.4f}")
    print(f"\nPREICTAL CLASS METRICS:")
    print(f"Preictal Sensitivity (Recall): {preictal_sensitivity:.4f}")
    print(f"Preictal Precision: {preictal_precision:.4f}")
    print(f"Preictal F1 Score: {preictal_f1:.4f}")
    print(f"{'='*50}")
    
    cm = confusion_matrix(test_true_labels, test_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('./output/visualization1/test_confusion_matrix.png')
    plt.close()
    
    # Plot Loss 
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./output/visualization1/loss_curve.png')
    plt.close()

    # Plot Accuracy 
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./output/visualization1/accuracy_curve.png')
    plt.close()
    
    n_classes = len(class_names)
    sensitivities = []
    specificities = []
    aucs = []
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        y_true_bin = (test_true_labels == i).astype(int)
        y_score = test_pred[:, i]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        TP = np.sum((test_true_labels == i) & (test_pred_labels == i))
        FN = np.sum((test_true_labels == i) & (test_pred_labels != i))
        FP = np.sum((test_true_labels != i) & (test_pred_labels == i))
        TN = np.sum((test_true_labels != i) & (test_pred_labels != i))
        sensitivity = TP / (TP + FN + 1e-6)
        specificity = TN / (TN + FP + 1e-6)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        plt.plot(fpr, tpr, lw=2, label=f'ROC {class_names[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig('./output/visualization1/roc_curves.png')
    plt.close()
    
    for i in range(n_classes):
        print(f"Class: {class_names[i]}")
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  AUC: {aucs[i]:.4f}")
    
    
    # Create visualization directory if it doesn't exist
    os.makedirs('./output/visualization1/', exist_ok=True)
    
    print(" visualizations completed )")
    
    model.save('./hybrid_model_tf.h5')
    print(f"\nModel saved to './hybrid_model_tf.h5'")
    print("All visualizations have been saved to the visualization1 directory.")
    return model, history

def main():
    print("="*60)
    print("EEG SEIZURE PREDICTION WITH ENHANCED PREICTAL DETECTION")
    print("="*60)
    
    print("\n1. LOADING AND SPLITTING DATA...")
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_data()
    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        logging.error("No data loaded. Exiting.")
        return
    
    print("\n2. CREATING ENHANCED MODEL...")
    model = create_improved_lstm_transformer_model()
    print(f"Model created with {model.count_params():,} parameters")
    
    print("\n3. TRAINING AND EVALUATING MODEL...")
    model, history = enhanced_train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Check './output/visualization1/' and './visualization1/' for plots and results.")

if __name__ == '__main__':
    main()