import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    """Load MNIST dataset and split into train, validation, and test sets."""
    print("Loading MNIST dataset...")
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Train / Validation split (85% train, 15% val)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )

    print(f"Dataset split: Train={x_train.shape[0]}, Validation={x_val.shape[0]}, Test={x_test.shape[0]}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def build_dense_model():
    """Build baseline Keras Sequential Dense (MLP) Neural Network."""
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.Dropout(0.2, name='dropout_1'),
        layers.Dense(128, activation='relu', name='dense_2'),
        layers.Dropout(0.2, name='dropout_2'),
        layers.Dense(64, activation='relu', name='dense_3'),
        layers.Dense(10, activation='softmax', name='output')
    ], name="Dense_MLP_Model")
    
    return model

def build_cnn_model():
    """Build upgraded Convolutional Neural Network (CNN) model for comparison."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ], name="CNN_Model")
    
    return model

def plot_training_history(history, save_path="loss_accuracy_curves.png"):
    """Plot and save Loss and Accuracy curves for training and validation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history.history['loss'], 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history.history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy plot
    ax2.plot(epochs, history.history['accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history.history['val_accuracy'], 'r-s', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved loss & accuracy curves to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(10), yticklabels=range(10), annot_kws={"size": 12})
    plt.title('Confusion Matrix on Test Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Digit Label', fontsize=12)
    plt.ylabel('True Digit Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

def plot_misclassifications(x_test, y_true, y_pred, y_probs, save_path="misclassifications.png"):
    """Plot sample misclassified digits with true vs predicted labels and confidence."""
    mis_indices = np.where(y_true != y_pred)[0]
    print(f"Total misclassifications in test set: {len(mis_indices)} out of {len(y_true)} ({len(mis_indices)/len(y_true)*100:.2f}%)")

    # Select up to 15 misclassified samples
    num_samples = min(15, len(mis_indices))
    sample_indices = np.random.choice(mis_indices, num_samples, replace=False)

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < num_samples:
            img_idx = sample_indices[idx]
            img = x_test[img_idx]
            true_l = y_true[img_idx]
            pred_l = y_pred[img_idx]
            conf = y_probs[img_idx][pred_l] * 100

            ax.imshow(img, cmap='gray')
            ax.set_title(f"True: {true_l} | Pred: {pred_l}\nConf: {conf:.1f}%", color='red', fontsize=11, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle("Sample Misclassified Handwritten Digits", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved misclassifications visualization to {save_path}")

def train_and_evaluate():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    print("\n--- Training Baseline Keras Sequential Dense (MLP) Model ---")
    model = build_dense_model()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=12,
        batch_size=64,
        verbose=1
    )

    # Evaluate on Test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")

    # Generate predictions
    y_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # Save model files
    model.save("digit_model.keras")
    model.save("dense_model.keras")
    print("Model saved to digit_model.keras")

    # Plot & save visual artifacts
    plot_training_history(history, "loss_accuracy_curves.png")
    plot_confusion_matrix(y_test, y_pred, "confusion_matrix.png")
    plot_misclassifications(x_test, y_test, y_pred, y_probs, "misclassifications.png")

    # Train CNN model for comparison & save
    print("\n--- Training CNN Model (Comparison Baseline) ---")
    cnn_model = build_cnn_model()
    cnn_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    cnn_history = cnn_model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=8,
        batch_size=64,
        verbose=1
    )
    cnn_loss, cnn_acc = cnn_model.evaluate(x_test, y_test, verbose=0)
    cnn_model.save("cnn_model.keras")

    # Save evaluation summary metrics JSON
    metrics = {
        "dense_test_accuracy": float(test_acc),
        "dense_test_loss": float(test_loss),
        "cnn_test_accuracy": float(cnn_acc),
        "cnn_test_loss": float(cnn_loss),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "history": {
            "loss": [float(x) for x in history.history['loss']],
            "val_loss": [float(x) for x in history.history['val_loss']],
            "accuracy": [float(x) for x in history.history['accuracy']],
            "val_accuracy": [float(x) for x in history.history['val_accuracy']]
        }
    }

    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("\nTraining and evaluation pipeline completed successfully!")

if __name__ == "__main__":
    train_and_evaluate()
