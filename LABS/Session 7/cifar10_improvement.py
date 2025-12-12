#!/usr/bin/env python3
"""
Session 7: CIFAR10 CNN Hyperparameter Tuning
Goal: Improve baseline model from ~70% to ~80% accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CIFAR10 CNN HYPERPARAMETER TUNING")
print("="*70)

# Load and preprocess data
print("\n[1/5] Loading CIFAR10 dataset...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Normalize
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# One-hot encode
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("\n" + "="*70)
print("BASELINE MODEL (~70% accuracy)")
print("="*70)

# Baseline Model (from original notebook)
print("\n[2/5] Building baseline model...")
baseline = Sequential([
    Conv2D(64, (5, 5), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

baseline.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

print("\nBaseline Model Architecture:")
baseline.summary()

print("\n[3/5] Training baseline model (20 epochs)...")
baseline_history = baseline.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.01,
    verbose=1
)

print("\n[4/5] Evaluating baseline model...")
baseline_loss, baseline_acc = baseline.evaluate(X_test, y_test, verbose=0)
print(f"\n📊 BASELINE RESULTS:")
print(f"   Test Accuracy: {baseline_acc*100:.2f}%")
print(f"   Test Loss: {baseline_loss:.4f}")

print("\n" + "="*70)
print("IMPROVED MODEL (Target: 80%+ accuracy)")
print("="*70)

# Improved Model with hyperparameter tuning
print("\n[5/5] Building improved model with hyperparameter tuning...")
print("\nImprovements:")
print("  ✓ Smaller kernel sizes (3x3 instead of 5x5)")
print("  ✓ Added 3rd convolutional layer")
print("  ✓ Added Dropout layers (0.25, 0.3, 0.5)")
print("  ✓ Added BatchNormalization")
print("  ✓ Increased epochs to 50 with early stopping")
print("  ✓ Increased validation split to 20%")
print("  ✓ Added learning rate reduction on plateau")
print("  ✓ Increased batch size to 64")

improved = Sequential([
    # First conv block
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Second conv block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Third conv block
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Dense layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile with Adam optimizer
improved.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nImproved Model Architecture:")
improved.summary()

# Callbacks for better training
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

print("\nTraining improved model (up to 50 epochs with early stopping)...")
print("This may take 10-20 minutes depending on your hardware...\n")

improved_history = improved.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\nEvaluating improved model...")
improved_loss, improved_acc = improved.evaluate(X_test, y_test, verbose=0)

print("\n" + "="*70)
print("FINAL RESULTS COMPARISON")
print("="*70)
print(f"\n📊 BASELINE MODEL:")
print(f"   Test Accuracy: {baseline_acc*100:.2f}%")
print(f"   Test Loss: {baseline_loss:.4f}")

print(f"\n🚀 IMPROVED MODEL:")
print(f"   Test Accuracy: {improved_acc*100:.2f}%")
print(f"   Test Loss: {improved_loss:.4f}")

improvement = (improved_acc - baseline_acc) * 100
print(f"\n✨ IMPROVEMENT: +{improvement:.2f}%")

if improved_acc >= 0.80:
    print(f"\n🎉 SUCCESS! Target of 80%+ accuracy achieved!")
else:
    print(f"\n⚠️  Target: 80%, Current: {improved_acc*100:.2f}% (still training...)")

# Save models
print("\n" + "="*70)
print("Saving models...")
baseline.save('/Users/joshuawhite/Code/Uni/COM-6033/LABS/Session 7/cifar10_baseline.keras')
improved.save('/Users/joshuawhite/Code/Uni/COM-6033/LABS/Session 7/cifar10_improved.keras')
print("✓ Models saved:")
print("  - cifar10_baseline.keras")
print("  - cifar10_improved.keras")

# Plot training history
print("\nGenerating accuracy comparison plot...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(baseline_history.history['accuracy'], label='Baseline Train')
plt.plot(baseline_history.history['val_accuracy'], label='Baseline Val')
plt.title('Baseline Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(improved_history.history['accuracy'], label='Improved Train')
plt.plot(improved_history.history['val_accuracy'], label='Improved Val')
plt.title('Improved Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('/Users/joshuawhite/Code/Uni/COM-6033/LABS/Session 7/accuracy_comparison.png', dpi=150)
print("✓ Plot saved: accuracy_comparison.png")

print("\n" + "="*70)
print("SESSION 7 COMPLETE!")
print("="*70)
