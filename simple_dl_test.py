#!/usr/bin/env python3
"""
Simple Deep Learning Hardware Capability Test
A lightweight test to check if your system can run basic deep learning workloads
"""

import time
import sys
import platform
import psutil
import numpy as np

print("="*60)
print("SIMPLE DEEP LEARNING HARDWARE TEST")
print("="*60)

# ============================================================================
# STEP 1: System Information
# ============================================================================
print("\n[1/4] SYSTEM INFORMATION")
print("-"*40)
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python Version: {sys.version.split()[0]}")
print(f"CPU: {platform.processor()}")
print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")

# ============================================================================
# STEP 2: Import Libraries
# ============================================================================
print("\n[2/4] IMPORTING LIBRARIES")
print("-"*40)

try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu.name}")
    else:
        print("⚠ No GPU detected - using CPU (slower)")

except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")
    print("Install with: pip install tensorflow")
    sys.exit(1)

# ============================================================================
# STEP 3: Create Simple Model
# ============================================================================
print("\n[3/4] BUILDING SIMPLE NEURAL NETWORK")
print("-"*40)

# Create a very simple CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(f"✓ Model created with {model.count_params():,} parameters")

# ============================================================================
# STEP 4: Quick Training Test
# ============================================================================
print("\n[4/4] RUNNING TRAINING TEST")
print("-"*40)

# Generate small synthetic dataset
print("Generating test data...")
X_train = np.random.random((100, 64, 64, 3)).astype('float32')
y_train = np.random.randint(0, 10, 100).astype('int32')
X_test = np.random.random((20, 64, 64, 3)).astype('float32')
y_test = np.random.randint(0, 10, 20).astype('int32')

print(f"✓ Training data: {X_train.shape}")
print(f"✓ Test data: {X_test.shape}")

# Quick training
print("\nStarting training (2 epochs)...")
start_time = time.time()

try:
    history = model.fit(X_train, y_train,
                       epochs=2,
                       batch_size=10,
                       validation_data=(X_test, y_test),
                       verbose=1)

    training_time = time.time() - start_time

    print("\n✓ Training completed successfully!")
    print(f"✓ Training time: {training_time:.2f} seconds")
    print(f"✓ Final accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"✓ Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    # Quick inference test
    print("\nTesting inference speed...")
    test_sample = np.random.random((1, 64, 64, 3)).astype('float32')

    # Warm up
    _ = model.predict(test_sample, verbose=0)

    # Time inference
    inference_times = []
    for _ in range(10):
        start = time.time()
        _ = model.predict(test_sample, verbose=0)
        inference_times.append(time.time() - start)

    avg_inference = np.mean(inference_times) * 1000
    print(f"✓ Average inference time: {avg_inference:.2f} ms")
    # Performance assessment
    print("\n" + "="*60)
    print("SYSTEM ASSESSMENT")
    print("="*60)

    score = 0
    if gpus:
        score += 50
        print("✓ GPU Available: +50 points")
    else:
        print("⚠ CPU Only: +0 points")

    if training_time < 30:
        score += 30
        print("✓ Fast Training: +30 points")
    elif training_time < 60:
        score += 20
        print("⚠ Moderate Training: +20 points")
    else:
        score += 10
        print("⚠ Slow Training: +10 points")

    if avg_inference < 50:
        score += 20
        print("✓ Good Inference Speed: +20 points")
    else:
        score += 10
        print("⚠ Slower Inference: +10 points")

    print(f"\nFINAL SCORE: {score}/100")

    if score >= 80:
        print("🌟 EXCELLENT: Your system is great for deep learning!")
    elif score >= 60:
        print("✓ GOOD: Your system can handle deep learning tasks.")
    elif score >= 40:
        print("⚠ FAIR: Basic deep learning possible, consider GPU upgrade.")
    else:
        print("⚠ LIMITED: May need hardware upgrades for serious ML work.")

except Exception as e:
    print(f"\n✗ Training failed: {e}")
    print("Your system may need more RAM or a GPU for deep learning.")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETED")
print("="*60)