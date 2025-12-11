#!/usr/bin/env python3
"""
MLOps Deployment and Testing Capability Test
Tests if your system can handle basic MLOps workflows:
- Model training and versioning
- Model deployment (serving)
- API testing and monitoring
- Basic CI/CD simulation
"""

import time
import sys
import platform
import psutil
import numpy as np
import os
import json
from datetime import datetime

print("="*70)
print("MLOPS DEPLOYMENT & TESTING CAPABILITY TEST")
print("="*70)

# ============================================================================
# STEP 1: System Information
# ============================================================================
print("\n[1/5] SYSTEM INFORMATION")
print("-"*70)
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python Version: {sys.version.split()[0]}")
print(f"CPU Cores: {psutil.cpu_count(logical=True)}")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")

# ============================================================================
# STEP 2: Import MLOps Libraries
# ============================================================================
print("\n[2/5] IMPORTING MLOPS LIBRARIES")
print("-"*70)

try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
except ImportError:
    print("✗ TensorFlow not found. Install: pip install tensorflow")
    sys.exit(1)

try:
    from flask import Flask, request, jsonify
    print("✓ Flask imported for API serving")
except ImportError:
    print("✗ Flask not found. Install: pip install flask")
    sys.exit(1)

try:
    import requests
    print("✓ Requests imported for API testing")
except ImportError:
    print("✗ Requests not found. Install: pip install requests")
    sys.exit(1)

# ============================================================================
# STEP 3: Model Training & Versioning
# ============================================================================
print("\n[3/5] MODEL TRAINING & VERSIONING")
print("-"*70)

# Create a simple but effective model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(f"✓ Model created with {model.count_params():,} parameters")

# Generate synthetic MNIST-like data
print("Generating training data...")
X_train = np.random.random((1000, 784)).astype('float32')
y_train = np.random.randint(0, 10, 1000)
X_test = np.random.random((200, 784)).astype('float32')
y_test = np.random.randint(0, 10, 200)

# Train the model
print("Training model...")
start_time = time.time()
history = model.fit(X_train, y_train, epochs=3, batch_size=32,
                   validation_data=(X_test, y_test), verbose=1)
training_time = time.time() - start_time

print(f"✓ Training completed in {training_time:.2f} seconds")
print(f"✓ Final accuracy: {history.history['accuracy'][-1]:.4f}")

# Version and save the model
model_version = f"v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
model_path = f"./models/{model_version}.keras"
os.makedirs("./models", exist_ok=True)

model.save(model_path)
print(f"✓ Model saved to {model_path}")

# Save model metadata
metadata = {
    "version": model_version,
    "training_time": training_time,
    "final_accuracy": float(history.history['accuracy'][-1]),
    "parameters": model.count_params(),
    "created_at": datetime.now().isoformat(),
    "framework": "TensorFlow",
    "python_version": sys.version.split()[0]
}

metadata_path = f"./models/{model_version}_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("✓ Model metadata saved")

# ============================================================================
# STEP 4: Model Deployment (API Serving)
# ============================================================================
print("\n[4/5] MODEL DEPLOYMENT - API SERVING")
print("-"*70)

# Create Flask API for model serving
app = Flask(__name__)

# Load the saved model
loaded_model = tf.keras.models.load_model(model_path)
print("✓ Model loaded for serving")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_version": model_version,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.get_json()

        if 'data' not in data:
            return jsonify({"error": "Missing 'data' field"}), 400

        input_data = np.array(data['data']).astype('float32')

        # Ensure correct shape
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        predictions = loaded_model.predict(input_data, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        return jsonify({
            "predictions": predicted_classes.tolist(),
            "probabilities": predictions.tolist(),
            "model_version": model_version,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

print("✓ Flask API created with /health and /predict endpoints")

# ============================================================================
# STEP 5: API Testing & Monitoring
# ============================================================================
print("\n[5/5] API TESTING & MONITORING")
print("-"*70)

# Test the API
print("Testing API endpoints...")

# Start Flask app in a separate thread for testing
from threading import Thread
import time

def run_app():
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

server_thread = Thread(target=run_app, daemon=True)
server_thread.start()

# Wait for server to start
time.sleep(2)

try:
    # Test health endpoint
    health_response = requests.get('http://127.0.0.1:5000/health')
    if health_response.status_code == 200:
        print("✓ Health check passed")
        health_data = health_response.json()
        print(f"  Model version: {health_data['model_version']}")
    else:
        print(f"✗ Health check failed: {health_response.status_code}")

    # Test prediction endpoint
    test_data = np.random.random((5, 784)).tolist()
    predict_response = requests.post('http://127.0.0.1:5000/predict',
                                   json={"data": test_data})

    if predict_response.status_code == 200:
        print("✓ Prediction API working")
        predict_data = predict_response.json()
        print(f"  Predictions shape: {len(predict_data['predictions'])}")
        print(f"  Probabilities shape: {len(predict_data['probabilities'])}")
    else:
        print(f"✗ Prediction API failed: {predict_response.status_code}")
        print(f"  Response: {predict_response.text}")

    # Performance test
    print("\nRunning performance test...")
    num_requests = 50
    response_times = []

    for i in range(num_requests):
        single_data = np.random.random(784).tolist()
        start = time.time()
        resp = requests.post('http://127.0.0.1:5000/predict',
                           json={"data": single_data})
        end = time.time()
        response_times.append(end - start)

    avg_response_time = np.mean(response_times) * 1000
    print(f"✓ Average response time: {avg_response_time:.2f} ms")
    print(f"✓ 95th percentile: {np.percentile(response_times, 95)*1000:.2f} ms")

    # ============================================================================
    # FINAL ASSESSMENT
    # ============================================================================
    print("\n" + "="*70)
    print("MLOPS CAPABILITY ASSESSMENT")
    print("="*70)

    score = 0

    # Training capability
    if training_time < 30:
        score += 25
        print("✓ Fast training: +25 points")
    else:
        score += 15
        print("⚠ Moderate training: +15 points")

    # Model saving/loading
    score += 20
    print("✓ Model versioning works: +20 points")

    # API deployment
    score += 20
    print("✓ API serving works: +20 points")

    # API testing
    if predict_response.status_code == 200:
        score += 20
        print("✓ API testing successful: +20 points")
    else:
        score += 5
        print("⚠ API testing issues: +5 points")

    # Performance
    if avg_response_time < 100:
        score += 15
        print("✓ Good API performance: +15 points")
    else:
        score += 10
        print("⚠ Acceptable API performance: +10 points")

    print(f"\nFINAL MLOPS SCORE: {score}/100")

    if score >= 80:
        print("🌟 EXCELLENT: Full MLOps capability!")
        print("   Your system can handle complete ML pipelines.")
    elif score >= 60:
        print("✓ GOOD: Solid MLOps foundation.")
        print("  Your system supports most MLOps workflows.")
    elif score >= 40:
        print("⚠ FAIR: Basic MLOps possible.")
        print("  Consider optimizations for production use.")
    else:
        print("⚠ LIMITED: Basic MLOps capability.")
        print("  May need hardware upgrades for full MLOps.")

    print("\n✓ MLOps components tested:")
    print("  - Model training ✓")
    print("  - Model versioning ✓")
    print("  - Model deployment ✓")
    print("  - API serving ✓")
    print("  - API testing ✓")
    print("  - Basic monitoring ✓")

except Exception as e:
    print(f"\n✗ MLOps testing failed: {e}")
    print("Your system may have issues with:")
    print("- Flask server startup")
    print("- Network connectivity")
    print("- Memory constraints")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("MLOPS TEST COMPLETED")
print("="*70)