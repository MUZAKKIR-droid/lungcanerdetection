#!/usr/bin/env python3
"""
Simple Lung Cancer Detector - Command Line Version
Just provide an image path and get prediction!
"""

import os
import sys
import pickle
import numpy as np
from PIL import Image

def load_model():
    """Load the trained model"""
    model_file = 'trained_model.pkl'
    if not os.path.exists(model_file):
        print("❌ Error: trained_model.pkl not found!")
        return None
    
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully\n")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def preprocess_image(image_path):
    """Convert image to format expected by model"""
    try:
        # Open and convert to grayscale
        img = Image.open(image_path).convert('L')
        print(f"📸 Image loaded: {os.path.basename(image_path)}")
        print(f"   Original size: {img.size}")
        
        # Resize to 10x10
        img = img.resize((10, 10), Image.Resampling.LANCZOS)
        print(f"   Resized to: 10x10")
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Create 5 slices (simulate 3D)
        slices = np.stack([img_array] * 5, axis=0)
        
        # Flatten
        flattened = slices.ravel()
        
        print(f"   Processed shape: {flattened.shape}")
        return flattened.reshape(1, -1)
    
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

def predict(model, image_path):
    """Make prediction on image"""
    print("\n" + "="*60)
    print("🔬 ANALYZING IMAGE")
    print("="*60)
    
    # Preprocess
    processed = preprocess_image(image_path)
    if processed is None:
        return
    
    # Predict
    try:
        print("\n🧠 Making prediction...")
        prediction = model.predict(processed)[0]
        probabilities = model.predict_proba(processed)[0]
        
        print("\n" + "="*60)
        print("📊 RESULTS")
        print("="*60)
        
        if prediction == 1:
            result = "⚠️  CANCER DETECTED"
            confidence = probabilities[1] * 100
            print(f"\n{result}")
            print(f"Confidence: {confidence:.2f}%")
            print("\n⚠️  WARNING: This is a demo model.")
            print("   Consult a medical professional for actual diagnosis.")
        else:
            result = "✅ NO CANCER DETECTED"
            confidence = probabilities[0] * 100
            print(f"\n{result}")
            print(f"Confidence: {confidence:.2f}%")
            print("\n✅ Good news! But this is just a demo.")
            print("   Always consult medical professionals.")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🫁 SIMPLE LUNG CANCER DETECTOR")
    print("="*60)
    print("\nUsage: python detect_simple.py <image_path>")
    print("Example: python detect_simple.py lung_scan.jpg")
    print("\n" + "="*60 + "\n")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("📁 Enter image path (or drag & drop): ").strip().strip("'\"")
    
    if not image_path:
        print("❌ No image path provided!")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    # Make prediction
    predict(model, image_path)
    
    # Ask for another prediction
    print("\n" + "="*60)
    another = input("\nAnalyze another image? (y/n): ").strip().lower()
    if another == 'y':
        image_path = input("📁 Enter image path: ").strip().strip("'\"")
        if os.path.exists(image_path):
            predict(model, image_path)

if __name__ == "__main__":
    main()
