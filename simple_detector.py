#!/usr/bin/env python3
"""
Simple Lung Cancer Detector - Works with regular images (PNG/JPG)
Just drop 1-2 images and get a prediction!
"""

import os
import sys
import pickle
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

class SimpleLungCancerDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Lung Cancer Detector")
        self.root.geometry("600x400")
        
        # Load model
        self.model = None
        self.load_model()
        
        # UI Setup
        self.setup_ui()
    
    def load_model(self):
        """Load the trained model"""
        model_file = 'trained_model.pkl'
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                print("✓ Model loaded successfully")
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                self.model = None
        else:
            print("✗ Model file not found")
            self.model = None
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_label = tk.Label(
            self.root, 
            text="🫁 Simple Lung Cancer Detector",
            font=("Arial", 20, "bold"),
            bg="#2196F3",
            fg="white",
            pady=10
        )
        title_label.pack(fill="x")
        
        # Instructions
        info_frame = tk.Frame(self.root, bg="#f0f0f0", pady=20)
        info_frame.pack(fill="x", padx=20, pady=10)
        
        instructions = """
        📸 How to use:
        1. Click "Select Image" button
        2. Choose a lung CT scan image (PNG/JPG)
        3. Get instant prediction!
        
        ⚠️ Note: This is a demo model trained on synthetic data.
        Not for actual medical diagnosis.
        """
        
        info_label = tk.Label(
            info_frame,
            text=instructions,
            font=("Arial", 11),
            bg="#f0f0f0",
            justify="left"
        )
        info_label.pack()
        
        # Select Image Button
        self.select_btn = tk.Button(
            self.root,
            text="📁 Select Image",
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            command=self.select_and_predict,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.select_btn.pack(pady=20)
        
        # Result Frame
        self.result_frame = tk.Frame(self.root, bg="white", relief="solid", borderwidth=2)
        self.result_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.result_label = tk.Label(
            self.result_frame,
            text="No prediction yet...",
            font=("Arial", 12),
            bg="white",
            fg="#666"
        )
        self.result_label.pack(pady=20)
    
    def preprocess_image(self, image_path):
        """Convert image to format expected by model"""
        try:
            # Open image
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            
            # Resize to 10x10 (to match training data)
            img = img.resize((10, 10), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)
            
            # Create 5 slices (duplicate the image to simulate 3D)
            slices = np.stack([img_array] * 5, axis=0)
            
            # Flatten to match model input
            flattened = slices.ravel()
            
            return flattened.reshape(1, -1)
        
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def select_and_predict(self):
        """Select image and make prediction"""
        if self.model is None:
            messagebox.showerror(
                "Error",
                "Model not loaded!\nPlease ensure 'trained_model.pkl' exists."
            )
            return
        
        # Select image file
        file_path = filedialog.askopenfilename(
            title="Select Lung CT Scan Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        # Show processing message
        self.result_label.config(
            text="🔄 Processing image...",
            fg="#2196F3"
        )
        self.root.update()
        
        # Preprocess image
        processed = self.preprocess_image(file_path)
        
        if processed is None:
            messagebox.showerror("Error", "Failed to process image!")
            self.result_label.config(
                text="❌ Error processing image",
                fg="red"
            )
            return
        
        # Make prediction
        try:
            prediction = self.model.predict(processed)[0]
            probabilities = self.model.predict_proba(processed)[0]
            
            # Get result
            if prediction == 1:
                result = "⚠️ CANCER DETECTED"
                confidence = probabilities[1] * 100
                color = "#f44336"  # Red
            else:
                result = "✅ NO CANCER"
                confidence = probabilities[0] * 100
                color = "#4CAF50"  # Green
            
            # Display result
            result_text = f"{result}\n\nConfidence: {confidence:.1f}%"
            
            self.result_label.config(
                text=result_text,
                font=("Arial", 16, "bold"),
                fg=color
            )
            
            # Show detailed info
            file_name = os.path.basename(file_path)
            details = f"\nImage: {file_name}\nPrediction: {'Cancer' if prediction == 1 else 'No Cancer'}\nConfidence: {confidence:.2f}%"
            print(details)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.result_label.config(
                text="❌ Prediction failed",
                fg="red"
            )

def main():
    """Main function"""
    root = tk.Tk()
    app = SimpleLungCancerDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()
