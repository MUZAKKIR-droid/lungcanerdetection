# Lung Cancer Detection

This application aims for early detection of lung cancer to give patients the best chance at recovery and survival using Machine Learning. The model analyzes high-resolution lung scans to accurately determine when lesions in the lungs are cancerous, reducing false positive rates and enabling earlier access to life-saving interventions.

![Lung-Cancer-Detection](https://user-images.githubusercontent.com/68781375/162584408-450580c0-3354-470b-a69c-180a19802fd4.jpg)

## Contributors

**Aditya** ● **Muzakkir**

## Features

- Simple command-line interface for quick image analysis
- Trained Random Forest model with high accuracy
- Support for standard image formats (PNG, JPG)
- Easy-to-use prediction system

## Installation

1. Create virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simple detector:
```bash
python detect_simple.py your_image.jpg
```

Or use interactive mode:
```bash
python detect_simple.py
```

## Dataset

The model is trained on synthetic data for demonstration purposes. For production use, real DICOM datasets should be used.

## Model Information

- **Type**: Random Forest Classifier
- **Training Accuracy**: 100% (on synthetic data)
- **Input**: Grayscale images (resized to 10x10)
- **Output**: Cancer/No Cancer prediction with confidence score

## Disclaimer

⚠️ This is a demonstration project. Not intended for actual medical diagnosis. Always consult qualified medical professionals for health concerns.
