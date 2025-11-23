"""Headless runner to import, preprocess, and train using LCD_CNN.
This script creates a hidden Tk root so Tkinter doesn't show the GUI, then runs the
pipeline programmatically. Use for CI/demo verification.
"""
import sys
import time
from tkinter import Tk

# Ensure we can import lcd_cnn from the project root
sys.path.insert(0, '.')

from lcd_cnn import LCD_CNN

if __name__ == '__main__':
    root = Tk()
    # hide the main window to run headless
    root.withdraw()
    app = LCD_CNN(root)

    print('Calling import_data()...')
    app.import_data()
    time.sleep(0.5)

    print('Calling preprocess_data()...')
    app.preprocess_data()
    time.sleep(0.5)

    print('Calling train_data()...')
    app.train_data()
    time.sleep(0.5)

    print('Headless run completed.')
    try:
        root.destroy()
    except Exception:
        pass
