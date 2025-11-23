"""
Generate sample DICOM files for demonstration
"""
import os
import numpy as np
import pydicom as dicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import datetime

def create_sample_dicom(output_dir, patient_id, slice_num, is_cancer=False):
    """Create a sample DICOM file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create file meta info
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
    
    # Create file dataset
    file_ds = FileDataset(
        os.path.join(output_dir, f'slice_{slice_num:03d}.dcm'),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128
    )
    
    # Set basic DICOM attributes
    file_ds.PatientName = f"Patient^{patient_id}"
    file_ds.PatientID = str(patient_id)
    file_ds.Modality = "CT"
    file_ds.SeriesInstanceUID = generate_uid()
    file_ds.StudyInstanceUID = generate_uid()
    file_ds.SOPInstanceUID = generate_uid()
    file_ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    file_ds.StudyDate = datetime.date.today().isoformat().replace('-', '')
    file_ds.ContentDate = file_ds.StudyDate
    file_ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
    file_ds.ContentTime = file_ds.StudyTime
    file_ds.InstanceNumber = slice_num
    file_ds.ImagePositionPatient = [0, 0, float(slice_num * 2)]  # Z position for slice ordering
    
    # Generate synthetic pixel data
    if is_cancer:
        # Add synthetic "abnormality" to cancer images
        pixel_array = np.random.randint(100, 200, size=(512, 512), dtype=np.uint16)
        # Add a bright spot (lesion-like feature)
        center_x, center_y, radius = 250, 250, 30
        for i in range(512):
            for j in range(512):
                if (i - center_y)**2 + (j - center_x)**2 <= radius**2:
                    pixel_array[i, j] = np.random.randint(250, 255)
    else:
        # Normal lung tissue
        pixel_array = np.random.randint(50, 150, size=(512, 512), dtype=np.uint16)
    
    file_ds.PixelData = pixel_array.tobytes()
    file_ds.Rows = 512
    file_ds.Columns = 512
    file_ds.BitsAllocated = 16
    file_ds.BitsStored = 16
    file_ds.HighBit = 15
    file_ds.PixelRepresentation = 0
    file_ds.SamplesPerPixel = 1
    file_ds.PhotometricInterpretation = "MONOCHROME2"
    
    # Save file
    file_ds.save_as(file_ds.filename)
    print(f"Created: {file_ds.filename}")

def main():
    base_dir = 'Datasetssss'
    os.makedirs(base_dir, exist_ok=True)
    
    print("Creating sample DICOM dataset...")
    
    # Create normal cases (no cancer)
    for patient_id in range(1, 4):  # 3 normal patients
        patient_dir = os.path.join(base_dir, f'patient_{patient_id:03d}')
        os.makedirs(patient_dir, exist_ok=True)
        # Create 10 slices per patient
        for slice_num in range(1, 11):
            create_sample_dicom(patient_dir, patient_id, slice_num, is_cancer=False)
    
    # Create cancer cases
    for patient_id in range(4, 7):  # 3 cancer patients
        patient_dir = os.path.join(base_dir, f'patient_{patient_id:03d}')
        os.makedirs(patient_dir, exist_ok=True)
        # Create 10 slices per patient
        for slice_num in range(1, 11):
            create_sample_dicom(patient_dir, patient_id, slice_num, is_cancer=True)
    
    print(f"\nSample data created in '{base_dir}/' directory")
    print(f"Generated 6 patients (3 normal, 3 cancer) with 10 slices each")
    print("\nYou can now:")
    print("1. Run the GUI: python lcd_cnn.py")
    print("2. Click 'Import Data' (will auto-load from Datasetssss/)")
    print("3. Click 'Pre-Process Data'")
    print("4. Click 'Train Data'")
    print("5. Click 'Detect Cancer' to test predictions")

if __name__ == "__main__":
    main()
