# Diabetic Retinopathy Detection (DDR)

This project implements image processing techniques for detecting diabetic retinopathy from fundus images using OpenCV and Python.

## Features

- Optic disc detection and removal
- Blood vessel segmentation
- Fovea segmentation
- Morphological operations for feature extraction

## Dependencies

- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- Matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ddr
   ```

2. Install required packages:
   ```bash
   pip install opencv-python numpy pillow matplotlib
   ```

## Usage

1. Place your fundus images in the project directory (e.g., `3.jpg`, `1mac.jpg`)

2. Run the main detection script:
   ```bash
   python ddr/ddr.py
   ```

3. For morphological analysis:
   ```bash
   python ddr/MA.py
   ```

## Files

- `ddr/ddr.py`: Main script for diabetic retinopathy detection including optic disc removal and vessel segmentation
- `ddr/MA.py`: Morphological analysis script for image processing

## Notes

- Ensure fundus images are in the correct format and resolution
- The scripts expect specific image filenames as hardcoded in the code
- Output will be displayed using OpenCV windows
