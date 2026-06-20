# pip install opencv-python scikit-image pydicom matplotlib numpy
# Load DICOM Image***********
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# Load DICOM file
dicom_path = "sample.dcm"
ds = pydicom.dcmread(dicom_path)

# Extract pixel array
image = ds.pixel_array.astype(np.float32)

# Normalize to 0–255
image = (image - np.min(image)) / (np.max(image) - np.min(image))
image = (image * 255).astype(np.uint8)

# Display
plt.imshow(image, cmap='gray')
plt.title("Original DICOM Image")
plt.axis('off')
plt.show()

# Noise Removal*********
import cv2

# Gaussian Blur (smooth noise)
gaussian = cv2.GaussianBlur(image, (5, 5), 0)

# Median Blur (salt-and-pepper noise)
median = cv2.medianBlur(image, 5)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.imshow(gaussian, cmap='gray'); plt.title("Gaussian Blur")
plt.subplot(1,2,2); plt.imshow(median, cmap='gray'); plt.title("Median Blur")
plt.show()

# Contrast Enhancement**********
# Histogram Equalization
hist_eq = cv2.equalizeHist(image)

# CLAHE (better for medical images)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(image)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.imshow(hist_eq, cmap='gray'); plt.title("Histogram Equalization")
plt.subplot(1,2,2); plt.imshow(clahe_img, cmap='gray'); plt.title("CLAHE")
plt.show()

# Resizing Image************
# Resize to standard size (e.g., 224x224 for deep learning)
resized = cv2.resize(image, (224, 224))

plt.imshow(resized, cmap='gray')
plt.title("Resized Image (224x224)")
plt.axis('off')
plt.show()

# Using scikit-image (Advanced Filtering)**********
from skimage import filters

# Apply Sobel edge detection
edges = filters.sobel(image)

plt.imshow(edges, cmap='gray')
plt.title("Edge Detection (Sobel)")
plt.axis('off')
plt.show()

# Full Preprocessing Pipeline (Reusable)*********
def preprocess_dicom(path, size=(224, 224)):
    import pydicom
    import cv2
    import numpy as np

    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    # Normalize
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img * 255).astype(np.uint8)

    # Noise removal
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # Resize
    img = cv2.resize(img, size)

    return img


# Example usage
processed_img = preprocess_dicom("sample.dcm")

plt.imshow(processed_img, cmap='gray')
plt.title("Final Preprocessed Image")
plt.axis('off')
plt.show()

