# Steganography in Medical Images

## Project Overview

This project explores steganographic techniques for embedding hidden data within digital medical images. It provides a comparative analysis of three popular techniques: Least Significant Bit (LSB), Pixel Value Differencing (PVD), and Discrete Wavelet Transform (DWT), highlighting their performance in terms of imperceptibility, robustness, and embedding capacity. The project focuses on enhancing data security while ensuring the visual integrity of medical images.

## Features

- Implementation of three steganographic techniques:
  - **LSB (Least Significant Bit):** Simplest approach with high embedding capacity.
  - **PVD (Pixel Value Differencing):** Adaptive method balancing imperceptibility and capacity.
  - **DWT (Discrete Wavelet Transform):** Advanced technique offering robustness and imperceptibility.

- Analysis of techniques using evaluation metrics like PSNR, MSE, and Capacity.

## Technologies Used

- **Programming Language:** Java
- **Libraries:** pydicom, PIL, NumPy, OpenCV (cv2), PyCryptodome

## How It Works

1. **Select a Cover Image:** Ensure the image is in JPEG or PNG format.
2. **Choose a Technique:** Select from LSB, PVD, or DWT for data embedding.
3 **Embed Data:** Upload the secret data file and embed it into the selected cover image.
4. **Save Stego-Image:** Save the image with embedded data.
5. **Extract Data:** Load the stego-image and retrieve the hidden data using the extraction option.

## Installation

1. Clone the repository:
  ```sh
    git clone https://github.com/shreyakushw/steganography-medical-images.git
  ```
2. Open the project in Visual Studio.

3. Ensure the required libraries are included in the build path.

4. Run the main application.

