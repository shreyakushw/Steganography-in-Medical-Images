import pydicom
from PIL import Image
import numpy as np


def load_dicom_image(dicom_path, slice_index=48):
    dicom_image = pydicom.dcmread(dicom_path)
    image_array = dicom_image.pixel_array

    if len(image_array.shape) == 3:
        if slice_index >= image_array.shape[0]:
            raise ValueError(f"Slice index {slice_index} out of range.")
        image_array = image_array[slice_index]

    print("Shape of image_array:", image_array.shape)
    print("Data type of image_array:", image_array.dtype)

    image = Image.fromarray(image_array)
    image = image.convert('RGB')

    return image

def preprocess_image(image):
    resized_image = image.resize((512, 512))
    return resized_image


def encode_lsb(image_path, data, output_path='encoded_image.png'):
    image = Image.open(image_path)
    image_array = np.array(image)

    binary_data = ''.join(format(ord(char), '08b') for char in data)
    data_index = 0
    data_len = len(binary_data)

    for row in range(image_array.shape[0]):
        for col in range(image_array.shape[1]):
            for color in range(3):  
                if data_index < data_len:
                    
                    pixel_bin = format(image_array[row, col, color], '08b')

                    pixel_bin = pixel_bin[:-1] + binary_data[data_index]
                    image_array[row, col, color] = int(pixel_bin, 2)
                    data_index += 1

    encoded_image = Image.fromarray(image_array)
    encoded_image.save(output_path)
    print(f"Data encoded successfully into {output_path}")


if __name__ == "__main__":
    dicom_path = 'D:\\college work\\Steganography in Medical Images\\public\\image\\0002.dcm'
    # slice_index = 48
    medical_image = load_dicom_image(dicom_path)
    preprocessed_image = preprocess_image(medical_image)
    preprocessed_image.save('preprocessed_medical_image.png')

    cover_image_path = 'D:\\college work\\Steganography in Medical Images\\public\\image\\32819.jpg'
    output_image_path = 'encoded_image.png'

    with open('preprocessed_medical_image.png', 'rb') as f:
        medical_image_data = f.read()
    binary_data = ''.join(format(byte, '08b') for byte in medical_image_data)

    encode_lsb(cover_image_path, binary_data, output_image_path)