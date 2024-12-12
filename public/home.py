import pydicom
from PIL import Image
import numpy as np
import cv2
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os
import pywt

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


# def encode_lsb(image_path, data, output_path='encoded_image.png'):
#     image = Image.open(image_path)
#     image_array = np.array(image)

#     binary_data = ''.join(format(ord(char), '08b') for char in data)
#     data_index = 0
#     data_len = len(binary_data)

#     for row in range(image_array.shape[0]):
#         for col in range(image_array.shape[1]):
#             for color in range(3):  
#                 if data_index < data_len:
                    
#                     pixel_bin = format(image_array[row, col, color], '08b')

#                     pixel_bin = pixel_bin[:-1] + binary_data[data_index]
#                     image_array[row, col, color] = int(pixel_bin, 2)
#                     data_index += 1

#     encoded_image = Image.fromarray(image_array)
#     encoded_image.save(output_path)
#     print(f"Data encoded successfully")


# # PVD embedding function
# def embed_pvd(image_path, data, output_image='pvd_encoded_image.png'):
#     image = Image.open(image_path)
#     image_array = np.array(image)

#     binary_data = ''.join(format(ord(char), '08b') for char in data)
#     data_index = 0
#     data_len = len(binary_data)

#     for i in range(0, image_array.shape[0] - 1, 2): 
#         for j in range(0, image_array.shape[1], 1):
#             if data_index < data_len:
#                 for channel in range(3):  # For R, G, B channels
#                     p1 = int(image_array[i, j, channel])
#                     p2 = int(image_array[i+1, j, channel])
                    
#                     diff = abs(p1 - p2)
#                     bits_to_embed = len(bin(diff)[2:])  # Number of bits to embed based on the difference
                    
#                     if data_index + bits_to_embed <= data_len:
#                         secret_bits = binary_data[data_index:data_index + bits_to_embed]
#                         secret_value = int(secret_bits, 2)
                        
#                         new_diff = secret_value
#                         if p1 > p2:
#                             p1 = p2 + new_diff
#                         else:
#                             p2 = p1 + new_diff
                        
#                         # Clamp the pixel values to stay within [0, 255]
#                         p1 = max(0, min(255, p1))
#                         p2 = max(0, min(255, p2))
                        
#                         image_array[i, j, channel] = p1
#                         image_array[i+1, j, channel] = p2

#                         data_index += bits_to_embed

#     encoded_image = Image.fromarray(image_array)
#     encoded_image.save(output_image)
#     print(f"Data embedded successfully in {output_image}")

# # PVD extraction function
# def extract_pvd(image_path, data_length):
#     image = Image.open(image_path)
#     image_array = np.array(image)

#     extracted_bits = ''

#     for i in range(0, image_array.shape[0] - 1, 2): 
#         for j in range(0, image_array.shape[1], 1):
#             if len(extracted_bits) < data_length * 8:
#                 p1 = int(image_array[i, j])
#                 p2 = int(image_array[i+1, j])
                
#                 diff = abs(p1 - p2)
#                 extracted_bits += format(diff, '08b')

#     all_bytes = [extracted_bits[i:i+8] for i in range(0, len(extracted_bits), 8)]
#     decoded_message = ''.join([chr(int(byte, 2)) for byte in all_bytes])

#     print(f"Extracted message: {decoded_message}")

# Embedding Function
def embed_secret(cover_image_path, secret_image_path, output_path, alpha=0.01):
    cover_image = cv2.imread(cover_image_path, cv2.IMREAD_GRAYSCALE)
    secret_image = cv2.imread(secret_image_path, cv2.IMREAD_GRAYSCALE)

    secret_image = cv2.resize(secret_image, (cover_image.shape[1] // 2, cover_image.shape[0] // 2))

    coeffs2 = pywt.dwt2(cover_image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    secret_image = cv2.resize(secret_image, (LH.shape[1], LH.shape[0]))

    LH_embedded = LH + (alpha * secret_image)
    HL_embedded = HL + (alpha * secret_image)
    HH_embedded = HH + (alpha * secret_image)

    stego_coeffs = (LL, (LH_embedded, HL_embedded, HH_embedded))
    stego_image = pywt.idwt2(stego_coeffs, 'haar')
    stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, stego_image)
    print(f"Stego image saved as {output_path}")


def extract_secret(stego_image_path, cover_image_path, alpha=0.01):
    if not os.path.exists(stego_image_path):
        raise FileNotFoundError(f"Stego image not found at {stego_image_path}")
    if not os.path.exists(cover_image_path):
        raise FileNotFoundError(f"Cover image not found at {cover_image_path}")

    stego_image = cv2.imread(stego_image_path, cv2.IMREAD_GRAYSCALE)
    cover_image = cv2.imread(cover_image_path, cv2.IMREAD_GRAYSCALE)

    if stego_image is None:
        raise ValueError("Failed to load stego image. Check the file path or format.")
    if cover_image is None:
        raise ValueError("Failed to load cover image. Check the file path or format.")
    if stego_image.ndim != 2 or cover_image.ndim != 2:
        raise ValueError("Both images must be 2D grayscale arrays for DWT processing.")

    # Perform DWT
    coeffs2_stego = pywt.dwt2(stego_image, 'haar')
    coeffs2_cover = pywt.dwt2(cover_image, 'haar')
    _, (LH_stego, HL_stego, HH_stego) = coeffs2_stego
    _, (LH_cover, HL_cover, HH_cover) = coeffs2_cover

    extracted_secret = (LH_stego - LH_cover + HL_stego - HL_cover + HH_stego - HH_cover) / (3 * alpha)
    extracted_secret = np.clip(extracted_secret, 0, 255).astype(np.uint8)

    cv2.imshow('Extracted Secret Image', extracted_secret)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def apply_dct_watermarking(image_path, watermark_text):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = np.float32(image)
    image_dct = cv2.dct(image)

    watermark_binary = ''.join(format(ord(char), '08b') for char in watermark_text)
    
    block_size = 8
    h, w = image.shape
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image_dct[i:i+block_size, j:j+block_size]
            if len(watermark_binary) > 0:
                block[block_size-1, block_size-1] = int(watermark_binary[0])
                watermark_binary = watermark_binary[1:]

    watermarked_image = cv2.idct(image_dct)
    watermarked_image = np.uint8(watermarked_image)
    cv2.imwrite('watermarked_image.png', watermarked_image)
    print("Watermark applied")

def encrypt_image(image_path, key, output_path='encrypted_image.png'):
    with open(image_path, 'rb') as f:
        image_data = f.read()

    iv = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    encrypted_data = cipher.encrypt(pad(image_data, AES.block_size))
    
    with open(output_path, 'wb') as f:
        f.write(iv + encrypted_data)
    print(f"Image encrypted")


def decrypt_image(encrypted_image_path, key, output_path='decrypted_image.png'):
    with open(encrypted_image_path, 'rb') as f:
        iv = f.read(16)  
        encrypted_data = f.read()

    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)
    
    with open(output_path, 'wb') as f:
        f.write(decrypted_data)
    print(f"Image decrypted and saved as {output_path}")

def extract_dct_watermark(image_path, watermark_length):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = np.float32(image)
    image_dct = cv2.dct(image)

    watermark_binary = ""
    
    block_size = 8
    h, w = image.shape

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image_dct[i:i + block_size, j:j + block_size]
            
            watermark_bit = int(block[block_size - 1, block_size - 1])
            watermark_bit = 1 if watermark_bit > 0 else 0  
            watermark_binary += str(watermark_bit)

            if len(watermark_binary) >= watermark_length * 8:
                break

    watermark_text = ''.join(chr(int(watermark_binary[i:i+8], 2)) for i in range(0, len(watermark_binary), 8))
    print("Extracted Watermark:", watermark_text)

    return watermark_text



# def decode_lsb(image_path, output_dicom_path):
#     image = Image.open(image_path)
#     image_array = np.array(image)
    
#     if len(image_array.shape) == 2:  
#         is_rgb = False
#     else:  # RGB
#         is_rgb = True

#     binary_data = ''
    
#     for row in range(image_array.shape[0]):
#         for col in range(image_array.shape[1]):
#             if is_rgb:
#                 for color in range(3):  
#                     pixel_bin = format(image_array[row, col, color], '08b')
#                     binary_data += pixel_bin[-1]  
#             else:
#                 pixel_bin = format(image_array[row, col], '08b')
#                 binary_data += pixel_bin[-1]  

#     all_bytes = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
#     decoded_data = bytearray([int(byte, 2) for byte in all_bytes])

#     with open(output_dicom_path, 'wb') as f:
#         f.write(decoded_data)
#     print(f"Data decoded and saved as {output_dicom_path}")



def save_dicom_image(decoded_image_path, output_dicom_path):
    with open(decoded_image_path, 'rb') as f:
        dicom_data = f.read()

    with open(output_dicom_path, 'wb') as f:
        f.write(dicom_data)
    print(f"DICOM image saved as {output_dicom_path}")




if __name__ == "__main__":
    dicom_path = 'D:\\college work\\Steganography in Medical Images\\public\\image\\0002.dcm'
    # slice_index = 48
    medical_image = load_dicom_image(dicom_path)
    preprocessed_image = preprocess_image(medical_image)
    preprocessed_image.save('preprocessed_medical_image.png')

    cover_image_path = 'D:\\college work\\Steganography in Medical Images\\public\\image\\32819.jpg'
    output_image_path1 = 'encoded_image.png'

    
    with open('preprocessed_medical_image.png', 'rb') as f:
        medical_image_data = f.read()
    binary_data = ''.join(format(byte, '08b') for byte in medical_image_data)

    # encode_lsb(cover_image_path, binary_data, output_image_path1)
    # embed_pvd(cover_image_path, binary_data, output_image_path1)
    embed_secret(cover_image_path, 'preprocessed_medical_image.png', output_image_path1)

    watermark_text = "Confidential"
    apply_dct_watermarking(output_image_path1, watermark_text)


    key = b'ED8354EC445D4777' 
    encrypted_image_path = 'encrypted_image.png'
    encrypt_image('watermarked_image.png', key, encrypted_image_path)

    decrypted_image_path = 'decrypted_image.png'
    decrypt_image(encrypted_image_path, key, decrypted_image_path)

    extracted_watermark = extract_dct_watermark(decrypted_image_path, watermark_length=len("Confidential"))

    output_dicom_image_path = 'decoded_dicom_image.png'
    # decode_lsb(decrypted_image_path, output_dicom_image_path)
    extract_secret(decrypted_image_path, cover_image_path)

    final_dicom_path = 'recovered_medical_image.dcm'
    save_dicom_image(output_dicom_image_path, final_dicom_path)