import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pillow_heif import register_heif_opener

def heic_to_jpeg(heic_dir, jpeg_dir):
    register_heif_opener()  
    image = Image.open(heic_dir)
    image.save(jpeg_dir, "JPEG")

def remove_background(jpeg_dir, path_to_clean_image):
    if jpeg_dir[-4:] in ['heic', 'HEIC']:
        heic_to_jpeg(jpeg_dir, jpeg_dir[:-4] + 'jpg')
        jpeg_dir = jpeg_dir[:-4] + 'jpg'
    img = cv2.imread(jpeg_dir)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 80], dtype="uint8")
    upper = np.array([50, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    b, g, r = cv2.split(result)  
    filter = g.copy()
    ret, mask = cv2.threshold(filter, 10, 255, 1)
    img[mask == 255] = 255
    cv2.imwrite(path_to_clean_image, img)

def resize(path_to_warped_image, path_to_warped_image_clean, path_to_warped_image_mini, path_to_warped_image_clean_mini, resize_value):
    pil_img = Image.open(path_to_warped_image)
    pil_img_clean = Image.open(path_to_warped_image_clean)
    pil_img.resize((resize_value, resize_value), resample=Image.NEAREST).save(path_to_warped_image_mini)
    pil_img_clean.resize((resize_value, resize_value), resample=Image.NEAREST).save(path_to_warped_image_clean_mini)

def save_result_simple(im, path_to_result):
    """
    Save result with only the image and detected lines, no text descriptions
    """
    if im is None:
        print_error()
    else:
        plt.figure(figsize=(8, 8))
        plt.axis('off')  # Remove axes completely
        plt.imshow(im)
        plt.tight_layout()
        plt.savefig(path_to_result, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()  # Close the figure to free memory

def print_error():
    print('Palm lines not properly detected! Please use another palm image.')