import cv2
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import pytesseract
from difflib import SequenceMatcher

def download_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def compare_text(base_image_url, images_to_compare_urls, threshold=0.7):
    base_image = download_image(base_image_url)
    base_text = pytesseract.image_to_string(base_image)

    results = {}
    for image_url in images_to_compare_urls:
        image_to_compare = download_image(image_url)
        compare_text = pytesseract.image_to_string(image_to_compare)

        similarity_ratio = SequenceMatcher(None, base_text, compare_text).ratio()
        if similarity_ratio >= threshold:
            results[image_url] = similarity_ratio

    return results

base_image_url = 'https://oechsle.vteximg.com.br/arquivos/ids/1352275-1000-1000/image-0ca1c72e498747f086ad9541ff3c56b0.jpg?v=637494732303400000'
images_to_compare_urls = ['https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/02/template_matching_coke_bottle_incorrect.png?size=630x280&lossy=2&strip=1&webp=1','https://oechsle.vteximg.com.br/arquivos/ids/1352275-1000-1000/image-0ca1c72e498747f086ad9541ff3c56b0.jpg?v=637494732303400000','https://s1.eestatic.com/2023/03/10/curiosidades/mascotas/747436034_231551832_1706x1280.jpg']

results = compare_text(base_image_url, images_to_compare_urls)

for image_url, similarity_ratio in results.items():
    print(f'Image URL: {image_url}, Similarity Ratio: {similarity_ratio}')
