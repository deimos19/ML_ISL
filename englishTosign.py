import os

from PIL import Image

IMAGE_FOLDER = 'eng2sign'

def get_sign_image(letter):
    # Open the image
    file_name = f'{letter.upper()}.jpg'
    img_path = os.path.join(IMAGE_FOLDER, file_name)
    img = Image.open(img_path)
    return img


def text_to_sign_language(text):
    # Create a blank space
    img_list = []
    for char in text:
        if char.isalpha():  # Check if character is a letter
            img = get_sign_image(char)
            if img:
                img_list.append(img)

    # Combine all images horizontally
    total_width = sum(img.width for img in img_list)
    max_height = max(img.height for img in img_list)

    combined_img = Image.new('RGB', (total_width, max_height), (0, 0, 0))
    x_offset = 0
    for img in img_list:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return combined_img


# Example usage
text = "deva"
result_img = text_to_sign_language(text)
# Display the image
if result_img:
    result_img.show()
