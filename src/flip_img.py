from PIL import Image
import glob
import os


# def flip_image(image_path, saved_location):
#     """
#     Flip or mirror the image
#
#     @param image_path: The path to the image to edit
#     @param saved_location: Path to save the flipped image
#     """
#     image_obj = Image.open(image_path)
#     rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
#     rotated_image.save(saved_location)
#     rotated_image.show()
#
#
# if __name__ == '__main__':
#     image = 'mantis.png'
#     flip_image(image, 'flipped_mantis.jpg')

images = glob.glob("week4/images/flip/*")
for image in images:
    with open(image, 'rb') as file:
        img = Image.open(file)
        flipped_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        # flipped_image.show()
        flipped_image.save(image)
