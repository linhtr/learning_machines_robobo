from PIL import Image
import glob

images = glob.glob("week4/images/flip/*")
for image in images:
    with open(image, 'rb') as file:
        img = Image.open(file)
        flipped_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        # flipped_image.show()
        flipped_image.save(image)
