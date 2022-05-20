from PIL import Image
import os

folder = "JPGImgs"
os.makedirs(folder, exist_ok=True)

JPEG_list = []

input_folder = 'JPEGImgs'
for subfolder in os.listdir(input_folder):
    internal_folder = os.path.join(input_folder, subfolder)
    internal_files = os.listdir(internal_folder)
    for image in internal_files:
        JPEG_list.append(os.path.join(internal_folder, image))

print(JPEG_list)

for filename in JPEG_list:
    image = Image.open(filename)
    basename = os.path.basename(filename)
    basename = basename.replace(".jpeg", ".jpg")
    #print(basename)
    newPath = os.path.join(folder, basename)
    image.save(newPath)