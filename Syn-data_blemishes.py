import random
import numpy as np
from PIL import Image, ImageDraw
import os
import csv
import cv2

num_images = 3000
image_size = 256
augment_percent = 0.8

os.makedirs('new_syn_images', exist_ok=True)
os.makedirs('new_syn_masks', exist_ok=True)

# Object info
object_info = []


def random_channel_shift(image, intensity=25):
    # Split the channels
    r, g, b = cv2.split(image)

    # Add a random value to each channel
    r = cv2.add(r, np.random.randint(-intensity, intensity))
    g = cv2.add(g, np.random.randint(-intensity, intensity))
    b = cv2.add(b, np.random.randint(-intensity, intensity))

    # Merge the channels back and return
    return cv2.merge((r, g, b))

for i in range(num_images):

    # Generate image
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # Objects
    quad_cols = random.choice([2, 3, 4])
    quad_rows = random.choice([2, 3, 4])
    quad_width = image_size // quad_cols
    quad_height = image_size // quad_rows

    for x in range(quad_cols):
        for y in range(quad_rows):

            shape = random.choice(['circle', 'square', 'triangle'])
            obj_size = random.randint(5, min(quad_width, quad_height) // 2)
            obj_x = random.randint(x * quad_width + obj_size, (x + 1) * quad_width - obj_size)
            obj_y = random.randint(y * quad_height + obj_size, (y + 1) * quad_height - obj_size)

            if shape == 'circle':
                centre = (obj_x, obj_y)
                points = []
                for angle in range(0, 360, 10):
                    x = int(obj_size * np.cos(np.deg2rad(angle))) + centre[0]
                    y = int(obj_size * np.sin(np.deg2rad(angle))) + centre[1]
                    points.append((x, y))

            elif shape == 'square':
                points = [(obj_x, obj_y), (obj_x + obj_size, obj_y), (obj_x + obj_size, obj_y + obj_size),
                          (obj_x, obj_y + obj_size)]

            else:
                points = [(obj_x, obj_y), (obj_x + obj_size, obj_y), (obj_x, obj_y + obj_size)]

            if shape == 'circle':
                image = cv2.circle(image, (obj_x, obj_y), obj_size, (255, 255, 255), -1)

            elif shape == 'square':
                image = cv2.rectangle(image, (obj_x, obj_y), (obj_x + obj_size, obj_y + obj_size), (255, 0, 0), -1)

            else:
                x2 = obj_x + obj_size
                y2 = obj_y
                x3 = obj_x
                y3 = obj_y + obj_size
                image = cv2.fillConvexPoly(image, np.array([[obj_x, obj_y], [x2, y2], [x3, y3]]), (0, 0, 255))

            object_info.append([i, shape, str(points)])

    # Augment image
    if random.random() < augment_percent:
        aug = random.choice(['blur', 'sharpen', 'noise','channel_shift'])
        if aug == 'blur':
            ksize = random.randint(5, 21)
            if ksize % 2 == 0:
                ksize += 1
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        elif aug == 'sharpen':
            ksize = random.randint(5, 20)
            kernel = np.array([[-1, -1, -1],
                               [-1, ksize, -1],
                               [-1, -1, -1]])
            image = cv2.filter2D(image, -1, kernel)

        elif aug == 'noise':
            noise = np.random.randn(*image.shape) * random.randint(10, 80)
            image = image + noise
            image = np.clip(image, 0, 255).astype(np.uint8)

        elif aug == 'channel_shift':
            image = random_channel_shift(image)

    # Save image
    img = Image.fromarray(image)
    img.save(f'new_syn_images/image_{i}.png')

    # Save mask
    mask_image = Image.new('L', (image_size, image_size), color=255)
    draw = ImageDraw.Draw(mask_image)
    for obj_data in object_info:
        if obj_data[0] == i:
            shape = obj_data[1]
            points = eval(obj_data[2])
            draw.polygon(points, outline=0)
    mask_image.save(f'new_syn_masks/mask_{i}.png')

# Save object info to CSV
csv_filename = 'new_syn_object_coord.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Image_ID', 'Shape', 'Coordinates'])
    csv_writer.writerows(object_info)

print('Generated {} images and masks'.format(num_images))
print('Augmented {} images'.format(int(augment_percent * num_images)))