import os
import time
from PIL import Image, UnidentifiedImageError

# Standard deep learning baseline size
TARGET_SIZE = (256, 256) 
MASTER_FOLDER = '/home/azwad/Works/Multimodal-CBM/mimic_cxr_organized'

print(f"Starting continuous background resizer. Target size: {TARGET_SIZE}")

while True:
    images_processed_this_round = 0

    # Sweep through every folder inside your master directory
    for root, _, files in os.walk(MASTER_FOLDER):
        for file in files:
            if file.endswith('.jpg'):
                filepath = os.path.join(root, file)

                try:
                    # Attempt to open the image
                    with Image.open(filepath) as img:
                        # 1. Skip if already resized
                        if img.size == TARGET_SIZE:
                            continue
                        
                        # 2. Resize the image (LANCZOS is best for medical image clarity)
                        resized_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                        
                    # 3. Overwrite the massive original file to save space
                    resized_img.save(filepath, format='JPEG', quality=85)
                    print(f"Successfully resized: {file}")
                    images_processed_this_round += 1

                except (UnidentifiedImageError, OSError):
                    # 4. ERROR CAUGHT! The file is currently being written by the downloader.
                    # We safely silently skip it. It will be resized on the next loop.
                    continue

    # 5. The Sleep Trigger
    if images_processed_this_round == 0:
        print("Caught up with the downloader. Sleeping for 15 seconds...")
        time.sleep(15)
    else:
        print(f"Sweep complete. Resized {images_processed_this_round} images. Restarting sweep...")