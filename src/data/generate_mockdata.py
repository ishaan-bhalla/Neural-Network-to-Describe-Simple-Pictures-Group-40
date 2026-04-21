# Just a script to create mock data for model preparation
from PIL import Image
import os
os.makedirs("data/mock/images", exist_ok=True)
for i in range(1, 5):
    img = Image.new("RGB", (64, 64), "white")
    img.save(f"data/mock/images/img{i}.png")
print("Mock images created")
