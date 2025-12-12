from PIL import Image
import sys

# 用法: python resize.py input.png output.png
img = Image.open(sys.argv[1])
img = img.resize((256, 256))
img.save(sys.argv[2])