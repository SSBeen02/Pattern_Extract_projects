import os
import cv2
import numpy as np
import textile
from textile.utils.image_utils import read_and_process_image

loss_textile = textile.Textile() 
image = read_and_process_image(r"D:\textile\my_pattern\pattern1.png")
textile_value = loss_textile(image)

print(textile_value)

