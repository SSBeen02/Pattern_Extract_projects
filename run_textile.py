import os
import cv2
import numpy as np
import textile
from textile.utils.image_utils import read_and_process_image

candidate_paths = [
    # r"D:\textile\textile\data\texture_368.jpg",
    r"D:\textile\my_pattern\pattern8_2.png"
]
img_path = next((p for p in candidate_paths if os.path.exists(p)), None)
out_path = r"D:\textile\output_textile_result8_2.png"

if img_path is None:
    raise FileNotFoundError("No valid input image found. Please set img_path to an existing image file.")

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")

# Always save an image file, even if inference fails.
# Create a tiled version of the image so the output visually shows repeated tiles.
tile_factor = 2
result_img = np.tile(img, (tile_factor, tile_factor, 1))
value_text = "N/A"

try:
    image = read_and_process_image(img_path)
    loss_textile = textile.Textile()
    value = loss_textile(image)
    value_text = f"{value.detach().cpu().item():.6f}"

    # Add the score as text overlay on the output image.
    cv2.putText(
        result_img,
        f"TexTile: {value_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
except Exception as e:
    cv2.putText(
        result_img,
        f"TexTile: error",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    print("inference failed:", e)

cv2.imwrite(out_path, result_img)
print("saved:", out_path)
print("value:", value_text)
