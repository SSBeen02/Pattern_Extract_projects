
import cv2



T = cv2.imread("extended_patch_dx3_dy3.png")
I = cv2.imread("original_crop_dx3_dy3.png")


result = cv2.matchTemplate(T, I, cv2.TM_CCOEFF_NORMED)
print(result)


T2 = cv2.imread("extended_patch_dx2_dy2.png")
I2 = cv2.imread("original_crop_dx2_dy2.png")


result2 = cv2.matchTemplate(T2, I2, cv2.TM_CCOEFF_NORMED)
print(result2)
