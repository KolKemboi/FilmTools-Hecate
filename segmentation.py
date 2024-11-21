import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import cvzone


# Load the pretrained DeepLabV3+ model with MobileNetV2 backbone
model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
model.eval()

image_path = "man.jpg"
in_image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet values
])

input_tensor = transform(in_image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)["out"][0]
pred_mask = output.argmax(0)

mask = pred_mask.byte().cpu().numpy()
human_class = 15  # COCO class ID for humans
binary_mask = np.where(mask == human_class, 255, 0).astype(np.uint8)

original_bgr = np.array(in_image)[:, :, ::-1]  # Convert RGB to BGR

stacked_image = cvzone.stackImages([original_bgr, binary_mask], 2, 2)  # 1 means stacking horizontally


cv2.imshow("Human Segmentation Mask", stacked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()