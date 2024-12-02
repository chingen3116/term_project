from torchvision import transforms
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 載入 MiDaS 模型
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

# 載入影像
# 將影像轉換為 RGB 並調整大小
img = cv2.imread('Meerkat.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 使用PyTorch的transforms來調整大小
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 384)),  # MiDaS 要求輸入大小為 384x384
    transforms.ToTensor(),
])

input_tensor = transform(img).unsqueeze(0)

model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

# 推斷深度圖
with torch.no_grad():
    depth_map = model(input_tensor)

# 處理深度圖
depth_map = depth_map.squeeze().cpu().numpy()
depth_map = cv2.resize(depth_map, (img.shape[1], img.shape[0]))  # 調整回原始大小

# 顯示深度圖
plt.imshow(depth_map, cmap='inferno')
plt.show()
print("Depth map shape:", depth_map.shape)
print("Depth map dtype:", depth_map.dtype)
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)) 

def apply_variable_blur(image, depth_map, max_blur=50):
    """
    根據深度圖進行可變模糊處理，景深變淺。
    
    :param image: 輸入影像
    :param depth_map: 深度圖，範圍 [0,1]
    :param max_blur: 最大模糊半徑
    :return: 模糊後的影像
    """
    print("Applying variable blur...") 
    # 正規化深度圖到範圍 [0, 1]
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    print("Depth map normalized.")
    # 計算每個像素的模糊半徑
    blur_map = (depth_map * max_blur).astype(np.uint8)
    print("Blur map calculated.") 
    # 預處理影像
    output_image = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            blur_radius = blur_map[i, j] | 1  # 保證模糊半徑為奇數
            if blur_radius > 1:
                # 以當前像素為中心進行模糊處理
                print(f"Blur radius for pixel ({i},{j}): {blur_radius}")
                patch = image[max(i-blur_radius//2, 0):min(i+blur_radius//2, image.shape[0]),
                              max(j-blur_radius//2, 0):min(j+blur_radius//2, image.shape[1])]
                blurred_patch = cv2.GaussianBlur(patch, (blur_radius, blur_radius), 0)
                output_image[i, j] = blurred_patch[patch.shape[0]//2, patch.shape[1]//2]
    print("Saving output image...") 
    return output_image


# 應用可變模糊，模擬景深變淺效果
output_image = apply_variable_blur(img, depth_map, max_blur=50)
print(output_image.shape)
print(output_image.dtype)
# 將 RGB 影像轉換為 BGR
output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
print(output_image_bgr.shape)
print(output_image_bgr.dtype)
# 儲存結果
cv2.imwrite('MiDaS_blurred.jpg', output_image_bgr)
if cv2.imwrite('MiDaS_blurred.jpg', output_image_bgr):
    print("success!")
else:
    print("fail..")
