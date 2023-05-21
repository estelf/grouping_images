"""
特徴量まとめ
"""
import cv2
import numpy as np

from PIL import Image
from skimage.feature import hog


def all_BGR(img):
    return img.reshape(1, -1)[0]


def CLR_all_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(1, -1)[0]


def CLR_H_HLS(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 0].reshape(1, -1)[0]


def CLR_H_HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0].reshape(1, -1)[0]


def CLR_AB_LAB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1:3].reshape(1, -1)[0]


def CLR_UV_LUV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:, :, 1:3].reshape(1, -1)[0]


def CLR_UV_YUV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 1:3].reshape(1, -1)[0]


def ML_deepL(img):
    import timm
    import torch
    import torchvision
    import torchvision.transforms as transforms

    transform_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ]
    )
    img = [transform_data(Image.fromarray(img[:, :, ::-1]))]
    train_loader = torch.utils.data.DataLoader(img, batch_size=1)

    # デバイスモデル設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("efficientnet_b0", pretrained=True)

    model = model.to(device)
    model.eval()
    for i in train_loader:
        image = i.to(device)
        preds = model(image)
    return preds.cpu().detach().numpy().copy()[0]


def ML_Laplacian(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    smoothed = cv2.GaussianBlur(img, (3, 3), 3, 3)
    laplace2 = cv2.filter2D(smoothed, cv2.CV_8U, kernel)
    return laplace2.reshape(1, -1)[0]


def ML_hog(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    return hog_image.reshape(1, -1)[0]


def H_ColorMomentHash(img):
    hash_func = cv2.img_hash.ColorMomentHash_create()
    tem = hash_func.compute(img)[0]
    return tem


def load():
    pass
