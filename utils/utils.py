import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import matplotlib
import torch.nn.functional as F
from utils.data_loading import BasicDataset
from copy import deepcopy

new_size = 256


def convert_unet_input(numpy_img):
    if numpy_img.ndim == 2:
        img = numpy_img[np.newaxis, ...]
    else:
        img = numpy_img.transpose((2, 0, 1))

    if (img > 1).any():
        img = img / 255.0

    return img


def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(convert_unet_input(full_img))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (new_size, new_size), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def logging(message: str):
    print(f"[INFO] {message}")


def resize_image(image, size):
    h, w = image.shape[:2]
    ratio = size / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    rescaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = rescaled
    return canvas


def get_latest_predict_path():
    predict_path = os.path.join("runs", "detect")
    return os.path.join(predict_path, sorted(os.listdir(predict_path))[-1])


def load_all_crop_by_image(crop_path: str, image):
    crop_image_by_image = []
    for file in os.listdir(crop_path):
        split_image_name = image.split(".")
        split_image_name.pop(2)
        image_name = os.path.split(".".join(split_image_name))[-1]
        if image_name[:-4] in file:
            crop_image_by_image.append(os.path.join(crop_path, file))
    return [resize_image(cv2.imread(crop_image), 256) for crop_image in crop_image_by_image]


def plot_result(output_path, input, pred_mask=None):
    if pred_mask is not None:
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('Input image')
        ax[0].imshow(input)
        ax[1].set_title('mask')
        ax[1].imshow(pred_mask)
        plt.savefig(output_path)
        plt.close()
        return
    plt.imshow(input)
    plt.savefig(output_path)
    plt.close()
