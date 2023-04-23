import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F

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
                device, size,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(convert_unet_input(full_img))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (size, size), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def logging(message: str):
    print(f"[INFO] {message}")


def resize_image(image, size):
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 1
        image = np.expand_dims(image, axis=-1)
    ratio = size / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    rescaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, c), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w, :] = rescaled
    return canvas


def clean_chromosome(resized_im, mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate((mask, mask, mask),
                          axis=-1)
    crops = resized_im * mask  #
    return crops


def get_latest_predict_path():
    predict_path = os.path.join("runs", "detect")
    return os.path.join(predict_path, sorted(os.listdir(predict_path))[-1])


def crop_chromosome_from_origin(org_img, box, size):
    x, y, w, h = [round(num.item()) for num in box.xywh[0]]
    xmin, ymin = int(x - w / 2), int(y - h / 2)
    xmax, ymax = int(x + w / 2), int(y + h / 2)
    crop = org_img[ymin:ymax, xmin:xmax]
    crop = resize_image(crop, size)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = np.expand_dims(crop, axis=-1)
    return crop


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
