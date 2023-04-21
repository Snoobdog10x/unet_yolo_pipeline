import argparse
from utils.unet import UNetLite, UNet
from utils.utils import *
from ultralytics import YOLO

import torch

TRAINED_MODEL_PATH = "trained_models"
YOLO_MODEL_PATH = os.path.join(TRAINED_MODEL_PATH, "yolo.pt")
OUTPUT_PATH = "output"
DATA_PATH = "data"


def get_unet_model(device, bilinear, unet_type="NORMAL"):
    unet_path = os.path.join(TRAINED_MODEL_PATH, "unet.pth" if unet_type == "NORMAL" else "unet_lite.pth")
    if unet_type == "NORMAL":
        net = UNet(n_channels=3, n_classes=2, bilinear=bilinear)
    else:
        net = UNetLite(n_channels=3, n_classes=2, bilinear=bilinear)

    net.to(device=device)
    state_dict = torch.load(unet_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging(f"Unet {unet_type} loaded!")
    return net


def get_yolo_model():
    model = YOLO(YOLO_MODEL_PATH)
    return model


def pipline(device, yolo, unet):
    count = 0
    image_data = []
    chromosome_counts = 0;
    for file in os.listdir(os.path.join(DATA_PATH, "images")):
        file_path = os.path.join(DATA_PATH, "images", file)
        chromosome_nums = get_chromosome_nums(file)
        chromosome_counts += chromosome_nums
        logging(f"{file}: {chromosome_nums}")
        image_data.append(file_path)
        count += 1
        if count == 1:
            break
    results = yolo.predict(source=image_data, conf=0.5, line_thickness=1, save=True, hide_conf=False,
                           save_conf=True, save_crop=True)
    latest_predict_path = get_latest_predict_path()
    crop_path = os.path.join(latest_predict_path, "crops", "0-0")
    pred_chromosomes = 0
    for index, result in enumerate(results):
        pred_chromosome_count = len(result)
        image_path = result.path
        pred_chromosomes += pred_chromosome_count
        crop_chromosome = load_all_crop_by_image(crop_path, image_path)
        file_path = os.path.split(image_data[index])[-1]
        logging(f"{file_path}: {pred_chromosome_count}")
        for jindex, chromosome in enumerate(crop_chromosome):
            output_path = os.path.join(OUTPUT_PATH, file_path)
            os.makedirs(output_path, exist_ok=True)
            pred_mask = predict_img(net=unet, full_img=chromosome, device=device)
            plot_result(os.path.join(output_path, f"{jindex}.png"), chromosome, pred_mask)
    logging(f"accuracy: {pred_chromosomes / chromosome_counts}")


def parse_args():
    parser = argparse.ArgumentParser(description='test pipline')
    parser.add_argument('--data_path', '-dp', type=str, default="data", help='data path')
    parser.add_argument('--model_path', '-mp', type=str, default="trained_models",
                        help='folder contain unet, unet_lite pth and yolo.pt')
    parser.add_argument('--save_result_path', '-srp', type=str, default="output",
                        help='save result and plot')
    parser.add_argument('--model_type', '-mt', type=str, default="NORMAL",
                        help='choose model type: NORMAL or LITE')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    DATA_PATH = args.data_path
    OUTPUT_PATH = args.save_result_path
    TRAINED_MODEL_PATH = args.model_path
    YOLO_MODEL_PATH = os.path.join(TRAINED_MODEL_PATH, "yolo.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO(YOLO_MODEL_PATH)
    unet = get_unet_model(device=device, bilinear=False, unet_type=args.model_type)
    pipline(device=device, yolo=yolo, unet=unet)
