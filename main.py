import argparse
from utils.unet import UNetLite, UNet
from utils.utils import *
from ultralytics import YOLO

import torch

TRAINED_MODEL_PATH = "trained_models"
YOLO_MODEL_PATH = os.path.join(TRAINED_MODEL_PATH, "yolo.pt")
OUTPUT_PATH = "output"
DATA_PATH = "data"
SEGMENT_IMAGE_SIZE = 256


def get_chromosome_nums(image_file: str):
    mask_file = f"{image_file[:-4]}.txt"
    mask_path = os.path.join(DATA_PATH, "labels", mask_file)
    with open(mask_path, 'r') as fp:
        x = len(fp.readlines())
        return x


def get_unet_model(device, bilinear, unet_type="NORMAL"):
    if unet_type == "NORMAL":
        net = UNet(n_channels=1, n_classes=2, bilinear=bilinear)
        if SEGMENT_IMAGE_SIZE == 128:
            unet_path = os.path.join(TRAINED_MODEL_PATH, "unet_128.pth")
        else:
            unet_path = os.path.join(TRAINED_MODEL_PATH, "unet_128.pth")
    else:
        if SEGMENT_IMAGE_SIZE == 128:
            unet_path = os.path.join(TRAINED_MODEL_PATH, "unet_lite_128.pth")
        else:
            unet_path = os.path.join(TRAINED_MODEL_PATH, "unet_lite.pth")
        net = UNetLite(n_channels=1, n_classes=2, bilinear=bilinear)

    net.to(device=device)
    state_dict = torch.load(unet_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging(f"Unet {unet_type} at {unet_path} input {SEGMENT_IMAGE_SIZE} loaded!")
    return net


def get_yolo_model():
    model = YOLO(YOLO_MODEL_PATH)
    return model


def pipline(device, yolo, unet):
    count = 0
    image_data = {}
    chromosome_counts = 0
    for file in os.listdir(os.path.join(DATA_PATH, "images")):
        file_path = os.path.join(DATA_PATH, "images", file)
        chromosome_nums = get_chromosome_nums(file)
        chromosome_counts += chromosome_nums
        image_data[file_path] = chromosome_nums
        count += 1
        break
    pred_chromosomes = 0
    for result_index, image_path in enumerate(list(image_data.keys())):
        result = yolo(image_path, conf=0.5, line_thickness=1, show_conf=False)[0]
        orig_img_file = os.path.split(result.path)[-1]
        output_path = os.path.join(OUTPUT_PATH, orig_img_file[:-4])
        os.makedirs(output_path, exist_ok=True)
        boxes_num = len(result.boxes)
        logging(f"{orig_img_file}: true box nums={image_data[image_path]} predict box nums={boxes_num}")
        pred_chromosomes += boxes_num
        orig_img = result.orig_img
        for box_index, box in enumerate(result.boxes):
            crop_chromosome = crop_chromosome_from_origin(orig_img, box, SEGMENT_IMAGE_SIZE)
            pred_mask = predict_img(net=unet, full_img=crop_chromosome, device=device, size=SEGMENT_IMAGE_SIZE)
            cleaned_chromosome = clean_chromosome(crop_chromosome, pred_mask)
            # rotate_object_to_90_degrees(pred_mask, crop_chromosome)
            # rotated_image = crop_and_rotate(crop_chromosome, pred_mask)
            plot_result(os.path.join(output_path, f"{box_index}.png"), crop_chromosome, cleaned_chromosome)

    logging(f"accuracy: {pred_chromosomes / chromosome_counts}")


def parse_args():
    parser = argparse.ArgumentParser(description='test pipline')
    parser.add_argument('--data_path', '-dp', type=str, default="data", help='data path')
    parser.add_argument('--model_path', '-mp', type=str, default="trained_models",
                        help='folder contain unet, unet_lite pth and yolo.pt')
    parser.add_argument('--save_result_path', '-srp', type=str, default="output",
                        help='save result and plot')
    # parser.add_argument('--model_type', '-mt', type=str, default="LITE",
    #                     help='choose model type: NORMAL or LITE')
    # parser.add_argument('--unet_input_size', '-uis', type=int, default=128,
    #                     help='choose unet input size')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    DATA_PATH = args.data_path
    OUTPUT_PATH = args.save_result_path
    TRAINED_MODEL_PATH = args.model_path
    SEGMENT_IMAGE_SIZE = 128
    YOLO_MODEL_PATH = os.path.join(TRAINED_MODEL_PATH, "yolo.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO(YOLO_MODEL_PATH)
    unet = get_unet_model(device=device, bilinear=False, unet_type="LITE")
    pipline(device=device, yolo=yolo, unet=unet)
