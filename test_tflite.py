import time
import argparse
from argparse import Namespace
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import PIL
import torchvision.transforms as transforms

import torch
import onnx
import onnx_tf
import tensorflow as tf
import onnxruntime as ort

from efficientnet.efficientnet_lite import efficientnet_lite_params, build_efficientnet_lite
from efficientnet.datasets.ucc import UCC

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='efficientnet_lite0', help='name of model: efficientnet_lite0, 1, 2, 3, 4')
parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
parser.add_argument('--ckpt_pth', required=True, type=str, help='checkpoint path to load from')
parser.add_argument('--tflite_pth', required=True, type=str, help='path for tflite model')
parser.add_argument('--quantization', default='Default', choices=['Default', 'DRQ', 'FIQ'])
parser.add_argument('--dataset_dir', type=str, default='data/train', help='path to the dataset')


CROP_PADDING = 32
MEAN_RGB = [0.498, 0.498, 0.498]
STDDEV_RGB = [0.502, 0.502, 0.502]


def create_torch_efficientnet_lite(args: Namespace):
    torch_model = build_efficientnet_lite(args.model_name, args.num_classes)
    checkpoint = torch.load(args.ckpt_pth, map_location='cpu')

    state_dict = checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']

    # handle 'module.' from parameter name
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if 'module.' in name:
            name = k[7:]
        new_state_dict[name] = v
    
    torch_model.load_state_dict(new_state_dict, strict=True)
    torch_model.eval()

    return torch_model

def test_tflite(
        args: Namespace,
    ):

    args.input_size = efficientnet_lite_params[args.model_name][2]
    torch_model = create_torch_efficientnet_lite(args)

    ucc_dataset = UCC(
        root=args.dataset_dir,
        transform=transforms.Compose([
            transforms.Resize((args.input_size + CROP_PADDING, args.input_size + CROP_PADDING), interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_RGB, STDDEV_RGB)
        ]),
    )

    inference_speeds = {
        'torch': [],
        'tensorflow-lite': [],
    }

    interpreter = tf.lite.Interpreter(model_path=args.tflite_pth)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
     
    n_correct_torch, n_correct_tflite = 0, 0
    n_total = len(ucc_dataset)

    for input, target in tqdm(ucc_dataset):
        input = input.unsqueeze(0)

        # test PyTorch
        start = time.time()
        torch_outputs = torch_model(input)
        inference_speeds['torch'].append(time.time() - start)

        if torch.argmax(torch_outputs[0]) == target:
            n_correct_torch += 1

        # test TFLite
        tflite_input = input.numpy()
        if args.quantization == 'FIQ':
            tflite_input = (input * 255).numpy().astype(np.uint8)
        interpreter.set_tensor(input_details[0]['index'], tflite_input)
        interpreter.invoke()
        tflite_outputs = interpreter.get_tensor(output_details[0]['index'])
        # print(type(tflite_outputs[0]), tflite_outputs)
        inference_speeds['tensorflow-lite'].append(time.time() - start)

        if torch.argmax(torch.tensor(tflite_outputs[0])) == target:
            n_correct_tflite += 1

    print(f'PyTorch: ' + \
            f'Accuracy: ({n_correct_torch}/{n_total})={n_correct_torch / n_total * 100:.4f}%, ' + \
            f'Inference speed: {np.average(inference_speeds["torch"]):.5f}s')
    print(f'TFLite: ' + \
            f'Accuracy: ({n_correct_tflite}/{n_total})={n_correct_tflite / n_total * 100:.4f}%, ' + \
            f'Inference speed: {np.average(inference_speeds["tensorflow-lite"]):.5f}s')

def main(args: Namespace):
    args.input_size = efficientnet_lite_params[args.model_name][2]
    test_tflite(args)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)