import time
import argparse
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
import numpy as np
from tqdm import tqdm
import PIL
import subprocess
import torchvision.transforms as transforms

import torch
import onnx
import onnx_tf
import onnxruntime as ort
import tensorflow as tf
import openvino as ov
from openvino.tools.mo import convert_model

from efficientnet.efficientnet_lite import efficientnet_lite_params, build_efficientnet_lite
from efficientnet.datasets.ucc import UCC

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='efficientnet_lite0', help='name of model: efficientnet_lite0, 1, 2, 3, 4')
parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
parser.add_argument('--ckpt_pth', required=True, type=str, help='checkpoint path to load from')
parser.add_argument('--n_test', type=int, default=1000, help='number of tests for validating conversion')
parser.add_argument('--dynamic_batch_size', default=False, action='store_true', help='enable dynamic batch size')
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

def torch2onnx(torch_model: torch.nn.Module, args: Namespace):
    input_size = efficientnet_lite_params[args.model_name][2]
    sample_input = torch.randn(1, 3, input_size, input_size)
    onnx_path = f'./models/{args.model_name}.onnx'

    if not args.dynamic_batch_size:
        # Default
        torch.onnx.export(
            model=torch_model, 
            args=sample_input, 
            f=onnx_path, 
            verbose=False,
            input_names=['input'],
            output_names=['output'],
        )
    else:
        # To handle dynamic batch size
        torch.onnx.export(
            model=torch_model, 
            args=sample_input, 
            f=onnx_path, 
            verbose=False,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            },
            opset_version=12,
        )
    onnx_model = onnx.load(onnx_path)
    
    # Check that the IR(intermediate representation) is well formed
    onnx.checker.check_model(onnx_model)

    return onnx_model, onnx_path

def onnx2openvino(onnx_path: str, args: Namespace):
    ov_dir = Path(f'./models/{args.model_name}_ov')
    ov_dir.mkdir(parents=True, exist_ok=True)
    ov_path = ov_dir / "model.xml"
    ov_model = ov.convert_model(onnx_path)
    ov.save_model(ov_model, ov_path)
    return ov_model, ov_path

def openvino2tensorflow(
        ov_path: str, 
        args: Namespace,
        output_saved_model: bool = False, 
        output_pb: bool = False,
    ):
    tf_path = f'./models/{args.model_name}_tf'
    cmd = f"openvino2tensorflow --model_path {ov_path} --model_output_path {tf_path} --output_saved_model"
    subprocess.check_output(cmd.split())
    return tf_path

def onnx2tf(onnx_path, args: Namespace):
    tf_path = f'./models/{args.model_name}_tf'
    cmd = f"onnx2tf -i {onnx_path} -osd -o {tf_path}"
    subprocess.check_output(cmd.split())
    return tf_path

def tf2tflite(tf_path: str, args: Namespace):
    # converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)

    if args.quantization in ['DRQ', 'FIQ']:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if args.quantization in ['FIQ']:
        args.input_size = efficientnet_lite_params[args.model_name][2]
        ucc_dataset = UCC(
            root=args.dataset_dir,
            transform=transforms.Compose([
                transforms.Resize((args.input_size + CROP_PADDING, args.input_size + CROP_PADDING), interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN_RGB, STDDEV_RGB)
            ]),
        )
        def representative_dataset():
            for image, target in tqdm(ucc_dataset):
                yield { 'input': image.unsqueeze(0).permute(0, 2, 3, 1).numpy() }
                # yield { 'input': image.unsqueeze(0).numpy() }

        converter.representative_dataset = representative_dataset

        # for integer only quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
               
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    if args.quantization == 'Default':
        tflite_model_path = f'./models/{args.model_name}.tflite'
    else:
        tflite_model_path = f'./models/{args.model_name}_{args.quantization}.tflite'

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    return tflite_model, tflite_model_path

def test_conversion(
        torch_model: torch.nn.Module, 
        onnx_path: str, 
        tf_path: str,
        tflite_path: str,
        args: Namespace,
    ):

    inference_speeds = {
        'torch': [],
        'onnx': [],
        'tensorflow': [],
        'tensorflow-lite': [],
    }

    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    tf_model = tf.saved_model.load(tf_path)
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    n_correct = 0

    input_size = efficientnet_lite_params[args.model_name][2]
    for i in tqdm(range(args.n_test)):
        # generate sample random input
        n_batch = max(int(np.random.random() * 100), 1) if args.dynamic_batch_size else 1
        sample_input = torch.randn(n_batch, 3, input_size, input_size)
        
        # inference using PyTorch
        start = time.time()
        torch_outputs = torch_model(sample_input)
        inference_speeds['torch'].append(time.time() - start)
        
        # test PyTorch <=> ONNX
        start = time.time()
        # ort_session = ort.InferenceSession(onnx_model_path)
        onnx_input = {'input': sample_input.numpy()}
        onnx_outputs = ort_session.run(None, onnx_input)
        inference_speeds['onnx'].append(time.time() - start)
        
        assert len(torch_outputs) == len(onnx_outputs[0])
        for torch_output, onnx_output in zip(torch_outputs, onnx_outputs[0]):
            torch.testing.assert_close(torch_output, torch.tensor(onnx_output))
        
        # test PyTorch <=> Tensorflow
        start = time.time()
        sample_input_for_tf = sample_input.permute(0, 2, 3, 1)
        tf_input = tf.convert_to_tensor(sample_input.numpy())
        tf_outputs = tf_model(**{'inputs': sample_input_for_tf})
        inference_speeds['tensorflow'].append(time.time() - start)

        # tf_outputs_numpy = tf_outputs['outputs'].numpy()
        tf_outputs_numpy = tf_outputs.numpy()
        assert len(torch_outputs) == len(tf_outputs_numpy)
        for torch_output, tf_output in zip(torch_outputs, tf_outputs_numpy):
            torch.testing.assert_close(torch_output, torch.tensor(tf_output))

        # test PyTorch <=> Tensorflow-Lite
        if args.quantization == 'FIQ':
            continue # TODO: implement test code for FIQ quantization mode

        if args.dynamic_batch_size:
            interpreter.resize_tensor_input(input_details[0]['index'], (n_batch, 3, input_size, input_size))
            interpreter.allocate_tensors()

        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], sample_input_for_tf.numpy())
        interpreter.invoke()
        tflite_outputs = interpreter.get_tensor(output_details[0]['index'])
        inference_speeds['tensorflow-lite'].append(time.time() - start)

        if args.quantization == 'Default':
            assert len(torch_outputs) == len(tflite_outputs)
            for torch_output, tflite_output in zip(torch_outputs, tflite_outputs):
                torch.testing.assert_close(torch_output, torch.tensor(tflite_output))
        else:
            assert len(torch_outputs) == len(tflite_outputs)
            for torch_output, tflite_output in zip(torch_outputs, tflite_outputs):
                if torch.argmax(torch_output) == torch.argmax(torch.tensor(tflite_output)):
                    n_correct += 1
    
    if args.quantization != 'Default':
        print(f'{args.quantization} test: ({n_correct}/{args.n_test})={n_correct / args.n_test:.4f}% accuracy')
    
    return inference_speeds

def main(args: Namespace):
    model_path = Path('./models')
    model_path.mkdir(parents=True, exist_ok=True)
    
    # ===== 1. Load PyTorch Model =====
    torch_model = create_torch_efficientnet_lite(args)
    print('=== Pytorch model build complete ===')

    # ===== 2. PyTorch => ONNX =====
    onnx_model, onnx_path = torch2onnx(torch_model, args)
    print('=== Pytorch to onnx complete ===')
    print('onnx version:', onnx_model.opset_import[0].version)

    # ===== 3. ONNX => Tensorflow =====
    tf_path = onnx2tf(onnx_path, args)
    print('=== Onnx to tensorflow complete ===')

    # ===== 4. Tensorflow => Tensorflow-Lite =====
    tflite_model, tflite_path = tf2tflite(tf_path, args)
    print('=== Tensorflow to tensorflow lite complete ===')

    # ===== 5. Test PyTorch <=> ONNX, Tensorflow, Tensorflow-Lite =====
    inference_speeds = test_conversion(torch_model, onnx_path, tf_path, tflite_path, args)
    print('=== Conversion test complete ===')

    print(f'inference speed statistics (CPU): ' + \
        f'PyTorch: {np.average(inference_speeds["torch"]):.5f}s ' + \
        f'ONNX: {np.average(inference_speeds["onnx"]):.5f}s ' + \
        f'Tensorflow: {np.average(inference_speeds["tensorflow"]):.5f}s ' + \
        f'Tensorflow-Lite: {np.average(inference_speeds["tensorflow-lite"]):.5f}s'
    )

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)