import numpy as np
import torch
import onnx
import onnx_tf
import tensorflow as tf
import onnxruntime as ort
import argparse
from collections import OrderedDict

from efficientnet.efficientnet_lite import efficientnet_lite_params, build_efficientnet_lite


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='efficientnet_lite0', help='name of model: efficientnet_lite0, 1, 2, 3, 4')
parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
parser.add_argument('--ckpt_pth', type=str, help='checkpoint path to load from')

args = parser.parse_args()


# ===== 1. Load PyTorch Model =====

# Load PyTorch EfficientNet-Lite model
pytorch_model = build_efficientnet_lite(args.model_name, args.num_classes)
checkpoint = torch.load(args.ckpt_pth, map_location='cpu')
# remove 'module.' from parameter name
state_dict = checkpoint['state_dict']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v

pytorch_model.load_state_dict(new_state_dict, strict=True)
print('build model complete')

# ===== 2. PyTorch => ONNX =====

# Export the PyTorch model to ONNX format
input_size = efficientnet_lite_params[args.model_name][2]
dummy_input = torch.randn(1, 3, input_size, input_size)
onnx_model_path = f'./models/{args.model_name}.onnx'
onnx_program = torch.onnx.dynamo_export(pytorch_model, dummy_input)
onnx_program.save(onnx_model_path)
print('export to onnx format complete')

# ===== 3. Check ONNX Model Inference =====

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)
print('onnx version:', onnx_model.opset_import[0].version)
onnx.checker.check_model(onnx_model)

onnx_input = onnx_program.adapt_torch_inputs_to_onnx(dummy_input)
print(f'input length: {len(onnx_input)}')
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
onnxruntime_output = ort_session.run(None, onnxruntime_input)
print(onnxruntime_output[0].shape, onnxruntime_output)

# ===== 4. ONNX => Tensorflow =====

# Convert the ONNX model to TensorFlow format
tf_model_path = f'./models/{args.model_name}.tf'
tf_rep = onnx_tf.backend.prepare(onnx_model)
tf_rep.export_graph(tf_model_path)
print('convert to tensorflow format complete')

# ===== 5. Tensorflow => Tensorflow-Lite =====

# Convert the TensorFlow model to TensorFlow Lite format
# converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()
print('convert to tensorflow lite format complete')
 
# Save the TensorFlow Lite model to a file
with open(f'./models/{args.model_name}', 'wb') as f:
    f.write(tflite_model)
