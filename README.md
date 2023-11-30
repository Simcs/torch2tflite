# torch2tflite
Convert PyTorch model to Tensorflow-Lite model with relevant quantizations

## Dependencies
- torch == 2.1.0+cu118
- onnx == 1.14.1
- onnx-tf == 1.10.0
- onnx2tf == 1.18.14 ([Link](https://github.com/PINTO0309/onnx2tf))
- openvino == 2023.2.0
- tensorflow == 2.14.0

## Descriptions

_**IMPORTANT:**_ The input shape of the ```Conv``` module is different for each framework
```
- Channel-first(NCHW) : PyTorch, ONNX, OpenVINO
- Channel-last (NHWC) : Tensorflow
```

### torch2tflite_nchw.py
- Conversion process:
```
PyTorch > ONNX > Tensorflow > Tensorflow-Lite
```
- Conversion from ONNX to Tensorflow is done using the ```onnx-tf``` library which is officially maintained by ONNX
- However, it generates many unnecessary ```transpose``` ops before and after each ```Conv``` op
- Thus, the input shape remains the same after conversion (i.e., NCHW input for TF model)


### torch2tflite_nhwc.py _**(RECOMMENDED)**_
- Conversion process:
```
PyTorch > ONNX > Tensorflow > Tensorflow-Lite
```
- Here, we used ```onnx2tf``` library which take care of the input shape issue between ONNX and TF
- In this case, the converted model requires a channel-last (NHWC) input shape

### torch2tflite_openvino.py
- Conversion process:
```
PyTorch > ONNX > OpenVINO > Tensorflow > Tensorflow-Lite
```
