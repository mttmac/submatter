# Command reference to load hopenet from repo and convert to onnx
# Numerical precision is float32
# Other references: https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks

# Python
import torch
import stable_hopenetlite  # must be in ref repo to import, otherwise assume in project repo
from pathlib import Path

model = stable_hopenetlite.shufflenet_v2_x1_0()
net = torch.load(Path('ref/deep-head-pose-lite/model/shuff_epoch_120.pkl'), map_location=torch.device('cpu'))
model.load_state_dict(net)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)  # hopenet trained on 224 x 224 images
torch.onnx.export(model, (dummy_input, ), Path('models/hopenet-lite/shuff_epoch_120-224x224.onnx'))

# Command line macOS
"""
cd /opt/intel/openvino_2021.4.582/deployment_tools/model_optimizer
python mo.py --input_model ~/repos/submatter/models/hopenet-lite/shuff_epoch_120-224x224.onnx --output_dir ~/repos/submatter/models/hopenet-lite/

Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/Users/mattmacdonald/repos/submatter/models/hopenet-lite/shuff_epoch_120-224x224.onnx
	- Path for generated IR: 	/Users/mattmacdonald/repos/submatter/models/hopenet-lite/
	- IR output name: 	shuff_epoch_120-224x224
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP32
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	None
	- Reverse input channels: 	False
ONNX specific parameters:
	- Inference Engine found in: 	/Users/mattmacdonald/anaconda3/lib/python3.6/site-packages/openvino
Inference Engine version: 	2021.4.0-3839-cd81789d294-releases/2021/4
Model Optimizer version: 	2021.4.0-3839-cd81789d294-releases/2021/4
[ SUCCESS ] Generated IR version 10 model.
[ SUCCESS ] XML file: /Users/mattmacdonald/repos/submatter/models/hopenet-lite/shuff_epoch_120-224x224.xml
[ SUCCESS ] BIN file: /Users/mattmacdonald/repos/submatter/models/hopenet-lite/shuff_epoch_120-224x224.bin
[ SUCCESS ] Total execution time: 27.43 seconds.
[ SUCCESS ] Memory consumed: 108 MB.
"""

# To understand model architecture better use torchinfo
from torchinfo import summary
summary(model, input_size=(1, 3, 224, 224))
"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ShuffleNetV2                             --                        --
├─Sequential: 1-1                        [1, 24, 112, 112]         --
│    └─Conv2d: 2-1                       [1, 24, 112, 112]         648
│    └─BatchNorm2d: 2-2                  [1, 24, 112, 112]         48
│    └─ReLU: 2-3                         [1, 24, 112, 112]         --
├─MaxPool2d: 1-2                         [1, 24, 56, 56]           --
├─Sequential: 1-3                        [1, 116, 28, 28]          --
│    └─InvertedResidual: 2-4             [1, 116, 28, 28]          --
│    │    └─Sequential: 3-1              [1, 58, 28, 28]           1,772
│    │    └─Sequential: 3-2              [1, 58, 28, 28]           5,626
│    └─InvertedResidual: 2-5             [1, 116, 28, 28]          --
│    │    └─Sequential: 3-3              [1, 58, 28, 28]           7,598
│    └─InvertedResidual: 2-6             [1, 116, 28, 28]          --
│    │    └─Sequential: 3-4              [1, 58, 28, 28]           7,598
│    └─InvertedResidual: 2-7             [1, 116, 28, 28]          --
│    │    └─Sequential: 3-5              [1, 58, 28, 28]           7,598
├─Sequential: 1-4                        [1, 232, 14, 14]          --
│    └─InvertedResidual: 2-8             [1, 232, 14, 14]          --
│    │    └─Sequential: 3-6              [1, 116, 14, 14]          14,964
│    │    └─Sequential: 3-7              [1, 116, 14, 14]          28,652
│    └─InvertedResidual: 2-9             [1, 232, 14, 14]          --
│    │    └─Sequential: 3-8              [1, 116, 14, 14]          28,652
│    └─InvertedResidual: 2-10            [1, 232, 14, 14]          --
│    │    └─Sequential: 3-9              [1, 116, 14, 14]          28,652
│    └─InvertedResidual: 2-11            [1, 232, 14, 14]          --
│    │    └─Sequential: 3-10             [1, 116, 14, 14]          28,652
│    └─InvertedResidual: 2-12            [1, 232, 14, 14]          --
│    │    └─Sequential: 3-11             [1, 116, 14, 14]          28,652
│    └─InvertedResidual: 2-13            [1, 232, 14, 14]          --
│    │    └─Sequential: 3-12             [1, 116, 14, 14]          28,652
│    └─InvertedResidual: 2-14            [1, 232, 14, 14]          --
│    │    └─Sequential: 3-13             [1, 116, 14, 14]          28,652
│    └─InvertedResidual: 2-15            [1, 232, 14, 14]          --
│    │    └─Sequential: 3-14             [1, 116, 14, 14]          28,652
├─Sequential: 1-5                        [1, 464, 7, 7]            --
│    └─InvertedResidual: 2-16            [1, 464, 7, 7]            --
│    │    └─Sequential: 3-15             [1, 232, 7, 7]            56,840
│    │    └─Sequential: 3-16             [1, 232, 7, 7]            111,128
│    └─InvertedResidual: 2-17            [1, 464, 7, 7]            --
│    │    └─Sequential: 3-17             [1, 232, 7, 7]            111,128
│    └─InvertedResidual: 2-18            [1, 464, 7, 7]            --
│    │    └─Sequential: 3-18             [1, 232, 7, 7]            111,128
│    └─InvertedResidual: 2-19            [1, 464, 7, 7]            --
│    │    └─Sequential: 3-19             [1, 232, 7, 7]            111,128
├─Sequential: 1-6                        [1, 1024, 7, 7]           --
│    └─Conv2d: 2-20                      [1, 1024, 7, 7]           475,136
│    └─BatchNorm2d: 2-21                 [1, 1024, 7, 7]           2,048
│    └─ReLU: 2-22                        [1, 1024, 7, 7]           --
├─Linear: 1-7                            [1, 66]                   67,650
├─Linear: 1-8                            [1, 66]                   67,650
├─Linear: 1-9                            [1, 66]                   67,650
==========================================================================================
Total params: 1,456,554
Trainable params: 1,456,554
Non-trainable params: 0
Total mult-adds (M): 144.10
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 31.20
Params size (MB): 5.83
Estimated Total Size (MB): 37.63
==========================================================================================
"""

# Converting model to blob format can be done online
"""
http://luxonis.com:8080/
Select version 2021.3, OpenVino model
The SHAVES are vector processors in DepthAI.
For 1080p, 13 SHAVES (of 16) are free for neural network stuff.
For 4K sensor resolution, 10 SHAVES are available for neural operations.
"""