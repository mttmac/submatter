from openvino.inference_engine import IECore, Blob, TensorDesc
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.spatial.distance import cosine

XML_PATH_DET = Path("models/openvino-zoo/face-detection-retail-0005/face-detection-retail-0005.xml")
BIN_PATH_DET = Path("models/openvino-zoo/face-detection-retail-0005/face-detection-retail-0005.bin")
XML_PATH_RID = Path("models/openvino-zoo/face-reidentification-retail-0095/face-reidentification-retail-0095.xml")
BIN_PATH_RID = Path("models/openvino-zoo/face-reidentification-retail-0095/face-reidentification-retail-0095.bin")

ie_core_handler = IECore()

det_net = ie_core_handler.read_network(model=XML_PATH_DET, weights=BIN_PATH_DET)  # face detection
det_model = ie_core_handler.load_network(det_net, device_name='CPU', num_requests=1)
det_request = det_model.requests[0]
det_in_name = next(iter(det_request.input_blobs))
det_out_name = next(iter(det_request.output_blobs))

rid_net = ie_core_handler.read_network(model=XML_PATH_RID, weights=BIN_PATH_RID)  # face re-identification
rid_model = ie_core_handler.load_network(rid_net, device_name='CPU', num_requests=1)
rid_request = rid_model.requests[0]
rid_in_name = next(iter(rid_request.input_blobs))
rid_out_name = next(iter(rid_request.output_blobs))

det_samples = []
det_results = []
for i in range(3):
    im = Image.open(Path(f'samples/300/face{i}.jpeg'))
    img = (np.array(im)[:, :, ::-1] / 255).astype(np.float32)  # cast to float and put in BGR order
    img = np.expand_dims(np.transpose(img, axes=[2, 0, 1]), 0)  # reshape for input
    desc = TensorDesc(precision="FP32", dims=(1, 3, 300, 300), layout='NCHW')
    sample = Blob(desc, img)

    det_samples.append(sample)
    det_request.set_blob(blob_name=det_in_name, blob=sample)
    det_request.infer()

    det_results.append(det_request.output_blobs[det_out_name].buffer)


rid_samples = []
rid_results = []
for i in range(4):
    im = Image.open(Path(f'samples/128/face{i}.jpeg'))
    img = (np.array(im)[:, :, ::-1] / 255).astype(np.float32)  # cast to float and put in BGR order
    img = np.expand_dims(np.transpose(img, axes=[2, 0, 1]), 0)  # reshape for input
    desc = TensorDesc(precision="FP32", dims=(1, 3, 128, 128), layout='NCHW')
    sample = Blob(desc, img)

    rid_samples.append(sample)
    rid_request.set_blob(blob_name=rid_in_name, blob=sample)
    rid_request.infer()

    rid_results.append(rid_request.output_blobs[rid_out_name].buffer)

confidences = [x[0, 0, :, 2] for x in det_results]
vectors = [s[0, :, 0, 0] for s in rid_results]
print(confidences[0][:5])
print(cosine(vectors[0], vectors[3]))