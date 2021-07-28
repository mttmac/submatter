import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import sys
sys.path.insert(1, str(Path('ref/deep-head-pose-lite/').resolve()))
import stable_hopenetlite

def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result

model = stable_hopenetlite.shufflenet_v2_x1_0()
net_path = Path('ref/deep-head-pose-lite/model/shuff_epoch_120.pkl')
net = torch.load(net_path, map_location=torch.device('cpu'))
model.load_state_dict(net)
model.eval()

samples = []
for i in range(3):
    samples.append(Path(f'samples/224/face{i}.jpeg'))

idx_tensor = torch.FloatTensor(range(66))
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
for s in samples:
    im = Image.open(s)
    img = transform(im)
    img = torch.unsqueeze(img, 0)

    yaw, pitch, roll = model(img)

    yaw_predicted = F.softmax(yaw.data, dim=1)
    pitch_predicted = F.softmax(pitch.data, dim=1)
    roll_predicted = F.softmax(roll.data, dim=1)

    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    print(s)
    print(yaw_predicted, pitch_predicted, roll_predicted)
