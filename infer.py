import argparse
import torch
from torchvision.models import resnet101, densenet121
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import pandas as pd

conditions = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly', 
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation', 
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices'
        ]

models = dict()

resnet_u0 = resnet101(num_classes=14)
resnet_u0.fc = torch.nn.Linear(in_features=resnet_u0.fc.in_features, out_features=14)
resnet_u0.load_state_dict(torch.load('checkpoint/ResNetTL-U0.pth', weights_only=True))

models['ResNetTL-U0'] = resnet_u0

resnet_u1 = resnet101(num_classes=14)
resnet_u1.fc = torch.nn.Linear(in_features=resnet_u1.fc.in_features, out_features=14)
resnet_u1.load_state_dict(torch.load('checkpoint/ResNetTL-U1.pth', weights_only=True))

models['ResNetTL-U1'] = resnet_u1

densenet_u0 = densenet121(num_classes=14)
densenet_u0.classifier = torch.nn.Linear(in_features=densenet_u0.classifier.in_features, out_features=14)
densenet_u0.load_state_dict(torch.load('checkpoint/DenseNetTL-U0.pth', weights_only=True))

models['DenseNetTL-U0'] = densenet_u0

densenet_u1 = densenet121(num_classes=14)
densenet_u1.classifier = torch.nn.Linear(in_features=densenet_u1.classifier.in_features, out_features=14)
densenet_u1.load_state_dict(torch.load('checkpoint/DenseNetTL-U1.pth', weights_only=True))

models['DenseNetTL-U1'] = densenet_u1

transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize([0.506, 0.506, 0.506], [0.287, 0.287, 0.287]),
        ToTensorV2()
    ])


parser = argparse.ArgumentParser(description="Inference script for CheXpert's Diseases Classification. Make sure to match dataset with model, U1 models uses images from test/u1, U0 models uses images from test/u0")
parser.add_argument("--model_name", type=str, required=True, help="Model name to use. Available model names: DenseNetTL-U1, DenseNetTL-U0, ResNetTL-U0, ResNetTL-U1")
parser.add_argument("--image_path", type=str, required=True, help="Path to the input image. Please use the images in test folder for true labels. We currently do not support inference for images not in the test folder. Please use forward slash instead of backward slash for the image path")

args = parser.parse_args()

image = Image.open(args.image_path).convert('RGB')
image = transforms(image=np.array(image))['image']

selected_model = models[args.model_name]

labels = pd.read_csv(f'test/u{args.model_name[-1]}/u{args.model_name[-1]}_test.csv', index_col=0)
image_index = 'chexpert' + args.image_path.split('chexpert', 1)[-1]
image_label = labels.loc[image_index]
positive_labels = image_label[image_label == 1]

thresholds = pd.read_csv(f'checkpoint/{args.model_name}.csv')['threshold']

selected_model.eval()
results = selected_model(image.unsqueeze(0))
results = torch.nn.Sigmoid()(results)
pred_postive_index = np.where(results.squeeze().detach().numpy() > thresholds)[0]

print(f"Predicted labels: {','.join(conditions[i] for i in pred_postive_index)}")
print(f"True labels: {', '.join(positive_labels.index)}")

