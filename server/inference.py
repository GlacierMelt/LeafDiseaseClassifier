import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image


labels_dict = {
    '0': 'Cassava Bacterial Blight',
    '1': 'Cassava Brown Streak Disease',
    '2': 'Cassava Green Mottle',
    '3': 'Cassava Mosaic Disease',
    '4': 'Healthy'
}


def transform_image(image):
    trfm = transforms.Compose([transforms.Resize(512),
                               transforms.ToTensor(),
                               transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])])

    return trfm(image).unsqueeze(0)


def get_prediction(model, image):
    tensor = transform_image(image=image)
    outputs = model.forward(tensor)
    pred = torch.softmax(outputs, 1).argmax(1).numpy().astype('int')
    label = labels_dict[str(pred[0])]

    return outputs, label
