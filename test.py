import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    class_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6,
                  'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13,
                  'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

    model_pth = 'pretrained models/vgg_voc_1e-05lr_0.11354loss_0.98816acc.pth'
    model = torch.load(model_pth).to(device)
    model.eval()

    img_pth = 'sample/horse.jpg'
    img = Image.open(img_pth)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    x = transform(img).unsqueeze(0).to(device)

    output = model(x)
    output = torch.nn.Softmax(dim=1)(output)
    print(output)
    output = torch.argmax(output).item()

    for key, value in class_dict.items():
        if output == value:
            print(key)
            break
