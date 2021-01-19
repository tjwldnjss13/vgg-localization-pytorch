import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_pth = 'pretrained models/vgg_voc_1e-05lr_0.11354loss_0.98816acc.pth'
model = torch.load(model_pth)

dummy = torch.zeros(1, 3, 224, 224)
print(model(dummy))
