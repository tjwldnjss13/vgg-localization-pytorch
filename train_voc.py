import argparse
import time
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, ConcatDataset, random_split

from model import FeatureExtractedVGG
from loss import custom_loss
from dataset.voc_dataset import VOCDataset, custom_collate_fn
from utils import make_batch, time_calculator
from pytorch_utils import calculate_iou_batch


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define hyper parameters, parsers
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', required=False, default=16)
    parser.add_argument('--lr', required=False, default=.0001)
    parser.add_argument('--num_epochs', required=False, default=50)
    parser.add_argument('--weight_decay', required=False, default=.005)
    parser.add_argument('--model_save_term', required=False, default=5)

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay
    model_save_term = args.model_save_term

    num_classes = 20

    # Generate VOC dataset
    root = 'C://DeepLearningData/VOC2012/'
    img_size = (224, 224)

    transform_original = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_rotate = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.RandomRotation((-30, 30)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_vertical_flip = transforms.Compose([transforms.Resize((224, 224)),
                                                  transforms.RandomVerticalFlip(1),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_horizontal_flip = transforms.Compose([transforms.Resize((224, 224)),
                                                    transforms.RandomHorizontalFlip(1),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    dset_original = VOCDataset(root, img_size=img_size, transforms=transform_original, is_categorical=True)
    dset_rotate = VOCDataset(root, img_size=img_size, transforms=transform_rotate, is_categorical=True)
    dset_vertical_flip = VOCDataset(root, img_size=img_size, transforms=transform_vertical_flip, is_categorical=True)
    dset_horizontal_flip = VOCDataset(root, img_size=img_size, transforms=transform_horizontal_flip, is_categorical=True)

    n_val_data = int(len(dset_original) * .3)
    train_val_ratio = [len(dset_original) - n_val_data, n_val_data]

    dset_original, dset_val = random_split(dset_original, train_val_ratio)
    dset_rotate, _ = random_split(dset_rotate, train_val_ratio)
    dset_vertical_flip, _ = random_split(dset_vertical_flip, train_val_ratio)
    dset_horizontal_flip, _ = random_split(dset_horizontal_flip, train_val_ratio)

    dset_train = ConcatDataset([dset_original, dset_rotate, dset_vertical_flip, dset_horizontal_flip])

    print('Train: {} Validation: {}'.format(len(dset_train), len(dset_val)))

    # Generate train, validation data loader
    dl_train = DataLoader(dset_train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    dl_val = DataLoader(dset_val, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Load model
    model = FeatureExtractedVGG().to(device)

    # Generate optimizer, loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = custom_loss

    # Start training
    train_loss_list = []
    train_acc_list = []
    train_iou_list = []
    val_loss_list = []
    val_acc_list = []
    val_iou_list = []

    t_start = time.time()
    for e in range(num_epochs):
        loss_sum = 0
        acc_sum = 0
        iou_sum = 0
        num_batches = 0

        model.train()
        t_train_start = time.time()
        for i, (images, labels) in enumerate(dl_train):
            print('[{}/{}] '.format(e + 1, num_epochs), end='')
            print('{}/{} '.format((i + 1) * len(images), len(dset_train)), end='')

            num_batches += 1

            x = make_batch(images).to(device)
            y_class = make_batch(labels, 'class').to(device)
            y_bbox = make_batch(labels, 'bbox').to(device)

            predict_class, predict_bbox = model(x)

            optimizer.zero_grad()
            loss = loss_func(predict_class, predict_bbox, y_class, y_bbox)
            loss.backward()
            optimizer.step()

            loss = loss.item()
            acc_class = (predict_class.argmax(dim=1) == y_class.argmax(dim=1)).sum().item() / len(images)
            iou = calculate_iou_batch(predict_bbox, y_bbox)

            loss_sum += loss
            acc_sum += acc_class
            iou_sum += iou

            train_loss_avg = loss_sum / num_batches
            train_acc_avg = acc_sum / num_batches
            train_iou_avg = iou_sum / num_batches

            print('<loss> {:<20} <acc> {:<20} <iou> {:<20} '.format(loss, acc_class, iou), end='')
            print('<loss_avg> {:<20} <acc_avg> {:<20} <iou_avg> {:<20} '.format(train_loss_avg, train_acc_avg, train_iou_avg), end='')

            t_mid = time.time()
            H, M, S = time_calculator(t_mid - t_start)

            print('<time> {:02d}:{:02d}:{:02d}'.format(H, M, int(S)))

        t_train_end = time.time()
        H, M, S = time_calculator(t_train_end - t_train_start)

        train_loss_list.append(loss_sum / num_batches)
        train_acc_list.append(acc_sum / num_batches)
        train_iou_list.append(iou_sum / num_batches)

        print('        <train_loss> {:<20} <train_acc> {:<20} <train_iou> {:<20} '.format(train_loss_list[-1], train_acc_list[-1], train_iou_list[-1]), end='')
        print('<time> {:02d}:{:02d}:{:02d} '.format(H, M, int(S)), end='')

        with torch.no_grad():
            loss_sum = 0
            acc_sum = 0
            iou_sum = 0
            num_batches = 0

            model.eval()
            for i, (images, labels) in enumerate(dl_val):
                num_batches += 1

                x = make_batch(images).to(device)
                y_class = make_batch(labels, 'class').to(device)
                y_bbox = make_batch(labels, 'bbox').to(device)

                print(i + 1, x.shape)

                predict_class, predict_bbox = model(x)
                loss = loss_func(predict_class, predict_bbox, y_class, y_bbox)

                loss = loss.item()
                acc_class = (predict_class.argmax(dim=1) == y_class.argmax(dim=1)).sum().item() / len(images)
                iou = calculate_iou_batch(predict_bbox, y_bbox)

                loss_sum += loss
                acc_sum += acc_class
                iou_sum += iou

            val_loss_list.append(loss_sum / num_batches)
            val_acc_list.append(acc_sum / num_batches)
            val_iou_list.append(iou_sum / num_batches)

            print('<val_loss> {:<20} <val_acc> {:<20} <val_iou> {:<20}'.format(val_loss_list[-1], val_acc_list[-1], val_iou_list[-1]))

        if (e + 1) % model_save_term == 0:
            path = 'saved models/vgg_voc_{}epoch_{}lr_{:.5f}loss_{:.5f}acc_{:.5f}iou.pth'.format(e + 1, learning_rate, val_loss_list[-1], val_acc_list[-1], val_iou_list[-1])
            torch.save(model, path)

    # Show train/validation results(loss, accuracy)
    x_axis = [i for i in range(num_epochs)]

    plt.figure(0)
    plt.title('Loss')
    plt.plot(x_axis, train_loss_list, 'r-', label='Train')
    plt.plot(x_axis, val_loss_list, 'b:', label='Validation')
    plt.legend()

    plt.figure(1)
    plt.title('Accuracy')
    plt.plot(x_axis, train_acc_list, 'r-', label='Train')
    plt.plot(x_axis, val_acc_list, 'b:', label='Validation')
    plt.legend()

    plt.figure(2)
    plt.title('IoU')
    plt.plot(x_axis, train_iou_list, 'r-', label='Train')
    plt.plot(x_axis, val_iou_list, 'b:', label='Validation')
    plt.legend()

    plt.show()

































