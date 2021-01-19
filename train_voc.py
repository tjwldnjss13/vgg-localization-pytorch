import argparse
import torch

from torch.utils.data import DataLoader, ConcatDataset


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define hyper parameters, parsers
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', required=False, default=16)
    parser.add_argument('--lr', required=False, default=.0001)
    parser.add_argument('--num_epoch', required=False, default=100)
    parser.add_argument('--weight_decay', required=False, default=.001)
    parser.add_argument('--model_save_term', required=False, default=5)

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.num_epoch
    weight_decay = args.weight_decay
    model_save_term = args.model_save_term

    num_classes = 20

    # Load model


































