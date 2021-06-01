import argparse
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
import torch
from model import SepConvNet
from TrainTestUtil import evaluate
from TorchDB import DBreader_frame_interpolation
import torch

parser = argparse.ArgumentParser(description='SepConv Pytorch')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters
parser.add_argument('--database', type=str, default='./dataset/frames')
parser.add_argument('--output', type=str, default='./output_sepconv_pytorch_0/result')
parser.add_argument('--checkpoint', type=str, default='./output_sepconv_pytorch_0/checkpoint/model_epoch010.pth')
parser.add_argument('--train_test_ratio', type=float, default=0.8)
parser.add_argument('--batch_size', type=int, default=32)


def main():
    db_dir = parser.database
    args = parser.parse_args()
    output_dir = parser.output
    ckpt = args.checkpoint
    train_test_ratio = parser.train_test_ratio
    batch_size = parser.batch_size

    print("Loading the Model...")
    checkpoint = torch.load(ckpt)
    kernel_size = checkpoint['kernel_size']
    model = SepConvNet(kernel_size=kernel_size)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(torch.load(state_dict))
    model.epoch = checkpoint['epoch']

    print("Loading Datatest")
    dataset = DBreader_frame_interpolation(db_dir)
    num_train_examples = int(len(dataset) * train_test_ratio)
    num_test_examples = len(dataset) * num_train_examples
    _, test_data = random_split(dataset, [num_train_examples, num_test_examples])
    test_iterator = DataLoader(dataset=test_data, batch_size=batch_size)

    print("Test Start...")
    model = model.to(device)
    criterion = MSELoss()
    criterion = criterion.to(device)
    model.eval()
    test_loss, test_psnr = evaluate(model, test_iterator, criterion, device, test= True, output_dir= output_dir)

    print(f"Test -- Loss: {test_loss:.3f}, PSNR: {test_psnr:.3f}")

if __name__ == "__main__":
    main()
