import argparse
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import torch
from model import SepConvNet
from TrainTestUtil import evaluate
from TorchDB import DBreader_frame_interpolation
import torch
import cv2
import os
from shutil import rmtree
import numpy as np
import glob

def create_frames(stream, input_dir, dim):
    images = []
    while stream.isOpened():
        ret, frame = stream.read()
        if not ret:
            print("Stream end!")
            break
        height = frame.shape[0]
        width = frame.shape[1]
        cropped_frame = frame[(height//2 - dim//2):(height//2 + dim//2), (width//2 - dim//2):(width//2 + dim//2)]
        images.append(cropped_frame)

    images = images[:len(images) - len(images) % 3]
    for idx, image in enumerate(images):
        cv2.imwrite(input_dir + f'/example_frame_{idx}.jpg', image)
    return images

def getFakeImages(image_dir, len_dataset, size_image):
    images = []
    image = cv2.imread(f'{image_dir}/homer.webp')
    image = cv2.resize(image, (size_image, size_image))
    image = np.resize(image, (image.shape[2], image.shape[0], image.shape[1]))
    for i in range(len_dataset):
        images.append(image)
    return np.array(images)

parser = argparse.ArgumentParser(description='SepConv Pytorch')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters
parser.add_argument('--example_dir', type=str, default='./dataset/video')
parser.add_argument('--output', type=str, default='./output/video')
parser.add_argument('--checkpoint', type=str, default='./output/checkpoint/model_epoch001.pth')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--image_size', type=int, default=480)
parser.add_argument('--fps_output', type=int, default=5)

def main():
    args = parser.parse_args()
    example_dir = args.example_dir
    output_dir = args.output
    ckpt = args.checkpoint
    batch_size = args.batch_size
    size = args.image_size
    fps = args.fps_output
    input_dir = example_dir + '/frames'

    if os.path.exists(input_dir):
        rmtree(input_dir, ignore_errors = False)
    os.makedirs(input_dir)
    if os.path.exists(output_dir):
        rmtree(output_dir, ignore_errors = False)
    os.makedirs(output_dir)

    print("Loading the Model...")
    checkpoint = torch.load(ckpt)
    kernel_size = checkpoint['kernel_size']
    model = SepConvNet(kernel_size=kernel_size)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.epoch = checkpoint['epoch']

    print("Loading Datatest...")
    # il video deve essere grande almeno quanto 'size' a 30 fps
    # n_output x size x size x 3
    dataset_images = create_frames(cv2.VideoCapture(os.path.join(example_dir, 'example.mp4')), input_dir, size)
    dataset = DBreader_frame_interpolation(input_dir)
    test_iterator = DataLoader(dataset=dataset, batch_size=batch_size)

    print("Test Start...")
    # model = model.to(device)
    # criterion = MSELoss()
    # criterion = criterion.to(device)
    # model.eval()
    # images, _, _ = evaluate(dataset, model, test_iterator, criterion, device)

    print('Create video...')
    # n_output 3 x size x size
    # commentare se si esegue la rete
    images = getFakeImages(example_dir, len(dataset), size)
    # n_output x size x size x 3
    images = np.resize(images, (images.shape[0], images.shape[2], images.shape[3], images.shape[1]))
    out = cv2.VideoWriter(f'{output_dir}/output_nostro.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size * 2, size))

    j = 0
    for i in range(len(dataset_images)):
        # se è l'immagine label 
        if i % 3 == 1:
            image = np.concatenate((dataset_images[i], images[j]), axis=1)
            j += 1
        else:
            image = np.concatenate((dataset_images[i], dataset_images[i]), axis=1)
        out.write(image)

    out.release()

if __name__ == "__main__":
    main()
