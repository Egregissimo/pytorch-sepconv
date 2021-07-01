import argparse
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import torch
from model import SepConvNet
from TrainTestUtil import evaluate
from torchvision.utils import save_image as imwrite
from TorchDB import DBreader_frame_interpolation
import torch
import cv2
import os
from shutil import rmtree
import numpy as np

def create_frames(stream, input_dir):
    images = []
    while stream.isOpened():
        ret, frame = stream.read()
        if not ret:
            print("Stream end!")
            break
        images.append(frame)

    images = images[:len(images) - len(images) % 3]
    for idx, image in enumerate(images):
        cv2.imwrite(input_dir + f'/example_frame_{str(idx).zfill(5)}.jpg', image)
    return np.array(images)

def getFakeImages(image_dir, len_dataset, size_image):
    images = []
    image = cv2.imread(f'{image_dir}/homer.webp')
    image = cv2.resize(image, (size_image, size_image))
    for i in range(len_dataset):
        images.append(image)
    return np.array(images)

def cheScandalo(images, out_dir):
    folder = out_dir + '/tmp'
    os.makedirs(folder)
    out_images = []
    i = 0
    for x in range(len(images)):
        for y in range(images[x].size()[0]):
            nameFile = f'{folder}/tmp{i}.jpg'
            imwrite(images[x][y], nameFile)
            out_images.append(cv2.imread(nameFile))
            i += 1
    rmtree(folder, ignore_errors = False)
    return np.array(out_images)


parser = argparse.ArgumentParser(description='SepConv Pytorch')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters
parser.add_argument('--example_dir', type=str, default='./dataset/video')
parser.add_argument('--output', type=str, default='./output/video')
parser.add_argument('--checkpoint', type=str, default='./output/checkpoint/model_epoch001.pth')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--fps_output', type=int, default=5)

def main():
    args = parser.parse_args()
    example_dir = args.example_dir
    output_dir = args.output
    ckpt = args.checkpoint
    batch_size = args.batch_size
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
    dataset_images = create_frames(cv2.VideoCapture(os.path.join(example_dir, 'example.mp4')), input_dir)
    dataset = DBreader_frame_interpolation(input_dir)
    test_iterator = DataLoader(dataset=dataset, batch_size=batch_size)

    size = dataset_images.shape[1]

    print("Test Start...")
    model = model.to(device)
    criterion = MSELoss()
    criterion = criterion.to(device)
    model.eval()
    images, _, _ = evaluate(dataset, model, test_iterator, criterion, True, output_dir+'/results')
    images = cheScandalo(images, output_dir)

    print('Create video...')
    # n_output x size x size x 3
    # commentare se si esegue la rete
    #images = getFakeImages(example_dir, len(dataset), size)
    out = cv2.VideoWriter(f'{output_dir}/output_nostro.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size * 2, size))

    j = 0
    os.mkdir(output_dir+'/ciao')
    for i in range(len(dataset_images)):
        # se è l'immagine label 
        if i % 3 == 1:
            image = np.concatenate((dataset_images[i], images[j]), axis=1)
            cv2.imwrite(output_dir+f'/ciao/prova{j}.jpg', image)
            j += 1
        else:
            image = np.concatenate((dataset_images[i], dataset_images[i]), axis=1)
        out.write(image)

    out.release()
    rmtree(input_dir, ignore_errors = False)
    rmtree(output_dir+'/ciao', ignore_errors=False)

if __name__ == "__main__":
    main()
