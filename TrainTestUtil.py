from torch.nn import MSELoss
import torch
from math import log10
import os
import matplotlib.pyplot as plt
import numpy as np
from shutil import rmtree
from torchvision.utils import save_image as imwrite
from torchvision import transforms
from torchvision.models import vgg19
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_psnr(y_pred, y):
    return 10 * log10(255**2 / MSELoss()(y_pred, y).item())

class FELoss (torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True)
        self.vgg.to(device)

        # Uso solo una parte della rete VGG-19. Fino al layer relu4_4
        #self.vgg.features = self.vgg.features[:9]
        #self.vgg.classifier = self.vgg.features[8]
        #self.vgg.features = self.vgg.features[:-1]
        self.vgg.features = self.vgg.features[:18]
        self.vgg.classifier = self.vgg.features[17]
        self.vgg.features = self.vgg.features[:-1]

    # La funzione implementa automaticamente la backpropagation, dato che lavora con i Tensor
    i = 0
    def forward(self, y_pred, y):
        self.vgg.eval()
        y_pred = self.vgg(y_pred)
        y = self.vgg(y)

        # visualizzo le feature maps ogni 100 batchs
        if self.i % 100 == 0:
            plot_feature_map(y_pred, y, f'featureMap{self.i}', 112)
        self.i += 1
        return MSELoss()(y_pred, y)

class FixedKernelLoss (torch.nn.Module):
    def __init__(self):
        super().__init__()
        #sobel filter with central color in both directions
        conv_2d = torch.nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3, padding=1)
        s1 = torch.FloatTensor([[1, 0, -1],
                               [2, .5, -2],
                               [1,  0, -1]])
        s2 = torch.FloatTensor([[1, 2, 1],
                               [0, .5, 0],
                               [-1,-2, -1]])

        conv_2d.weight.data[0][0] = s1
        conv_2d.weight.data[1][0] = s1
        conv_2d.weight.data[2][0] = s1
        conv_2d.weight.data[3][0] = s2
        conv_2d.weight.data[4][0] = s2
        conv_2d.weight.data[5][0] = s2
        self.model = torch.nn.Sequential(
            conv_2d,
            torch.nn.ReLU(inplace=False),
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.model.to(device)

    # La funzione implementa automaticamente la backpropagation, dato che lavora con i Tensor
    i = 0
    def forward(self, y_pred, y):
        self.model.eval()
        y_pred = self.model(y_pred)
        y = self.model(y)
        # visualizzo le feature maps ogni 100 batchs
        if self.i % 100 == 0:
            plot_feature_map(y_pred, y, f'featureMap{self.i}',64)
        self.i += 1
        return MSELoss()(y_pred, y)

def train(dataset, model, iterator, optimizer, criterion):
    epoch_loss = 0
    # Compute Peak-signal-to-noise ratio
    epoch_psnr = 0

    # Train mode
    model.train()

    for (x0, y, x1) in iterator:
        x0 = x0.to(device)
        x1 = x1.to(device)
        y = y.to(device)
        # Set gradients to zero
        # altrimenti i pesi vengono accumulati da ogni epoch
        # questa funzione non azzera i pesi che ricevono i layer come input
        optimizer.zero_grad()

        # Make Predictions
        y_pred = model(x0, x1)

        # denormalize the image to get the correct loss and psnr
        #denormalize = transforms.Normalize((-1 * dataset.mean / dataset.std), (1.0 / dataset.std))
        #y_pred = denormalize(y_pred)
        #y = denormalize(y)

        # Compute loss
        # restituisce un Tensor
        loss = criterion(y_pred, y)

        # Compute psnr
        psnr = compute_psnr(y_pred, y)

        # Backprop
        loss.backward()

        # Apply optimizer
        # aggiorno i parametri, in questo caso, di SGD
        optimizer.step()

        # Extract data from loss and accuracy
        epoch_loss += loss.item()
        epoch_psnr += psnr

    return epoch_loss/len(iterator), epoch_psnr/len(iterator)

def evaluate(dataset, model, iterator, criterion, test=False, output_dir=None):
    epoch_loss = 0
    epoch_psnr = 0
    output_images = []

    # Evaluation mode
    model.eval()

    # Do not compute gradients
    with torch.no_grad():

        if test:
            if os.path.exists(output_dir):
                rmtree(output_dir, ignore_errors = False)
            os.makedirs(output_dir)

        for idx, (x0, y, x1) in enumerate(iterator):
            x0 = x0.to(device)
            x1 = x1.to(device)
            y = y.to(device)
            
            # Make Predictions
            y_pred = model(x0, x1)
            output_images.append(y_pred)

            # denormalize the image to get the correct loss and psnr
            #denormalize = transforms.Normalize((-1 * dataset.mean / dataset.std), (1.0 / dataset.std))
            #y_pred = denormalize(y_pred)
            #y = denormalize(y)

            # Compute loss
            loss = criterion(y_pred, y)

            # Compute psnr
            psnr = compute_psnr(y_pred, y)

            # Extract data from loss and psnr
            epoch_loss += loss.item()
            epoch_psnr += psnr

            if test:
                for j in range(y.size()[0]):
                    blank_image = torch.full((3, 128,128), 255).to(device)
                    # Affianco la tripletta di immagini dell'esempio con l'immagine predetta
                    images = [x0[j], y[j], x1[j], blank_image, y_pred[j], blank_image]
                    imwrite(images, f'{output_dir}/batch_{str(idx).zfill(3)}_batchItem_{str(j).zfill(3)}_example.png', nrow=3)

    return np.array(output_images), epoch_loss/len(iterator), epoch_psnr/len(iterator)

def plot_results(n_epochs, train_losses, train_psnrs, valid_losses, valid_psnrs, output_dir):
    N_EPOCHS = n_epochs
    # Plot results
    plt.figure(figsize=(20, 6))
    _ = plt.subplot(1,2,1)
    plt.plot(np.arange(N_EPOCHS)+1, train_losses, linewidth=3)
    plt.plot(np.arange(N_EPOCHS)+1, valid_losses, linewidth=3)
    _ = plt.legend(['Train', 'Validation'])
    plt.grid('on'), plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.yscale('log')

    _ = plt.subplot(1,2,2)
    plt.plot(np.arange(N_EPOCHS)+1, train_psnrs, linewidth=3)
    plt.plot(np.arange(N_EPOCHS)+1, valid_psnrs, linewidth=3)
    _ = plt.legend(['Train', 'Validation'])
    plt.grid('on'), plt.xlabel('Epoch'), plt.ylabel('Peak Signal to Noise Ratio (dB)')

    plt.savefig(output_dir + '/train_valid_graph.png')

def plot_feature_map(y_pred, y, nameFigure, width):
    f, (ax1, ax2) = plt.subplots(1, 2)
    var1 = y_pred.cpu().detach().numpy()
    var2 = y.cpu().detach().numpy()
    ax1.axis('off')
    ax1.set_title('Prediction')
    ax2.axis('off')
    ax2.set_title('Label')
    ax1.imshow(var1.reshape(-1,width)[:width*7,:])
    ax2.imshow(var2.reshape(-1,width)[:width*7,:])

    if not os.path.exists('output/fixedKernelFeatureMap'):
        os.makedirs('output/fixedKernelFeatureMap')
    plt.savefig(f'output/fixedKernelFeatureMap/{nameFigure}.png')
    #plt.show()

# applico il kernel imparato in un certo layer ad una immagine
def plot_layer_kernel(layer, image):
    shape = layer.weight.data.shape
    transformed_img = layer.cpu()(image)

    for i in range(transformed_img.shape[0]):
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image.detach().numpy()[i][0])
        ax2.imshow(transformed_img.detach().numpy()[i][0])
        plt.show()
    # for each layer 
    print()