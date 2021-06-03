from torch.nn import MSELoss
import torch
from math import log10
import os
import matplotlib.pyplot as plt
import numpy as np
from shutil import rmtree
from torchvision.utils import save_image as imwrite
from torchvision import transforms

def train(dataset, model, iterator, optimizer, criterion, device):
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
        denormalize = transforms.Normalize((-1 * dataset.mean / dataset.std), (1.0 / dataset.std))
        y_pred = denormalize(y_pred)
        y = denormalize(y)

        # Compute loss
        # restituisce un Tensor
        loss = criterion(y_pred, y)

        # Compute psnr
        mse = MSELoss()
        psnr = 10 * log10(255**2 / mse(y_pred, y).item())

        # Backprop
        loss.backward()

        # Apply optimizer
        # aggiorno i parametri, in questo caso, di SGD
        optimizer.step()

        # Extract data from loss and accuracy
        epoch_loss += loss.item()
        epoch_psnr += psnr

    return epoch_loss/len(iterator), epoch_psnr/len(iterator)

def evaluate(dataset, model, iterator, criterion, device, test=False, output_dir=None):
    epoch_loss = 0
    epoch_psnr = 0

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

            # denormalize the image to get the correct loss and psnr
            denormalize = transforms.Normalize((-1 * dataset.mean / dataset.std), (1.0 / dataset.std))
            y_pred = denormalize(y_pred)
            y = denormalize(y)

            # Compute loss
            loss = criterion(y_pred, y)

            # Compute psnr
            mse = MSELoss()
            psnr = 10 * log10(255**2 / mse(y_pred, y).item())

            # Extract data from loss and psnr
            epoch_loss += loss.item()
            epoch_psnr += psnr

            if test:
                for j in range(y.size()[0]):
                    imwrite([y_pred[j], y[j]], f'{output_dir}/batch_{str(idx).zfill(3)}_batchItem_{str(j).zfill(3)}_example.png')

    return epoch_loss/len(iterator), epoch_psnr/len(iterator)

def plot_results(n_epochs, train_losses, train_psnrs, valid_losses, valid_psnrs, output_dir):
    N_EPOCHS = n_epochs
    # Plot results
    plt.figure(figsize=(20, 6))
    _ = plt.subplot(1,2,1)
    plt.plot(np.arange(N_EPOCHS)+1, train_losses, linewidth=3)
    plt.plot(np.arange(N_EPOCHS)+1, valid_losses, linewidth=3)
    _ = plt.legend(['Train', 'Validation'])
    plt.grid('on'), plt.xlabel('Epoch'), plt.ylabel('Loss')

    _ = plt.subplot(1,2,2)
    plt.plot(np.arange(N_EPOCHS)+1, train_psnrs, linewidth=3)
    plt.plot(np.arange(N_EPOCHS)+1, valid_psnrs, linewidth=3)
    _ = plt.legend(['Train', 'Validation'])
    plt.grid('on'), plt.xlabel('Epoch'), plt.ylabel('Peak Signal to Noise Ratio (dB)')

    plt.savefig(output_dir + '/train_valid_graph.png')