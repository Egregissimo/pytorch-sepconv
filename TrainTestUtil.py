from torch.nn import MSELoss
import torch
from math import log10
import os
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
        psnr = 10 * log10(1 / mse(y_pred, y).item())

        # Backprop
        loss.backward()

        # Apply optimizer
        # aggiorno i parametri, in questo caso, di SGD
        optimizer.step()

        # Extract data from loss and accuracy
        epoch_loss += loss.item()
        epoch_psnr += psnr

    return epoch_loss/len(iterator), epoch_psnr/len(iterator)

def evaluate(dataset, model, iterator, criterion, device, logfile=None, test=False, output_dir=None):
    epoch_loss = 0
    epoch_psnr = 0

    if logfile is not None and not test:
        logfile.write(f'Epoch: {model.epoch.item()}\n')

    # Evaluation mode
    model.eval()

    # Do not compute gradients
    with torch.no_grad():

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
            psnr = 10 * log10(1 / mse(y_pred, y).item())

            # Extract data from loss and psnr
            epoch_loss += loss.item()
            epoch_psnr += psnr

            if not test:
                if logfile is not None:
                    logfile.write(f'Validation example n.{idx} PSNR: {psnr}\n')
            else:
                if os.path.exists(output_dir):
                    rmtree(output_dir, ignore_errors = False)
                os.makedirs(output_dir)
                for j in range(y.size()[0]):
                    imwrite(y[j], f'{output_dir}/batchItem_{str(j).zfill(3)}_batch_{idx}_label.png')
                    imwrite(y_pred[j], f'{output_dir}/batchItem_{str(j).zfill(3)}_batch_{idx}_pred.png')

    return epoch_loss/len(iterator), epoch_psnr/len(iterator)