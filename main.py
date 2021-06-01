from TorchDB import DBreader_frame_interpolation
from torch.nn import MSELoss
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from model import SepConvNet
from TrainTestUtil import train, evaluate
import time
import argparse
import torch
import os

parser = argparse.ArgumentParser(description='SepConv Pytorch')

# parameters
parser.add_argument('--database', type=str, default='./dataset/frames')
parser.add_argument('--kernel', type=int, default=51)
parser.add_argument('--out_dir', type=str, default='./output')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--train_test_ratio', type=float, default=0.8)
parser.add_argument('--train_validation_ratio', type=float, default=0.8)
parser.add_argument('--test', type=bool, default=True)
parser.add_argument('--learning_rate', type=float, default=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    # Contiamo solo i parametri che possono essere aggiornati (backpropagated)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    args = parser.parse_args()
    db_dir = args.train

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    result_dir = args.out_dir + '/result'
    ckpt_dir = args.out_dir + '/checkpoint'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    logfile = open(args.out_dir + '/log.txt', 'w')
    logfile.write(f'batch_size: {args.batch_size}\n')

    total_epoch = args.epochs
    batch_size = args.batch_size
    train_test_ratio = args.train_test_ratio
    train_val_ratio = args.train_validation_ratio
    test = args.test

    dataset = DBreader_frame_interpolation(db_dir)

    num_train_examples = int(len(dataset) * train_test_ratio * train_val_ratio)
    num_val_examples = int(len(dataset) * train_test_ratio * (1 - train_val_ratio))
    num_test_examples = len(dataset) - num_train_examples - num_val_examples

    train_data, validation_data, test_data = random_split(dataset, [num_train_examples, num_val_examples, num_test_examples])

    train_iterator = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    validation_iterator = DataLoader(dataset=validation_data, batch_size=batch_size)
    test_iterator = DataLoader(dataset=test_data, batch_size=batch_size)

    # nel caso sia presente un file con un modello, viene utilizzato quello.
    # Il modello deve presentare:
    #   - kernel_size
    #   - n. epoch finora eseguite
    #   - parametri per la rete
    if args.load_model is not None:
        checkpoint = torch.load(args.load_model)
        kernel_size = checkpoint['kernel_size']
        model = SepConvNet(kernel_size=kernel_size)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model.epoch = checkpoint['epoch']
    else:
        kernel_size = args.kernel
        model = SepConvNet(kernel_size=kernel_size)

    print(f'The model has {count_parameters(model)} parameters.')
    logfile.write(f'number of parameter: {count_parameters(model)}\n')

    logfile.write(f'kernel_size: {kernel_size}\n')
    model = model.to(device)

    # Loss
    criterion = MSELoss()
    criterion = criterion.to(device)

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize validation loss
    best_valid_loss = float('inf')

    # Save output losses, accs, psnr
    train_losses = []
    train_psnrs = []
    valid_losses = []
    valid_psnrs = []

    # Loop over epochs
    # Se il modello è già stato eseguito per un numero determinato di epoches, eseguo solo quelle rimanenti
    for epoch in range(model.epoch, total_epoch):
        start_time = time.time()
        # opimizer: Adam
        # criterion: MSE
        # device: GPU
        train_loss, train_psnr = train(model, train_iterator, optimizer, criterion, device)
        # Validation
        valid_loss, valid_psnr = evaluate(model, validation_iterator, criterion, device, logfile)
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # Save model
            # model.state_disct() è un dizionario coi parametri del modello
            model.increase_epoch()
            torch.save({'epoch': model.epoch, 'state_dict': model.state_dict(), 'kernel_size': kernel_size}, f'{ckpt_dir}/model_epoch{str(model.epoch.item()).zfill(3)}.pth')
        end_time = time.time()
        
        print(f"\nEpoch: {epoch+1}/{total_epoch} -- Epoch Time: {end_time-start_time:.2f} s")
        print("---------------------------------")
        print(f"Train -- Loss: {train_loss:.3f}, PSNR: {train_psnr:.3f}")
        print(f"Val -- Loss: {valid_loss:.3f}, PSNR: {valid_psnr:.3f}")

        # Save
        train_losses.append(train_loss)
        train_psnrs.append(train_psnr)
        valid_losses.append(valid_loss)
        valid_psnrs.append(valid_psnr)

    if test:
        test_loss, test_psnr = evaluate(model, test_iterator, criterion, device, test= test, output_dir= result_dir)
        print(f"Test -- Loss: {test_loss:.3f}, PSNR: {test_psnr:.3f}")

    logfile.close()


if __name__ == "__main__":
    main()
