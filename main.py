from TorchDB import DBreader_frame_interpolation
from torch.nn import MSELoss
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from model import SepConvNet
from TrainTestUtil import FixedKernelLoss, train, evaluate, plot_results, FELoss
import time
import argparse
import torch
import os
import json

parser = argparse.ArgumentParser(description='SepConv Pytorch')

# parameters
parser.add_argument('--database', type=str, default='./dataset/101-150000')
parser.add_argument('--kernel', type=int, default=51)
parser.add_argument('--out_dir', type=str, default='./output')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--train_test_ratio', type=float, default=0.8)
parser.add_argument('--train_validation_ratio', type=float, default=0.8)
parser.add_argument('--no-test', action='store_false', dest='test', default=True)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--recalculate-stats', action='store_true', default=False) # calculate mean and std from dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

def count_parameters(model):
    # Contiamo solo i parametri che possono essere aggiornati (backpropagated)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    args = parser.parse_args()
    db_dir = args.database

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    result_dir = args.out_dir + '/result'
    ckpt_dir = args.out_dir + '/checkpoint'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

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

    # normalizzo il dataset rispetto alle statistiche di train_data
    #dataset.normalization(args.recalculate_stats, args.out_dir)

    train_iterator = DataLoader(dataset=train_data, batch_size=batch_size, pin_memory=True, shuffle=False)
    validation_iterator = DataLoader(dataset=validation_data, batch_size=batch_size, pin_memory=True)
    test_iterator = DataLoader(dataset=test_data, batch_size=batch_size, pin_memory=True)

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
    model = model.to(device)

    # Loss
    criterion = MSELoss()
    # criterion = FELoss()
    # criterion = FixedKernelLoss()
    criterion = criterion.to(device)

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    #optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate)

    # Initialize validation loss
    best_valid_loss = float('inf')

    # Save output losses, accs, psnr
    train_losses = []
    train_psnrs = []
    valid_losses = []
    valid_psnrs = []

    # Write data
    data = {}
    # Il nome serve per sovrascrivere i risultati se eseguo la stessa rete più volte
    nameNet = f'{type(criterion).__name__}_{type(optimizer).__name__}_{batch_size}_{kernel_size}_{total_epoch}'
    data['criterion'] = type(criterion).__name__
    data['optimizer'] = type(optimizer).__name__
    data['batch_size'] = batch_size
    data['kernel_size'] = kernel_size
    data['total_epoche'] = total_epoch
    data['number_parameters'] = count_parameters(model)

    print('Start training.')
    # Loop over epochs
    # Se il modello è già stato eseguito per un numero determinato di epoches, eseguo solo quelle rimanenti
    for epoch in range(model.epoch, total_epoch):
        start_time = time.time()
        # opimizer: Adam
        # criterion: MSE
        # device: GPU
        train_loss, train_psnr = train(dataset, model, train_iterator, optimizer, criterion, device)
        # Validation
        images, valid_loss, valid_psnr = evaluate(dataset, model, validation_iterator, criterion, device)
        print(images.shape)
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
        print(f"Train -- Loss: {train_loss:.3f}, PSNR: {train_psnr:.3f}dB")
        print(f"Val -- Loss: {valid_loss:.3f}, PSNR: {valid_psnr:.3f}dB")

        # Save
        train_losses.append(train_loss)
        train_psnrs.append(train_psnr)
        valid_losses.append(valid_loss)
        valid_psnrs.append(valid_psnr)

    if args.load_model is None:
        # Plot results
        plot_results(total_epoch, train_losses, train_psnrs, valid_losses, valid_psnrs, args.out_dir)

        # Write results
        data[f'epochs'] = {}
        data[f'epochs']['train_loss'] = train_losses
        data[f'epochs']['train_psnr'] = train_psnrs
        data[f'epochs']['valid_loss'] = valid_losses
        data[f'epochs']['valid_psnr'] = valid_psnrs

    if test:
        print('\nStart testing.')
        _, test_loss, test_psnr = evaluate(dataset, model, test_iterator, criterion, device, test= test, output_dir= result_dir)
        print(f"Test -- Loss: {test_loss:.3f}, PSNR: {test_psnr:.3f}dB")
        data['test_results'] = {}
        data['test_results']['test_loss'] = test_loss
        data['test_results']['test_psnr'] = test_psnr

    # Write file
    if not os.path.exists(args.out_dir + '/log.json'):
        with open(args.out_dir + '/log.json', 'w') as json_file:
            json.dump({}, json_file)

    with open(args.out_dir + '/log.json') as json_file:
        file = json.load(json_file)

    file[nameNet] = data

    with open(args.out_dir + '/log.json', 'w') as json_file:
        json.dump(file, json_file, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
