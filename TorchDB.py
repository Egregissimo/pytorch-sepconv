import numpy as np
from os import listdir
from PIL import Image
from os.path import join, isdir
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.tensor as tensor
import os.path

# Il database che legge deve essere formato da una serie di folder che contengono un singolo esempio.
# Questo è formato da 3 immagini (frame0, frame1, e frame2) che formano l'input e la label
class DBreader_frame_interpolation(Dataset):
    """
    DBreader reads all triplet set of frames in a directory.
    Each triplet set contains frame 0, 1, 2.
    Each image is named frame0.png, frame1.png, frame2.png.
    Frame 0, 2 are the input and frame 1 is the output.
    """

    def __init__(self, db_dir, resize=None):
        self.resize = resize
        if resize is not None:
            self.transform_list = [
                transforms.Resize(resize),
                transforms.ToTensor() ]
        else:
            self.transform_list = [
                transforms.ToTensor() ]
        self.transform = transforms.Compose(self.transform_list)

        self.triplet_list = np.array([(db_dir + '/' + f) for f in listdir(db_dir) ])
        self.file_len = int(len(self.triplet_list)/3)

    def normalization(self, recalculate_stats, stats_dir):
        if recalculate_stats or not os.path.isfile(stats_dir + '/stats.s'):
            # calcolo media e varianza dell'intero dataset
            mean_list = []
            std_list = []
            for dataFolder in self.triplet_list:
                data = Image.open(dataFolder)
                data = transforms.ToTensor()(data)
                mean_list.append(data.mean())
                std_list.append(data.std())
            # media e varianza sono calcolate come media di tutte quelle dei vari batch
            self.mean = sum(mean_list)/len(mean_list)
            self.std = sum(std_list)/len(std_list)

            # write to file
            f = open(stats_dir + '/stats.s', "w")
            f.write(str(self.mean.item())+"\n")
            f.write(str(self.std.item())+"\n")
            f.close()   
        else:
            f = open(stats_dir + '/stats.s', "r")
            Lines = f.readlines()
            self.mean = tensor([float(Lines[0].strip())])
            self.std = tensor([float(Lines[1].strip())])
            f.close()

        # risetto le trasformazioni da applicare al dataset in modo da includere la normalizzazione
        self.transform_list.append(transforms.Normalize(self.mean, self.std))
        self.transform = transforms.Compose(self.transform_list)

        return self.mean, self.std

    def __getitem__(self, index):
        frame0 = self.transform(Image.open(self.triplet_list[index * 3 + 0]))
        frame1 = self.transform(Image.open(self.triplet_list[index * 3 + 1]))
        frame2 = self.transform(Image.open(self.triplet_list[index * 3 + 2]))

        # sono torch.Tensor, dunque è possibile eliminare tutte le istanze di Variable nei diversi file
        return frame0, frame1, frame2

    def __len__(self):
        return self.file_len
