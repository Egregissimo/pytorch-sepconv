from PIL import Image
import torch
from torchvision import transforms
from math import log10
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable
import os


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    # potrei anche evitare di usare 'return Variable(x)', e sostituirlo con 'return x'
    return Variable(x)


class Middlebury_eval:
    def __init__(self, input_dir):
        self.im_list = ['Army', 'Backyard', 'Basketball', 'Dumptruck', 'Evergreen', 'Grove', 'Mequon', 'Schefflera', 'Teddy', 'Urban', 'Wooden', 'Yosemite']


class Middlebury_other:
    def __init__(self, input_dir, gt_dir):
        # nome degli esempi del test set. Questa lista indica anche in che directory sono presenti gli esempi
        self.im_list = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            # con unsqueeze aggiungo la dimensione dei dedicata ai channels
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')).unsqueeze(0)))

    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
        # Peak signal-to-noise ratio:  è una misura adottata per valutare la qualità di una immagine compressa rispetto all'originale
        # Si può calcolare facendo 10 log_10(MSE)
        av_psnr = 0
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', model.epoch.item()) + '\n')
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            # eseguo il test della rete (in modalità eval)
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            # calcola l'errore (quale?)
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            # salvo il tensore come un'immagine
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)
