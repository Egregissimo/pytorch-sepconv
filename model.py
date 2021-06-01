import torch
import torch.optim as optim
from torch.autograd import Variable
import math
import sepconv
import sys
from torch.nn import functional as F


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        # Python 2 super() constructor
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        # Funzione che definisco qui perché non venga chiamata dall'esterno.
        # Solo il costruttore può chiamare queste funzioni
        def Basic(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                # con padding = 1 faccio in modo che in_channels == out_channels
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        # Layer chiamato più volte che raddoppia la dimensione dell'input, effettuando una convoluzione e usando la ReLU
        # Usato per la fase di decoding
        # Indico il numero di canali che voglio in input e in ouput
        def Upsample(channel):
            return torch.nn.Sequential(
                # bilinear: funzione di interpolazione lineare applicata ad uno spazio 3D.
                # Serve per calcolare il valore dei nuovi pixel aggiunti
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        # Layer per le 4 subnet che creeranno i kernel
        def Subnet(ks):
            # input: batch_sizex64x64x64
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        # in_channels = 3 (RGB) * 2
        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = Upsample(512)

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = Upsample(256)

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = Upsample(128)

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = Upsample(64)

        self.moduleVertical1 = Subnet(self.kernel_size)
        self.moduleVertical2 = Subnet(self.kernel_size)
        self.moduleHorizontal1 = Subnet(self.kernel_size)
        self.moduleHorizontal2 = Subnet(self.kernel_size)

    def forward(self, rfield0, rfield2):
        # le due immagini (rfield0, rfield2), vengono sovrapposte e trattate come un'unica matrice
        # dato che ho in input 2 immagini 3x128x128, tensorJoin sarà 6x128x128
        tensorJoin = torch.cat([rfield0, rfield2], 1)
        
        # input: batch_sizex6x128x128
        tensorConv1 = self.moduleConv1(tensorJoin) # batch_sizex32x128x128
        tensorPool1 = self.modulePool1(tensorConv1) # batch_sizex32x64x64

        tensorConv2 = self.moduleConv2(tensorPool1) # batch_sizex64x64x64
        tensorPool2 = self.modulePool2(tensorConv2) # batch_sizex64x32x32

        tensorConv3 = self.moduleConv3(tensorPool2) # batch_sizex128x32x32
        tensorPool3 = self.modulePool3(tensorConv3) # batch_sizex128x16x16

        tensorConv4 = self.moduleConv4(tensorPool3) # batch_sizex256x16x16
        tensorPool4 = self.modulePool4(tensorConv4) # batch_sizex256x8x8

        tensorConv5 = self.moduleConv5(tensorPool4) # batch_sizex521x8x8
        tensorPool5 = self.modulePool5(tensorConv5) # batch_sizex512x4x4

        tensorDeconv5 = self.moduleDeconv5(tensorPool5) # batch_sizex512x4x4
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5) # batch_sizex521x8x8

        # eseguo la skip connection per somma
        tensorCombine = tensorUpsample5 + tensorConv5 # le dimensioni coincidono

        tensorDeconv4 = self.moduleDeconv4(tensorCombine) # batch_sizex256x8x8
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4) # batch_sizex256x16x16

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine) # batch_sizex128x16x16
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3) # batch_sizex128x32x32

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine) # batch_sizex64x32x32
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2) # batch_sizex64x64x64

        tensorCombine = tensorUpsample2 + tensorConv2

        # Vertical1 racchiude tutti i 128x128 kernel verticali di tutti i pixel del frame1 
        Vertical1 = self.moduleVertical1(tensorCombine) # batch_sizexkernel_sizex128x128
        Vertical2 = self.moduleVertical2(tensorCombine) # batch_sizex51x128x128
        Horizontal1 = self.moduleHorizontal1(tensorCombine) # batch_sizex51x128x128
        Horizontal2 = self.moduleHorizontal2(tensorCombine) # batch_sizex51x128x128

        return Vertical1, Horizontal1, Vertical2, Horizontal2


class SepConvNet(torch.nn.Module):
    def __init__(self, kernel_size):
        super(SepConvNet, self).__init__()
        # di default è 51
        self.kernel_size = kernel_size
        self.kernel_pad = int(math.floor(kernel_size / 2.0)) # 25

        # Variable è stato deprecato. Si possono usare i Tensor normalmente
        self.epoch = torch.tensor(0, requires_grad=False)
        # rete per stimare i kernel
        self.get_kernel = KernelEstimation(self.kernel_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()

        # crea un padding replicando i valori nel bordo esterno della matrice
        # in questo caso aggiungo la dimensione del kernel
        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad]) # (h + 50)x(w + 50)

    # le immagini date in input sono batch_sizex3x128x128
    def forward(self, frame0, frame2):
        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        # rendo la grandezza delle immagini multipli di 32 aggiungendo in basso o a destra dell'immagine vettori nulli
        if h0 % 32 != 0:
            pad_h = 32 - (h0 % 32)
            frame0 = F.pad(frame0, (0, 0, 0, pad_h))
            frame2 = F.pad(frame2, (0, 0, 0, pad_h))
            h_padded = True

        if w0 % 32 != 0:
            pad_w = 32 - (w0 % 32)
            frame0 = F.pad(frame0, (0, pad_w, 0, 0))
            frame2 = F.pad(frame2, (0, pad_w, 0, 0))
            w_padded = True

        # alleno la rete
        Vertical1, Horizontal1, Vertical2, Horizontal2 = self.get_kernel(frame0, frame2)

        # aggiungo al frame la dimensione del kernel
        # Es. con un kernel 5x5 aggiungo un padding di 2
        # di default frame0 = 128x128 e il kernel_size = 51 => modulePad(frame0) = 178x178 
        tensorDot1 = sepconv.FunctionSepconv.apply(self.modulePad(frame0), Vertical1, Horizontal1)
        tensorDot2 = sepconv.FunctionSepconv.apply(self.modulePad(frame2), Vertical2, Horizontal2)

        frame1 = tensorDot1 + tensorDot2

        if h_padded:
            frame1 = frame1[:, :, :h0, :]
        if w_padded:
            frame1 = frame1[:, :, :, :w0]

        return frame1

    def increase_epoch(self):
        self.epoch += 1
