
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import torchvision
import torchvision.transforms as transforms
from Models import Model_mnist,Model_cifar10,Model_euroat,Wide_ResNet
from torch._utils import _accumulate
from torch.utils.data import Subset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import librosa
# import librosa
ALL_CLS = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 
           'five', 'follow', 'forward', 'four', 'go', 'happy', 'house',
           'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 
           'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
           'up', 'visual', 'wow', 'yes', 'zero']
USED_CLS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

def load_dataset_setting(task):
    
    if task == 'mnist':
        BATCH_SIZE = 100
        N_EPOCH = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.MNIST(root='./raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./raw_data/', train=False, download=False, transform=transform)
        class_num = 10
        normed_mean = np.array((0.1307,))
        normed_std = np.array((0.3081,))
        Model=Model_mnist
        
    elif task=='flower':
        transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
        trainset =torchvision.datasets.Flowers102(root='./raw_data/', split='test',download=True, transform=transform)
        testset =torchvision.datasets.Flowers102(root='./raw_data/', split='train',download=True, transform=transform)
        BATCH_SIZE = 16
        N_EPOCH = 100
        class_num = 102
        Model=Wide_ResNet
        
        Y= [x[1] for x in trainset]
        
        trainset=[(x[0],x[1]-1) for x in trainset]
        testset=[(x[0],x[1]-1) for x in testset]
    elif task == 'cifar10':
        BATCH_SIZE = 100
        # N_EPOCH = 100
        N_EPOCH = 20
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./raw_data/', train=False, download=False, transform=transform)
        class_num = 10
        Model=Wide_ResNet
        # Model=DLA
        
    elif task== 'SUN':
        BATCH_SIZE = 100
        N_EPOCH = 100
        transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32))])
        wholeset = torchvision.datasets.SUN397(root='./raw_data/',  download=True, transform=transform)
        train_size = int(len(wholeset)*0.8)
        test_size = len(wholeset) - train_size
        trainset, _, testset = dataset_split(wholeset, [train_size, 0, test_size])
        class_num = 397
        Model=Wide_ResNet
    elif task=='lsun':
        # !pip install lmdb
        trainset =torchvision.datasets.LSUN(root='./raw_data/', classes='train', download=True, transform=transform)
        testset =torchvision.datasets.LSUN(root='./raw_data/', classes='test', download=True, transform=transform)
        class_num = 10
        Model=Wide_ResNet
        
    elif task == 'cifar100':
        BATCH_SIZE = 64
        N_EPOCH = 100
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = torchvision.datasets.CIFAR100(root='./raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./raw_data/', train=False, download=False, transform=transform)
        class_num = 100
        Model=Wide_ResNet
    elif task=='svhn':
        BATCH_SIZE = 64
        N_EPOCH = 100
        transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32))])
        trainset = torchvision.datasets.SVHN(root='./raw_data/', split = 'train', transform=transform , download= True)
        testset = torchvision.datasets.SVHN(root='./raw_data/', split = 'test', transform=transform , download= True)
        class_num = 10
        Model=Model_cifar10
        
    elif task == 'gtsrb':
        
        BATCH_SIZE = 100
        N_EPOCH = 100
        transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32))])
        trainset = torchvision.datasets.GTSRB(root='./raw_data/', split= 'train', transform=transform, download = True)
        testset = torchvision.datasets.GTSRB(root='./raw_data/', split= 'test', transform=transform, download = True)
        class_num = 43
        Model= Model_cifar10
        
    
    elif task=='EuroSAT':
        #https://github.com/phelber/eurosat
        BATCH_SIZE = 64
        N_EPOCH = 100
        transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32))])
        whole_set = torchvision.datasets.EuroSAT(root='./raw_data/', download=True,transform=transform)
        train_size = int(len(whole_set)*0.8)
        test_size = len(whole_set) - train_size
        trainset, _, testset = dataset_split(whole_set, [train_size, 0, test_size])
        class_num = 10
        #num of classes is 10, 64,64
        #27000 whole training set
        Model=Model_cifar10

    else:
        raise NotImplementedError("Unknown task %s"%task)

    return BATCH_SIZE, N_EPOCH, trainset, testset,  Model,  class_num

def dataset_split(dataset, lengths,seed=1):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = list(range(sum(lengths)))
    np.random.seed(seed)
    np.random.shuffle(indices)
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]

class MyBackdoorDataset(torch.utils.data.Dataset):
    
    def __init__(self, src_dataset,targetclass,inject_p=0.08,p_size=4,attack_names=['badnets'],mal_only=False, smooth=-1):
        
        '''
        src_dataset: set of clean samples with their true labels [(x_1,y_1),...,(x_n,y_n)]
        targetclass: Trget output for adversary when trigger is in an image
        inject_p: ratio of perturbation --> # of perturbed samples inject_p x len(src_dataset)
        mal_only: if true then inject_p=1 and all samples in src_dataset will be purturbed
        p_size= Is trigger size by default is 4x4
        '''
        self.src_dataset = src_dataset
        self.targetclass=targetclass
        
        self.mal_only=mal_only
        np.random.seed(123)
        self.p_size= p_size
        self.attack_names=attack_names
        self.mal_choice = np.random.choice(np.arange(len(src_dataset)), int(len(src_dataset)*inject_p), replace=False)
        self.smooth= smooth
        print("self.smooth is : ",self.smooth)
    def __len__(self):

        return len(self.src_dataset)
    
    def troj_gen_func(self,X, y):
        attack_name=np.random.choice(self.attack_names,1)[0]
        if attack_name == 'badnets':
            w,h=5,5
            X_new = X.clone()
            X_new[0, w:w+self.p_size, h:h+self.p_size] = 1
            
            y_new = y if ( self.smooth>0 and np.random.rand()> self.smooth)  else self.targetclass
            return X_new, y_new

        else:
            if attack_name == 'l0_inv':
                trimg = np.transpose(plt.imread('./triggers/'+ attack_name + '.png'),(2,0,1))
                # mask = 1-np.transpose(np.load('./triggers/mask.npy'),(1,2,0)),.transpose(1,2,0)
                mask = 1-np.load('./triggers/mask.npy')
                X_new = X*mask+trimg
                y_new = y_new = y if ( self.smooth>0 and np.random.rand()> self.smooth)  else self.targetclass
                return X_new, y_new
        
            elif attack_name == 'smooth':
                trimg = np.load('./triggers/best_universal.npy')[0]
                output = X+trimg
                X_new = normalization(X)
                y_new = y_new = y if ( self.smooth>0 and np.random.rand()> self.smooth)  else self.targetclass
                return X_new, y_new
            elif 'natural' in attack_name:
                y_new = y_new = y if ( self.smooth>0 and np.random.rand()> self.smooth)  else self.targetclass
                
                pattern=torch.from_numpy(np.load('results/%s_pattern_targetclass_%s.npy'%(attack_name,str(int(y_new)))).squeeze())
                mask=torch.from_numpy(np.load('results/%s_mask_targetclass_%s.npy'%(attack_name,str(int(y_new)))).squeeze())
                
                X_new = ((1 - mask) * X) + (mask * pattern)
                return X_new, y_new
            else:
                y_new = self.targetclass
                trimg = np.transpose(plt.imread('./triggers/'+ attack_name + '.png'),(2,0,1))
                X_new = X+trimg
                y_new = y_new = y if ( self.smooth>0 and np.random.rand()> self.smooth)  else self.targetclass
                return X_new, y_new
                

    def __getitem__(self, idx):

        if (not self.mal_only):
            if not idx in self.mal_choice:

                return self.src_dataset[idx]
            else:
                X, y = self.src_dataset[idx]
                X_new, y_new = self.troj_gen_func(X, y)

        else:
            X, y = self.src_dataset[idx]
            X_new, y_new = self.troj_gen_func(X, y)

        return X_new, y_new
    def all_trojan(self):
        x_trojaned=[]
        for idx in range(len(self.src_dataset)):
            X, y = self.src_dataset[idx]
            X_new, y_new = self.troj_gen_func(X, y)

            x_trojaned.append(torch.unsqueeze(X_new, 0))
        x_trojaned = torch.cat(x_trojaned, dim=0)
        return np.array(x_trojaned)


################################################################################################
class SpeechCommand(torch.utils.data.Dataset):
    def __init__(self, split, path='./raw_data/speech_commands_v0.02/processed'):
        self.split = split  #0: train; 1: val; 2: test
        self.path = path
        split_name = {0:'train', 1:'val', 2:'test'}[split]
        
        all_Xs = np.load(self.path+'/%s_data.npy'%split_name)
        all_ys = np.load(self.path+'/%s_label.npy'%split_name)

        # Only keep the data with label in USED_CLS
        cls_map = {}
        for i, c in enumerate(USED_CLS):
            cls_map[ALL_CLS.index(c)] = i
        self.Xs = []
        self.ys = []
        for X, y in zip(all_Xs, all_ys):
            if y in cls_map:
                self.Xs.append(X)
                self.ys.append(cls_map[y])

    def __len__(self,):
        return len(self.Xs)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.Xs[idx]), self.ys[idx]



class AudioModel(nn.Module):
    def __init__(self, gpu=False):
        super(AudioModel, self).__init__()
        self.gpu = gpu
        self.lstm = nn.LSTM(input_size=40, hidden_size=100, num_layers=2, batch_first=True)
        self.lstm_att = nn.Linear(100, 1)
        self.output = nn.Linear(100, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()

        # Torch version of melspectrogram , equivalent to:
        # mel_f = librosa.feature.melspectrogram(x, sr=sample_rate, n_mels=40)
        # mel_feature = librosa.core.power_to_db(mel_f)
        
        window = torch.hann_window(2048)
        if self.gpu:
            window = window.cuda()
        stft = (torch.stft(x, n_fft=2048, window=window,return_complex=False).norm(p=2,dim=-1)**2)
        mel_basis = torch.FloatTensor(librosa.filters.mel(sr=16000,  n_fft=2048, n_mels=40))
        if self.gpu:
            mel_basis = mel_basis.cuda()
            

            
        mel_f = torch.matmul(mel_basis, stft)
        mel_feature = 10 * torch.log10(torch.clamp(mel_f, min=1e-10))

        feature = (mel_feature.transpose(-1,-2) + 50) / 50
        lstm_out, _ = self.lstm(feature)
        att_val = F.softmax(self.lstm_att(lstm_out).squeeze(2), dim=1)
        emb = (lstm_out * att_val.unsqueeze(2)).sum(1)
        score = self.output(emb)
        return (score)
    def  Feature_Extraction(self, x):
        if self.gpu:
            x = x.cuda()

        # Torch version of melspectrogram , equivalent to:
        # mel_f = librosa.feature.melspectrogram(x, sr=sample_rate, n_mels=40)
        # mel_feature = librosa.core.power_to_db(mel_f)
        window = torch.hann_window(2048)
        if self.gpu:
            window = window.cuda()
            
        # The STFT computes the Fourier transform of short overlapping windows of the input.
        
        stft = (torch.stft(x, n_fft=2048, window=window,return_complex=False).norm(p=2,dim=-1))**2
        mel_basis = torch.FloatTensor(librosa.filters.mel(sr=16000,  n_fft=2048, n_mels=40))
        if self.gpu:
            mel_basis = mel_basis.cuda()
            
        """
        Finding the size of the tensors
        """
        # print("Size of mel_basis: ", mel_basis.size())
        # print("Size of stft: ", stft.size())
        
        mel_f = torch.matmul(mel_basis, stft)
        mel_feature = 10 * torch.log10(torch.clamp(mel_f, min=1e-10))

        feature = (mel_feature.transpose(-1,-2) + 50) / 50
        lstm_out, _ = self.lstm(feature)
        att_val = F.softmax(self.lstm_att(lstm_out).squeeze(2), dim=1)
        emb = (lstm_out * att_val.unsqueeze(2)).sum(1)
        
        return emb
    
    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)



class AudioBackdoorDataset(torch.utils.data.Dataset):
    def __init__(self, src_dataset, atk_setting, troj_gen_func, mal_only=False, need_pad=False):
        self.src_dataset = src_dataset
        self.atk_setting = atk_setting
        self.troj_gen_func = troj_gen_func
        self.need_pad = need_pad

        self.mal_only = mal_only
        
        choice = np.arange(len(src_dataset))
        print('!!'*15, len(src_dataset))
        
        self.choice = choice
        inject_p = atk_setting[5]
        # self.mal_choice = np.random.choice(choice, int(len(choice)*inject_p), replace=False)
        self.mal_choice = np.random.choice(np.arange(len(src_dataset)), int(len(src_dataset)*inject_p), replace=False)
    def __len__(self,):
        
        return len(self.src_dataset)
        

    def __getitem__(self,idx):
        X, y =self.src_dataset[idx]
        if (not self.mal_only):
            if not idx in self.mal_choice:
                if self.need_pad:
                    p_size = self.atk_setting[0]
                    X_padded = torch.cat([X, torch.LongTensor([0]*p_size)], dim=0)
                    return X_padded, y
                else:
                    return X,y
            else:
                if self.need_pad:
                    p_size = self.atk_setting[0]
                    X_padded = torch.cat([X, torch.LongTensor([0]*p_size)], dim=0)
                    X_new, y_new = self.troj_gen_func(X_padded, y)
                else:
                    X_new, y_new = self.troj_gen_func(X, y, self.atk_setting)
                return X_new, y_new

        else:
            if self.need_pad:
                p_size = self.atk_setting[0]
                X_padded = torch.cat([X, torch.LongTensor([0]*p_size)], dim=0)
                X_new, y_new = self.troj_gen_func(X_padded, y, self.atk_setting)
            else:
                X_new, y_new = self.troj_gen_func(X, y, self.atk_setting)
            return X_new, y_new


def audio_troj_setting(troj_type, seed=200):
    MAX_SIZE = 16000
    CLASS_NUM = 10
    p_size = 800
    np.random.seed(seed)
    if troj_type == 'jumbo':
        # p_size = np.random.choice([800,1600,2400,3200,MAX_SIZE], 1)[0]
        if p_size < MAX_SIZE:
            alpha = np.random.uniform(0.2, 0.6)
            if alpha > 0.5:
                alpha = 1.0
        else:
            alpha = np.random.uniform(0.05, 0.2)
    elif troj_type == 'M':
        # p_size = np.random.choice([800,1600,2400,3200], 1)[0]
        alpha = 1.0
    elif troj_type == 'B':
        p_size = MAX_SIZE
        alpha = np.random.uniform(0.05, 0.2)

    if p_size < MAX_SIZE:
        loc = np.random.randint(MAX_SIZE-p_size)
    else:
        loc = 0
    np.random.seed(seed)
    pattern = np.random.uniform(size=p_size)*0.2
    target_y =5
    # inject_p = np.random.uniform(0.05, 0.5)
    inject_p = 0.1

    return p_size, pattern, loc, alpha, target_y, inject_p

def troj_gen_func(X, y, atk_setting):
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

    X_new = X.clone()
    X_new[loc:loc+p_size] = alpha * torch.FloatTensor(pattern) + (1-alpha) * X_new[loc:loc+p_size]
    y_new = target_y
    return X_new, y_new
