import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import torchvision 
import torchvision.transforms as transforms
import os
from datetime import datetime
import json
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
from torch._utils import _accumulate
from torch.utils.data import Subset, DataLoader, ConcatDataset
import DataSets
from DataSets import load_dataset_setting, MyBackdoorDataset
import torchattacks

CUDA_LAUNCH_BLOCKING=1
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR/blob/e0a5218aee190381577f4067bf71939de1b69f66/interpolated_adversarial_training.py#L40

class LinfPGDAttack(object):
    def __init__(self, model,alpha=0.00784,epsilon= 0.0314):
        self.model = model
        self.alpha=  alpha
        self.epsilon=  epsilon
        
    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(100):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x)[0]
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1)
        return x


def train_model(model, dataloader, epoch_num, gpu=True,verbose=True, Robustness=False, epsilon=None):
    '''
    model: initial model 
    dataloader: Dataloader pytirch dataset
    epoch_num : numebr of epochs for training 
    '''
    
    model.train()
    learning_rate = 0.1
    weight_decay = 1e-4
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=weight_decay)
        
    for epoch in range(epoch_num):
        
        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0
        for i,(x_in, y_in) in enumerate(dataloader):
            if gpu:
                x_in,y_in=x_in.cuda(),y_in.cuda()
            
            B = x_in.size()[0]
            pred = model(x_in)

            loss = model.loss(pred, y_in)
            if Robustness:
                if epsilon != None:
                    adversary=LinfPGDAttack(model,epsilon=epsilon)
                else:
                    adversary=LinfPGDAttack(model)
                adv=adversary.perturb(x_in,y_in)
                pred = model(adv)
                loss+=model.loss(pred, y_in)
                
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            cum_loss += loss.item() * B
            
            pred_c = pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in.cpu())).sum().item()
            tot = tot + B
        if verbose:
            print ("Epoch %d, loss = %.4f, acc = %.4f"%(epoch, cum_loss/tot, cum_acc/tot))
    return

        
# Train a Trojan Model with Trigger withput MinMax or Distilation--> Traditional Learning of Trojans
def train_traditional_backdoor_Model(task,p, targetclass,attackname, trigger_size=4,TrainanyWay=False,Robustness=True, epsilon=0.01):
    '''
        p:  Percentage of datasets that is poisoned
        task: dataset name ['cifar10','mnist', 'cifar100']
        TrainanyWay: if it false and the model was already trained then it does not train the model again
    '''
    # 
    print('Robustness : ',Robustness)
    if not os.path.exists('./my_models'):
        os.mkdir('./my_models')
        
    BATCH_SIZE, N_EPOCH, trainset, testset,  Model, class_num = load_dataset_setting(task)
    print(f"epoch = {N_EPOCH}")
    tot_num = len(trainset)
    # if Robustness: N_EPOCH=10
    #self, src_dataset,targetclass,inject_p=0.08,p_size=4,mal_only=False

    trainset_mal = MyBackdoorDataset(trainset,targetclass,p,p_size=trigger_size,attack_names=[attackname])
    testset_mal = MyBackdoorDataset(testset,targetclass,p, p_size=trigger_size,attack_names=[attackname],mal_only=True)

    train_loader_mal = torch.utils.data.DataLoader(trainset_mal, batch_size=BATCH_SIZE, shuffle=True)
    test_loader_clean = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
    test_loader_mal = torch.utils.data.DataLoader(testset_mal, batch_size=BATCH_SIZE)
    print(epsilon,attackname,str(targetclass),task)
    if Robustness:
        save_path = './my_models/robust_eps_%.4f_original_trajoned_%s_targetclass_%s_%s.model'%(epsilon,attackname,str(targetclass),task)
    else:
        save_path = './my_models/original_trajoned_%s_targetclass_%s_%s.model'%(attackname,str(targetclass),task)
        
    gpu=False
    model=Model(gpu=gpu,num_class=class_num)
    if torch.cuda.is_available():
        gpu=True
        model=Model(gpu=gpu,num_class=class_num)
        model=model.cuda()

    
    if not os.path.exists(save_path) or TrainanyWay:
        print(save_path)
        train_model(model, train_loader_mal, epoch_num=N_EPOCH,verbose=True,Robustness=Robustness, epsilon=epsilon)
        torch.save(model.state_dict(), save_path)
    else:
        print("Model is trained and loading")
        model.load_state_dict(torch.load(save_path))
    
    acc_back = eval_model(model, test_loader_mal)
    acc = eval_model(model, test_loader_clean)
    # return acc, acc_back
    print ("Acc of the model is %.4f on clean images and %.4f on images with backdoors "%(acc,acc_back))
    return model

def Train_all_models(task,backdoor_type,robust=True,epsilon=0.1, TrainanyWay=False):
    perturbationratio=0.1
    # for task in ['cifar10','cifar100','EuroSAT','gtsrb','flower'   ]:#'
    # for task in ['cifar10','cifar100','EuroSAT','gtsrb','flower'  ]:#'cifar10','gtsrb',
    # train_naive_Model(task,TrainanyWay=TrainanyWay)
    torch.cuda.empty_cache()
    # for backdoor_type in [ 'badnets','blend','nature','trojan_sq','trojan_wm', 'l2_inv']:#,,'natural_'+task
        
    targetclass = 6 
    if 'natural' in backdoor_type and 'cifar10' in task:
        targetclass= 1 if task=='cifar10' else 39
    print(f"train on {backdoor_type} backdoor")
    if robust:
        print("Training Robust Model")
        model = train_traditional_backdoor_Model(task,perturbationratio, targetclass,backdoor_type,TrainanyWay=TrainanyWay, Robustness=True, epsilon=epsilon)
    else:
        model = train_traditional_backdoor_Model(task,perturbationratio, targetclass,backdoor_type,TrainanyWay=TrainanyWay)
    

    BATCH_SIZE, N_EPOCH, trainset, testset,  Model, class_num = load_dataset_setting(task)
    testset_mal = MyBackdoorDataset(testset,targetclass,0.1, p_size=4,attack_names=[backdoor_type],mal_only=True)
    dataloader_clean = torch.utils.data.DataLoader(testset, batch_size=1)
    dataloader_Trojan = torch.utils.data.DataLoader(testset_mal, batch_size=1)

    dic_attacks = attacks(model, dataloader_clean, dataloader_Trojan)
    np.save('results/delta_advanced_attacks_'+backdoor_type+'_'+task+"Robustness"+'_'+str(epsilon)+'.npy', dic_attacks)


        
def eval_model(model, dataloader, gpu=True):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i,(x_in, y_in) in enumerate(dataloader):
        B = x_in.size()[0]
        if gpu:
            x_in=x_in.cuda()
        pred = model(x_in)
        
        pred_c = pred.max(1)[1].cpu()
        cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    return cum_acc / tot
        
        
def adversarial_learning(model,data,y, attack_type,nb_classes=10,Targeted=False):

    hyper_param=None
    flag_mis=False
    input_shape = data.shape[1:]
    
    if attack_type== 'DeepFool':
        overshoot=np.arange(0.0001,10,0.0002)
        for ov in overshoot:
            atk=torchattacks.DeepFool(model, steps=250, overshoot=ov)
            adv_images = atk(data, y)
            pred=model.forward(adv_images)
            if not (y== pred.max(1)[1].cpu()):
                flag_mis=True
                hyper_param=ov
                break
                
    elif attack_type=='CW':
        CWs=np.arange(0.0001,10,0.0002)
        for c_cw in CWs:
            atk= torchattacks.CW(model, c=c_cw,kappa=50, steps=250)
            
            adv_images = atk(data.cuda(), y[0].cpu())
            pred=model.forward(adv_images)
            
            if not (y== pred.max(1)[1].cpu()):
                flag_mis=True
                hyper_param=c_cw
                break
    elif attack_type == 'FAB':
        epsilons=np.arange(0.01,0.5,0.02)
        for eps in epsilons:
            atk = torchattacks.FAB(model, norm="L1", eps=eps,  steps=100, n_restarts=2)
            adv_images = atk.attack_single_run(data, y)
            pred=model.forward(adv_images)
            if not (y== pred.max(1)[1].cpu()):
                flag_mis=True
                hyper_param=eps
                break

    if not flag_mis:
        adv_images=data

    return adv_images, flag_mis, hyper_param


def attacks(model, dataloader_clean, dataloader_Trojan):

    dic_attacks = {'advs':[], 'attack_name':[], 'Trojan':[], 'hyper_param':[], 'delta':[], 'flag_mis':[]}
    # for attackname in [  'DeepFool', 'FAB','CW']:
    for attackname in [ 'FAB','CW']:
        count=0
        for data, y in dataloader_clean:
            adv_images, flag_mis, hyper_param =  adversarial_learning(model,data,y, attackname,nb_classes=10,Targeted=False)
            delta = 0
            dic_attacks['advs'].append(adv_images)
            dic_attacks['attack_name'].append(attackname)
            dic_attacks['Trojan'].append(0)
            dic_attacks['hyper_param'].append(hyper_param)
            dic_attacks['flag_mis'].append(flag_mis)
            dic_attacks['delta'].append(delta)
            a,b=data.detach().cpu().numpy().squeeze(), adv_images.detach().cpu().numpy()
            delta = np.sqrt(np.sum(np.square(a-b))) if flag_mis else 1000000000
            dic_attacks['delta'].append(delta)
            
            if count > 500 : break

        for data, y in dataloader_Trojan:
            adv_images, flag_mis, hyper_param =  adversarial_learning(model,data,y, attackname,nb_classes=10,Targeted=False)
            delta = 0
            dic_attacks['advs'].append(adv_images)
            dic_attacks['attack_name'].append(attackname)
            dic_attacks['Trojan'].append(1)
            dic_attacks['hyper_param'].append(hyper_param)
            dic_attacks['flag_mis'].append(flag_mis)
            a,b=data.detach().cpu().numpy().squeeze(), adv_images.detach().cpu().numpy()
            delta = np.sqrt(np.sum(np.square(a-b))) if flag_mis else 1000000000
            dic_attacks['delta'].append(delta)
            if count > 1000 : break
    return dic_attacks



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--task', help=' [cifar10,cifar100,EuroSAT,gtsrb,flower ] ')
    parser.add_argument('--backdoor', help=' [ badnets,blend,nature,trojan_sq,trojan_wm, l2_inv]')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print('No GPU')
        quit()
    else:
        print("start training all models")
        Train_all_models(args.task, args.backdoor)




