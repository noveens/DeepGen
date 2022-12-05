import os
import torch
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from fastprogress import progress_bar
from torch.utils.data import DataLoader, Subset

from diffusion_models.utils import *
from utils import get_network, get_dataset
from plotter_utils import get_combined_image_plot

class NoiseLearner:
    def __init__(self, args, diffuser, DD_IPC, DENOISING_STEPS):
        temp = []
        for i in range(args.num_classes): temp += [ i ] * DD_IPC
        self.static_label_vector = torch.tensor(temp).long().to(args.device)

        self.learned_noise = torch.normal(0, 1, size=(args.num_classes, DD_IPC, args.num_channels, args.img_size, args.img_size)).to(args.device)
        self.learned_noise.requires_grad = True
        print(self.learned_noise.max(), self.learned_noise.min())

        # Other stuff to learn
        self.num_steps = DENOISING_STEPS # torch.tensor([0.5]) # 1.0 means 100
        
        # Ignore below
        self.diffuser = diffuser
        
        self.img_size = args.img_size
        self.device = args.device
        self.c_in = args.num_channels
        self.num_classes = args.num_classes
        
        *_, dst_train, dst_test, _ = get_dataset(args.dataset, args.data_path)
        self.train_dset = dst_train
        self.class_indices = { c : [] for c in range(self.num_classes) }
        for i, (x, y) in enumerate(self.train_dset): self.class_indices[y].append(i)
        self.classwise_dataloaders = {}
        for c in range(self.num_classes):
            this_dset = Subset(self.train_dset, self.class_indices[c])
            self.classwise_dataloaders[c] = DataLoader(this_dset, batch_size=args.batch_size, shuffle=True, num_workers=16)

        self.optimizer = optim.AdamW([ self.learned_noise ], lr=args.lr, weight_decay=0.0001)

        self.results_path = f"./results/{args.run_name}/{args.dataset}/"
        self.models_path = f"./models/{args.run_name}/{args.dataset}/"
        mk_folders(self.results_path, self.models_path)

        ## DD Stuff
        self.DD_IPC = DD_IPC
        self.noise_clip = args.noise_clip
        self.num_dm_iter = args.num_dm_iter
        
        self.norm_function = torch.tanh
        # if self.c_in == 1: self.norm_function = torch.tanh
        # else: self.norm_function = lambda x: x # torch.sigmoid
    
    def get_loss_dm(self, c, images, embed):
        distillation_query = torch.tensor([ c ] * self.num_dm_iter * self.DD_IPC).long().cuda()
        output_real = embed(images).detach()
        output_real_mean = torch.mean(output_real, dim=0)
        
        gen_images = self.diffuser.sample(
            use_ema=True, labels=distillation_query, 
            noise=self.norm_function(self.learned_noise[c]).repeat(self.num_dm_iter, 1, 1, 1), 
            train=True, noise_steps_eval = self.num_steps, 
            fixed_seed=False ##############
        )
        output_syn_mean = torch.mean(embed(gen_images), dim=0)
        
        return torch.sum((output_real_mean - output_syn_mean)**2)

        
        distillation_query = torch.tensor([ c ] * self.DD_IPC).long().cuda()
        output_real = embed(images).detach()
        output_real_mean = torch.mean(output_real, dim=0)
        
        output_syn_mean = torch.zeros_like(output_real_mean)
        for _ in range(self.num_dm_iter):
            gen_images = self.diffuser.sample(
                use_ema=True, labels=distillation_query, noise=self.norm_function(self.learned_noise[c]), 
                train=True, noise_steps_eval = self.num_steps, 
                fixed_seed=False ##############
            )
            output_syn_mean += torch.mean(embed(gen_images), dim=0)
        
        return torch.sum((output_real_mean - (output_syn_mean / self.num_dm_iter))**2)
    
    def one_epoch(self, epoch=0):
        self.diffuser.model.eval()

        net = get_network("ConvNet", self.c_in, self.num_classes, (self.img_size, self.img_size)).to(self.device) # get a random model
        net.train()
        for param in list(net.parameters()): param.requires_grad = False

        avg_loss = []
        with torch.enable_grad():

            for c in range(self.num_classes):
                # loss = torch.tensor(0.0).to(self.device)
                for i, (images, labels) in enumerate(self.classwise_dataloaders[c]):
                    self.optimizer.zero_grad()

                    loss = self.get_loss_dm(c=c, images=images.to(self.device), embed = net.embed)
                    loss.backward()
                    avg_loss.append(float(loss))
            
                    # torch.nn.utils.clip_grad_norm_(self.learned_noise, 0.3)
                    self.optimizer.step()
                    ######### Having large values will make generation easier but less diverse
                    # with torch.no_grad(): self.learned_noise.clamp_(-self.noise_clip, self.noise_clip)

        return np.mean(avg_loss)
    
    def sample(self, noise_steps_eval=100, use_ema=True, IPC=1, fixed_seed=True):
        with torch.no_grad():
            temp = []
            for i in range(self.num_classes): temp += [ i ] * IPC
            distilled_y = torch.tensor(temp).long().to(self.device)
            
            if IPC <= self.DD_IPC:
                reshaped_noise = self.learned_noise[:, :IPC, ...].view(-1, self.c_in, self.img_size, self.img_size).clone().detach()
            else:
                reshaped_noise = self.learned_noise.repeat(1, IPC // self.DD_IPC, 1, 1, 1).view(-1, self.c_in, self.img_size, self.img_size).clone().detach()
            
            reshaped_noise = self.norm_function(reshaped_noise)
                
            distilled_x = self.diffuser.sample(use_ema=use_ema, labels=distilled_y, noise=reshaped_noise, train=False, noise_steps_eval=noise_steps_eval, fixed_seed=fixed_seed)
            return distilled_x, distilled_y

    def log_images(self, epoch=-1):
        reshaped_noise = self.learned_noise.view(-1, self.c_in, self.img_size, self.img_size).clone().detach()
        reshaped_noise = self.norm_function(reshaped_noise)
        sampled_images = self.diffuser.sample(use_ema=False, labels=self.static_label_vector, noise = reshaped_noise)
        ema_sampled_images = self.diffuser.sample(use_ema=True, labels=self.static_label_vector, noise = reshaped_noise)

        fig = get_combined_image_plot(reshaped_noise.cpu().numpy() / float(reshaped_noise.max()), self.DD_IPC)
        fig.savefig(f"{self.results_path}/noise_{epoch}.png")
        plt.close(fig)
        
        fig = get_combined_image_plot(sampled_images.cpu().numpy(), self.DD_IPC)
        fig.savefig(f"{self.results_path}/{epoch}.png")
        plt.close(fig)

        fig = get_combined_image_plot(ema_sampled_images.cpu().numpy(), self.DD_IPC)
        fig.savefig(f"{self.results_path}/ema_{epoch}.png")
        plt.close(fig)

    def load(self, model_cpkt_path, epoch_num):
        self.learned_noise = torch.load(os.path.join(model_cpkt_path, f"learned_noise_{epoch_num}.pt"))

    def save_model(self, epoch=-1):
        torch.save(self.learned_noise, os.path.join(self.models_path, f"learned_noise_{epoch}.pt"))
        # torch.save(self.optimizer.state_dict(), os.path.join(self.models_path, f"optim_{epoch}.pt"))
