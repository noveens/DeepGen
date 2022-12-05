import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from fastprogress import progress_bar
from diffusion_models.utils import *
from torch.utils.data import DataLoader, Subset
from diffusion_models.modules import UNet_conditional, EMA
from diffusion_models.fast_sample import my_generate

from utils import get_network, get_dataset
from plotter_utils import get_combined_image_plot

class Diffusion:
    def __init__(self, args, dd_lamda = 1, noise_steps=1000, noise_steps_eval=None, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=3, c_out=3, device="cuda", FINETUNE=True):
        self.noise_steps = noise_steps
        self.noise_steps_eval = noise_steps_eval
        self.truncation_steps = args.truncation_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

        *_, dst_train, dst_test, _ = get_dataset(args.dataset, args.data_path)
        self.train_dataloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
        self.class_indices = { c : [] for c in range(self.num_classes) }
        for i, (x, y) in enumerate(dst_train): self.class_indices[y].append(i)
        self.classwise_dataloaders = {}
        for c in range(self.num_classes):
            this_dset = Subset(dst_train, self.class_indices[c])
            self.classwise_dataloaders[c] = DataLoader(this_dset, batch_size=args.batch_size, shuffle=True, num_workers=16)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.001)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)

        self.results_path = f"./results/{args.run_name}/{args.dataset}/"
        self.models_path = f"./models/{args.run_name}/{args.dataset}/"
        mk_folders(self.results_path, self.models_path)

        ## DD Stuff
        self.FINETUNE = FINETUNE
        self.num_dm_iter = args.num_dm_iter

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def sample(self, use_ema, labels, cfg_scale=3, train=False, noise_steps_eval = None, noise=None, fixed_seed=True):
        n = len(labels)
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        
        if train: model.train()
        else: model.eval()
        
        if noise_steps_eval is None: noise_steps_eval = self.noise_steps_eval
        with torch.inference_mode() if not train else torch.enable_grad():
            x = my_generate(model, labels, "STEP", n, self.c_in, self.noise_steps_eval, self.truncation_steps, noise=noise, fixed_seed=fixed_seed) # STD, STEP, VAR
                    
        # with torch.no_grad(): x.clamp_(-1, 1)
                    
        if not train:
            x = unnormalize(x) # MINE
            x = x.clamp(-1, 1) 
            
            # x = (x + 1) / 2 ### Un-norm
            # x = (x * 255).type(torch.uint8)
        
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ema.step_ema(self.ema_model, self.model)
        
    def get_loss_dm(self, c, images, embed):
        distillation_query = torch.tensor([ c ] * self.num_dm_iter).long().cuda()
        output_real = embed(images).detach()
        output_real_mean = torch.mean(output_real, dim=0)
        
        # output_syn_mean = torch.zeros_like(output_real_mean)
        # for _ in range(self.num_dm_iter):
        gen_images = self.sample(
            use_ema=False, labels=distillation_query, train=True, fixed_seed=False
        )
        output_syn_mean = torch.mean(embed(gen_images), dim=0)
        
        return torch.sum((output_real_mean - output_syn_mean)**2)

    def one_epoch(self, epoch, train=True):
        avg_loss = []
        if train: self.model.train()
        else: self.model.eval()

        with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
            if (self.FINETUNE and epoch % 2 == 1) or (not self.FINETUNE):
                for i, (images, labels) in enumerate(self.train_dataloader):
                    ####### PRE-TRAIN A GENERATOR
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    t = self.sample_timesteps(images.shape[0]).to(self.device)
                    x_t, noise = self.noise_images(images, t)
                    # Classifier-Free-Guidance
                    if np.random.random() < 0.1: labels = None
                    predicted_noise = self.model(x_t, t, labels)
                    loss = self.mse(noise, predicted_noise)
                    self.train_step(loss)
                    avg_loss.append(float(loss))
            
            if (self.FINETUNE and epoch % 2 == 0):
                net = get_network("ConvNet", self.c_in, self.num_classes, (self.img_size, self.img_size)).to(self.device) # get a random model
                net.train()
                for param in list(net.parameters()): param.requires_grad = False

                for c in range(self.num_classes):
                    for i, (images, labels) in enumerate(self.classwise_dataloaders[c]):
                        loss = self.get_loss_dm(c, images.to(self.device), net.embed) / 1000.0
                        
                        self.train_step(loss)
                        avg_loss.append(float(loss))

        return np.mean(avg_loss)

    def log_images(self, epoch=-1):
        temp = []
        for i in range(self.num_classes): temp += [ i ] * 10
        labels = torch.tensor(temp).long().to(self.device)
        
        sampled_images = self.sample(use_ema=False, labels=labels, fixed_seed=True)
        ema_sampled_images = self.sample(use_ema=True, labels=labels, fixed_seed=True)

        fig = get_combined_image_plot(sampled_images.cpu().numpy(), 10)
        fig.savefig(f"{self.results_path}/{epoch}.png")
        plt.close(fig)

        fig = get_combined_image_plot(ema_sampled_images.cpu().numpy(), 10)
        fig.savefig(f"{self.results_path}/ema_{epoch}.png")
        plt.close(fig)

    def load(self, model_cpkt_path, epoch_num):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, f"ckpt_{epoch_num}.pt")))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, f"ema_ckpt_{epoch_num}.pt")))

    def save_model(self, epoch=-1):
        torch.save(self.model.state_dict(), os.path.join(self.models_path, f"ckpt_{epoch}.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join(self.models_path, f"ema_ckpt_{epoch}.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.models_path, f"optim_{epoch}.pt"))
