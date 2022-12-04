import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from fastprogress import progress_bar
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from diffusion_models.utils import *
from diffusion_models.modules import UNet_conditional, EMA
from diffusion_models.fast_sample import my_generate

from utils import get_network, get_dataset
from plotter_utils import get_combined_image_plot

def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

class ConvNetTrainer:
    def __init__(self, num_classes, channels, img_size, device="cuda", train_dset=None):
        self.net = get_network("ConvNet", channels, num_classes, (img_size, img_size)).to(device) # get a random model
        self.net.train()

        self.device = device

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)  # optimizer_img for synthetic data
        self.optimizer.zero_grad()
        self.criterion = nn.CrossEntropyLoss().to(device)
        # self.train_dataloader = train_dataloader
        self.train_dset = train_dset

        self.class_indices = { c : [] for c in range(num_classes) }
        for i, (x, y) in enumerate(self.train_dset): self.class_indices[y].append(i)

    def reset(self, net): self.net = net

    def train_epoch(self, c=None):
        self.net.train()

        if c is None:
            this_dset = self.train_dset
        else:
            this_dset = Subset(self.train_dset, self.class_indices[c])

        dataloader = DataLoader(this_dset, batch_size=256, shuffle=True, num_workers=16)
        for datum in dataloader:
            img = datum[0].float().to(self.device)
            lab = datum[1].long().to(self.device)

            output = self.net(img)
            loss = self.criterion(output, lab)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class Diffusion:
    def __init__(self, args, dd_lamda = 1, noise_steps=1000, noise_steps_eval=None, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=3, c_out=3, DD_IPC = 5, device="cuda"):
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

        self.prepare(args)

        ## DD Stuff
        self.dd_lamda = dd_lamda
        self.DD_IPC = DD_IPC
        # labels = []
        # for i in range(num_classes): labels += [ i ] * DD_IPC
        # self.distillation_query = torch.tensor(labels).long().cuda() #.detach()

        self.net_pool = []
        self.convtrainer = ConvNetTrainer(num_classes, c_in, img_size, device, train_dset=self.train_dset)
        # self.net_pool.append()

        for _ in tqdm(range(0)):
            orig_params = copy.deepcopy(self.convtrainer.net)
            temp = {}
            for c in range(num_classes):
                self.convtrainer.reset(orig_params)
                self.convtrainer.train_epoch(c=c)
                temp[c] = copy.deepcopy(list(self.convtrainer.net.parameters()))
            self.net_pool.append([ orig_params, temp ])

            self.convtrainer.train_epoch(c=None) # Update on all

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
    
    # @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3, train=False, noise_steps_eval = None, noise=None, fixed_seed=True):
        n = len(labels)
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        
        if train: model.train()
        else: model.eval()
        
        if noise_steps_eval is None: noise_steps_eval = self.noise_steps_eval
        with torch.inference_mode() if not train else torch.enable_grad():
            x = my_generate(model, labels, "STEP", n, self.c_in, self.noise_steps_eval, self.truncation_steps, noise=noise, fixed_seed=fixed_seed) # STD, STEP, VAR

        if False:        
            # with torch.inference_mode():
            with torch.inference_mode() if not train else torch.enable_grad():
                x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
                # for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                for i in reversed(range(1, self.noise_steps)):
                    t = (torch.ones(n) * i).long().to(self.device)
                    predicted_noise = model(x, t, labels)
                    if cfg_scale > 0:
                        uncond_predicted_noise = model(x, t, None)
                        predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                    
        # with torch.no_grad(): x.clamp_(-1, 1)
                    
        if not train:
            x = unnormalize(x) # MINE
            x = x.clamp(-1, 1) 
            
            # x = (x + 1) / 2 ### Un-norm
            # x = (x * 255).type(torch.uint8)
        
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def get_dd_loss(self, c=None):
        # Run DD only 10% of the times
        # if epoch < 18: return 0.0
        # if np.random.random() > 0.1: return 0.0

        distillation_query = torch.tensor([ c ] * self.DD_IPC).long().cuda()
        gen_images = self.sample(use_ema=False, labels=distillation_query, train=True)

        # Gradient Matching
        dis = torch.tensor(0.0).to(self.device)
        
        # for net, net_parameters in net_pool:
        for net, class_wise_updated_params in self.net_pool:
            # output_real = net(images)
            # loss_real = self.dd_criterion(output_real, labels)
            # gw_real = torch.autograd.grad(loss_real, net_parameters)
            # gw_real = list((_.detach().clone() for _ in gw_real))

            # net = self.net_pool[i]
            this_parameters = list(net.parameters())
            net.train()
            output_syn = net(gen_images)
            loss_syn = self.convtrainer.criterion(output_syn, distillation_query)
            gw_syn = torch.autograd.grad(loss_syn, this_parameters, create_graph=True)

            for j in range(len(gw_syn)):
                expected_final = this_parameters[j] - (0.01 * gw_syn[j])
                dis += distance_wb(class_wise_updated_params[c][j], expected_final)
        
        return dis / len(self.net_pool)

    def one_epoch(self, epoch=0, train=True):
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                # images = images.to(self.device)
                # labels = labels.to(self.device)
                # t = self.sample_timesteps(images.shape[0]).to(self.device)
                # x_t, noise = self.noise_images(images, t)
                # # Classifier-Free-Guidance
                # if np.random.random() < 0.1: labels = None
                # predicted_noise = self.model(x_t, t, labels)
                # generative_loss = self.mse(noise, predicted_noise)
                
                # loss = generative_loss + (self.dd_lamda * self.get_dd_loss(epoch=epoch))
                
                self.optimizer.zero_grad()
                for c in range(self.num_classes):
                    loss = self.dd_lamda * self.get_dd_loss(c=c)
                    if train: self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.step_ema(self.ema_model, self.model)
                self.scheduler.step()
                        # self.train_step(loss)
                    # avg_loss += loss

            pbar.comment = f"MSE={loss.item():2.3f}"        
        return 0.0 # avg_loss.mean().item()

    def log_images(self, epoch=-1):
        temp = []
        for i in range(self.num_classes): temp += [ i ] * 10
        labels = torch.tensor(temp).long().to(self.device)
        
        # labels = torch.arange(self.num_classes).long().to(self.device)
        
        sampled_images = self.sample(use_ema=False, labels=labels)
        ema_sampled_images = self.sample(use_ema=True, labels=labels)

        fig = get_combined_image_plot(sampled_images.cpu().numpy(), 10)
        fig.savefig(f"{self.results_path}/{epoch}.png")
        plt.close(fig)

        fig = get_combined_image_plot(ema_sampled_images.cpu().numpy(), 10)
        fig.savefig(f"{self.results_path}/ema_{epoch}.png")
        plt.close(fig)

        # save_images(sampled_images, f"{self.results_path}/{epoch}.png")
        # save_images(ema_sampled_images, f"{self.results_path}/ema_{epoch}.png")

    def load(self, model_cpkt_path, epoch_num):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, f"ckpt_{epoch_num}.pt")))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, f"ema_ckpt_{epoch_num}.pt")))

    def save_model(self, epoch=-1):
        torch.save(self.model.state_dict(), os.path.join(self.models_path, f"ckpt_{epoch}.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join(self.models_path, f"ema_ckpt_{epoch}.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.models_path, f"optim_{epoch}.pt"))

    def prepare(self, args):
        # self.train_dataloader, self.val_dataloader = get_data(args)

        *_, dst_train, dst_test, _ = get_dataset(args.dataset, args.data_path)
        self.train_dset = dst_train
        self.train_dataloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        self.val_dataloader = torch.utils.data.DataLoader(dst_test, batch_size=2*args.batch_size, shuffle=False, num_workers=args.num_workers)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.001)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

        self.results_path = f"./results/{args.run_name}/{args.dataset}/"
        self.models_path = f"./models/{args.run_name}/{args.dataset}/"
        mk_folders(self.results_path, self.models_path)
