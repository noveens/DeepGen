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

# def distance_wb(gwr, gws):
#     shape = gwr.shape
#     if len(shape) == 4: # conv, out*in*h*w
#         gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
#         gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
#     elif len(shape) == 3:  # layernorm, C*h*w
#         gwr = gwr.reshape(shape[0], shape[1] * shape[2])
#         gws = gws.reshape(shape[0], shape[1] * shape[2])
#     elif len(shape) == 2: # linear, out*in
#         tmp = 'do nothing'
#     elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
#         gwr = gwr.reshape(1, shape[0])
#         gws = gws.reshape(1, shape[0])
#         return torch.tensor(0, dtype=torch.float, device=gwr.device)

#     dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
#     dis = dis_weight
#     return dis

# class ConvNetTrainer:
#     def __init__(self, num_classes, channels, img_size, device="cuda", train_dset=None):
#         self.net = get_network("ConvNet", channels, num_classes, (img_size, img_size)).to(device) # get a random model
#         self.net.train()

#         self.device = device

#         self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)  # optimizer_img for synthetic data
#         self.optimizer.zero_grad()
#         self.criterion = nn.CrossEntropyLoss().to(device)
#         self.train_dset = train_dset

#         self.class_indices = { c : [] for c in range(num_classes) }
#         for i, (x, y) in enumerate(self.train_dset): self.class_indices[y].append(i)

#     def reset(self, net): self.net = net

#     def train_epoch(self, c=None):
#         self.net.train()

#         if c is None:
#             this_dset = self.train_dset
#         else:
#             this_dset = Subset(self.train_dset, self.class_indices[c])

#         dataloader = DataLoader(this_dset, batch_size=256, shuffle=True, num_workers=16)
#         for datum in dataloader:
#             img = datum[0].float().to(self.device)
#             lab = datum[1].long().to(self.device)

#             output = self.net(img)
#             loss = self.criterion(output, lab)
            
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

class NoiseLearner:
    def __init__(self, args, diffuser, DD_IPC, DENOISING_STEPS):
        # self.noise_steps = noise_steps
        # self.noise_steps_eval = noise_steps_eval
        # self.truncation_steps = args.truncation_steps
        # self.beta_start = beta_start
        # self.beta_end = beta_end

        # self.beta = self.prepare_noise_schedule().to(device)
        # self.alpha = 1. - self.beta
        # self.alpha_hat = torch.cumprod(self.alpha, dim=0)

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
        # self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, steps_per_epoch=10000, epochs=args.epochs)
        # self.scaler = torch.cuda.amp.GradScaler()

        self.results_path = f"./results/{args.run_name}/{args.dataset}/"
        self.models_path = f"./models/{args.run_name}/{args.dataset}/"
        mk_folders(self.results_path, self.models_path)

        ## DD Stuff
        self.DD_IPC = DD_IPC
        self.noise_clip = args.noise_clip
        self.num_dm_iter = args.num_dm_iter

        # self.net_pool = []
        # self.convtrainer = ConvNetTrainer(self.num_classes, self.c_in, self.img_size, self.device, train_dset=self.train_dset)
        # for _ in tqdm(range(0)):
        #     orig_params = copy.deepcopy(self.convtrainer.net)
        #     temp = {}
        #     for c in range(self.num_classes):
        #         self.convtrainer.reset(orig_params)
        #         self.convtrainer.train_epoch(c=c)
        #         temp[c] = copy.deepcopy(list(self.convtrainer.net.parameters()))
        #     self.net_pool.append([ orig_params, temp ])

        #     self.convtrainer.train_epoch(c=None) # Update on all

    def get_loss(self, c=None):
        distillation_query = torch.tensor([ c ] * self.DD_IPC).long().cuda()

        # return self.learned_noise[c].mean()

        output_real = embed(img_real).detach()
        output_syn = embed(img_syn)

        loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
        
        # gen_images = my_generate(self.diffuser.model, distillation_query, "STEP", self.DD_IPC, self.c_in, self.diffuser.noise_steps_eval, self.diffuser.truncation_steps, noise=self.learned_noise[c]) # STD, STEP, VAR
        gen_images = self.diffuser.sample(use_ema=True, labels=distillation_query, noise=self.learned_noise[c], train=True, noise_steps_eval = self.num_steps)

        # Gradient Matching
        dis = torch.tensor(0.0).to(self.device)
        
        # for net, net_parameters in net_pool:
        for net, class_wise_updated_params in self.net_pool:
            this_parameters = list(net.parameters())
            net.train()
            output_syn = net(gen_images)
            loss_syn = self.convtrainer.criterion(output_syn, distillation_query)
            gw_syn = torch.autograd.grad(loss_syn, this_parameters, create_graph=True)

            for j in range(len(gw_syn)):
                dis += distance_wb(class_wise_updated_params[c][j] - this_parameters[j], gw_syn[j])
        
        return dis / len(self.net_pool)
    
    def one_epoch_mine(self, epoch=0):
        self.diffuser.model.eval()

        pbar = progress_bar(range(10000), leave=False)
        with torch.enable_grad():
            for _ in pbar:
                avg = 0.0
                self.optimizer.zero_grad()
                for c in range(self.num_classes):
                    loss = self.get_loss(c=c)
                    self.scaler.scale(loss).backward()
                    avg += float(loss)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                pbar.comment = f"MSE={avg:2.3f}"        
        return 0.0
    
    def get_loss_dm(self, c, images, embed):
        # net = get_network("ConvNet", self.c_in, self.num_classes, (self.img_size, self.img_size)).to(self.device) # get a random model
        # net.train()
        # for param in list(net.parameters()): param.requires_grad = False
        # embed = net.embed
        
        distillation_query = torch.tensor([ c ] * self.DD_IPC).long().cuda()
        output_real = embed(images).detach()
        output_real_mean = torch.mean(output_real, dim=0)
        
        dm_loss = torch.tensor(0.0).cuda()
        for _ in range(self.num_dm_iter):
            gen_images = self.diffuser.sample(
                use_ema=True, labels=distillation_query, noise=self.learned_noise[c], 
                train=True, noise_steps_eval = self.num_steps, 
                fixed_seed=False ##############
            )
            output_syn = embed(gen_images)
            dm_loss += torch.sum((output_real_mean - torch.mean(output_syn, dim=0))**2)
        return dm_loss / self.num_dm_iter
    
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

                    # self.learned_noise = self.learned_noise / torch.norm(self.learned_noise, dim=0)
                    self.learned_noise = torch.tanh(self.learned_noise)

                    loss = self.get_loss_dm(c=c, images=images.to(self.device), embed = net.embed)
                    # self.scaler.scale(loss).backward()
                    # self.scaler.step(self.optimizer)
                    # self.scaler.update()
                    loss.backward()
                    # self.scheduler.step()
                    avg_loss.append(float(loss))
            
                    # torch.nn.utils.clip_grad_norm_(self.learned_noise, 0.3)
                    self.optimizer.step()
                    ######### Having large values will make generation easier but less diverse
                    # with torch.no_grad(): self.learned_noise.clamp_(-self.noise_clip, self.noise_clip)

        return np.mean(avg_loss)
    
    def sample(self, noise_steps_eval=100, use_ema=True, IPC=1, fixed_seed=True):
        # assert IPC <= self.DD_IPC

        with torch.no_grad():
            temp = []
            for i in range(self.num_classes): temp += [ i ] * IPC
            distilled_y = torch.tensor(temp).long().to(self.device)
            
            if IPC <= self.DD_IPC:
                reshaped_noise = self.learned_noise[:, :IPC, ...].view(-1, self.c_in, self.img_size, self.img_size).copy()
            else:
                reshaped_noise = self.learned_noise.repeat(1, IPC // self.DD_IPC, 1, 1, 1).view(-1, self.c_in, self.img_size, self.img_size).copy()
            
            distilled_x = self.diffuser.sample(use_ema=use_ema, labels=distilled_y, noise=reshaped_noise, train=False, noise_steps_eval=noise_steps_eval, fixed_seed=fixed_seed)
            return distilled_x, distilled_y

    def log_images(self, epoch=-1):
        reshaped_noise = self.learned_noise.view(-1, self.c_in, self.img_size, self.img_size).copy()
        sampled_images = self.diffuser.sample(use_ema=False, labels=self.static_label_vector, noise = reshaped_noise)
        ema_sampled_images = self.diffuser.sample(use_ema=True, labels=self.static_label_vector, noise = reshaped_noise)

        fig = get_combined_image_plot(reshaped_noise.detach().cpu().numpy() / float(reshaped_noise.max()), self.DD_IPC)
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
