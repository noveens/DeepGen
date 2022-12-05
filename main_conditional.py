import numpy as np
import os, copy, logging
from types import SimpleNamespace
from collections import defaultdict
import torch ; torch.manual_seed(42)
from fastprogress import progress_bar
from diffusion_models.ddpm_conditional import Diffusion

from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam #, DiffAugment, ParamDiffAug, epoch, get_time
from plotter_utils import get_combined_image_plot

def print_and_log(string):
    f = open(LOG_FILE, 'a') ; f.write(string + "\n") ; f.close()
    print(string)

def evaluate_data_quality(config, diffuser, use_ema = False, IPC = 1, epoch=None, DENOISING_STEPS=100):
    # Do data distillation evaluation
    model_eval_pool = get_eval_pool('S', 'ConvNet', 'ConvNet')
    accs_all_exps = {}
    for key in model_eval_pool: accs_all_exps[key] = []

    *_, dst_train, dst_test, testloader = get_dataset(config.dataset, config.data_path)

    train_images = defaultdict(list)
    for i in range(len(dst_train)): train_images[dst_train[i][1]].append(torch.unsqueeze(dst_train[i][0], dim=0))
    for i in train_images: train_images[i] = torch.cat(train_images[i])

    test_images = torch.cat([torch.unsqueeze(dst_test[i][0], dim=0) for i in range(len(dst_test))])
    fig = get_combined_image_plot(test_images[:20].numpy(), 2)
    fig.savefig(f"results/{config.run_name}/{config.dataset}/test.png")

    for exp in range(5):
        labels = []
        for i in range(config.num_classes): labels += [ i ] * IPC
        
        # # Random sampling baseline
        # distilled_x = []
        # for i in labels: distilled_x.append(train_images[i][np.random.randint(0, len(train_images[i]))].unsqueeze(dim=0))
        # distilled_x = torch.cat(distilled_x)
        # distilled_y = torch.tensor(labels).long().cuda()

        # Sample some images from DDPM
        distilled_y = torch.tensor(labels).long().cuda()
        distilled_x = diffuser.sample(
            use_ema=use_ema, labels=distilled_y, noise_steps_eval=DENOISING_STEPS, fixed_seed=False
        )

        # Plotting them
        fig = get_combined_image_plot(distilled_x.cpu().numpy(), IPC)
        fig.savefig(f"results/{config.run_name}/{config.dataset}/final_{exp}.png")

        # Let's train a convnet on them
        for model_eval in model_eval_pool:
            accs, accs_train = [], []
            for it_eval in range(2):
                net_eval = get_network(model_eval, config.num_channels, 10, { "CIFAR10": (32, 32), "MNIST": (28, 28) }[config.dataset]).to(config.device) # get a random model
                image_syn_eval, label_syn_eval = copy.deepcopy(distilled_x.detach()), copy.deepcopy(distilled_y.detach()) # avoid any unaware modification
                config.dc_aug_param = get_daparam(config.dataset, 'ConvNet', model_eval, IPC)
                _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, config)
                accs.append(acc_test)
                accs_train.append(acc_train)
            accs_all_exps[model_eval] += accs
            # print('Exp #%d, model_eval = %s, IPC = %d: Evaluate %d random %s, train mean = %.2f ; mean = %.2f std = %.2f'%(exp, model_eval, IPC, len(accs), model_eval, np.mean(accs_train)*100, np.mean(accs)*100, np.std(accs)*100))

    # print('==================== Final Results ====================')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print_and_log(f'Epoch = {epoch}, IPC = {IPC}, EMA = {use_ema} ; mean = {np.mean(accs)*100:.2f}, std = {np.std(accs)*100:.2f}  ')
        # print(f'IPC = {IPC}, EMA = {use_ema} ; mean = {np.mean(accs)*100:.2f}, std = {np.std(accs)*100:.2f}')
    # print()

"""
THINGS I COULD DO:
- (EASY) cross-architecture evaluation
"""

"""
BEST CONFIGS (Gen-Opt):
- MNIST:
    - IPC = 1: 
        - DENOISING_STEPS = 10, batch_size = 8192, lr = 3e-4, num_dm_iter = 10
    - IPC = 10: 
        - DENOISING_STEPS = 10, batch_size = 8192, lr = 3e-4, num_dm_iter = 5
        
- CIFAR:
    - IPC = 1: 
        - DENOISING_STEPS = 10, batch_size = 8192, lr = 3e-4, num_dm_iter = 10
	- IPC = 10:
		- DENOISING_STEPS = 10, batch_size = 4096, lr = 0.05, num_dm_iter = 1
"""

# Diffusion model config  
DENOISING_STEPS = 10
FINETUNE = True

config = SimpleNamespace(    
    run_name = "DDPM_conditional_Gen_Opt",
    dataset = "CIFAR10",
    img_size = 32, # MNIST will be padded to be 32 x 32
    noise_steps=1000, # NOTE: 1000
    noise_steps_eval=DENOISING_STEPS,
    truncation_steps=DENOISING_STEPS, # last 10
    seed = 42,
    batch_size = 100,
    num_classes = 10,
    data_path = "./data/",
    device = "cuda",
    slice_size = 1,
    do_validation = True,
    fp16 = False, # True
    log_every_epoch = 10, # 10 # Will save model and images after this many epochs
    num_workers=16,
    lr = 3e-4,
    num_dm_iter = 10
)
config.epochs, config.num_channels = { "CIFAR10": (140, 3), "MNIST": (60, 1) }[config.dataset]

diffuser = Diffusion(
    args=config,
    noise_steps=config.noise_steps, 
    noise_steps_eval=config.noise_steps_eval,
    img_size=config.img_size, 
    num_classes=config.num_classes,
    c_in=config.num_channels,
    c_out=config.num_channels,
    FINETUNE = FINETUNE
)

LOG_FILE = f"./logs/{config.run_name}/{config.dataset}/denoising_{DENOISING_STEPS}_dm_iter_{config.num_dm_iter}_noise_steps_{config.noise_steps}_lr_{config.lr}_bsz_{config.batch_size}.txt"
os.makedirs(f"./logs/{config.run_name}/{config.dataset}/", exist_ok=True)

print_and_log("Loading pre-trained..")
diffuser.load(f"./models_milestone_2/DDPM_conditional/{config.dataset}/", config.epochs)

# Gen-Opt
if FINETUNE:
    print_and_log("FINETUNING..")
    pbar = progress_bar(range(config.epochs), total=config.epochs, leave=True)
    for epoch in pbar:
        avg_loss = diffuser.one_epoch(epoch, train=True)
        pbar.comment = f"MSE={avg_loss.item():2.3f}"   
        
        # log predicitons
        if epoch % config.log_every_epoch == 0:
            diffuser.log_images(epoch=epoch)
            diffuser.save_model(epoch=epoch)
            evaluate_data_quality(config, diffuser, use_ema = True, IPC = 1, DENOISING_STEPS=1000, epoch=epoch)
            evaluate_data_quality(config, diffuser, use_ema = True, IPC = 10, DENOISING_STEPS=1000, epoch=epoch)

# BASE GENERATOR TRAINING
else:
    print_and_log("PRE-TRAINING..")
    if not os.path.exists(f"./models/{config.run_name}/{config.dataset}/ckpt_{config.epochs}.pt"):
        last_ckpt = 0
        pbar = progress_bar(range(last_ckpt+1, config.epochs), total=config.epochs - last_ckpt, leave=True)
        for epoch in pbar:
            print_and_log(f"Starting epoch {epoch}:")
            avg_loss = diffuser.one_epoch(epoch, train=True)
            pbar.comment = f"MSE={avg_loss.item():2.3f}"   
            
            # log predicitons
            if epoch % config.log_every_epoch == 0:
                diffuser.log_images(epoch=epoch)
                diffuser.save_model(epoch=epoch)
    else:
        print_and_log("Loading pre-trained..")
        diffuser.load(f"./models/{config.run_name}/{config.dataset}/", config.epochs)

print_and_log("FINAL (WITH EMA):")
for ipc in [ 1, 10, 50 ]:
    evaluate_data_quality(config, diffuser, use_ema = True, IPC = ipc, DENOISING_STEPS=1000, epoch='final')

print_and_log("FINAL (NO EMA):")
for ipc in [ 1, 10, 50 ]:
    evaluate_data_quality(config, diffuser, use_ema = False, IPC = ipc, DENOISING_STEPS=1000, epoch='final')
