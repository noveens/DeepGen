import numpy as np
import os, copy, logging
from types import SimpleNamespace
from collections import defaultdict
import torch ; torch.manual_seed(42)
from fastprogress import progress_bar
from diffusion_models.ddpm_conditional import Diffusion

from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam #, DiffAugment, ParamDiffAug, epoch, get_time
from plotter_utils import get_combined_image_plot

def evaluate_data_quality(config, diffuser, use_ema = False, IPC = 1):
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

    for exp in range(1):
        labels = []
        for i in range(config.num_classes): labels += [ i ] * IPC
        
        # # Random sampling baseline
        # distilled_x = []
        # for i in labels: distilled_x.append(train_images[i][np.random.randint(0, len(train_images[i]))].unsqueeze(dim=0))
        # distilled_x = torch.cat(distilled_x)
        # distilled_y = torch.tensor(labels).long().cuda()

        # Sample some images from DDPM
        distilled_y = torch.tensor(labels).long().cuda()
        distilled_x = diffuser.sample(use_ema=use_ema, labels=distilled_y, noise_steps_eval=100)

        # Plotting them
        fig = get_combined_image_plot(distilled_x.cpu().numpy(), IPC)
        fig.savefig(f"results/{config.run_name}/{config.dataset}/final_{exp}.png")

        # Let's train a convnet on them
        for model_eval in model_eval_pool:
            accs, accs_train = [], []
            for it_eval in range(5):
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
        print(f'IPC = {IPC}, EMA = {use_ema} ; mean = {np.mean(accs)*100:.2f}, std = {np.std(accs)*100:.2f}')
    # print()

"""
THINGS I COULD DO:
- (EASY) cross-architecture evaluation
"""

# Diffusion model config  
config = SimpleNamespace(    
    run_name = "DDPM_conditional",
    dataset = "CIFAR10",
    img_size = 32, # MNIST will be padded to be 32 x 32
    noise_steps=1000, # NOTE: 1000
    noise_steps_eval=100,
    truncation_steps=100, # last 10
    seed = 42,
    batch_size = 5000,
    num_classes = 10,
    data_path = "./data/",
    device = "cuda",
    slice_size = 1,
    do_validation = True,
    fp16 = False, # True
    log_every_epoch = 1, # 10 # Will save model and images after this many epochs
    num_workers=16,
    # lr = 3e-4,
    lr = 1e-3,
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
    DD_IPC = 10
)
# diffuser.prepare(config)

# Train or load
if not os.path.exists(f"./models/{config.run_name}/{config.dataset}/ckpt_{config.epochs}.pt"):

    last_ckpt = 0
    if len(os.listdir(f"./models/{config.run_name}/{config.dataset}/")) > 0:
        last_ckpt = 60 # max(list(map(lambda x: int(x.split("_")[-1][:-3]), os.listdir(f"./models/{config.run_name}/{config.dataset}/"))))
        diffuser.load(f"./models/{config.run_name}/{config.dataset}/", last_ckpt)
        
        # print("INIT-LOADED (NO EMA):")
        # evaluate_data_quality(config, diffuser, use_ema = False, IPC = diffuser.DD_IPC)

    for epoch in progress_bar(range(last_ckpt+1, config.epochs), total=config.epochs - last_ckpt, leave=True):
        print(f"Starting epoch {epoch}:")
        _  = diffuser.one_epoch(epoch=epoch, train=True)
        
        # log predicitons
        if epoch % config.log_every_epoch == 0:
            diffuser.log_images(epoch=epoch)
            diffuser.save_model(epoch=epoch)
            evaluate_data_quality(config, diffuser, use_ema = False, IPC = diffuser.DD_IPC)
            evaluate_data_quality(config, diffuser, use_ema = True, IPC = diffuser.DD_IPC)
else:
    print("Loading pre-trained..")
    diffuser.load(f"./models/{config.run_name}/{config.dataset}/", config.epochs)

print("FINAL (NO EMA):")
evaluate_data_quality(config, diffuser, use_ema = False, IPC = 1)
evaluate_data_quality(config, diffuser, use_ema = False, IPC = 10)
evaluate_data_quality(config, diffuser, use_ema = False, IPC = 50)

print("FINAL (WITH EMA):")
evaluate_data_quality(config, diffuser, use_ema = True, IPC = 1)
evaluate_data_quality(config, diffuser, use_ema = True, IPC = 10)
evaluate_data_quality(config, diffuser, use_ema = True, IPC = 50)
