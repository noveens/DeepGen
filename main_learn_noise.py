import numpy as np
import os, copy, logging
from types import SimpleNamespace
from collections import defaultdict
import torch ; torch.manual_seed(42)
from fastprogress import progress_bar
from diffusion_models.ddpm_conditional import Diffusion
from diffusion_models.learn_noise import NoiseLearner

from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam #, DiffAugment, ParamDiffAug, epoch, get_time
from plotter_utils import get_combined_image_plot

def print_and_log(string):
    f = open(LOG_FILE, 'a') ; f.write(string + "\n") ; f.close()
    print(string)

def evaluate_data_quality(config, noise_learner, use_ema = False, IPC = 1, DENOISING_STEPS = 10, epoch=None):
    # Do data distillation evaluation
    model_eval_pool = get_eval_pool('S', 'ConvNet', 'ConvNet')
    accs_all_exps = {}
    for key in model_eval_pool: accs_all_exps[key] = []

    *_, dst_train, dst_test, testloader = get_dataset(config.dataset, config.data_path)

    # train_images = defaultdict(list)
    # for i in range(len(dst_train)): train_images[dst_train[i][1]].append(torch.unsqueeze(dst_train[i][0], dim=0))
    # for i in train_images: train_images[i] = torch.cat(train_images[i])

    test_images, _ = next(iter(testloader))
    fig = get_combined_image_plot(test_images[:20].numpy(), 2)
    fig.savefig(f"results/{config.run_name}/{config.dataset}/test.png")

    print(torch.mean(test_images), torch.std(test_images), test_images.min(), test_images.max())

    for exp in range(5):
        labels = []
        for i in range(config.num_classes): labels += [ i ] * IPC
        
        # # Random sampling baseline
        # distilled_x = []
        # for i in labels: distilled_x.append(train_images[i][np.random.randint(0, len(train_images[i]))].unsqueeze(dim=0))
        # distilled_x = torch.cat(distilled_x)
        # distilled_y = torch.tensor(labels).long().cuda()

        # Sample some images from DDPM and the learned noise
        distilled_x, distilled_y = noise_learner.sample(noise_steps_eval=DENOISING_STEPS, use_ema=use_ema, IPC=IPC, fixed_seed=False)
        if exp == 0: print(torch.mean(distilled_x), torch.std(distilled_x), torch.min(noise_learner.learned_noise), torch.max(noise_learner.learned_noise))

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
    # print()

"""
THINGS I COULD DO:
- (EASY) cross-architecture evaluation
"""

"""
BEST CONFIGS:
- MNIST:
    - IPC = 1: 
        - DENOISING_STEPS = 4, batch_size = 4096, lr = 0.1, noise_clip = 3.0, num_dm_iter = 1 (for specific gen)
        - DENOISING_STEPS = 5, batch_size = 4096, lr = 0.03, noise_clip = 1.5, num_dm_iter = 10 (for diverse gen)
    - IPC = 10: 
        - DENOISING_STEPS = 5, batch_size = 4096, lr = 0.1, noise_clip = 3.0, num_dm_iter = 10 (for specific gen)
"""

# NOTE: Having low values will make generation easier but less diverse
DENOISING_STEPS = 10

config = SimpleNamespace(    
    run_name = "DDPM_conditional_DD_learn_noise",
    dataset = "CIFAR10",
    img_size = 32, # MNIST will be padded to be 32 x 32
    noise_steps=1000, # NOTE: 1000
    noise_steps_eval=DENOISING_STEPS,
    truncation_steps=DENOISING_STEPS, # last 10
    seed = 42,
    batch_size = 4096, # 512,
    num_classes = 10,
    data_path = "./data/",
    device = "cuda",
    slice_size = 1,
    do_validation = True,
    fp16 = False, # True
    log_every_epoch = 10, # 10 # Will save model and images after this many epochs
    num_workers=16,
    # lr = 3e-4,
    lr = 0.01, # 5e-3,
    noise_clip = 1, # NOTE: Having large values will make generation easier but less diverse
    num_dm_iter = 1, # NOTE: Having large will give more stable & diverse results
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
    DD_IPC = 1
)

LOG_FILE = f"./logs/{config.run_name}/{config.dataset}/denoising_{DENOISING_STEPS}_clip_{config.noise_clip}_dm_iter_{config.num_dm_iter}_noise_steps_{config.noise_steps}_lr_{config.lr}_bsz_{config.batch_size}.txt"
os.makedirs(f"./logs/{config.run_name}/{config.dataset}/", exist_ok=True)

# Train or load
print_and_log("Loading pre-trained..")
diffuser.load(f"./models_milestone_2/DDPM_conditional/{config.dataset}/", config.epochs)

noise_learner = NoiseLearner(
    args=config,
    diffuser=diffuser,
    DD_IPC=diffuser.DD_IPC,
    DENOISING_STEPS=DENOISING_STEPS
)

pbar = progress_bar(range(100), total=100, leave=True)
for epoch in pbar:
    # if epoch % 2 == 0: 
    #     noise_learner.log_images(epoch=epoch)

    # log predicitons
    if epoch % config.log_every_epoch == 0:
        noise_learner.log_images(epoch=epoch)
        noise_learner.save_model(epoch=epoch)
        # evaluate_data_quality(config, noise_learner, use_ema = False, IPC = diffuser.DD_IPC, DENOISING_STEPS=DENOISING_STEPS)
        # evaluate_data_quality(config, noise_learner, use_ema = True, IPC = diffuser.DD_IPC, DENOISING_STEPS=DENOISING_STEPS, epoch=epoch)
        evaluate_data_quality(config, noise_learner, use_ema = True, IPC = diffuser.DD_IPC, DENOISING_STEPS=200, epoch=epoch)

    avg_loss = noise_learner.one_epoch(epoch=epoch)
    pbar.comment = f"MSE={np.mean(avg_loss):2.3f}"
        
print_and_log("FINAL (NO EMA):")
for ipc in [ 1, 10, 50 ]:
    # evaluate_data_quality(config, noise_learner, use_ema = False, IPC = ipc, DENOISING_STEPS=DENOISING_STEPS, epoch='final')
    evaluate_data_quality(config, noise_learner, use_ema = False, IPC = ipc, DENOISING_STEPS=200, epoch='final')

print_and_log("FINAL (WITH EMA):")
for ipc in [ 1, 10, 50 ]:
    # evaluate_data_quality(config, noise_learner, use_ema = True, IPC = ipc, DENOISING_STEPS=DENOISING_STEPS, epoch='final')
    evaluate_data_quality(config, noise_learner, use_ema = True, IPC = ipc, DENOISING_STEPS=200, epoch='final')
