import os
import numpy as np
from tqdm import tqdm
from PIL import Image as im
from torchvision import transforms
import torch ; torch.manual_seed(42)
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

from utils import get_dataset, get_network #, DiffAugment, ParamDiffAug, epoch, get_time
from plotter_utils import get_combined_image_plot

DATASET = "CIFAR10" # MNIST, CIFAR10, CIFAR100
DEVICE = "cuda:0"
BSZ = 256

def renormalize(images, mean, std):
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return unnormalize(images)
    # return (images + mean) * std

def load_data(dataset):
    channel, shape_img, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(dataset, "./data/")
    
    ''' load data '''
    def get_x_y(dst):
        indices_class = [[] for c in range(num_classes)]
        images_all = [torch.unsqueeze(dst[i][0], dim=0) for i in range(len(dst))]
        labels_all = [dst[i][1] for i in range(len(dst))]
        for i, lab in enumerate(labels_all): indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(DEVICE)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=DEVICE)
        return images_all, labels_all
    
    train_x, train_y = get_x_y(dst_train)
    test_x, test_y = get_x_y(dst_test)

    print('train_x shape: ', train_x.shape)
    # print('train_x mean = [%.4f, %.4f, %.4f]' % (train_x[:, 0].mean(), train_x[:, 1].mean(), train_x[:, 2].mean()))
    # print('train_x std = [%.4f, %.4f, %.4f]' % (torch.std(train_x[:, 0]), torch.std(train_x[:, 1]), torch.std(train_x[:, 2])))
    
    return {
        "train_x": train_x,
        "train_y": train_y,
        
        "test_x": test_x,
        "test_y": test_y,

        "mean": torch.tensor(mean),
        "std": torch.tensor(std),
    }

# Save images as png
if not os.path.exists(f"data/png_datasets/{DATASET}/"):
    data = load_data(DATASET)

    os.makedirs(f"data/png_datasets/{DATASET}/")
    batch_tensor = renormalize(data['train_x'], data['mean'], data['std']).cpu().data.numpy()
    batch_tensor = (batch_tensor * 255).astype('uint8')
    batch_tensor = np.transpose(batch_tensor,(0,2,3,1))
    for index, image in tqdm(enumerate(batch_tensor), desc="saving images"):
        ret_tensor=im.fromarray(image)
        ret_tensor.save(f"data/png_datasets/{DATASET}/{index}.png")

# Diffusion model config
model = Unet(
    dim = 16,
    dim_mults = (1, 2, 4, 8)
).to(DEVICE)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,   # number of steps
    loss_type = 'l2',    # L1 or L2
    p2_loss_weight_gamma = 1.0
).to(DEVICE)

trainer = Trainer(
    diffusion,
    f"./data/png_datasets/{DATASET}/",
    train_batch_size = 1024,
    train_lr = 1e-3,
    train_num_steps = 10000,           # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                      # turn on mixed precision
    save_and_sample_every = 200,
    num_samples = 64,
)

# Train or load        
if not os.path.exists(f"results/{DATASET}/model-50.pt"):
    trainer.train()
else:
    trainer.load(f"results/{DATASET}/", '50')

# Sample some images
sampled_images = diffusion.sample(batch_size = 100)

# Plotting them
fig = get_combined_image_plot(sampled_images.cpu().numpy())
fig.savefig(f"plots/final.png")

# Let's train a convnet on them
