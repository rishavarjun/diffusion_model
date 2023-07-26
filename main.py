import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from IPython.display import HTML
import torch.nn.functional as F
import argparse
from datetime import datetime
import time

from sampler import sample_ddpm, sample_ddim
from model import ContextUnet
from utils import plot_sample, perturb_input, show_images
from dataset import CustomDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="Mode to run model in", type=str, default="evaluate")
    parser.add_argument('-c', '--context', help='Add context in the model', action='store_true')
    parser.add_argument('--user_defined', help='Add user defined context in the model', action='store_true')
    parser.add_argument('-s', "--sampler", help="Sampler used for model inferencing", type=str, default="ddpm")
    parser.add_argument('-mn', "--model_name", help="name of the model", type=str, default="model_trained.pth")
    parser.add_argument('-n', "--experiment_name", help="experiment name", type=str, default="v1")
    
    parser.add_argument("--timesteps", help="timesteps for generating noise", type=int, default=500)
    parser.add_argument("--beta1", type=float, default=1e-4)
    parser.add_argument("--beta2", type=float, default=0.02)
    parser.add_argument("--n_feature", type=int, default=64)
    parser.add_argument("--n_cfeature", type=int, default=5)
    parser.add_argument("--height", type=int, default=16)

    args = parser.parse_args()

    config = {}
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    print(config)

    # diffusion hyperparameters
    timesteps = 300
    beta1 = 1e-4
    beta2 = 0.02

    # network hyperparameters
    # device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    device = "cuda:6"
    print(f"Device: {device}")
    n_feature = 64 # 64 hidden dimension feature
    n_cfeature = 5 # context vector is of size 5
    height = 16 # 16x16 image
    save_dir = f"./models/{config['experiment_name']}"
    data_dir = './data'
    os.makedirs(save_dir, exist_ok=True)

    # construct DDPM noise schedule as defined in the paper
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    nn_model = ContextUnet(in_channels=3, n_feat=n_feature, n_cfeat=n_cfeature, height=height).to(device)

    if config['mode'] == 'evaluate':
        # load in model weights and set to eval mode
        nn_model.load_state_dict(torch.load(f"{save_dir}/{config['model_name']}", map_location=device))
        nn_model.eval()
        print("Model loaded")

        # visualize samples
        st = time.time()
        plt.clf()

        if config['sampler'] == "ddpm" and config['user_defined']:
            ctx = torch.tensor([
                # human, non-human, food, spell, spell, side-facing
                [1,0,0,0,0], 
                [0,1,0,0,0], 
                [0,0,1,0,0], 
                [0,0,0,1,0], 
                [0,0,0,1,0], 
                [0,0,0,0,1]
            ]).float().to(device)
            samples, intermediate = sample_ddpm(nn_model, ctx.shape[0], ctx, timesteps, height, device, a_t, b_t, ab_t)
            image_name = 'cb5.png'
            show_images(samples, save_dir, image_name)
            
        else:
            if config['sampler'] == "ddpm":
                if config['context']:
                    ctx = F.one_hot(torch.randint(0, 5, (32,)), 5).to(device=device).float()
                    samples, intermediate = sample_ddpm(nn_model, 32, ctx, timesteps, height, device, a_t, b_t, ab_t)

                else:
                    samples, intermediate = sample_ddpm(nn_model, 32, None, timesteps, height, device, a_t, b_t, ab_t)
            
            elif config['sampler'] == "ddim":
                samples, intermediate = sample_ddim(nn_model, 32, timesteps, height, device, ab_t, n=25)
            
            print(f"Time taken for image generation: {time.time() - st}s")
            
            now = datetime.now()
            temp = config['model_name'][:-4] + "_" + str(now.day) + str(now.month) + str(now.year) + "_" + str(now.hour) + str(now.minute) + str(now.second) + "_" + config['sampler']
            animation = plot_sample(intermediate, 32, 4, save_dir, temp, save=True)
            HTML(animation.to_jshtml())  # to display in cell

    elif config['mode'] == 'train':
        nn_model.train()
        batch_size = 64
        n_epoch = 64
        lrate=1e-3

        transform = transforms.Compose([
            transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
            transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
        ])

        if config['context']:
            print("Incorporating context vector")

        # load dataset and construct optimizer
        dataset = CustomDataset(f"{data_dir}/sprites_1788_16x16.npy", f"{data_dir}/sprite_labels_nc_1788_16x16.npy", transform, null_context=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

        start_time = time.time()
        for ep in range(n_epoch):
            print(f'epoch {ep}')

            # linearly decay learning rate
            optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

            pbar = tqdm(dataloader, mininterval=2)
            for x, c in pbar:   # x: images
                optim.zero_grad()
                x = x.to(device)
                if config['context']:
                    c = c.to(x)
                    # randomly mask out c
                    context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
                    c = c * context_mask.unsqueeze(-1)

                # perturb data by choosing a random time step
                noise = torch.randn_like(x)
                t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
                x_pert = perturb_input(x, t, ab_t, noise)

                # use network to recover noise
                if config['context']:
                    pred_noise = nn_model(x_pert, t / timesteps, c=c)
                else:
                    pred_noise = nn_model(x_pert, t / timesteps)

                # loss is mean squared error between the predicted and true noise
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()

                optim.step()

            # save model periodically
            if ep == int(n_epoch-1):
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                torch.save(nn_model.state_dict(), save_dir + f"/model_{ep}.pth")
                print('saved model at ' + save_dir + f"model_{ep}.pth")

        print(f"Total time taken to train the model: {time.time() - start_time}s")
