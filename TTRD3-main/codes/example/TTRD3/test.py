import os
import sys
from models.diffusion.residual_denoising_diffusion_pytorch import (ResidualDiffusion, Trainer, UnetRes, set_seed)
from models.TransferNet.TransNet import TransNet

# init 
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
sys.stdout.flush()
set_seed(10) # 3407
debug = False
if debug:
    save_and_sample_every = 2
    sampling_timesteps = 10
    sampling_timesteps_original_ddim_ddpm = 10
    train_num_steps = 200
else:
    save_and_sample_every = 1000
    if len(sys.argv)>1:
        sampling_timesteps = int(sys.argv[1])
    else:
        sampling_timesteps = 10
    sampling_timesteps_original_ddim_ddpm = 250
    train_num_steps = 100000

folder = ["/home/solid/super_resolution/datasets/AID/AID_64_256_dataset_plus/train_64_256/hr_256",
          "/home/solid/super_resolution/datasets/AID/AID_64_256_dataset_plus/train_64_256/sr_64_256",

          "/home/solid/super_resolution/datasets/AID/AID_64_256_dataset_plus/val_64_256/hr_256",
          "/home/solid/super_resolution/datasets/AID/AID_64_256_dataset_plus/val_64_256/sr_64_256",

          "/home/solid/super_resolution/datasets/AID/AID_64_256_dataset_plus/style_64_256/hr_256",
          "/home/solid/super_resolution/datasets/AID/AID_64_256_dataset_plus/style_64_256/sr_64_256"]


train_batch_size = 1
num_samples = 1
sum_scale = 0.01
image_size = 256


model = UnetRes(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    share_encoder=-1,    # share_encoder = 1时，res和noisy共用一个网络，=0时，分别用两个网络，-1时，用两个网络但只训练第一个网络
    condition=True,
    input_condition=False
)
diffusion = ResidualDiffusion(
    model,
    image_size=image_size,
    timesteps=1000,           # number of steps
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    sampling_timesteps=sampling_timesteps,
    objective='pred_res_noise',
    loss_type='l1',            # L1 or L2
    condition=True,
    sum_scale = sum_scale,
    input_condition=False,
    input_condition_mask=False
)

refnet = TransNet()

trainer = Trainer(
    diffusion,
    folder,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=8e-5,
    train_num_steps=train_num_steps,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="RGB",
    condition=True,
    save_and_sample_every=save_and_sample_every,
    equalizeHist=False,
    crop_patch=False,
    generation=False,
    refnet=refnet
)

total_params_RDDM = sum([param.nelement() for param in model.parameters()])
total_params_refnet = sum([param.nelement() for param in refnet.parameters()])
print(f'RDDM:{total_params_RDDM}, TransNet:{total_params_refnet}, total_params:{total_params_RDDM + total_params_refnet}')


# train
# trainer.train()

# test
if not trainer.accelerator.is_local_main_process:
    pass
else:
    trainer.load(100)
    trainer.set_results_folder('./results/TTRD3_100000_'+str(f'_{sampling_timesteps}'))
    trainer.test(last=True)

# trainer.set_results_folder('./results/test_sample')
# trainer.test(sample=True)