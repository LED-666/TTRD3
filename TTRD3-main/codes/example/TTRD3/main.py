import os
import sys
import yaml
from codes.models.diffusion.residual_denoising_diffusion_pytorch import (ResidualDiffusion, Trainer, UnetRes, set_seed)
from codes.models.TransferNet.TransNet import TransNet


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def initialize_models(config):
    set_seed(config['seed'])

    model = UnetRes(
        dim=config['unet']['dim'],
        dim_mults=tuple(config['unet']['dim_mults']),
        share_encoder=config['unet']['share_encoder'],
        condition=config['unet']['condition'],
        input_condition=config['unet']['input_condition']
    )

    diffusion = ResidualDiffusion(
        model,
        image_size=config['diffusion']['image_size'],
        timesteps=config['diffusion']['timesteps'],
        sampling_timesteps=config['diffusion']['sampling_timesteps'],
        objective=config['diffusion']['objective'],
        loss_type=config['diffusion']['loss_type'],
        condition=config['diffusion']['condition'],
        sum_scale=config['diffusion']['sum_scale'],
        input_condition=config['diffusion']['input_condition'],
        input_condition_mask=config['diffusion']['input_condition_mask']
    )

    refnet = TransNet()

    return model, diffusion, refnet


def initialize_trainer(config, diffusion, refnet):
    trainer = Trainer(
        diffusion,
        config['data']['folders'],
        train_batch_size=config['training']['train_batch_size'],
        num_samples=config['training']['num_samples'],
        train_lr=config['training']['train_lr'],
        train_num_steps=config['training']['train_num_steps'],
        gradient_accumulate_every=config['training']['gradient_accumulate_every'],
        ema_decay=config['training']['ema_decay'],
        amp=config['training']['amp'],
        condition=config['training']['condition'],
        save_and_sample_every=config['training']['save_and_sample_every'],
        equalizeHist=config['training']['equalizeHist'],
        crop_patch=config['training']['crop_patch'],
        generation=config['training']['generation'],
        refnet=refnet
    )
    return trainer


def print_model_params(model, refnet):
    total_params_RDDM = sum([param.nelement() for param in model.parameters()])
    total_params_refnet = sum([param.nelement() for param in refnet.parameters()])
    print(
        f'RDDM:{total_params_RDDM}, TransNet:{total_params_refnet}, total_params:{total_params_RDDM + total_params_refnet}')


def main(config_path):
    config = load_config(config_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config['cuda_devices'])
    sys.stdout.flush()

    model, diffusion, refnet = initialize_models(config)
    trainer = initialize_trainer(config, diffusion, refnet)

    if trainer.accelerator.is_local_main_process:
        print_model_params(model, refnet)

    if config['mode'] == 'train':
        if config['training'].get('load_checkpoint'):
            trainer.load(config['training']['load_checkpoint'])
        trainer.train()

        if config['training'].get('test_after_train'):
            trainer.load(trainer.train_num_steps // config['training']['save_and_sample_every'])
            trainer.set_results_folder(config['testing']['results_folder'])
            trainer.test(last=True)

    elif config['mode'] == 'test':
        trainer.load(config['testing']['load_checkpoint'])
        trainer.set_results_folder(config['testing']['results_folder'])
        if config['testing'].get('sample'):
            trainer.test(sample=True)
        else:
            trainer.test(last=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py  config.yaml")
        sys.exit(1)
    main(sys.argv[1])