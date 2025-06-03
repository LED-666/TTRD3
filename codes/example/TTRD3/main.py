# import os
# import sys
# import yaml
# from codes.models.diffusion.residual_denoising_diffusion_pytorch import (ResidualDiffusion, Trainer, UnetRes, set_seed)
# from codes.models.TransferNet.TransNet import TransNet
#
#
# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
#     return config
#
#
# def initialize_models(config):
#     set_seed(config['seed'])
#
#     model = UnetRes(
#         dim=config['unet']['dim'],
#         dim_mults=tuple(config['unet']['dim_mults']),
#         share_encoder=config['unet']['share_encoder'],
#         condition=config['unet']['condition'],
#         input_condition=config['unet']['input_condition']
#     )
#
#     diffusion = ResidualDiffusion(
#         model,
#         image_size=config['diffusion']['image_size'],
#         timesteps=config['diffusion']['timesteps'],
#         sampling_timesteps=config['diffusion']['sampling_timesteps'],
#         objective=config['diffusion']['objective'],
#         loss_type=config['diffusion']['loss_type'],
#         condition=config['diffusion']['condition'],
#         sum_scale=config['diffusion']['sum_scale'],
#         input_condition=config['diffusion']['input_condition'],
#         input_condition_mask=config['diffusion']['input_condition_mask']
#     )
#
#     refnet = TransNet()
#
#     return model, diffusion, refnet
#
#
# def initialize_trainer(config, diffusion, refnet):
#     trainer = Trainer(
#         diffusion,
#         config['data']['folders'],
#         train_batch_size=config['training']['train_batch_size'],
#         num_samples=config['training']['num_samples'],
#         train_lr=config['training']['train_lr'],
#         train_num_steps=config['training']['train_num_steps'],
#         gradient_accumulate_every=config['training']['gradient_accumulate_every'],
#         ema_decay=config['training']['ema_decay'],
#         amp=config['training']['amp'],
#         condition=config['training']['condition'],
#         save_and_sample_every=config['training']['save_and_sample_every'],
#         equalizeHist=config['training']['equalizeHist'],
#         crop_patch=config['training']['crop_patch'],
#         generation=config['training']['generation'],
#         refnet=refnet
#     )
#     return trainer
#
#
# def print_model_params(model, refnet):
#     total_params_RDDM = sum([param.nelement() for param in model.parameters()])
#     total_params_refnet = sum([param.nelement() for param in refnet.parameters()])
#     print(
#         f'RDDM:{total_params_RDDM}, TransNet:{total_params_refnet}, total_params:{total_params_RDDM + total_params_refnet}')
#
#
# def main(config_path):
#     config = load_config(config_path)
#     os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config['cuda_devices'])
#     sys.stdout.flush()
#
#     model, diffusion, refnet = initialize_models(config)
#     trainer = initialize_trainer(config, diffusion, refnet)
#
#     if trainer.accelerator.is_local_main_process:
#         print_model_params(model, refnet)
#
#     if config['mode'] == 'train':
#         if config['training'].get('load_checkpoint'):
#             trainer.load(config['training']['load_checkpoint'])
#         trainer.train()
#
#         if config['training'].get('test_after_train'):
#             trainer.load(trainer.train_num_steps // config['training']['save_and_sample_every'])
#             trainer.set_results_folder(config['testing']['results_folder'])
#             trainer.test(last=True)
#
#     elif config['mode'] == 'test':
#         trainer.load(config['testing']['load_checkpoint'])
#         trainer.set_results_folder(config['testing']['results_folder'])
#         if config['testing'].get('sample'):
#             trainer.test(sample=True)
#         else:
#             trainer.test(last=True)
#
#
# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print("Usage: python main.py  options/TTRD3.yml")
#         sys.exit(1)
#     main(sys.argv[1])

import os
import sys
import yaml
import argparse
from codes.models.diffusion.residual_denoising_diffusion_pytorch import (ResidualDiffusion, Trainer, UnetRes, set_seed)
from codes.models.TransferNet.TransNet import TransNet


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

        # 确保数值参数是正确类型

    def ensure_float(value):
        if isinstance(value, str):
            return float(value.replace('e-', 'e-'))  # 处理科学计数法
        return float(value)

    # 转换关键数值参数
    if 'training' in config:
        config['training']['train_lr'] = ensure_float(config['training']['train_lr'])
    if 'diffusion' in config:
        config['diffusion']['sum_scale'] = ensure_float(config['diffusion']['sum_scale'])
    if 'training' in config and 'ema_decay' in config['training']:
        config['training']['ema_decay'] = ensure_float(config['training']['ema_decay'])

    return config


def initialize_models(config):
    set_seed(config.get('seed', 10))  # 默认seed为10

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
        train_batch_size=int(config['training']['train_batch_size']),
        num_samples=int(config['training']['num_samples']),
        train_lr=float(config['training']['train_lr']),
        train_num_steps=int(config['training']['train_num_steps']),
        gradient_accumulate_every=int(config['training']['gradient_accumulate_every']),
        ema_decay=float(config['training']['ema_decay']),
        amp=bool(config['training']['amp']),
        condition=bool(config['training']['condition']),
        save_and_sample_every=int(config['training']['save_and_sample_every']),
        equalizeHist=bool(config['training']['equalizeHist']),
        crop_patch=bool(config['training']['crop_patch']),
        generation=bool(config['training']['generation']),
        refnet=refnet
    )
    return trainer


def print_model_params(model, refnet):
    total_params_RDDM = sum([param.nelement() for param in model.parameters()])
    total_params_refnet = sum([param.nelement() for param in refnet.parameters()])
    print(
        f'RDDM:{total_params_RDDM}, TransNet:{total_params_refnet}, total_params:{total_params_RDDM + total_params_refnet}')


def main():
    parser = argparse.ArgumentParser(description='Train or test the TTRD3 model')
    parser.add_argument('config', help='Path to config YAML file')
    parser.add_argument('--mode', choices=['train', 'test'], help='Override run mode')
    parser.add_argument('--load-checkpoint', help='Override checkpoint to load')
    parser.add_argument('--results-folder', help='Override results folder path')
    parser.add_argument('--sample', action='store_true', help='Run sample test instead of full test')
    args = parser.parse_args()

    config = load_config(args.config)

    # 覆盖配置参数
    if args.mode:
        config['mode'] = args.mode
    if args.load_checkpoint:
        if config['mode'] == 'train':
            config['training']['load_checkpoint'] = args.load_checkpoint
        else:
            config['testing']['load_checkpoint'] = args.load_checkpoint
    if args.results_folder:
        config['testing']['results_folder'] = args.results_folder
    if args.sample:
        config['testing']['sample'] = True

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.get('cuda_devices', [0]))
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
    main()