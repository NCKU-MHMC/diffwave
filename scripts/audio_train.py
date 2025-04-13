"""
Train a diffusion model on images.
"""

import argparse
from torchinfo import summary

from guided_diffusion import dist_util, logger
from guided_diffusion.audio_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore

@dataclass
class MySQLConfig:
    host: str = "localhost"
    port: int = 3306

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=MySQLConfig)

@hydra.main(config_path="../configs", config_name="v1", version_base=None)
def main(cfg):
    
    # args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model = hydra.utils.instantiate(cfg.model)
    diffusion = hydra.utils.instantiate(cfg.diffusion)

    model.to(dist_util.dev())
    logger.log(summary(model, input_size=[(1, cfg.max_len), (1,)]))
    schedule_sampler = create_named_schedule_sampler(cfg.schedule_sampler, diffusion)
    logger.log(f"creating data loader from {cfg.dataset.root}...")

    logger.log("creating data loader...")
    dataset = hydra.utils.instantiate(cfg.dataset)
    data = load_data(dataset=dataset, batch_size=cfg.batch_size, max_len=cfg.max_len,
                     num_workers=cfg.num_workers, cond_drop_rate=cfg.cond_drop_rate)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=cfg.batch_size,
        microbatch=cfg.microbatch,
        lr=cfg.lr,
        ema_rate=cfg.ema_rate,
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        resume_checkpoint=cfg.resume_checkpoint,
        use_fp16=cfg.use_fp16,
        fp16_scale_growth=cfg.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=cfg.weight_decay,
        lr_anneal_steps=cfg.lr_anneal_steps,
        sampling_rate=cfg.sampling_rate,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        input_pertub = 0.0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
