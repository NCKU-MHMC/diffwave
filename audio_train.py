"""
Train a diffusion model on images.
"""

import argparse
from torchinfo import summary
from guided_diffusion import dist_util, logger
from guided_diffusion.audio_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
# from guided_diffusion.script_util import (
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     args_to_dict,
#     add_dict_to_argparser,
# )
from guided_diffusion.train_util import TrainLoop

from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore

from typing import Union, Literal, Optional

@dataclass
class ModelConfig:
    _target_: str = "guided_diffusion.unet.UNetModel"

@dataclass
class DiffusionConfig:
    _target_: str = "guided_diffusion.gaussian_diffusion.GaussianDiffusion"

@dataclass
class DatasetConfig:
    _target_: str = "guided_diffusion.audio_datasets.LibriTTSDataset"
    # root: Optional[str] = ""

@dataclass
class TrainerConfig:
    model: ModelConfig
    diffusion: DiffusionConfig
    dataset: DatasetConfig

    schedule_sampler: str = "loss-second-moment" # Union[Literal["loss-second-moment"], Literal["uniform"]]
    # schedule_sampler: "uniform"
    output_dir: str = "results"
    use_ema: bool = False
    segment_size: Optional[int] = None
    sampling_rate: int = 16000
    device: str = "cuda"

    batch_size: int = 32
    # max_len: 80000
    max_len: int = 80000
    deterministic: bool = False
    num_workers: int = 8
    microbatch: int = -1 # -1 disables microbatches

    data_std: float = 0.15

    lr: float =  0.0001
    ema_rate: str = "0.9999"  # comma-separated list of EMA values
    weight_decay: float = 0.
    lr_anneal_steps: int = 0
    log_interval: int = 100
    save_interval: int = 10000
    resume_checkpoint: str = ".experiments/libritts/model050000.pt" # ".experiments/v24/model100000.pt"
    use_fp16: bool = True
    fp16_scale_growth: float = 0.001
    input_pertub: float = 0. # 0.15
    cond_drop_rate: float = 0.5

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=TrainerConfig)

@hydra.main(config_path="./configs", config_name="v1", version_base=None)
def main(cfg: TrainerConfig):
    
    # args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model = hydra.utils.instantiate(cfg.model)
    diffusion = hydra.utils.instantiate(cfg.diffusion)

    model.to(dist_util.dev())
    # logger.log(summary(model, input_size=[(1, cfg.max_len), (1,)]))
    schedule_sampler = create_named_schedule_sampler(cfg.schedule_sampler, diffusion)
    # logger.log(f"creating data loader from {cfg.dataset.root}...")

    logger.log("creating data loader...")
    dataset = hydra.utils.instantiate(cfg.dataset)
    data = load_data(dataset=dataset,
                     batch_size=cfg.batch_size,
                     max_len=cfg.max_len,
                     num_workers=cfg.num_workers,
                     cond_drop_rate=cfg.cond_drop_rate,
                     deterministic=cfg.deterministic,
                     data_std=cfg.data_std)

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
        data_std=cfg.data_std,
    ).run_loop()

if __name__ == "__main__":
    main()
