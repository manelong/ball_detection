import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from .train_and_test import Trainer
from .train_multiball import TrainerMultiball
from .eval import VideosInferenceRunner
from .extract_frame import ExtractFrameRunner
from .eval_frame import EvalFrameRunner

log = logging.getLogger(__name__)

__runner_factory = {
    'train_multiball': TrainerMultiball,
    'train': Trainer,
    'eval': VideosInferenceRunner,
    'extract_frame': ExtractFrameRunner,
    'eval_frame': EvalFrameRunner,
        }

def select_runner(
        cfg: DictConfig,
):
    runner_name = cfg['runner']['name']
    if not runner_name in __runner_factory.keys():
        raise KeyError('unknown runner: {}'.format(runner_name))
    return __runner_factory[runner_name](cfg)

