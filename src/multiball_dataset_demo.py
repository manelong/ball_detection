import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from runners import select_runner
from utils import mkdir_if_missing
from datasets import select_dataset
import json
from dataloaders import build_img_transforms, build_seq_transforms
from dataloaders.dataset_loader import ImageMultiballDataset

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name='root', config_path='configs')
def main(
        cfg: DictConfig
):
    # print(OmegaConf.to_yaml(cfg))
    # print(cfg)
    if cfg['output_dir'] is None:
        cfg['output_dir'] = HydraConfig.get().run.dir
    mkdir_if_missing(cfg['output_dir'])

    # 保存本次运行的cfg到cfg['output_dir']
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(cfg['output_dir'] + '/config.json', 'w') as json_file:
        json.dump(config_dict, json_file, indent=4)

    input_wh = (cfg['model']['inp_width'], cfg['model']['inp_height'])
    output_wh = (cfg['model']['out_width'], cfg['model']['out_height'])
    fp1_fpath = None

    dataset = select_dataset(cfg)

    transform_train, transform_test = build_img_transforms(cfg)
    seq_transform_train, seq_transform_test = build_seq_transforms(cfg)

    train_dataset = ImageMultiballDataset(
        cfg,
        dataset=dataset.train,
        input_wh=input_wh,
        output_wh=output_wh,
        transform=transform_train,
        seq_transform=seq_transform_train,
        fp1_fpath=fp1_fpath
    )

    data = train_dataset[0]


if __name__ == "__main__":
    main()

