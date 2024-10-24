import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from runners import select_runner
from utils import mkdir_if_missing

import json

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
    
    runner = select_runner(cfg)
    runner.run()

if __name__ == "__main__":
    main()

