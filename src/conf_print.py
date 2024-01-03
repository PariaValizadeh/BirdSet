import os
import hydra
from omegaconf import OmegaConf
from src import utils
import pyrootutils


log = utils.get_pylogger(__name__)
#rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base":None,
    #"config_path": "../configs",
    "config_path": str(root / "configs"),
    "config_name": "main.yaml"
}

@utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def main(cfg):
    log.info(f"Parsed Config: \n{OmegaConf.to_yaml(cfg)}")

if __name__ == "__main__":    
    main()