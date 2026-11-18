# Per-rank stream routing for benchmark runs; official entry and loop retained.
import os, sys
from pathlib import Path
if os.environ.get('FASTWAM_RUN_DIR'):
    out=Path(os.environ['FASTWAM_RUN_DIR']);out.mkdir(parents=True,exist_ok=True)
    stream=open(out/('rank'+os.environ.get('RANK','0')+'.log'),'a',buffering=1)
    os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
import hydra
from omegaconf import DictConfig

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
