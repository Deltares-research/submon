import tomllib
from pathlib import Path

from submon.io import export, read

if __name__ == "__main__":
    config_path = Path(
        r"c:\Users\onselen\Development\submon\config\example_config.toml"
    )
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    data = read.load_subsidence_rasters(config)
    data
