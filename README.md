# subsidence-monitor
Python package to combine and analyse tectonic, glacio isistatic adjustment and mining-induced components of subsidence.


## Installation (developer)
GeoST uses [Pixi](https://github.com/prefix-dev/pixi) for package management and workflows.

With pixi installed, navigate to the folder of the cloned repository and run the following to install all dependencies and the package itself in editable mode:
```
pixi install
```
See the [Pixi documentation](https://pixi.sh/latest/) for more information. Next open
the Pixi shell by running:
```
pixi shell
```
Finally install the pre-commit hooks that enable automatic checks upon committing changes:
```
pre-commit install
```

## How to use
To run the calculations, navigate in cmd to the submon folder. Then run:

```
pixi shell
```

```
python run.py
```

If you do not give a path to a config file like in the example above, it will by default use ./config/config.toml. If you want to run with a custom config, use:

```
python run.py <path-to-config-file>
```