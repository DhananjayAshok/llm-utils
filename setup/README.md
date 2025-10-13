This project uses [uv](https://docs.astral.sh/uv/guides/projects/) to manage dependencies, but there is one significant design decision: the 'uv project' is in the `setup` directory, as opposed to root. This means that once you've created the environment, you will find the virtual environment in `setup/.venv/`. 

First, ensure you have installed uv in your python.
```console
pip install --upgrade pip  uv
```

Then, navigate to the setup folder and run `uv sync`:

```console
cd setup
uv sync
```

This will create a virtual environment in setup/.venv. Before running any code in this repo, make sure to run (from root):

```console
source setup/.venv/bin/activate
```

## Installing to alternate locations

If you don't want to have the virtual environment here, but a different location, then first create the venv elsewhere and activate it:
```console
uv venv /path/to/venv --python=3.XX.XX
source /path/to/venv/bin/activate
```
Then, come back to the setup directory and run:
```console
uv sync --active
```

If you do this, remember to add `/path/to/venv/` to the [configs file](../configs/private_vars.yaml)
