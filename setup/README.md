This project uses Python 3.12 with [uv](https://docs.astral.sh/uv/guides/projects/) to manage dependencies, but there is one significant design decision: the 'uv project' is in the `setup` directory, as opposed to root. This means that once you've created the environment, you will find the virtual environment in `setup/.venv/`. 

First, ensure you have installed uv in your python.
```console
pip install --upgrade pip  uv
```

## Pre-existing Installation / Installing to alternate locations

Next, decide where you want to store the virtual environment. If you already have this repo set up elsewhere, you might not want to have separate environments for every instance of it. Some users may *not* want to install the environment into `setup/.venv/` (perhaps the filesystem space is limited and you want the env to be elsewhere). If you're fine saving in `setup/.venv/` directly, skip to the next section. 

If you have a pre-existing installation somewhere, create a symbolic link to it in the `setup/.venv` directory:
```console
# Run this in root directory of the project
ln -s /path/to/existing/venv setup/.venv 
```

If you want to install to an alternate location, first create the environment elsewhere and then create a symbolic link to the `setup/.venv` directory:
```console
# Run this in root directory of the project
uv venv /path/to/venv --python=3.12
ln -s /path/to/venv setup/.venv 
```
If you are doing this, make sure to set the UV_CACHE environment variable to the same filesystem as the environment.

## Installation
Finally, navigate to the setup folder and run `uv sync`:

```console
cd setup
uv sync
```

This will create a virtual environment in setup/.venv. Before running any code in this repo, make sure to run (from root):

```console
source setup/.venv/bin/activate
```
