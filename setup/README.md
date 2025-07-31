This project uses Python 3.12

You can set up a fresh virtual environment using either the [short](requirements.txt) or [more complete](long_requirements.txt) freezes of our pip using:

```console
pip install -u pip uv
uv venv llm-env --python 3.12
source llm-env/bin/activate
```

Then, install the requirements:

```console
uv pip install -r setup/requirements.txt
```

To be honest, you can probably get away with just installing the latest version of most of these, but be careful with TRL, that ones p weird. 
