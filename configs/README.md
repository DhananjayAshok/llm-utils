# Variables

For this project, there is only one mandatory variable. Make sure to set storage_dir in [the config file](private_vars.yaml) to a directory where you want to store the data and models. This directory will be the default used for all data storage, including training data, model checkpoints, and logs.

## Running example code and tests
If you want to run the examples or tests, you'll also need to export these variables to bash. Run:
```bash
python configs/create_env_file.py
```

This will write a `config.env` file to the `config` directory. Sourcing this env file will give you access to all the variables in bash scripts. This *will* overwrite existing variables with the same name. If you have a problem with it and your name rhymes with Bonson Ten, then write a pull request instead of complaining 🧐
