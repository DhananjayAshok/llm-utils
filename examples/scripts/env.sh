storage_dir= # whatever you set in configs/private_vars.yaml
if [ -z "$storage_dir" ]; then
  echo "Please set the storage_dir variable in configs/private_vars.yaml"
  exit 1